import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List
import random
from itertools import chain
import math

import numpy as np
import cv2 as cv
import json5
import torch
from serde import from_dict, serde
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as VF
import torch.nn.functional as F


def load_data(
    data_dir: Path,
    batch_size: int,
    piece_size: int,
    puzzle_size: int,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = JigsawDataset(
        data_dir=data_dir,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        augment=not deterministic,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

    while True:
        yield from loader


@serde
@dataclass
class PuzzleInfo:
    name: str
    n_pieces: int


@serde
@dataclass
class Manifest:
    puzzles: List[PuzzleInfo]


class JigsawDataset(Dataset):
    position: List[torch.FloatTensor]
    piece_images: List[torch.FloatTensor]
    adjacent_map: List[List[List[int]]]
    n_samples: int
    augment: bool
    piece_size: int
    puzzle_size: int

    def __init__(
        self, data_dir: Path, piece_size: int, puzzle_size: int, augment: bool
    ):
        self.piece_images, self.positions, self.adjacent_maps = self.load_data_dir(
            data_dir, piece_size
        )
        self.n_samples = len(self.positions)
        self.augment = augment
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

    def __getitem__(self, index):
        with torch.no_grad():
            position = self.positions[index]
            piece_image = self.piece_images[index]
            adjacent_map = self.adjacent_maps[index]

            if self.augment:
                position, piece_image = self.augment_sample(
                    position, piece_image, adjacent_map
                )

            cond = {"pieces": piece_image}
            return position, cond

    def __len__(self):
        return self.n_samples

    def load_data_dir(self, data_dir: Path, piece_size: int):
        # Load manifest.json5
        manifest_path = data_dir / "manifest.json5"
        with open(manifest_path, "r") as fp:
            manifest = from_dict(Manifest, json5.load(fp))
        samples = list(
            self.load_puzzle_dir(data_dir, puzzle_info, piece_size)
            for puzzle_info in manifest.puzzles
        )
        piece_images, positions, adjacent_maps = zip(*samples)
        return piece_images, positions, adjacent_maps

    def load_puzzle_dir(self, data_dir: Path, puzzle_info: PuzzleInfo, piece_size: int):
        puzzle_name = puzzle_info.name
        n_pieces = puzzle_info.n_pieces
        n_pieces = 25
        puzzle_dir = data_dir / puzzle_name
        assert n_pieces >= 1 and n_pieces <= 99

        # Load piece images
        def load_image(piece_idx: int):
            piece_path = puzzle_dir / "{:02}.png".format(piece_idx)
            piece_image = cv.imread(str(piece_path), cv.IMREAD_GRAYSCALE)
            assert piece_image.shape == (piece_size, piece_size)
            return (
                torch.from_numpy(piece_image).reshape(1, piece_size, piece_size) / 255.0
                - 0.5
            )

        piece_images = list(load_image(piece_idx) for piece_idx in range(n_pieces))
        piece_tensor = torch.stack(piece_images, 0)

        # Load the solution
        label_path = puzzle_dir / "label.csv"
        with open(label_path, "r") as fp:
            rows = csv.reader(fp)

            label_list = list(list(float(val) for val in row) for row in rows)
            assert len(label_list) == n_pieces
            label_tensor = torch.FloatTensor(label_list)

        # Load adjecnt.json5
        adjacent_path = puzzle_dir / "adjacent.json5"
        with open(adjacent_path, "r") as fp:
            adjacent_map = json5.load(fp)
            adjacent_map = list(
                list(int(neighbor) for neighbor in adjacent_map[str(idx)])
                for idx in range(n_pieces)
            )

        return piece_tensor, label_tensor, adjacent_map

    def augment_sample(
        self,
        position: torch.FloatTensor,
        piece_image: torch.FloatTensor,
        adjacent_map: List[List[int]],
    ):
        pi_2 = math.pi * 2
        n_pieces = piece_image.shape[0]

        # 2/3 chance to sample a subset of pieces
        # if random.random() < 2 / 3:
        if random.random() < 2 / 3:
            pivot_idx = random.randrange(n_pieces)
            selected_set = set([pivot_idx])
            selected_set.update(adjacent_map[pivot_idx])

            # 1/2 chance to sample up to 2 levels of neighbors
            if random.random() < 0.5:
                neighbors_of_neighbors = list(
                    chain(*(adjacent_map[idx] for idx in selected_set))
                )
                selected_set.update(neighbors_of_neighbors)

            non_selected_set = list(
                filter(lambda idx: idx not in selected_set, range(n_pieces))
            )

            # Erase non-selected piece images
            xy = position[non_selected_set, :2]
            xy = (
                torch.rand(xy.shape) * (self.puzzle_size - self.piece_size)
                + self.piece_size / 2.0
            )

            rot = position[non_selected_set, 2:3]
            rot = torch.rand(rot.shape) * pi_2

            piece_image[non_selected_set] = -0.5

        # Random per-piece rotation
        puzzle_angle_rad = random.random() * pi_2
        piece_angle_rad = np.random.rand(n_pieces) * pi_2
        total_angle_rad = (puzzle_angle_rad + piece_angle_rad) % pi_2

        puzzle_angle_deg = puzzle_angle_rad * 180 / math.pi
        total_angle_deg = total_angle_rad * 180.0 / math.pi

        position[:, 2] = (position[:, 2] + pi_2 - piece_angle_rad) % pi_2

        for idx in range(n_pieces):
            piece_image[idx] = VF.rotate(
                piece_image[idx], total_angle_deg[idx], fill=-0.5
            )

        # Random global translation
        x = position[:, 0]
        min_x = min(x)
        max_x = max(x)
        mid_x = (min_x + max_x) / 2
        shift_x = self.puzzle_size / 2 - mid_x

        y = position[:, 1]
        min_y = min(y)
        max_y = max(y)
        mid_y = (min_y + max_y) / 2
        shift_y = self.puzzle_size / 2 - mid_y

        translation = (
            torch.FloatTensor([[shift_x, shift_y]])
            + (torch.rand(1, 2) - 0.5) * self.piece_size
        )
        position[:, :2] += translation

        # Random global rotation
        center = (self.puzzle_size / 2, self.puzzle_size / 2)
        rot_mat = torch.from_numpy(
            cv.getRotationMatrix2D(center, puzzle_angle_deg, 1.0).astype(np.float32)
        )
        xy1 = F.pad(position[:, :2], pad=(0, 1, 0, 0), mode="constant", value=1.0)
        position[:, :2] = torch.matmul(xy1, rot_mat.transpose(0, 1))

        return position, piece_image
