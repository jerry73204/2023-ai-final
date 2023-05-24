import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2 as cv
import json5
import torch
from serde import from_dict, serde
from torch.utils.data import DataLoader, Dataset


def load_data(
    data_dir: Path,
    batch_size,
    piece_size,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = JigsawDataset(data_dir, piece_size)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
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
    def __init__(self, data_dir: Path, piece_size: int):
        self.piece_images, self.positions = self.load_data_dir(data_dir, piece_size)
        self.n_samples = len(self.positions)

    def __getitem__(self, index):
        position = self.positions[index]
        piece_image = self.piece_images[index]
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
        piece_images, positions = zip(*samples)
        return piece_images, positions

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
            return torch.from_numpy(piece_image).reshape(1, piece_size, piece_size)

        piece_images = list(load_image(piece_idx) for piece_idx in range(n_pieces))
        piece_tensor = torch.stack(piece_images, 0)

        # Load the solution
        label_path = puzzle_dir / "label.csv"
        with open(label_path, "r") as fp:
            rows = csv.reader(fp)

            label_list = list(list(float(val) for val in row) for row in rows)
            assert len(label_list) == n_pieces
            label_tensor = torch.FloatTensor(label_list)

        return piece_tensor, label_tensor
