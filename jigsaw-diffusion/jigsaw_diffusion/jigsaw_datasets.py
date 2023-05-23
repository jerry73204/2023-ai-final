from pathlib import Path
import json5
import torch
import cv2 as cv
import csv
import os
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List
from serde import serde, from_dict


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
    def __init__(self, data_dir, piece_size: int):
        self.x, self.y = self.load_data_dir(data_dir, piece_size)
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def load_data_dir(self, data_dir: Path, piece_size: int):
        # Load manifest.json5
        manifest_path = os.path.join(data_dir, "manifest.json5")
        with open(manifest_path, "r") as fp:
            manifest = from_dict(Manifest, json5.load(fp))
        samples = list(
            self.load_puzzle_dir(data_dir, puzzle_info, piece_size)
            for puzzle_info in manifest.puzzles
        )
        images, labels = zip(*samples)

        return images, labels

    def load_puzzle_dir(self, data_dir: Path, puzzle_info: PuzzleInfo, piece_size: int):
        puzzle_name = puzzle_info.name
        n_pieces = puzzle_info.n_pieces
        n_pieces = 25
        puzzle_dir = os.path.join(data_dir, puzzle_name)
        assert n_pieces >= 1 and n_pieces <= 99

        # Load piece images
        def load_image(piece_idx: int):
            piece_path = os.path.join(puzzle_dir, "{:02}.png".format(piece_idx))
            piece_image = cv.imread(piece_path, cv.IMREAD_GRAYSCALE)
            assert piece_image.shape == (piece_size, piece_size)
            return torch.from_numpy(piece_image)

        piece_images = list(load_image(piece_idx) for piece_idx in range(n_pieces))
        piece_tensor = torch.stack(piece_images, 0)

        # Load the solution
        label_path = os.path.join(puzzle_dir, "label.csv")
        with open(label_path, "r") as fp:
            rows = csv.reader(fp)
            label = []
            rx = 0
            for r in rows:
                rx += 1
                if rx > n_pieces:
                    break
                nr = [float(sr) for sr in r[0].split("\t")]
                label.append(nr)
            assert len(label) == n_pieces
            label = torch.FloatTensor(label)
        return piece_tensor, label
