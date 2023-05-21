from pathlib import Path
import json5
import torch
import cv2 as cv
import csv


def load_data(data_dir: Path, piece_size: int):
    # Load manifest.json5
    manifest_path = data_dir.join("manifest.json5")
    with open(manifest_path, "r") as fp:
        manifest = json5.load(fp)

    samples = list(
        load_puzzle_dir(data_dir, puzzle_info) for puzzle_info in manifest["puzzles"]
    )
    images, labels = zip(*samples)

    return images, labels


def load_puzzle_dir(data_dir: Path, puzzle_info):
    puzzle_name = puzzle_info["name"]
    n_pieces = puzzle_info["n_pieces"]
    puzzle_dir = data_dir.join(puzzle_name)
    assert n_pieces >= 1 and n_pieces <= 99

    # Load piece images
    def load_image(piece_idx: int):
        piece_path = puzzle_dir.join("{:02}.jpg".format(piece_idx))
        piece_image = cv.imread(piece_path)
        assert piece_image.shape == [1, 64, 64]
        return torch.from_numpy(piece_image)

    piece_images = list(load_image(piece_idx) for piece_idx in range(n_pieces))
    piece_tensor = torch.stack(piece_images, 0)

    # Load the solution
    label_path = puzzle_dir.join("label.csv")

    with open(label_path, "r") as fp:
        label = list([float(x), float(y)] for x, y in csv.reader(fp))
        assert len(label) == n_pieces
        label = torch.FloatTensor(label)

    return piece_tensor, label
