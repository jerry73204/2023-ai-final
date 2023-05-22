from pathlib import Path
import json5
import torch
import cv2 as cv
import csv
import os
import numpy
from torch.utils.data import Dataset
class JigsawDataset(Dataset):
    def __init__(self, data_dir,batch_size,piece_size):
        self.x, self.y = self.load_data(data_dir,batch_size, piece_size)
        self.n_samples = len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples

    def load_data(self,data_dir: Path, batch_size, piece_size: int):
            # Load manifest.json5
        manifest_path = os.path.join(data_dir, "manifest.json5")
        with open(manifest_path, "r") as fp:
            manifest = json5.load(fp)
        samples = list(
            self.load_puzzle_dir(data_dir, puzzle_info) for puzzle_info in manifest["puzzles"]
        )
        # print(len(samples))
        # print(samples[0][0].shape, samples[0][1].shape)
        # print(samples[1][0].shape, samples[1][1].shape)
        images, labels = zip(*samples)
        
        # print(images[0].shape, images[1].shape)
        # print(labels[0].shape, labels[1].shape)
        return images, labels

    def load_puzzle_dir(self,data_dir: Path, puzzle_info):
        puzzle_name = puzzle_info["name"]
        n_pieces = puzzle_info["n_pieces"]
        n_pieces = 25
        puzzle_dir = os.path.join(data_dir, puzzle_name)
        assert n_pieces >= 1 and n_pieces <= 99

        # Load piece images
        def load_image(piece_idx: int):
            piece_path = os.path.join(puzzle_dir, "{:02}.png".format(piece_idx))
            piece_image = cv.imread(piece_path)
            # print(piece_image.shape)
            # assert piece_image.shape == [1, 64, 64]
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
                if (rx > n_pieces):
                    break
                nr = [float(sr) for sr in r[0].split('\t')]
                label.append(nr)
            assert len(label) == n_pieces
            label = torch.FloatTensor(label)
        return piece_tensor, label
