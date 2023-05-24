import csv
import json
import math
import random
import shutil
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from subprocess import Popen
from typing import List

import cv2 as cv
import json5
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("OUTPUT_DIR")
    parser.add_argument("-s", "--piece_image_size", type=int, default=64)
    return parser.parse_args()


def check_piecemaker_output(input_dir: Path, image_size: int) -> bool:
    size_100_dir = input_dir / "size-100"
    pieces_path = size_100_dir / "pieces.json"

    with open(pieces_path, "r") as fp:
        piece_names = list(json.load(fp).keys())

    raster_dir = size_100_dir / "raster"

    for pid in piece_names:
        image_path = raster_dir / "{}.png".format(int(pid))
        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        h, w, _ = image.shape
        if h >= image_size or w >= image_size:
            return False

    return True


def main():
    args = parse_args()
    image_size = args.piece_image_size
    output_dir = Path(args.OUTPUT_DIR)
    output_dir.mkdir()

    with tempfile.TemporaryDirectory() as input_dir:
        input_dir = Path(input_dir)

        # Generate a puzzle using `piecemaker`
        background_size = image_size * 3
        white_image = np.full([background_size, background_size, 3], 255)
        white_image_path = input_dir / "white.png"

        is_successful = False
        for _ in range(10):
            cv.imwrite(str(white_image_path), white_image)
            proc = Popen(
                [
                    "piecemaker",
                    "--dir",
                    str(input_dir),
                    "-n",
                    "25",
                    str(white_image_path),
                ]
            )
            proc.wait(10)

            if check_piecemaker_output(input_dir, image_size):
                is_successful = True
                break
            else:
                print("Some pieces are too large. Retrying...", file=sys.stderr)
                shutil.rmtree(input_dir)
                input_dir.mkdir()

        if not is_successful:
            print("Unable to run `piecemaker`. Is it installed?", file=sys.stderr)
            return 1

        size_100_dir = input_dir / "size-100"

        # Load data generated from piecemaker
        pieces_path = size_100_dir / "pieces.json"
        with open(pieces_path, "r") as fp:
            pieces = list(json.load(fp).items())
            pieces.sort(key=lambda args: int(args[0]))

        in_adjacent_path = input_dir / "adjacent.json"
        with open(in_adjacent_path, "r") as fp:
            adjacent = json.load(fp)

        # Write label.csv
        def bbox_center(bbox: List[int]):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            center = [cx, cy]
            return center

        centers = (bbox_center(bbox) for pid, bbox in pieces)
        label_path = output_dir / "label.csv"
        with open(label_path, "w") as fp:
            writer = csv.writer(fp)

            for center in centers:
                writer.writerow(center)

        # Write adjacent.csv
        out_adjacent_path = output_dir / "adjacent.json5"
        with open(out_adjacent_path, "w") as fp:
            json5.dump(adjacent, fp)

        # Write images
        raster_dir = size_100_dir / "raster"
        for pid, _ in pieces:
            in_image_path = raster_dir / "{}.png".format(int(pid))
            orig_image = cv.imread(str(in_image_path), cv.IMREAD_UNCHANGED)
            new_image = process_piece_image(orig_image, image_size)
            out_image_path = output_dir / "{:02}.png".format(int(pid))
            cv.imwrite(str(out_image_path), new_image)

        # Generate the reassembled puzzle image
        reassembled_image = reassemble_puzzle(pieces, output_dir, image_size * 5)
        reassemble_image_path = output_dir / "reassemble.png"
        cv.imwrite(str(reassemble_image_path), reassembled_image)


def process_piece_image(orig_image, image_size: int):
    sh, sw, sc = orig_image.shape
    assert sc == 4
    pad_t = (image_size - sh) // 2
    pad_b = image_size - sh - pad_t
    pad_l = (image_size - sw) // 2
    pad_r = image_size - sw - pad_l

    image = cv.copyMakeBorder(
        orig_image,
        pad_t,
        pad_b,
        pad_l,
        pad_r,
        cv.BORDER_CONSTANT,
        None,
        [0, 0, 0, 0],
    )

    image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    return image


def reassemble_puzzle(pieces, output_dir: Path, output_size: int):
    # Load piece images
    def load_image(pid):
        image_path = output_dir / "{:02}.png".format(int(pid))
        return cv.imread(str(image_path))

    images = list(load_image(pid) for pid, bbox in pieces)

    # Load center coordinates
    label_path = output_dir / "label.csv"
    with open(label_path, "r") as fp:
        centers = list((float(x), float(y)) for x, y in csv.reader(fp))

    canvas = np.full([output_size, output_size, 3], 0, dtype=np.uint8)

    for image, center in zip(images, centers):
        mask = image != 0

        cx, cy = center
        h, w, _ = image.shape

        y1 = int(math.floor(cy - h / 2))
        y2 = y1 + h
        y3 = max(y1, 0)
        y4 = min(y2, output_size)

        x1 = int(math.floor(cx - w / 2))
        x2 = x1 + w
        x3 = max(x1, 0)
        x4 = min(x2, output_size)

        sub_canvas = canvas[y3:y4, x3:x4]
        sub_mask = mask[(y3 - y1) : (y4 - y1), (x3 - x1) : (x4 - x1)]
        sub_canvas[sub_mask] = random.randint(128, 255)

    return canvas
