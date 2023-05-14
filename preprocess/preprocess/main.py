import cv2 as cv
import csv
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from dataclasses import dataclass


@dataclass
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def width(self):
        return self.x_max - self.x_min

    def height(self):
        return self.y_max - self.y_min

    def x_med(self):
        return self.x_min + (self.width() + 1) // 2

    def y_med(self):
        return self.y_min + (self.height() + 1) // 2


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT_DIR")
    parser.add_argument("OUTPUT_DIR")
    parser.add_argument("--piece-size", default=64)
    parser.add_argument("--contour-size-thresh", default=300)
    args = parser.parse_args()

    input_dir = Path(args.INPUT_DIR)
    output_dir = Path(args.OUTPUT_DIR)
    output_dir.mkdir()

    for path in input_dir.iterdir():
        generate_puzzle(path, output_dir, args.piece_size, args.contour_size_thresh)


def generate_puzzle(
    input_file: Path, output_dir: Path, piece_size: int, contour_size_thresh: int
):
    # Load image file
    image = cv.imread(str(input_file), cv.IMREAD_GRAYSCALE)

    # Thresholding
    ret, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Remove noise points
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    # Detect contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bboxes = list(bbox_of_contour(cnt) for cnt in contours)

    # Remove small contours
    def is_bbox_large_enough(bbox: BBox):
        return (
            bbox.width() >= contour_size_thresh and bbox.height() >= contour_size_thresh
        )

    pairs = zip(contours, bboxes)
    pairs = filter(lambda args: is_bbox_large_enough(args[1]), pairs)
    pairs = list(pairs)

    max_width = max(bbox.width() for _, bbox in pairs)
    max_height = max(bbox.height() for _, bbox in pairs)
    max_size = max(max_width, max_height)
    scale = piece_size / max_size

    # Re-locate contour coordinates
    contours = list(
        relocate_contour(cnt, bbox, scale, piece_size) for cnt, bbox in pairs
    )

    # Generate puzzle piece images
    dir_name = input_file.stem
    sub_dir = output_dir / dir_name
    sub_dir.mkdir()

    for index, cnt in enumerate(contours):
        # Create a piece image
        piece_image = generate_piece(cnt, piece_size)
        image_file_name = f"{index:02}.png"
        image_output_file = sub_dir / image_file_name
        cv.imwrite(str(image_output_file), piece_image)

        # Write point coordinates to a CSV file
        csv_file_name = f"{index:02}.csv"
        csv_output_file = sub_dir / csv_file_name

        with open(csv_output_file, "w") as fp:
            csv_writer = csv.writer(fp)

            n_pts = cnt.shape[0]
            cnt = cnt.reshape(n_pts, 2)  # Nx1x2 -> Nx2
            for pt in cnt:
                x = pt[0]
                y = pt[1]
                csv_writer.writerow([x, y])


def generate_piece(cnt, piece_size: int):
    canvas = np.zeros((piece_size, piece_size, 1), dtype=np.uint8)
    canvas = cv.drawContours(canvas, [cnt], -1, 255, cv.FILLED)
    return canvas


def relocate_contour(cnt, bbox: BBox, scale: float, piece_size: int):
    n_pts = cnt.shape[0]
    cnt = cnt.reshape(n_pts, 2)  # Nx1x2 -> Nx2

    def relocate(pt):
        x = round((pt[0] - bbox.x_med()) * scale + piece_size / 2.0)
        y = round((pt[1] - bbox.y_med()) * scale + piece_size / 2.0)
        return np.array([x, y])

    cnt = np.stack(list(relocate(pt) for pt in cnt))
    cnt.reshape(n_pts, 1, 2)
    return cnt


def bbox_of_contour(cnt) -> BBox:
    n_pts = cnt.shape[0]
    cnt = cnt.reshape(n_pts, 2)  # Nx1x2 -> Nx2

    x_min = min(pt[0] for pt in cnt)
    x_max = max(pt[0] for pt in cnt)
    y_min = min(pt[1] for pt in cnt)
    y_max = max(pt[1] for pt in cnt)

    return BBox(x_min, y_min, x_max, y_max)


if __name__ == "__main__":
    main()
