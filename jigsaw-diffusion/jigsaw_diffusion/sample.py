"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from itertools import chain
from pathlib import Path
import csv

import torch as th
import torch.distributed as dist

from .jigsaw_datasets import load_data
from .jigsaw_script_util import reassemble_puzzle
import cv2 as cv  ## ADDED


from . import dist_util, logger

# model_and_diffusion_defaults,
# create_model_and_diffusion,
from .jigsaw_script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from .script_util import (
    # NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # Load data
    data_loader = load_data(
        data_dir=Path(args.data_dir),
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
        batch_size=args.batch_size,
        deterministic=True,
    )

    # Load model
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    all_positions = []
    all_pieces = []

    # List to store all generated images
    path_parent = os.path.join(logger.get_dir(), "samples")
    os.makedirs(path_parent, exist_ok=True)
    i = 0
    for position, cond in data_loader:
        pieces = cond["pieces"].to(dist_util.dev())

        model_kwargs = {"pieces": pieces}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            position.shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # Gather all samples across different nodes
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_positions = list(
            chain(*(sample.cpu().numpy() for sample in gathered_samples))
        )
        logger.log(f"created {len(all_positions) * args.batch_size} samples")

        # Gather all pieces across different nodes
        gathered_pieces = [th.zeros_like(pieces) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_pieces, pieces)  # gather not supported with NCCL
        all_pieces = list(chain(*(piece.cpu().numpy() for piece in gathered_pieces)))
        logger.log(f"created {len(all_pieces) * args.batch_size} pieces")

        for all_positions_i, all_pieces_i in zip(all_positions, all_pieces):
            image = reassemble_puzzle(all_positions_i, all_pieces_i)
            image_path = os.path.join(path_parent, f"image_{i}.png")
            cv.imwrite(image_path, image)

            position_path = os.path.join(path_parent, f"position_{i}.csv")
            with open(position_path, "w") as fp:
                writer = csv.writer(fp)
                for row in all_positions_i:
                    writer.writerow(row)

            i += 1

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        piece_size=64,
        puzzle_size=320,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../dataset/example/")
    parser.add_argument("--model_path")
    add_dict_to_argparser(parser, defaults)
    return parser
