"""
Train a diffusion model on puzzles.
"""

import argparse
from pathlib import Path

from . import dist_util, jigsaw_datasets, logger
from .jigsaw_train_util import TrainLoop
from .resample import create_named_schedule_sampler
from .script_util import (
    add_dict_to_argparser,
    args_to_dict,
)
from .jigsaw_script_util import model_and_diffusion_defaults, create_model_and_diffusion


def main():
    args = parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    jigsaw_dataloader = jigsaw_datasets.load_data(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
        deterministic=False,
    )
    logger.log("load dataset finish")
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=jigsaw_dataloader,
        puzzle_size=args.puzzle_size,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        show_gui=args.show_gui,
    ).run_loop()


def parse_args():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        piece_size=64,
        puzzle_size=320,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../dataset/example/")
    parser.add_argument("--show_gui", action="store_true")
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()
