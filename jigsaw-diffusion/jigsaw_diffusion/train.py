"""
Train a diffusion model on images.
"""

import argparse

from jigsaw_diffusion import dist_util, logger
from jigsaw_diffusion.jigsaw_datasets import JigsawDataset
from jigsaw_diffusion.resample import create_named_schedule_sampler
from jigsaw_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from jigsaw_diffusion.jigsaw_train_util import TrainLoop


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
    # images, labels
    jigsaw_dataset = JigsawDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        piece_size=args.piece_size,
    )
    logger.log("load dataset finish")
    #  TODO
    jigsaw_dataloader = jigsaw_dataset
    # exit()
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=jigsaw_dataloader,
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
        piece_size=64
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../dataset/example/")
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


def __main__():
    main()
