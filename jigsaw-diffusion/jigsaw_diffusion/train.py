"""
Train a diffusion model on puzzles.
"""

import argparse
from pathlib import Path

from . import dist_util, jigsaw_datasets, logger
from .jigsaw_train_util import TrainLoop
from .jigsaw_unet import JigsawUNetModel
from .resample import create_named_schedule_sampler
from .script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
)


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
    )
    logger.log("load dataset finish")

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
        piece_size=64,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../dataset/example/")
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


def create_model_and_diffusion(
    image_size,
    num_pieces,
    piece_channels,
    model_channels,
    learn_sigma,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size=image_size,
        num_pieces=num_pieces,
        piece_channels=piece_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_pieces,
    piece_channels,
    model_channels,
    num_res_blocks,
    channel_mult="",
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return JigsawUNetModel(
        image_size=image_size,
        num_pieces=num_pieces,
        piece_channels=piece_channels,
        model_channels=model_channels,
        position_channels=3,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


# def create_gaussian_diffusion(
#     *,
#     steps=1000,
#     learn_sigma=False,
#     sigma_small=False,
#     noise_schedule="linear",
#     use_kl=False,
#     predict_xstart=False,
#     rescale_timesteps=False,
#     rescale_learned_sigmas=False,
#     timestep_respacing="",
# ):
#     betas = gd.get_named_beta_schedule(noise_schedule, steps)
#     if use_kl:
#         loss_type = gd.LossType.RESCALED_KL
#     elif rescale_learned_sigmas:
#         loss_type = gd.LossType.RESCALED_MSE
#     else:
#         loss_type = gd.LossType.MSE
#     if not timestep_respacing:
#         timestep_respacing = [steps]
#     return SpacedDiffusion(
#         use_timesteps=space_timesteps(steps, timestep_respacing),
#         betas=betas,
#         model_mean_type=(
#             gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
#         ),
#         model_var_type=(
#             (
#                 gd.ModelVarType.FIXED_LARGE
#                 if not sigma_small
#                 else gd.ModelVarType.FIXED_SMALL
#             )
#             if not learn_sigma
#             else gd.ModelVarType.LEARNED_RANGE
#         ),
#         loss_type=loss_type,
#         rescale_timesteps=rescale_timesteps,
#     )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_pieces=25,
        piece_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def __main__():
    main()
