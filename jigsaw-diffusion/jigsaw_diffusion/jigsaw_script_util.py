from .jigsaw_unet import JigsawUNetModel
from .script_util import (
    create_gaussian_diffusion,
)
import numpy as np
import cv2 as cv
import math
import random


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
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def reassemble_puzzle(all_positions_i, all_pieces_i, output_size=320):
    # print(all_positions_i.shape,all_pieces_i.shape)
    # Create a new canvas for each image
    canvas = np.full([1, output_size, output_size], 0, dtype=np.uint8)

    for image, center in zip(all_pieces_i, all_positions_i):
        cx, cy, angle = center
        image = rotate_image(image, angle)
        mask = image != 0

        _, h, w = image.shape

        y1 = int(math.floor(cy - h / 2))
        y2 = y1 + h
        y3 = max(y1, 0)
        y4 = min(y2, output_size)

        x1 = int(math.floor(cx - w / 2))
        x2 = x1 + w
        x3 = max(x1, 0)
        x4 = min(x2, output_size)

        sub_canvas = canvas[:, y3:y4, x3:x4]
        sub_mask = mask[:, (y3 - y1) : (y4 - y1), (x3 - x1) : (x4 - x1)]
        sub_canvas[sub_mask] = random.randint(128, 255)

    canvas = np.transpose(canvas, [1, 2, 0])
    return canvas
