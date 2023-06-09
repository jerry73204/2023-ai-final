from .jigsaw_unet import JigsawUNetModel
from .script_util import (
    create_gaussian_diffusion,
)
import numpy as np
import cv2 as cv
import math
import random
import numpy.typing as npt
import torch


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
        diffusion_steps=500,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def rotate_image(image: npt.NDArray[np.uint8], angle_deg: float):
    sh, sw, sc = image.shape
    center = (sh / 2, sw / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    result = cv.warpAffine(image, rot_mat, (sh, sw), flags=cv.INTER_LINEAR)

    if sc == 1:
        assert result.shape == (sh, sw)
        result = result.reshape(sh, sw, 1)

    # cv.imshow("before", image)
    # cv.imshow("after", result)
    # cv.waitKey(1)
    return result


def reassemble_puzzle(
    positions: npt.NDArray[np.float32],
    pieces: npt.NDArray[np.uint8],
    puzzle_size: int,
):
    """
    Reassemble the piece images according to their poses into an image.
    The channel-last convension is used.
    """

    # Create a new canvas for each image
    _, _, _, nc = pieces.shape
    canvas = np.full([puzzle_size, puzzle_size, nc], 0, dtype=np.uint8)

    for image, center in zip(pieces, positions):
        cx, cy, angle_rad = center
        cx = float(cx)
        cy = float(cy)
        angle_deg = float(angle_rad) * 180 / math.pi

        image = rotate_image(image, angle_deg)
        mask = image != 0

        h, w, _ = image.shape

        y1 = int(math.floor(cy - h / 2))
        y2 = y1 + h
        y3 = min(max(y1, 0), puzzle_size)
        y4 = min(max(y2, 0), puzzle_size)

        x1 = int(math.floor(cx - w / 2))
        x2 = x1 + w
        x3 = min(max(x1, 0), puzzle_size)
        x4 = min(max(x2, 0), puzzle_size)

        sub_canvas = canvas[y3:y4, x3:x4]
        sub_mask = mask[(y3 - y1) : (y4 - y1), (x3 - x1) : (x4 - x1)]
        sub_canvas[sub_mask] = random.randint(128, 255)

    return canvas


def normalize_piece_images(images: torch.FloatTensor) -> torch.FloatTensor:
    return images.type(torch.FloatTensor) / 255 * 2 - 1


def denormalize_piece_images(images: torch.FloatTensor) -> torch.FloatTensor:
    return (images + 1) / 2 * 255


def normalize_piece_positions(positions: torch.FloatTensor, puzzle_size: int):
    # Normalize x, y to [-1, 1]

    positions[:, :2] = positions[:, :2] / puzzle_size * 2 - 1

    # Normalize rotation to [-1, 1]
    positions[:, 2] = positions[:, 2] / math.pi - 1

    return positions


def denormalize_piece_positions(positions: torch.FloatTensor, puzzle_size: int):
    # Denormalize x, y to [0, puzzle_size]
    positions[:, :2] = (positions[:, :2] + 1) / 2 * puzzle_size

    # Denormalize rotation to [0, 2π]
    positions[:, 2] = (positions[:, 2] + 1) * math.pi

    return positions
