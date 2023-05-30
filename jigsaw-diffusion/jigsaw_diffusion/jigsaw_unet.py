import torch as th
import torch.nn as nn
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from .unet import (
    Upsample,
    AttentionBlock,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
)


class JigsawUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size: int,
        num_pieces: int,
        piece_channels: int,
        model_channels: int,
        position_channels: int,
        num_res_blocks: int,
        attention_resolutions,
        dropout: float = 0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.num_pieces = num_pieces
        self.piece_channels = piece_channels
        self.model_channels = model_channels
        self.position_channels = position_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # Compute input/output channels
        in_channels = num_pieces * (piece_channels + position_channels)
        out_channels = num_pieces * position_channels

        # Create time embedding module
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.out = nn.Sequential(
            nn.Linear(ch, ch // 2),
            nn.SiLU(),
            nn.Linear(ch // 2, out_channels),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, positions, timesteps, pieces):
        """
        Apply the model to an input batch.
        `B` is the batch size. `N` is the number of pieces.
        `C` is the channel size for piece images.
        `P` is the number of parameters to describe the pose of a piece.
        `H` and `W` are the height and the width of piece images.

        :param positions: an [B x N x P] Tensor of inputs.
        :param pieces: an [B × N × C × H × W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [B x N x 3] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Check input tensor shapes
        sb, sn, sc, sh, sw = pieces.shape
        sb, sn, sp = positions.shape
        assert sc == self.piece_channels
        assert sp == self.position_channels

        # Rehape `positions` to [B x N x P x 1 x 1]
        positions = positions.view(sb, sn, sp, 1, 1)

        # Broadcast `positions` shape to [B x N x P x H x W]
        positions = positions.expand(sb, sn, sp, sh, sw)

        # Concatenate pieces and positions into a compound tensor with shape
        # [B x N x (C + P) x H x W]
        compound = th.cat([pieces, positions], 2)

        # Reshape the compound tensor to image-like shape [B x (Nx(C+P)) x H x W]
        x_in = compound.reshape(sb, sn * (sc + sp), sh, sw)

        # Downsampling blocks
        hs = []
        x_h = x_in.type(self.dtype)
        for module in self.input_blocks:
            x_h = module(x_h, emb)
            hs.append(x_h)

        # middle block
        x_h = self.middle_block(x_h, emb)

        # output
        x_h = x_h.mean([2, 3])
        x_out = self.out(x_h)
        x_result = x_out.view(sb, sn, sp)

        return x_result
