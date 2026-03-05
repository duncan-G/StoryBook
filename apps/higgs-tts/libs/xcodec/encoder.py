# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import torch.nn as nn

from .conv import Conv1d
from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = Conv1d(
            in_channels=input_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, bias=False
        )
        self.conv_blocks = nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x
