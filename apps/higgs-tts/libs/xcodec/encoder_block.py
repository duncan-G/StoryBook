# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import torch.nn as nn

from .conv import Conv1d
from .residual_unit import ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dilations=(1, 1), unit_kernel_size=3, bias=True
    ):
        super().__init__()
        self.res_units = nn.ModuleList()
        for dilation in dilations:
            self.res_units += [ResidualUnit(in_channels, in_channels, kernel_size=unit_kernel_size, dilation=dilation)]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3 if stride == 1 else (2 * stride),  # special case: stride=1, do not use kernel=2
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x
