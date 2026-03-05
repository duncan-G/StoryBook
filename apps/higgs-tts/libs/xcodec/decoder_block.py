# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import torch.nn as nn

from .conv import Conv1d, ConvTranspose1d
from .residual_unit import ResidualUnit


class DecoderBlock(nn.Module):
    """Decoder block (no up-sampling)"""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dilations=(1, 1), unit_kernel_size=3, bias=True
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,  # fix kernel=3 when stride=1 for unchanged shape
                stride=stride,
                bias=bias,
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )

        self.res_units = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            self.res_units += [
                ResidualUnit(out_channels, out_channels, kernel_size=unit_kernel_size, dilation=dilation)
            ]
        self.num_res = len(self.res_units)

    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x
