from __future__ import annotations

import torch
import torch.nn as nn


class IBN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()

        bn_channels = num_features // 2
        in_channels = num_features - bn_channels

        self.bn = nn.BatchNorm2d(bn_channels, eps=eps, momentum=momentum)

        self.inorm = nn.InstanceNorm2d(in_channels, affine=True, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn_channels = self.bn.num_features
        if bn_channels == 0:
            return self.inorm(x)

        in_channels = x.size(1) - bn_channels

        x_bn = x[:, :bn_channels, :, :]
        x_in = x[:, bn_channels:, :, :]

        out_bn = self.bn(x_bn)
        out_in = self.inorm(x_in)

        return torch.cat([out_bn, out_in], dim=1)
