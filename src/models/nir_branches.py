from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .ibn import IBN


def _make_norm(channels: int, norm_mode: str = "bn") -> nn.Module:
    if norm_mode == "bn":
        return nn.BatchNorm2d(channels)
    if norm_mode == "ibn":
        return IBN(channels)
    raise ValueError(f"Unknown norm_mode: {norm_mode}")


def _dw_sep_conv(in_ch: int, out_ch: int, stride: int = 1, *, depth_norm_mode: str = "bn",
                 point_norm_mode: str = "bn") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                  groups=in_ch, bias=False),
        _make_norm(in_ch, norm_mode=depth_norm_mode),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        _make_norm(out_ch, norm_mode=point_norm_mode),
        nn.ReLU6(inplace=True),
    )


def _conv_norm_relu(in_ch: int, out_ch: int, stride: int = 1, *, norm_mode: str = "bn") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        _make_norm(out_ch, norm_mode=norm_mode),
        nn.ReLU(inplace=True),
    )


class NIRStem(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 32, norm_mode: str = "bn") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_norm_relu(in_channels, out_channels // 2, norm_mode=norm_mode),
            _conv_norm_relu(out_channels // 2, out_channels, norm_mode=norm_mode),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class NIREncoderLite(nn.Module):
    _STAGE_CHANNELS = (24, 32, 64, 128, 192)

    def __init__(
            self,
            in_channels: int = 1,
            width_mult: float = 1.0,
            stem_norm_mode: str = "bn",
            stage1_norm_mode: str = "bn",
    ) -> None:
        super().__init__()
        chs = [max(8, int(c * width_mult)) for c in self._STAGE_CHANNELS]
        stem_ch = max(8, int(16 * width_mult))

        self.stem = _conv_norm_relu(in_channels, stem_ch, norm_mode=stem_norm_mode)
        self.stage1 = _dw_sep_conv(stem_ch, chs[0], stride=2, depth_norm_mode=stage1_norm_mode,
                                   point_norm_mode=stage1_norm_mode)
        self.stage2 = _dw_sep_conv(chs[0], chs[1], stride=2)
        self.stage3 = _dw_sep_conv(chs[1], chs[2], stride=2)
        self.stage4 = _dw_sep_conv(chs[2], chs[3], stride=2)
        self.stage5 = _dw_sep_conv(chs[3], chs[4], stride=2)

        self.out_channels: List[int] = chs  # [stage1_ch, ..., stage5_ch]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s = self.stem(x)
        f1 = self.stage1(s)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f1, f2, f3, f4, f5]


class NIRPyramidLite(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            width_mult: float = 1.0,
            stem_norm_mode: str = "bn",
            stage1_norm_mode: str = "bn",
    ) -> None:
        super().__init__()
        self._encoder = NIREncoderLite(in_channels=in_channels, width_mult=width_mult)
        stem_ch = max(8, int(16 * width_mult))
        self.stem_ch = stem_ch
        self.out_channels: List[int] = [stem_ch] + self._encoder.out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s = self._encoder.stem(x)
        f1 = self._encoder.stage1(s)
        f2 = self._encoder.stage2(f1)
        f3 = self._encoder.stage3(f2)
        f4 = self._encoder.stage4(f3)
        f5 = self._encoder.stage5(f4)
        return [s, f1, f2, f3, f4, f5]
