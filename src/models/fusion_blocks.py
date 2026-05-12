from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1x1(in_ch: int, out_ch: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)


def _conv3x3(in_ch: int, out_ch: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)


class ConcatFusion(nn.Module):
    def __init__(
            self,
            rgb_channels: int,
            nir_channels: int,
            out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or rgb_channels
        self.use_residual = out_channels == rgb_channels
        self.proj = _conv1x1(rgb_channels + nir_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_residual:
            nn.init.zeros_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        delta = self.bn(self.proj(torch.cat([rgb, nir], dim=1)))
        if self.use_residual:
            return F.relu(rgb + delta, inplace=True)
        return F.relu(delta, inplace=True)


class SumFusion(nn.Module):
    def __init__(self, rgb_channels: int, nir_channels: int) -> None:
        super().__init__()
        self.proj: Optional[nn.Module] = None
        if nir_channels != rgb_channels:
            self.proj = _conv1x1(nir_channels, rgb_channels)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            nir = self.proj(nir)
        return rgb + nir


class WeightedSumFusion(nn.Module):

    def __init__(self, rgb_channels: int, nir_channels: int) -> None:
        super().__init__()
        self.proj: Optional[nn.Module] = None
        if nir_channels != rgb_channels:
            self.proj = _conv1x1(nir_channels, rgb_channels)

        self.weights = nn.Parameter(torch.zeros(2))

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            nir = self.proj(nir)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * rgb + w[1] * nir


class SEFusion(nn.Module):
    def __init__(
            self,
            rgb_channels: int,
            nir_channels: int,
            out_channels: Optional[int] = None,
            reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = out_channels or rgb_channels
        self.use_residual = out_channels == rgb_channels
        self.proj = _conv1x1(rgb_channels + nir_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_residual:
            nn.init.zeros_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
        bottleneck = max(out_channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        delta = F.relu(self.bn(self.proj(torch.cat([rgb, nir], dim=1))), inplace=True)
        scale = self.se(delta).view(delta.size(0), delta.size(1), 1, 1)
        delta = delta * scale
        if self.use_residual:
            return F.relu(rgb + delta, inplace=True)
        return delta


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.shared_fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        avg = x.mean(dim=(2, 3))
        max_ = x.flatten(2).max(dim=2).values
        gate = torch.sigmoid(self.shared_fc(avg) + self.shared_fc(max_))
        return x * gate.view(b, c, 1, 1)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.max(dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * scale


class CBAMFusion(nn.Module):
    def __init__(
            self,
            rgb_channels: int,
            nir_channels: int,
            out_channels: Optional[int] = None,
            reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = out_channels or rgb_channels
        self.use_residual = out_channels == rgb_channels
        self.proj = _conv1x1(rgb_channels + nir_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_residual:
            nn.init.zeros_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
        self.channel_attn = _ChannelAttention(out_channels, reduction)
        self.spatial_attn = _SpatialAttention()

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        delta = F.relu(self.bn(self.proj(torch.cat([rgb, nir], dim=1))), inplace=True)
        delta = self.channel_attn(delta)
        delta = self.spatial_attn(delta)
        if self.use_residual:
            return F.relu(rgb + delta, inplace=True)
        return delta


class GatedFusion(nn.Module):

    def __init__(self, rgb_channels: int, nir_channels: int) -> None:
        super().__init__()
        self.nir_proj = _conv1x1(nir_channels, rgb_channels)
        self.gate_conv = nn.Sequential(
            _conv3x3(rgb_channels + rgb_channels, rgb_channels),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(rgb_channels)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        nir_p = self.nir_proj(nir)
        gate = self.gate_conv(torch.cat([rgb, nir_p], dim=1))
        return self.bn(rgb + gate * nir_p)


class CrossAttentionFusion(nn.Module):
    def __init__(self, rgb_channels: int, nir_channels: int, num_heads: int = 4) -> None:
        super().__init__()

        self.rgb_proj = _conv1x1(rgb_channels, rgb_channels)
        self.nir_proj = _conv1x1(nir_channels, rgb_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=rgb_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            _conv1x1(rgb_channels, rgb_channels),
            nn.BatchNorm2d(rgb_channels),
        )
        self.norm = nn.LayerNorm(rgb_channels)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb.shape
        q = self.rgb_proj(rgb).flatten(2).permute(0, 2, 1)
        k = self.nir_proj(nir).flatten(2).permute(0, 2, 1)
        v = k
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.norm(attn_out)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return rgb + self.out_proj(attn_out)


class ResidualAdapterFusion(nn.Module):
    def __init__(
            self,
            rgb_channels: int,
            nir_channels: int,
            bottleneck_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(nir_channels // 4, 8)
        self.down = _conv1x1(nir_channels, bottleneck_dim)
        self.act = nn.ReLU(inplace=True)
        self.up = _conv1x1(bottleneck_dim, rgb_channels)
        nn.init.zeros_(self.up.weight)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        adapter_out = self.up(self.act(self.down(nir)))
        return rgb + adapter_out


FUSION_BLOCK_REGISTRY: dict[str, type] = {
    "concat": ConcatFusion,
    "sum": SumFusion,
    "weighted_sum": WeightedSumFusion,
    "se": SEFusion,
    "cbam": CBAMFusion,
    "gated": GatedFusion,
    "cross_attention": CrossAttentionFusion,
    "residual_adapter": ResidualAdapterFusion,
}


def build_fusion_block(
        fusion_type: str,
        rgb_channels: int,
        nir_channels: int,
        out_channels: Optional[int] = None,
        **kwargs,
) -> nn.Module:
    if fusion_type not in FUSION_BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. "
            f"Choose from {list(FUSION_BLOCK_REGISTRY.keys())}"
        )
    cls = FUSION_BLOCK_REGISTRY[fusion_type]

    _accepts_out = fusion_type in ("concat", "se", "cbam")
    if _accepts_out and out_channels is not None:
        return cls(rgb_channels, nir_channels, out_channels, **kwargs)
    return cls(rgb_channels, nir_channels, **kwargs)
