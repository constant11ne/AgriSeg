from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_utils import (
    build_rgb_encoder,
    adapt_first_conv,
    freeze_encoder_full,
    freeze_encoder_stages,
    unfreeze_last_n_stages,
)
from .dual_stream_fpn import FPNDecoder
from .ibn import IBN


class MultiBranchConcatFusion(nn.Module):
    def __init__(self, rgb_ch: int, extra_chs: Sequence[int], norm_mode: str = "bn") -> None:
        super().__init__()
        in_ch = int(rgb_ch + sum(extra_chs))
        if norm_mode == "ibn":
            norm = IBN(rgb_ch)
        else:
            norm = nn.BatchNorm2d(rgb_ch)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, rgb_ch, kernel_size=1, bias=False),
            norm,
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, extra_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        aligned = [
            F.interpolate(feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
            for feat in extra_feats
        ]
        return self.proj(torch.cat([rgb_feat, *aligned], dim=1))


class MultiStreamFPN(nn.Module):
    def __init__(
            self,
            num_classes: int,
            branches: Tuple[str, ...] = ("nir", "ndvi"),
            fusion_method: str = "progressive_concat_rgb_nir_ndvi",
            encoder_name: str = "timm-efficientnet-b4",
            encoder_weights: Optional[str] = "imagenet",
            fpn_channels: int = 128,
            freeze_rgb: str = "none",
            freeze_rgb_stages: int = 3,
            partial_unfreeze_last_n: int = 2,
            output_size: tuple = (512, 512),
            depth: int = 5,
            progressive_level_indices: Optional[str] = None,
            align_norm_mode: str = "none",
    ) -> None:
        super().__init__()
        self.branches = tuple(branches)
        self.fusion_method = fusion_method
        self.output_size = output_size
        self.align_norm_mode = align_norm_mode
        self.use_ibn_align = "ibn" in str(align_norm_mode).lower() and str(align_norm_mode).lower() != "none"

        self.rgb_encoder, rgb_stage_channels = build_rgb_encoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            depth=depth,
        )
        rgb_feat_channels: List[int] = list(rgb_stage_channels[1:])

        self._apply_freeze(freeze_rgb, freeze_rgb_stages, partial_unfreeze_last_n)

        if "nir" in self.branches:
            self.nir_encoder, nir_stage_channels = build_rgb_encoder(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                depth=depth,
            )
            adapt_first_conv(self.nir_encoder, new_in_channels=1, init_mode="copy-r")
            self.nir_feat_channels: List[int] = list(nir_stage_channels[1:])
        else:
            self.nir_encoder = None
            self.nir_feat_channels = []

        if "ndvi" in self.branches:
            self.ndvi_encoder, ndvi_stage_channels = build_rgb_encoder(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                depth=depth,
            )
            adapt_first_conv(self.ndvi_encoder, new_in_channels=1, init_mode="copy-r")
            self.ndvi_feat_channels: List[int] = list(ndvi_stage_channels[1:])
        else:
            self.ndvi_encoder = None
            self.ndvi_feat_channels = []

        if progressive_level_indices:
            self.progressive_level_indices = [
                int(x) for x in str(progressive_level_indices).split(",") if str(x).strip() != ""
            ]
        else:
            self.progressive_level_indices = list(range(len(rgb_feat_channels)))

        self.rgb_align = nn.ModuleList([IBN(ch) if self.use_ibn_align else nn.Identity() for ch in rgb_feat_channels])
        self.nir_align = nn.ModuleList(
            [IBN(ch) if self.use_ibn_align else nn.Identity() for ch in self.nir_feat_channels])
        self.ndvi_align = nn.ModuleList(
            [IBN(ch) if self.use_ibn_align else nn.Identity() for ch in self.ndvi_feat_channels])

        fusion_norm = "ibn" if self.use_ibn_align else "bn"
        self.fusion_blocks = nn.ModuleList()
        for idx in self.progressive_level_indices:
            if not (0 <= idx < len(rgb_feat_channels)):
                continue
            extra_chs: List[int] = []
            if self.nir_encoder is not None:
                extra_chs.append(self.nir_feat_channels[idx])
            if self.ndvi_encoder is not None:
                extra_chs.append(self.ndvi_feat_channels[idx])
            self.fusion_blocks.append(MultiBranchConcatFusion(rgb_feat_channels[idx], extra_chs, norm_mode=fusion_norm))

        fused_channels = list(rgb_feat_channels)
        decode_levels = min(4, len(fused_channels), depth)
        decoder_in_channels = fused_channels[-decode_levels:]
        self.decode_levels = decode_levels
        self.decoder = FPNDecoder(
            in_channels_list=list(reversed(decoder_in_channels)),
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            output_size=output_size,
        )

    def _apply_freeze(self, freeze_rgb: str, freeze_rgb_stages: int, partial_unfreeze_last_n: int) -> None:
        if freeze_rgb == "none":
            return
        if freeze_rgb == "full":
            freeze_encoder_full(self.rgb_encoder)
        elif freeze_rgb == "stem":
            freeze_encoder_stages(self.rgb_encoder, freeze_n_stages=1)
        elif freeze_rgb == "first_N":
            freeze_encoder_stages(self.rgb_encoder, freeze_n_stages=freeze_rgb_stages)
        elif freeze_rgb == "partial":
            freeze_encoder_full(self.rgb_encoder)
            unfreeze_last_n_stages(self.rgb_encoder, n=partial_unfreeze_last_n)

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        rgb = x[:, :3]
        nir = None
        ndvi = None
        if "nir" in self.branches:
            if x.shape[1] < 5:
                raise RuntimeError(f"RGB+NIR+NDVI mid-fusion expected 5 channels, got {x.shape[1]}")
            nir = x[:, 3:4]
            ndvi = x[:, 4:5]
        else:
            if x.shape[1] >= 5:
                ndvi = x[:, 4:5]
            elif x.shape[1] >= 4:
                ndvi = x[:, 3:4]
            else:
                raise RuntimeError(f"RGB+NDVI mid-fusion expected 4 channels, got {x.shape[1]}")
        return rgb, nir, ndvi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, nir, ndvi = self._split_inputs(x)

        rgb_feats_raw = self.rgb_encoder(rgb)
        rgb_feats: List[torch.Tensor] = [align(feat) for align, feat in zip(self.rgb_align, list(rgb_feats_raw[1:]))]
        fused: List[torch.Tensor] = list(rgb_feats)

        nir_feats: Optional[List[torch.Tensor]] = None
        ndvi_feats: Optional[List[torch.Tensor]] = None
        if self.nir_encoder is not None and nir is not None:
            nir_feats = [align(feat) for align, feat in zip(self.nir_align, list(self.nir_encoder(nir)[1:]))]
        if self.ndvi_encoder is not None and ndvi is not None:
            ndvi_feats = [align(feat) for align, feat in zip(self.ndvi_align, list(self.ndvi_encoder(ndvi)[1:]))]

        active = [idx for idx in self.progressive_level_indices if 0 <= idx < len(fused)]
        for block, idx in zip(self.fusion_blocks, active):
            extras: List[torch.Tensor] = []
            if nir_feats is not None:
                extras.append(nir_feats[idx])
            if ndvi_feats is not None:
                extras.append(ndvi_feats[idx])
            fused[idx] = block(fused[idx], extras)

        decode_feats = fused[-self.decode_levels:]
        logits = self.decoder(list(reversed(decode_feats)))
        return logits
