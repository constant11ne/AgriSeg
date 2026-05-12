from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_blocks import (
    ConcatFusion, SumFusion, GatedFusion, SEFusion,
    CBAMFusion, CrossAttentionFusion, ResidualAdapterFusion,
)
from .nir_branches import NIREncoderLite, NIRPyramidLite
from .ibn import IBN
from .backbone_utils import (
    build_rgb_encoder, adapt_first_conv,
    freeze_encoder_full, freeze_encoder_stages, unfreeze_last_n_stages,
)


class FPNDecoder(nn.Module):

    def __init__(
            self,
            in_channels_list: List[int],
            fpn_channels: int,
            num_classes: int,
            output_size: tuple = (512, 512),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.nir_level_proj = None

        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list
        ])

        _pred_conv = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

        nn.init.normal_(_pred_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(_pred_conv.bias)
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels * len(in_channels_list), fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            _pred_conv,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        lat = [lat(f) for lat, f in zip(self.laterals, features)]

        for i in range(len(lat) - 1):
            lat[i + 1] = lat[i + 1] + F.interpolate(
                lat[i], size=lat[i + 1].shape[2:], mode="bilinear", align_corners=False
            )

        finest_size = lat[-1].shape[2:]
        outs = []
        for conv, l in zip(self.fpn_convs, lat):
            out = conv(l)
            out = F.interpolate(out, size=finest_size, mode="bilinear", align_corners=False)
            outs.append(out)
        merged = torch.cat(outs, dim=1)
        logits = self.seg_head(merged)

        if logits.shape[2:] != torch.Size(self.output_size):
            logits = F.interpolate(
                logits, size=self.output_size, mode="bilinear", align_corners=False
            )
        return logits


_BOTTLENECK_METHODS = {
    "bottleneck_concat",
    "bottleneck_sum",
    "cross_attn_bottleneck",
}
_MULTISCALE_METHODS = {
    "multiscale_concat",
    "multiscale_sum",
    "multiscale_gated",
    "multiscale_se",
    "multiscale_cbam",
    "adapter_multilevel",
    "adapter_residual",
    "adapter_input",
}
_FUSION_TYPE_MAP = {
    "bottleneck_concat": "concat",
    "bottleneck_sum": "sum",
    "multiscale_concat": "concat",
    "multiscale_sum": "sum",
    "multiscale_gated": "gated",
    "multiscale_se": "se",
    "multiscale_cbam": "cbam",
    "adapter_multilevel": "residual_adapter",
    "adapter_residual": "residual_adapter",
    "cross_attn_bottleneck": "cross_attention",
    "progressive_concat": "concat",
    "progressive_se": "se",
}


class DualStreamFPN(nn.Module):

    def __init__(
            self,
            num_classes: int,
            fusion_method: str = "multiscale_concat",
            encoder_name: str = "timm-efficientnet-b4",
            encoder_weights: Optional[str] = "imagenet",
            nir_width_mult: float = 0.5,
            fpn_channels: int = 128,
            freeze_rgb: str = "none",
            freeze_rgb_stages: int = 3,
            partial_unfreeze_last_n: int = 2,
            output_size: tuple = (512, 512),
            depth: int = 5,
            align_norm_mode: str = "none",
            nir_active_levels: int = 5,
            progressive_level_indices: Optional[str] = None,
            nir_in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.output_size = output_size
        self.nir_level_proj = None
        self.nir_in_channels = int(nir_in_channels)
        self.align_norm_mode = align_norm_mode
        self.nir_active_levels = max(1, min(5, int(nir_active_levels)))
        if progressive_level_indices:
            self.progressive_level_indices = [int(x) for x in str(progressive_level_indices).split(',') if
                                              str(x).strip() != '']
        else:
            self.progressive_level_indices = None

        self.rgb_encoder, rgb_stage_channels = build_rgb_encoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            depth=depth,
        )

        rgb_feat_channels: List[int] = list(rgb_stage_channels[1:])

        self._apply_freeze(freeze_rgb, freeze_rgb_stages, partial_unfreeze_last_n)

        nir_stem_norm = "ibn" if self._should_align_nir_stem() else "bn"
        nir_stage1_norm = "ibn" if self._should_align_nir_stage1() else "bn"
        if fusion_method == "adapter_input":
            from .nir_branches import NIRStem
            self.nir_encoder = NIRStem(in_channels=1, out_channels=rgb_feat_channels[0], norm_mode=nir_stem_norm)
            nir_feat_channels: List[int] = [rgb_feat_channels[0]]
        else:
            self.nir_encoder = NIREncoderLite(
                in_channels=self.nir_in_channels,
                width_mult=nir_width_mult,
                stem_norm_mode=nir_stem_norm,
                stage1_norm_mode=nir_stage1_norm,
            )
            nir_feat_channels = self.nir_encoder.out_channels

        self.rgb_align_layers = self._build_rgb_align_layers(rgb_feat_channels)

        self.fusion_blocks = self._build_fusion_blocks(
            rgb_feat_channels, nir_feat_channels, fusion_method
        )

        fused_channels = self._compute_fused_channels(
            rgb_feat_channels, nir_feat_channels, fusion_method
        )

        decode_levels = min(4, len(fused_channels), depth)
        decoder_in_channels = fused_channels[-decode_levels:]
        self.decode_levels = decode_levels

        self.decoder = FPNDecoder(
            in_channels_list=list(reversed(decoder_in_channels)),
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            output_size=output_size,
        )

    def _should_align_nir_stem(self) -> bool:
        return self.align_norm_mode in {"nir_ibn_stem", "nir_ibn_stem_s1", "rgb_nir_ibn_stem", "rgb_nir_ibn_stem_s1"}

    def _should_align_nir_stage1(self) -> bool:
        return self.align_norm_mode in {"nir_ibn_stem_s1", "rgb_nir_ibn_stem_s1"}

    def _build_rgb_align_layers(self, rgb_feat_channels: List[int]) -> nn.ModuleList:
        layers = [nn.Identity() for _ in rgb_feat_channels]
        if self.align_norm_mode in {"rgb_nir_ibn_stem", "rgb_nir_ibn_stem_s1"} and len(rgb_feat_channels) >= 1:
            layers[0] = IBN(rgb_feat_channels[0])
        if self.align_norm_mode == "rgb_nir_ibn_stem_s1" and len(rgb_feat_channels) >= 2:
            layers[1] = IBN(rgb_feat_channels[1])
        return nn.ModuleList(layers)

    def _apply_freeze(

            self,
            freeze_rgb: str,
            freeze_rgb_stages: int,
            partial_unfreeze_last_n: int,
    ) -> None:
        if freeze_rgb == "none":
            return
        elif freeze_rgb == "full":
            freeze_encoder_full(self.rgb_encoder)
        elif freeze_rgb == "stem":
            freeze_encoder_stages(self.rgb_encoder, freeze_n_stages=1)
        elif freeze_rgb == "first_N":
            freeze_encoder_stages(self.rgb_encoder, freeze_n_stages=freeze_rgb_stages)
        elif freeze_rgb == "partial":
            freeze_encoder_full(self.rgb_encoder)
            unfreeze_last_n_stages(self.rgb_encoder, n=partial_unfreeze_last_n)

    def _build_fusion_blocks(
            self,
            rgb_chs: List[int],
            nir_chs: List[int],
            method: str,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        fusion_type = _FUSION_TYPE_MAP.get(method, "concat")

        if method in _BOTTLENECK_METHODS:
            rgb_ch = rgb_chs[-1]
            nir_ch = nir_chs[-1] if len(nir_chs) > 0 else rgb_ch
            blocks.append(self._make_block(fusion_type, rgb_ch, nir_ch))

        elif method == "adapter_input":
            blocks.append(ResidualAdapterFusion(rgb_chs[0], nir_chs[0]))

        elif method in _MULTISCALE_METHODS or method in {"progressive_concat", "progressive_se"}:
            if method in {"progressive_concat", "progressive_se"} and self.progressive_level_indices is not None:
                for idx in self.progressive_level_indices:
                    if 0 <= idx < len(rgb_chs) and 0 <= idx < len(nir_chs):
                        blocks.append(self._make_block(fusion_type, rgb_chs[idx], nir_chs[idx]))
            else:
                max_levels = self.nir_active_levels if method in {"progressive_concat", "progressive_se"} else 4
                n_levels = min(max_levels, len(rgb_chs), len(nir_chs))
                rgb_offset = len(rgb_chs) - n_levels
                nir_offset = len(nir_chs) - n_levels
                for i in range(n_levels):
                    blocks.append(self._make_block(fusion_type, rgb_chs[rgb_offset + i], nir_chs[nir_offset + i]))

        return blocks

    def _make_block(self, fusion_type: str, rgb_ch: int, nir_ch: int) -> nn.Module:
        if fusion_type == "concat":
            return ConcatFusion(rgb_ch, nir_ch, rgb_ch)
        elif fusion_type == "sum":
            return SumFusion(rgb_ch, nir_ch)
        elif fusion_type == "gated":
            return GatedFusion(rgb_ch, nir_ch)
        elif fusion_type == "se":
            return SEFusion(rgb_ch, nir_ch, rgb_ch)
        elif fusion_type == "cbam":
            return CBAMFusion(rgb_ch, nir_ch, rgb_ch)
        elif fusion_type == "residual_adapter":
            return ResidualAdapterFusion(rgb_ch, nir_ch)
        elif fusion_type == "cross_attention":
            for nh in (8, 4, 2, 1):
                if rgb_ch % nh == 0:
                    return CrossAttentionFusion(rgb_ch, nir_ch, num_heads=nh)
        return ConcatFusion(rgb_ch, nir_ch, rgb_ch)

    def _compute_fused_channels(
            self,
            rgb_chs: List[int],
            nir_chs: List[int],
            method: str,
    ) -> List[int]:
        if method in _BOTTLENECK_METHODS or method == "adapter_input":
            return list(rgb_chs)

        return list(rgb_chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]
        nir = x[:, 3:3 + self.nir_in_channels]

        rgb_feats_raw = self.rgb_encoder(rgb)
        rgb_feats: List[torch.Tensor] = list(rgb_feats_raw[1:])
        rgb_feats = [align(feat) for align, feat in zip(self.rgb_align_layers, rgb_feats)]

        if self.fusion_method == "adapter_input":
            nir_stem = self.nir_encoder(nir)
            nir_feats: List[torch.Tensor] = [nir_stem]
        else:
            nir_feats = self.nir_encoder(nir)
            if self.nir_level_proj is not None:
                nir_feats = [proj(feat) for proj, feat in zip(self.nir_level_proj, nir_feats)]

        fused: List[torch.Tensor] = list(rgb_feats)

        if self.fusion_method in _BOTTLENECK_METHODS:
            nir_deep = nir_feats[-1] if len(nir_feats) > 0 else nir_feats[0]
            nir_deep = F.interpolate(nir_deep, size=fused[-1].shape[2:], mode="bilinear", align_corners=False)
            fused[-1] = self.fusion_blocks[0](fused[-1], nir_deep)

        elif self.fusion_method == "adapter_input":
            nir_aligned = F.interpolate(
                nir_feats[0], size=fused[0].shape[2:],
                mode="bilinear", align_corners=False
            )
            fused[0] = self.fusion_blocks[0](fused[0], nir_aligned)

        else:
            if self.fusion_method in {"progressive_concat",
                                      "progressive_se"} and self.progressive_level_indices is not None:
                active = [idx for idx in self.progressive_level_indices if
                          0 <= idx < len(rgb_feats) and 0 <= idx < len(nir_feats)]
                for block, idx in zip(self.fusion_blocks, active):
                    nir_aligned = F.interpolate(
                        nir_feats[idx], size=fused[idx].shape[2:],
                        mode="bilinear", align_corners=False
                    )
                    fused[idx] = block(fused[idx], nir_aligned)
            else:

                n_levels = min(len(self.fusion_blocks), len(rgb_feats), len(nir_feats))
                rgb_offset = len(rgb_feats) - n_levels
                nir_offset = len(nir_feats) - n_levels
                for i in range(n_levels):
                    rgb_idx = rgb_offset + i
                    nir_idx = nir_offset + i
                    nir_aligned = F.interpolate(
                        nir_feats[nir_idx], size=fused[rgb_idx].shape[2:],
                        mode="bilinear", align_corners=False
                    )
                    fused[rgb_idx] = self.fusion_blocks[i](fused[rgb_idx], nir_aligned)

        decode_feats = fused[-self.decode_levels:]

        decode_feats_rev = list(reversed(decode_feats))
        logits = self.decoder(decode_feats_rev)
        return logits
