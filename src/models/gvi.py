from __future__ import annotations

from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    smp = None
    HAS_SMP = False

from .backbone_utils import (
    build_rgb_encoder,
    adapt_first_conv,
    freeze_encoder_full,
    freeze_encoder_stages,
    unfreeze_last_n_stages,
)
from .dual_stream_fpn import FPNDecoder
from .multi_stream_fpn import MultiBranchConcatFusion


class LearnableRatioGVI(nn.Module):
    """Learnable vegetation index initialized close to known spectral indices.

    Input is expected to be normalized RGB+NIR in [-1, 1], channel order R,G,B,NIR.
    Internally we map it back to [0, 1] and compute
        GVI_k = learned_num_k(R,G,B,NIR) / (abs(learned_den_k(R,G,B,NIR)) + eps)

    Supported initial indices:
        ndvi  = (NIR - Red)   / (NIR + Red)
        gndvi = (NIR - Green) / (NIR + Green)
        ndwi  = (Green - NIR) / (Green + NIR)
        random: small random numerator over a near-constant denominator
    """

    def __init__(
        self,
        eps: float = 1e-4,
        clamp_output: bool = True,
        num_channels: int = 1,
        init_indices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.clamp_output = bool(clamp_output)
        self.num_channels = int(num_channels)
        if self.num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if init_indices is None:
            init_indices = ["ndvi"]
        self.init_indices = [str(x).strip().lower() for x in init_indices if str(x).strip()]
        if not self.init_indices:
            self.init_indices = ["ndvi"]
        while len(self.init_indices) < self.num_channels:
            self.init_indices.append("random")
        self.init_indices = self.init_indices[: self.num_channels]
        self.num = nn.Conv2d(4, self.num_channels, kernel_size=1, bias=True)
        self.den = nn.Conv2d(4, self.num_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.num.weight)
        nn.init.zeros_(self.num.bias)
        nn.init.zeros_(self.den.weight)
        nn.init.zeros_(self.den.bias)
        with torch.no_grad():
            for out_idx, index_name in enumerate(self.init_indices):
                self._init_one_channel(out_idx, index_name)

    def _init_ratio(self, out_idx: int, num_terms: Sequence[tuple[int, float]], den_terms: Sequence[tuple[int, float]]) -> None:
        for channel_idx, weight in num_terms:
            self.num.weight[out_idx, channel_idx, 0, 0] = float(weight)
        for channel_idx, weight in den_terms:
            self.den.weight[out_idx, channel_idx, 0, 0] = float(weight)

    def _init_one_channel(self, out_idx: int, index_name: str) -> None:
        # Channel order: R=0, G=1, B=2, NIR=3.
        if index_name == "ndvi":
            self._init_ratio(out_idx, [(3, 1.0), (0, -1.0)], [(3, 1.0), (0, 1.0)])
        elif index_name == "gndvi":
            self._init_ratio(out_idx, [(3, 1.0), (1, -1.0)], [(3, 1.0), (1, 1.0)])
        elif index_name == "ndwi":
            self._init_ratio(out_idx, [(1, 1.0), (3, -1.0)], [(1, 1.0), (3, 1.0)])
        elif index_name == "random":
            nn.init.normal_(self.num.weight[out_idx : out_idx + 1], mean=0.0, std=0.02)
            self.den.bias[out_idx] = 1.0
        else:
            raise ValueError(
                f"Unknown GVI init index '{index_name}'. "
                "Use any of: ndvi, gndvi, ndwi, random."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 4:
            raise RuntimeError(f"GVI requires RGB+NIR input with 4 channels, got {x.shape[1]}")
        x01 = torch.clamp((x[:, :4] + 1.0) * 0.5, 0.0, 1.0)
        numerator = self.num(x01)
        denominator = self.den(x01).abs() + self.eps
        gvi = numerator / denominator
        if self.clamp_output:
            gvi = torch.clamp(gvi, -1.0, 1.0)
        return gvi


class GVIOnlySegModel(nn.Module):
    """Segmentation model that first learns GVI from RGB+NIR, then segments from GVI."""

    def __init__(
        self,
        model_name: str,
        encoder_name: str,
        encoder_weights: Optional[str],
        num_classes: int,
        gvi_channels: int = 1,
        gvi_init_indices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if not HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required for GVIOnlySegModel")
        self.gvi_module = LearnableRatioGVI(num_channels=gvi_channels, init_indices=gvi_init_indices)
        model_name = model_name.lower()
        if model_name == "unet":
            self.seg_model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=gvi_channels,
                activation=None,
            )
        elif model_name == "fpn":
            self.seg_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=gvi_channels,
                activation=None,
            )
        elif model_name == "deeplabv3p":
            self.seg_model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=gvi_channels,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported model_name for GVI: {model_name}")

    @property
    def encoder(self) -> nn.Module:
        return self.seg_model.encoder

    @property
    def gvi_encoder(self) -> nn.Module:
        return self.seg_model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gvi = self.gvi_module(x[:, :4])
        return self.seg_model(gvi)


class RGBGVIStreamFPN(nn.Module):
    """Two-branch mid-fusion model: RGB encoder + learnable one/multi-GVI encoder."""

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "timm-efficientnet-b4",
        encoder_weights: Optional[str] = "imagenet",
        fusion_method: str = "progressive_concat_rgb_gvi",
        fpn_channels: int = 128,
        freeze_rgb: str = "none",
        freeze_rgb_stages: int = 3,
        partial_unfreeze_last_n: int = 2,
        output_size: tuple = (512, 512),
        depth: int = 5,
        progressive_level_indices: Optional[str] = None,
        align_norm_mode: str = "none",
        gvi_channels: int = 1,
        gvi_init_indices: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.output_size = output_size
        self.align_norm_mode = align_norm_mode
        self.gvi_channels = int(gvi_channels)
        self.use_ibn_align = "ibn" in str(align_norm_mode).lower() and str(align_norm_mode).lower() != "none"
        self.gvi_module = LearnableRatioGVI(num_channels=self.gvi_channels, init_indices=gvi_init_indices)

        self.rgb_encoder, rgb_stage_channels = build_rgb_encoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            depth=depth,
        )
        rgb_feat_channels: List[int] = list(rgb_stage_channels[1:])
        self._apply_freeze(freeze_rgb, freeze_rgb_stages, partial_unfreeze_last_n)

        self.gvi_encoder, gvi_stage_channels = build_rgb_encoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            depth=depth,
        )
        adapt_first_conv(self.gvi_encoder, new_in_channels=self.gvi_channels, init_mode="copy-mean")
        self.gvi_feat_channels: List[int] = list(gvi_stage_channels[1:])

        if progressive_level_indices:
            self.progressive_level_indices = [
                int(x) for x in str(progressive_level_indices).split(",") if str(x).strip() != ""
            ]
        else:
            self.progressive_level_indices = list(range(len(rgb_feat_channels)))

        try:
            from .ibn import IBN
            self.rgb_align = nn.ModuleList([IBN(ch) if self.use_ibn_align else nn.Identity() for ch in rgb_feat_channels])
            self.gvi_align = nn.ModuleList([IBN(ch) if self.use_ibn_align else nn.Identity() for ch in self.gvi_feat_channels])
        except Exception:
            self.rgb_align = nn.ModuleList([nn.Identity() for _ in rgb_feat_channels])
            self.gvi_align = nn.ModuleList([nn.Identity() for _ in self.gvi_feat_channels])
            self.use_ibn_align = False

        fusion_norm = "ibn" if self.use_ibn_align else "bn"
        self.fusion_blocks = nn.ModuleList()
        for idx in self.progressive_level_indices:
            if 0 <= idx < len(rgb_feat_channels):
                self.fusion_blocks.append(
                    MultiBranchConcatFusion(
                        rgb_feat_channels[idx],
                        [self.gvi_feat_channels[idx]],
                        norm_mode=fusion_norm,
                    )
                )

        decode_levels = min(4, len(rgb_feat_channels), depth)
        decoder_in_channels = rgb_feat_channels[-decode_levels:]
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
        else:
            raise ValueError(f"Unknown freeze_rgb mode: {freeze_rgb}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 4:
            raise RuntimeError(f"RGB+GVI mid-fusion expected RGB+NIR input with 4 channels, got {x.shape[1]}")
        rgb = x[:, :3]
        gvi = self.gvi_module(x[:, :4])

        rgb_feats = [align(feat) for align, feat in zip(self.rgb_align, list(self.rgb_encoder(rgb)[1:]))]
        gvi_feats = [align(feat) for align, feat in zip(self.gvi_align, list(self.gvi_encoder(gvi)[1:]))]
        fused: List[torch.Tensor] = list(rgb_feats)

        active = [idx for idx in self.progressive_level_indices if 0 <= idx < len(fused)]
        for block, idx in zip(self.fusion_blocks, active):
            fused[idx] = block(fused[idx], [gvi_feats[idx]])

        decode_feats = fused[-self.decode_levels:]
        logits = self.decoder(list(reversed(decode_feats)))
        return logits
