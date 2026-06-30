from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .nir_branches import NIREncoderLite
from .backbone_utils import build_rgb_encoder


class NIRLiteClassifier(nn.Module):
    def __init__(self, num_classes: int = 9, width_mult: float = 0.5) -> None:
        super().__init__()
        self.encoder = NIREncoderLite(in_channels=1, width_mult=width_mult)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.encoder.out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        last = feats[-1]
        pooled = self.pool(last).flatten(1)
        return self.head(pooled)


def _replace_first_conv_any(model: nn.Module, new_in_channels: int) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            old = module
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=(old.bias is not None),
                padding_mode=old.padding_mode,
            )
            with torch.no_grad():
                if new_in_channels == 1:
                    new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                elif new_in_channels > 3:
                    new_conv.weight[:, :3].copy_(old.weight)
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    for c in range(3, new_in_channels):
                        new_conv.weight[:, c:c + 1].copy_(mean_w)
                else:
                    raise ValueError(f"Unsupported new_in_channels={new_in_channels}")
                if old.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old.bias)
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return
    raise RuntimeError("Could not find first Conv2d with 3 input channels to replace.")


class SMPNIRClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b4",
        encoder_weights: Optional[str] = "imagenet",
        depth: int = 5,
        num_classes: int = 9,
    ) -> None:
        super().__init__()
        self.encoder, out_channels = build_rgb_encoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            depth=depth,
        )
        _replace_first_conv_any(self.encoder, new_in_channels=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        last = feats[-1]
        pooled = self.pool(last).flatten(1)
        return self.head(pooled)


class EncoderOnlyStateMixin:
    def encoder_state_dict(self) -> dict:
        if not hasattr(self, "encoder"):
            raise AttributeError("Model has no attribute 'encoder'.")
        return self.encoder.state_dict()


class NIRLiteClassifierExportable(EncoderOnlyStateMixin, NIRLiteClassifier):
    pass


class SMPNIRClassifierExportable(EncoderOnlyStateMixin, SMPNIRClassifier):
    pass
