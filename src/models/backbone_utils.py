from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp

    HAS_SMP = True
except Exception:
    smp = None
    HAS_SMP = False


def build_rgb_encoder(
        encoder_name: str = "timm-efficientnet-b4",
        encoder_weights: Optional[str] = "imagenet",
        depth: int = 5,
) -> Tuple[nn.Module, List[int]]:
    if not HAS_SMP:
        raise ImportError(
            "segmentation_models_pytorch is required for build_rgb_encoder. "
            "Install it with: pip install segmentation-models-pytorch"
        )
    encoder = smp.encoders.get_encoder(
        encoder_name,
        in_channels=3,
        depth=depth,
        weights=encoder_weights,
    )
    out_channels: List[int] = list(encoder.out_channels)
    return encoder, out_channels


def adapt_first_conv(
        model: nn.Module,
        new_in_channels: int,
        init_mode: str = "copy-mean",
) -> None:
    if new_in_channels == 3:
        return
    channel_map = {"copy-r": 0, "copy-g": 1, "copy-b": 2}
    copy_idx = channel_map.get(init_mode, None)

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
                if init_mode == "random":
                    if new_in_channels >= 3:
                        new_conv.weight[:, :3] = old.weight.data
                elif new_in_channels == 1:
                    if init_mode == "copy-mean" or copy_idx is None:
                        new_conv.weight[:, 0:1] = old.weight.data.mean(dim=1, keepdim=True)
                    else:
                        new_conv.weight[:, 0:1] = old.weight.data[:, copy_idx:copy_idx + 1]
                else:
                    new_conv.weight[:, :3] = old.weight.data
                    if init_mode == "copy-mean":
                        rgb_mean = old.weight.data.mean(dim=1, keepdim=True)
                        for c in range(3, new_in_channels):
                            new_conv.weight[:, c:c + 1] = rgb_mean
                    elif copy_idx is not None:
                        for c in range(3, new_in_channels):
                            new_conv.weight[:, c:c + 1] = old.weight.data[:, copy_idx:copy_idx + 1]
                if old.bias is not None and new_conv.bias is not None:
                    new_conv.bias.data = old.bias.data
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            return


def freeze_encoder_full(encoder: nn.Module) -> None:
    for p in encoder.parameters():
        p.requires_grad = False


def freeze_encoder_stages(encoder: nn.Module, freeze_n_stages: int) -> None:
    stages_found = 0
    for child_name, child in encoder.named_children():
        if stages_found >= freeze_n_stages:
            break
        for p in child.parameters():
            p.requires_grad = False
        stages_found += 1


def unfreeze_last_n_stages(encoder: nn.Module, n: int) -> None:
    children = list(encoder.named_children())
    for _, child in children[-n:]:
        for p in child.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
