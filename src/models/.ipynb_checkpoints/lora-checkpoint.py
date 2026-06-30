from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):

    def __init__(
        self,
        conv: nn.Conv2d,
        rank: int,
        alpha: int,
    ) -> None:
        super().__init__()
        assert rank > 0, "Rank must be positive"
        self.rank = rank
        self.alpha = alpha

        self.conv = conv

        for p in self.conv.parameters():
            p.requires_grad = False

        in_dim = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        out_dim = conv.out_channels

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

        import math
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.rank > 0:
            delta_w = (self.lora_B @ self.lora_A).view(
                self.conv.out_channels,
                self.conv.in_channels,
                self.conv.kernel_size[0],
                self.conv.kernel_size[1],
            )

            delta_w = delta_w * (self.alpha / self.rank)
            weight = self.conv.weight + delta_w
        else:
            weight = self.conv.weight
        return F.conv2d(
            x,
            weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: int = 8,
    module_filter: Optional[callable] = None,
    target_replace_module: Optional[type] = nn.Conv2d,
) -> None:
    import math

    def _replace(module: nn.Module, name: str) -> None:
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, target_replace_module) and (
                module_filter is None or module_filter(full_name, child)
            ):
                lora_layer = LoRAConv2d(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
            else:
                _replace(child, full_name)

    _replace(model, "")
