from __future__ import annotations

import torch.nn as nn


def freeze_except_biases(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False