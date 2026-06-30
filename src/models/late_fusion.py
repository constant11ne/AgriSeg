from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class LateFusionWrapper(nn.Module):
    def __init__(self, rgb_model: nn.Module, nir_model: Optional[nn.Module] = None, ndvi_model: Optional[nn.Module] = None, fusion_mode: str = "rgb_nir", freeze_branches: bool = False) -> None:
        super().__init__()
        self.rgb_model = rgb_model
        self.nir_model = nir_model
        self.ndvi_model = ndvi_model
        self.fusion_mode = fusion_mode
        self.branch_names = self._validate_and_get_branch_names()
        if freeze_branches:
            _freeze(self.rgb_model)
            if self.nir_model is not None:
                _freeze(self.nir_model)
            if self.ndvi_model is not None:
                _freeze(self.ndvi_model)

    def _validate_and_get_branch_names(self) -> List[str]:
        if self.fusion_mode == "rgb_nir":
            if self.nir_model is None:
                raise ValueError("rgb_nir requires nir_model")
            return ["rgb", "nir"]
        if self.fusion_mode == "rgb_ndvi":
            if self.ndvi_model is None:
                raise ValueError("rgb_ndvi requires ndvi_model")
            return ["rgb", "ndvi"]
        if self.fusion_mode == "rgb_nir_ndvi":
            if self.nir_model is None or self.ndvi_model is None:
                raise ValueError("rgb_nir_ndvi requires nir_model and ndvi_model")
            return ["rgb", "nir", "ndvi"]
        raise ValueError(f"Unknown late fusion mode: {self.fusion_mode}")

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        rgb = x[:, :3]
        nir = None
        ndvi = None
        if self.fusion_mode == "rgb_nir":
            if x.shape[1] < 4:
                raise RuntimeError(f"rgb_nir expected >=4 channels, got {x.shape[1]}")
            nir = x[:, 3:4]
        elif self.fusion_mode == "rgb_ndvi":
            if x.shape[1] == 4:
                ndvi = x[:, 3:4]
            elif x.shape[1] >= 5:
                ndvi = x[:, 4:5]
            else:
                raise RuntimeError(f"rgb_ndvi expected 4 or 5 channels, got {x.shape[1]}")
        elif self.fusion_mode == "rgb_nir_ndvi":
            if x.shape[1] < 5:
                raise RuntimeError(f"rgb_nir_ndvi expected >=5 channels, got {x.shape[1]}")
            nir = x[:, 3:4]
            ndvi = x[:, 4:5]
        return rgb, nir, ndvi

    def _branch_logits(self, x: torch.Tensor) -> List[torch.Tensor]:
        rgb, nir, ndvi = self._split_inputs(x)
        logits = [self.rgb_model(rgb)]
        if "nir" in self.branch_names:
            assert self.nir_model is not None and nir is not None
            logits.append(self.nir_model(nir))
        if "ndvi" in self.branch_names:
            assert self.ndvi_model is not None and ndvi is not None
            logits.append(self.ndvi_model(ndvi))
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(self._branch_logits(x), dim=0).mean(dim=0)


class WeightedLogitsFusion(LateFusionWrapper):
    def __init__(self, rgb_model: nn.Module, nir_model: Optional[nn.Module] = None, ndvi_model: Optional[nn.Module] = None, fusion_mode: str = "rgb_nir", freeze_branches: bool = True) -> None:
        super().__init__(rgb_model, nir_model, ndvi_model, fusion_mode, freeze_branches)
        self.weights = nn.Parameter(torch.zeros(len(self.branch_names)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._branch_logits(x)
        w = torch.softmax(self.weights, dim=0)
        out = 0.0
        for i, branch_logits in enumerate(logits):
            out = out + w[i] * branch_logits
        return out


class PerClassWeightedFusion(LateFusionWrapper):
    def __init__(self, rgb_model: nn.Module, nir_model: Optional[nn.Module] = None, ndvi_model: Optional[nn.Module] = None, num_classes: int = 9, fusion_mode: str = "rgb_nir", freeze_branches: bool = True) -> None:
        super().__init__(rgb_model, nir_model, ndvi_model, fusion_mode, freeze_branches)
        self.class_weights = nn.Parameter(torch.zeros(len(self.branch_names), num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._branch_logits(x)
        w = torch.softmax(self.class_weights, dim=0)
        out = 0.0
        for i, branch_logits in enumerate(logits):
            out = out + w[i].view(1, -1, 1, 1) * branch_logits
        return out


def build_late_fusion(fusion_method: str, rgb_model: nn.Module, nir_model: Optional[nn.Module] = None, ndvi_model: Optional[nn.Module] = None, num_classes: Optional[int] = None, freeze_branches: bool = True, fusion_mode: str = "rgb_nir") -> nn.Module:
    if fusion_method == "late_avg":
        return LateFusionWrapper(rgb_model, nir_model, ndvi_model, fusion_mode, freeze_branches)
    if fusion_method == "late_weighted":
        return WeightedLogitsFusion(rgb_model, nir_model, ndvi_model, fusion_mode, freeze_branches)
    if fusion_method == "late_per_class":
        if num_classes is None:
            raise ValueError("num_classes required for late_per_class")
        return PerClassWeightedFusion(rgb_model, nir_model, ndvi_model, num_classes, fusion_mode, freeze_branches)
    raise ValueError(f"Unknown late fusion method: {fusion_method}")
