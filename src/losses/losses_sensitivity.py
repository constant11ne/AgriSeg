from __future__ import annotations

from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bchw_like(logits: torch.Tensor, targets: torch.Tensor):
    if logits.ndim == 4 and targets.ndim == 4:
        c = logits.shape[1]
        if targets.shape[1] != c and targets.shape[-1] == c:
            targets = targets.permute(0, 3, 1, 2).contiguous()
    return logits.float(), targets.float()


def _flatten_probs_targets(logits: torch.Tensor, targets: torch.Tensor):
    logits, targets = _to_bchw_like(logits, targets)
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), probs.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)
    return probs, targets


class SoftBCEWithLogitsLossMultiLabel(nn.Module):

    def __init__(
        self,
        smooth: float = 0.1,
        class_weights: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        if class_weights is not None:
            cw = torch.tensor(list(class_weights), dtype=torch.float32)
            self.register_buffer("class_weights", cw.view(1, -1, 1, 1))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _to_bchw_like(logits, targets)
        if self.smooth > 0.0:
            targets = targets.clamp(self.smooth / 2.0, 1.0 - self.smooth / 2.0)
        loss = self.bce(logits, targets)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()


class BCEWithLogitsLossMultiLabel(nn.Module):
    def __init__(self, class_weights: Optional[Iterable[float]] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        if class_weights is not None:
            cw = torch.tensor(list(class_weights), dtype=torch.float32)
            self.register_buffer("class_weights", cw.view(1, -1, 1, 1))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _to_bchw_like(logits, targets)
        loss = self.bce(logits, targets)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    p: int = 1,
) -> torch.Tensor:
    probs, targets = _flatten_probs_targets(logits, targets)
    intersection = (probs * targets).sum(dim=2)
    union = probs.pow(p).sum(dim=2) + targets.pow(p).sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probs, targets = _flatten_probs_targets(logits, targets)
    tp = (probs * targets).sum(dim=2)
    fn = ((1.0 - probs) * targets).sum(dim=2)
    fp = (probs * (1.0 - targets)).sum(dim=2)
    t = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - t.mean()


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    logits, targets = _to_bchw_like(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = ((1.0 - pt).pow(gamma)) * bce
    if alpha is not None:
        alpha_t = torch.tensor(list(alpha), dtype=loss.dtype, device=loss.device).view(1, -1, 1, 1)
        loss = loss * alpha_t
    return loss.mean()


class MultiLoss(nn.Module):
    def __init__(
        self,
        mode: str = "bce_dice",
        class_weights: Optional[Iterable[float]] = None,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[Iterable[float]] = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        dice_smooth: float = 1e-6,
        tversky_smooth: float = 1e-6,
        soft_bce_smooth: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.class_weights = list(class_weights) if class_weights is not None else None
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        self.focal_alpha = list(focal_alpha) if focal_alpha is not None else None
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.dice_smooth = dice_smooth
        self.tversky_smooth = tversky_smooth
        self.soft_bce_smooth = soft_bce_smooth
        self.bce = BCEWithLogitsLossMultiLabel(class_weights=self.class_weights)
        self.soft_bce = SoftBCEWithLogitsLossMultiLabel(
            smooth=self.soft_bce_smooth, class_weights=self.class_weights
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "soft_bce_dice":
            return self.bce_weight * self.soft_bce(logits, targets) + self.dice_weight * binary_dice_loss(
                logits, targets, smooth=self.dice_smooth
            )
        if self.mode == "bce_dice":
            return self.bce_weight * self.bce(logits, targets) + self.dice_weight * binary_dice_loss(
                logits, targets, smooth=self.dice_smooth
            )
        if self.mode == "bce_tversky":
            return self.bce_weight * self.bce(logits, targets) + self.tversky_weight * tversky_loss(
                logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta, smooth=self.tversky_smooth
            )
        if self.mode == "dice_focal":
            return self.dice_weight * binary_dice_loss(
                logits, targets, smooth=self.dice_smooth
            ) + self.focal_weight * binary_focal_loss(
                logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha
            )
        if self.mode == "focal_tversky_mix":
            return self.focal_weight * binary_focal_loss(
                logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha
            ) + self.tversky_weight * tversky_loss(
                logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta, smooth=self.tversky_smooth
            )
        raise ValueError(f"This sensitivity script supports only soft_bce_dice, bce_dice, bce_tversky, dice_focal, focal_tversky_mix. Got: {self.mode}")
