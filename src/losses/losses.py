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


def _make_class_weight_tensor(
        class_weights: list,
        ndim: int,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    cw = torch.tensor(class_weights, dtype=dtype, device=device)
    if ndim == 2:
        return cw.view(1, -1)
    if ndim == 3:
        return cw.view(1, -1, 1)
    return cw.view(1, -1, 1, 1)


class BCEWithLogitsLossMultiLabel(nn.Module):
    def __init__(self, class_weights: Optional[Iterable[float]] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.class_weights = list(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _to_bchw_like(logits, targets)
        loss = self.bce(logits, targets)
        if self.class_weights is not None:
            cw = _make_class_weight_tensor(self.class_weights, loss.ndim, loss.dtype, loss.device)
            loss = loss * cw
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
    dice = (2.0 * intersection + smooth) / (union + smooth)
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
    loss = (1.0 - pt).pow(gamma) * bce
    if alpha is not None:
        alpha_t = _make_class_weight_tensor(list(alpha), loss.ndim, loss.dtype, loss.device)
        loss = loss * alpha_t
    return loss.mean()


def focal_tversky_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1e-6,
) -> torch.Tensor:
    probs, targets = _flatten_probs_targets(logits, targets)
    tp = (probs * targets).sum(dim=2)
    fn = ((1.0 - probs) * targets).sum(dim=2)
    fp = (probs * (1.0 - targets)).sum(dim=2)
    t = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1.0 - t).pow(gamma).mean()


def soft_bce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.0,
        class_weights: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    logits, targets = _to_bchw_like(logits, targets)
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if class_weights is not None:
        cw = _make_class_weight_tensor(list(class_weights), loss.ndim, loss.dtype, loss.device)
        loss = loss * cw
    return loss.mean()


def asymmetric_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
) -> torch.Tensor:
    logits, targets = _to_bchw_like(logits, targets)
    probs = torch.sigmoid(logits)
    xs_pos = probs
    xs_neg = 1.0 - probs
    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    los_pos = targets * torch.log(xs_pos.clamp(min=eps)) * (1.0 - xs_pos).pow(gamma_pos)
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps)) * probs.pow(gamma_neg)
    loss = -(los_pos + los_neg)
    return loss.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_multilabel(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits, targets = _to_bchw_like(logits, targets)
    losses = []
    for c in range(logits.shape[1]):
        logit_c = logits[:, c].reshape(-1)
        target_c = targets[:, c].reshape(-1)
        losses.append(_lovasz_hinge_flat(logit_c, target_c))
    return torch.stack(losses).mean()


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
            lovasz_weight: float = 0.5,
            label_smoothing: float = 0.0,
            asym_gamma_neg: float = 4.0,
            asym_gamma_pos: float = 1.0,
            asym_clip: float = 0.05,
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
        self.lovasz_weight = lovasz_weight
        self.label_smoothing = label_smoothing
        self.asym_gamma_neg = asym_gamma_neg
        self.asym_gamma_pos = asym_gamma_pos
        self.asym_clip = asym_clip
        self.bce = BCEWithLogitsLossMultiLabel(class_weights=self.class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        if mode == "bce":
            return self.bce(logits, targets)
        if mode == "dice":
            return binary_dice_loss(logits, targets)
        if mode == "bce_dice":
            return self.bce_weight * self.bce(logits, targets) + self.dice_weight * binary_dice_loss(logits, targets)
        if mode == "tversky":
            return tversky_loss(logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta)
        if mode == "bce_tversky":
            return self.bce_weight * self.bce(logits, targets) + self.tversky_weight * tversky_loss(
                logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta
            )
        if mode == "focal":
            return binary_focal_loss(logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha)
        if mode == "bce_focal":
            return self.bce_weight * self.bce(logits, targets) + self.focal_weight * binary_focal_loss(
                logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha
            )
        if mode == "dice_focal":
            return self.dice_weight * binary_dice_loss(logits, targets) + self.focal_weight * binary_focal_loss(
                logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha
            )
        if mode == "focal_tversky":
            return focal_tversky_loss(
                logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta, gamma=self.focal_gamma
            )
        if mode == "lovasz":
            return lovasz_hinge_multilabel(logits, targets)
        if mode == "bce_lovasz":
            return self.bce_weight * self.bce(logits, targets) + self.lovasz_weight * lovasz_hinge_multilabel(logits,
                                                                                                              targets)
        if mode == "asymmetric":
            return asymmetric_loss(
                logits, targets,
                gamma_neg=self.asym_gamma_neg,
                gamma_pos=self.asym_gamma_pos,
                clip=self.asym_clip,
            )
        if mode == "soft_bce_dice":
            return self.bce_weight * soft_bce_loss(
                logits, targets, label_smoothing=self.label_smoothing, class_weights=self.class_weights
            ) + self.dice_weight * binary_dice_loss(logits, targets)
        if mode == "focal_tversky_mix":
            return self.focal_weight * binary_focal_loss(
                logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha
            ) + self.tversky_weight * tversky_loss(
                logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta
            )
        raise ValueError(f"Unknown loss mode: {mode}")
