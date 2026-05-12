from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import torch


class SegmentationMetrics:
    def __init__(
            self,
            num_classes: int,
            threshold: float = 0.5,
            rare_class_ids: Optional[Iterable[int]] = None,
            device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        if rare_class_ids is None:
            self.rare_class_ids = list(range(1, num_classes))
        else:
            self.rare_class_ids = list(rare_class_ids)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        with torch.no_grad():
            if self.tp.device != preds.device:
                self.device = preds.device
                self.reset()
            if preds.ndim == 4 and targets.ndim == 4:
                c = preds.shape[1]
                if targets.shape[1] != c and targets.shape[-1] == c:
                    targets = targets.permute(0, 3, 1, 2).contiguous()
            probs = torch.sigmoid(preds)
            pred_bin = probs > self.threshold
            target_bin = targets > 0.5
            pred_flat = pred_bin.view(pred_bin.size(0), pred_bin.size(1), -1)
            target_flat = target_bin.view(target_bin.size(0), target_bin.size(1), -1)
            tp = (pred_flat & target_flat).sum(dim=2).sum(dim=0).to(torch.float64)
            fp = (pred_flat & (~target_flat)).sum(dim=2).sum(dim=0).to(torch.float64)
            fn = ((~pred_flat) & target_flat).sum(dim=2).sum(dim=0).to(torch.float64)
            union = (pred_flat | target_flat).sum(dim=2).sum(dim=0).to(torch.float64)
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.intersection += tp
            self.union += union

    def compute(self) -> Dict[str, float]:
        intersection = self.intersection.detach().cpu().numpy()
        union = self.union.detach().cpu().numpy()
        tp = self.tp.detach().cpu().numpy()
        fp = self.fp.detach().cpu().numpy()
        fn = self.fn.detach().cpu().numpy()

        iou = np.divide(
            intersection,
            union + 1e-10,
            out=np.zeros_like(intersection),
            where=(union != 0),
        )
        f1 = np.divide(
            2.0 * tp,
            2.0 * tp + fp + fn + 1e-10,
            out=np.zeros_like(tp),
            where=((2.0 * tp + fp + fn) != 0),
        )

        valid_iou = union > 0
        miou = float(iou[valid_iou].mean()) if valid_iou.any() else 0.0

        valid_f1 = (2.0 * tp + fp + fn) > 0
        macro_f1 = float(f1[valid_f1].mean()) if valid_f1.any() else 0.0

        rare_valid = [idx for idx in self.rare_class_ids if 0 <= idx < self.num_classes and union[idx] > 0]
        rare_miou = float(np.mean(iou[rare_valid])) if rare_valid else 0.0

        result: Dict[str, float] = {f"iou_class_{i}": float(v) for i, v in enumerate(iou)}
        result["miou"] = miou
        result["macro_f1"] = macro_f1
        result["rare_miou"] = rare_miou
        return result


IoUMetric = SegmentationMetrics
