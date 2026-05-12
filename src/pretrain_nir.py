from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.datamodules.agri_vision import AgricultureVisionDataset, ANOMALY_CLASSES
from src.models.nir_pretrain import NIRLiteClassifierExportable, SMPNIRClassifierExportable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NIRClassificationDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            split: str,
            img_size: Tuple[int, int],
            augment: bool,
    ) -> None:
        self.base = AgricultureVisionDataset(
            root=data_root,
            split=split,
            transform=None,
            img_size=img_size,
            use_nir=True,
        )
        from src.datamodules.agri_vision import get_transforms
        self.transform = get_transforms(img_size=img_size, augment=augment, use_nir=True)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, mask = self.base[idx]
        nir = image[3:4]
        labels = (mask.view(mask.shape[0], -1).amax(dim=1) > 0).float()
        return nir, labels


class NIRClassificationDatasetFast(Dataset):
    def __init__(self, data_root: str, split: str, img_size: Tuple[int, int], augment: bool) -> None:
        from src.datamodules.agri_vision import _load_image, _load_mask, get_transforms
        self.data_root = data_root
        self.split = split
        self.img_dir = os.path.join(data_root, split, "images")
        self.mask_dir = os.path.join(data_root, split, "masks")
        self.ids = []
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.startswith('.'):
                continue
            base, _ = os.path.splitext(filename)
            self.ids.append(base)
        self._load_image = _load_image
        self._load_mask = _load_mask
        self.transform = get_transforms(img_size=img_size, augment=augment, use_nir=True)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, sample_id + ".tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, sample_id + ".png")
        image = self._load_image(img_path, use_nir=True)
        mask = self._load_mask(os.path.join(self.mask_dir, sample_id))
        mask_hwc = np.transpose(mask, (1, 2, 0))
        out = self.transform(image, mask_hwc)
        image_t = out["image"]
        mask_t = out["mask"]
        nir = image_t[3:4]
        labels = (mask_t.view(mask_t.shape[0], -1).amax(dim=1) > 0).float()
        return nir, labels


def build_dataloaders(args: argparse.Namespace):
    train_ds = NIRClassificationDatasetFast(
        data_root=args.data_root,
        split="train",
        img_size=tuple(args.img_size),
        augment=True,
    )
    val_ds = NIRClassificationDatasetFast(
        data_root=args.data_root,
        split="val",
        img_size=tuple(args.img_size),
        augment=False,
    )

    dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = not args.no_persistent_workers
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader


def compute_pos_weight(loader: DataLoader, num_classes: int) -> torch.Tensor:
    pos = torch.zeros(num_classes, dtype=torch.float64)
    total = 0
    for _, y in loader:
        pos += y.sum(dim=0).double()
        total += y.shape[0]
    neg = total - pos
    return ((neg + 1.0) / (pos + 1.0)).float()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int | None = None,
             epochs: int | None = None) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    tp = None
    fp = None
    fn = None
    val_bar = tqdm(loader, desc=(f"Val   {epoch}/{epochs}" if epoch is not None and epochs is not None else "Val"),
                   ncols=120, leave=True)
    for x, y in val_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_value = float(loss.item())
        loss_sum += loss_value
        n_batches += 1
        val_bar.set_postfix(loss=loss_value)

        pred = (torch.sigmoid(logits) >= 0.5).float()
        batch_tp = (pred * y).sum(dim=0)
        batch_fp = (pred * (1.0 - y)).sum(dim=0)
        batch_fn = ((1.0 - pred) * y).sum(dim=0)
        if tp is None:
            tp, fp, fn = batch_tp, batch_fp, batch_fn
        else:
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "macro_f1": float(f1.mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
    }


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, metrics: Dict[str, float],
                    args: argparse.Namespace) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "args": vars(args),
        "classes": ANOMALY_CLASSES,
    }
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain NIR branches on image-level multilabel anomaly classification")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--nir-branch", type=str, choices=["lite", "smp"], required=True)
    parser.add_argument("--encoder", type=str, default="timm-efficientnet-b4")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--nir-width-mult", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--save-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp

    train_loader, val_loader = build_dataloaders(args)

    num_classes = len(ANOMALY_CLASSES)
    if args.nir_branch == "lite":
        model = NIRLiteClassifierExportable(num_classes=num_classes, width_mult=args.nir_width_mult)
    else:
        model = SMPNIRClassifierExportable(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            depth=args.depth,
            num_classes=num_classes,
        )
    model = model.to(device)

    print("[INFO] Computing class-wise pos_weight from the training loader...")
    pos_weight = compute_pos_weight(train_loader, num_classes=num_classes).to(device)
    print("[INFO] pos_weight ready")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = max(1, args.epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(
        f"[INFO] Device: {device} | AMP: {use_amp} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_name = args.save_name or f"nir_pretrain_{args.nir_branch}_{timestamp}"
    out_dir = os.path.join(args.runs_dir, save_name)
    os.makedirs(out_dir, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        train_bar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=120, leave=True)
        for x, y in train_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.item())
            train_loss_sum += loss_value
            n_batches += 1
            train_bar.set_postfix(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device, epoch=epoch, epochs=args.epochs)
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_batches, 1),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"val_macro_f1={record['val_macro_f1']:.4f} "
            f"lr={record['lr']:.6g}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, record, args)
        if record["val_macro_f1"] > best_f1:
            best_f1 = record["val_macro_f1"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, record, args)

        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    encoder_best = torch.load(best_path, map_location="cpu")
    torch.save(
        {
            "encoder_state_dict": encoder_best["encoder_state_dict"],
            "args": encoder_best["args"],
            "metrics": encoder_best["metrics"],
            "classes": encoder_best["classes"],
        },
        os.path.join(out_dir, "best_encoder_only.pt"),
    )
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
