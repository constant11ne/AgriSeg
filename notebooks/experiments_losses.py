from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.datamodules.agri_vision import ANOMALY_CLASSES, build_dataloaders
from src.losses.losses import MultiLoss
from src.models.lora import freeze_model, inject_lora
from src.train import build_model, init_first_conv_nir
from src.utils.metrics import SegmentationMetrics


@dataclass
class Config:
    data_root: str
    img_size: tuple = (512, 512)
    model_name: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    use_nir: bool = True
    nir_fusion: str = "early"
    nir_init_mode: str = "copy-r"
    norm_type: str = "bn"
    batch_size: int = 8
    num_workers: int = 4
    oversample_rare: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    max_train_steps: Optional[int] = 1000
    max_val_steps: Optional[int] = None
    class_weights: Optional[List[float]] = None
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: int = 8
    rare_class_ids: Optional[List[int]] = None
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


LOSS_EXPERIMENTS: List[Dict[str, Any]] = [
    {"name": "bce", "mode": "bce", "params": {}},
    {"name": "dice", "mode": "dice", "params": {}},
    {"name": "bce_dice", "mode": "bce_dice", "params": {"bce_weight": 0.5, "dice_weight": 0.5}},
    {"name": "tversky", "mode": "tversky", "params": {"tversky_alpha": 0.5, "tversky_beta": 0.5}},
    {
        "name": "bce_tversky",
        "mode": "bce_tversky",
        "params": {"bce_weight": 0.5, "tversky_weight": 0.5, "tversky_alpha": 0.5, "tversky_beta": 0.5},
    },
    {"name": "focal", "mode": "focal", "params": {"focal_gamma": 2.0}},
    {
        "name": "bce_focal",
        "mode": "bce_focal",
        "params": {"bce_weight": 0.5, "focal_weight": 0.5, "focal_gamma": 2.0},
    },
    {
        "name": "dice_focal",
        "mode": "dice_focal",
        "params": {"dice_weight": 0.5, "focal_weight": 0.5, "focal_gamma": 2.0},
    },
    {
        "name": "focal_tversky",
        "mode": "focal_tversky",
        "params": {"tversky_alpha": 0.5, "tversky_beta": 0.5},
    },
    {"name": "lovasz", "mode": "lovasz", "params": {}},
    {
        "name": "bce_lovasz",
        "mode": "bce_lovasz",
        "params": {"bce_weight": 0.5, "lovasz_weight": 0.5},
    },
    {
        "name": "asymmetric",
        "mode": "asymmetric",
        "params": {"asym_gamma_neg": 4.0, "asym_gamma_pos": 1.0, "asym_clip": 0.05},
    },
    {
        "name": "soft_bce_dice",
        "mode": "soft_bce_dice",
        "params": {"label_smoothing": 0.05, "bce_weight": 0.5, "dice_weight": 0.5},
    },
    {
        "name": "focal_tversky_mix",
        "mode": "focal_tversky_mix",
        "params": {"focal_weight": 0.5, "tversky_weight": 0.5, "focal_gamma": 2.0, "tversky_alpha": 0.5,
                   "tversky_beta": 0.5},
    },
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_criterion(cfg: Config, loss_mode: str, loss_params: Dict[str, Any]) -> MultiLoss:
    return MultiLoss(
        mode=loss_mode,
        class_weights=cfg.class_weights,
        tversky_alpha=loss_params.get("tversky_alpha", 0.5),
        tversky_beta=loss_params.get("tversky_beta", 0.5),
        focal_gamma=loss_params.get("focal_gamma", 2.0),
        focal_alpha=loss_params.get("focal_alpha", None),
        bce_weight=loss_params.get("bce_weight", 0.5),
        dice_weight=loss_params.get("dice_weight", 0.5),
        focal_weight=loss_params.get("focal_weight", 0.5),
        tversky_weight=loss_params.get("tversky_weight", 0.5),
        lovasz_weight=loss_params.get("lovasz_weight", 0.5),
        label_smoothing=loss_params.get("label_smoothing", 0.05),
        asym_gamma_neg=loss_params.get("asym_gamma_neg", 4.0),
        asym_gamma_pos=loss_params.get("asym_gamma_pos", 1.0),
        asym_clip=loss_params.get("asym_clip", 0.05),
    ).to(cfg.device)


def train_one(cfg: Config, experiment: Dict[str, Any], train_loader, val_loader) -> Dict[str, Any]:
    num_classes = len(ANOMALY_CLASSES)
    in_channels = 4 if cfg.use_nir else 3
    set_seed(cfg.seed)

    model = build_model(
        cfg.model_name,
        cfg.encoder_name,
        cfg.encoder_weights,
        num_classes,
        in_channels,
        nir_fusion=cfg.nir_fusion,
        norm_type=cfg.norm_type,
        use_nir=cfg.use_nir,
    )
    if cfg.use_nir and cfg.nir_fusion == "early" and cfg.nir_init_mode != "random":
        init_first_conv_nir(model, cfg.nir_init_mode)
    if cfg.use_lora:
        freeze_model(model)
        inject_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    model = model.to(cfg.device)

    criterion = build_criterion(cfg, experiment["mode"], experiment["params"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    metric = SegmentationMetrics(num_classes=num_classes, rare_class_ids=cfg.rare_class_ids)

    history: List[Dict[str, Any]] = []
    best_miou = 0.0
    best_macro_f1 = 0.0
    best_rare_miou = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_bar = tqdm(enumerate(train_loader), total=cfg.max_train_steps or len(train_loader),
                         desc=f"[{experiment['name']}] Epoch {epoch + 1}/{cfg.epochs} train")
        for i, (images, masks) in train_bar:
            if cfg.max_train_steps is not None and i >= cfg.max_train_steps:
                break
            images = images.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1
            train_bar.set_postfix(loss=float(loss.item()))
        avg_train_loss = train_loss_sum / max(1, train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        metric.reset()
        val_bar = tqdm(enumerate(val_loader), total=cfg.max_val_steps or len(val_loader),
                       desc=f"[{experiment['name']}] Epoch {epoch + 1}/{cfg.epochs} val")
        with torch.no_grad():
            for j, (images, masks) in val_bar:
                if cfg.max_val_steps is not None and j >= cfg.max_val_steps:
                    break
                images = images.to(cfg.device, non_blocking=True)
                masks = masks.to(cfg.device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_sum += float(loss.item())
                val_batches += 1
                metric.update(outputs, masks)
        avg_val_loss = val_loss_sum / max(1, val_batches)
        metrics = metric.compute()
        miou = float(metrics["miou"])
        macro_f1 = float(metrics["macro_f1"])
        rare_miou = float(metrics["rare_miou"])
        best_miou = max(best_miou, miou)
        best_macro_f1 = max(best_macro_f1, macro_f1)
        best_rare_miou = max(best_rare_miou, rare_miou)

        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **metrics,
        }
        history.append(history_row)
        print(
            f"[{experiment['name']}] epoch {epoch + 1}/{cfg.epochs} — "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"mIoU={miou:.4f} macroF1={macro_f1:.4f} rare_mIoU={rare_miou:.4f} "
            f"(best mIoU={best_miou:.4f})"
        )

    return {
        "loss_name": experiment["name"],
        "loss_mode": experiment["mode"],
        "best_miou": best_miou,
        "last_miou": history[-1]["miou"],
        "best_macro_f1": best_macro_f1,
        "best_rare_miou": best_rare_miou,
        "history": history,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the first benchmark series of segmentation losses")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--class-weights", type=str, default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rare-class-ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        class_weights=json.loads(args.class_weights) if args.class_weights else None,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        rare_class_ids=args.rare_class_ids,
        seed=args.seed,
    )

    train_loader, val_loader = build_dataloaders(
        data_root=cfg.data_root,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=True,
        use_nir=cfg.use_nir,
        oversample_rare=cfg.oversample_rare,
        class_weights=cfg.class_weights,
    )

    out_dir = os.path.join("runs", "experiments_losses")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    summary_rows = []
    for experiment in LOSS_EXPERIMENTS:
        print(f"\nTraining with loss: {experiment['name']} ({experiment['mode']})")
        result = train_one(cfg, experiment, train_loader, val_loader)
        pd.DataFrame(result["history"]).to_csv(os.path.join(out_dir, f"history_{experiment['name']}.csv"), index=False)
        summary_rows.append(
            {
                "loss_name": result["loss_name"],
                "loss_mode": result["loss_mode"],
                "best_miou": result["best_miou"],
                "last_miou": result["last_miou"],
                "best_macro_f1": result["best_macro_f1"],
                "best_rare_miou": result["best_rare_miou"],
                "epochs": cfg.epochs,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(by="best_miou", ascending=False).reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\nFirst loss benchmark completed. Summary")
    print(summary)


if __name__ == "__main__":
    main()
