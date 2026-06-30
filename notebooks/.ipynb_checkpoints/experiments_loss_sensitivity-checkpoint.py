"""Second-stage hyperparameter sensitivity experiments for top loss families.

This script runs compact parameter sweeps for:
  - Soft BCE + Dice
  - BCE + Dice
  - Focal + Tversky mix
  - Dice + Focal
  - BCE + Tversky

Each configuration is repeated multiple times with different seeds to reduce
randomness. The output includes:
  - per-repeat summaries
  - aggregated mean/std summaries across repeats
  - per-epoch histories for every repeat
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.datamodules.agri_vision import ANOMALY_CLASSES, build_dataloaders
from src.train import build_model, init_first_conv_nir
from src.models.lora import freeze_model, inject_lora
from src.models.bitfit import freeze_except_biases
from src.utils.metrics import SegmentationMetrics

from src.losses.losses_sensitivity import MultiLoss

from src.utils.loss_sensitivity_utils import build_loss_search_spaces, aggregate_repeat_summary


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
    freeze_encoder: bool = False
    use_lora: bool = False
    use_bitfit: bool = False
    class_weights: Optional[List[float]] = None
    batch_size: int = 8
    num_workers: int = 4
    oversample_rare: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    max_train_steps: Optional[int] = 1000
    max_val_steps: Optional[int] = None
    lora_rank: int = 4
    lora_alpha: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    base_seed: int = 42
    repeats: int = 3
    rare_class_ids: Optional[List[int]] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_repeat(
    cfg: Config,
    experiment_name: str,
    loss_mode: str,
    loss_params: Dict[str, Any],
    repeat_idx: int,
    seed: int,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    set_seed(seed)

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

    num_classes = len(ANOMALY_CLASSES)
    in_channels = 4 if cfg.use_nir else 3
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
    if cfg.freeze_encoder:
        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False
    if cfg.use_lora:
        freeze_model(model)
        inject_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    if cfg.use_bitfit:
        freeze_except_biases(model)
    model = model.to(cfg.device)

    criterion = MultiLoss(mode=loss_mode, class_weights=cfg.class_weights, **loss_params).to(cfg.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    metrics = SegmentationMetrics(num_classes=num_classes, rare_class_ids=cfg.rare_class_ids)

    history_rows: List[Dict[str, Any]] = []
    best_miou = -1.0
    best_macro_f1 = -1.0
    best_rare_miou = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_bar = tqdm(
            enumerate(train_loader),
            total=cfg.max_train_steps or len(train_loader),
            desc=f"[{experiment_name}] repeat {repeat_idx+1}/{cfg.repeats} epoch {epoch+1}/{cfg.epochs} train",
        )
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
        metrics.reset()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.inference_mode():
            val_bar = tqdm(
                enumerate(val_loader),
                total=cfg.max_val_steps or len(val_loader),
                desc=f"[{experiment_name}] repeat {repeat_idx+1}/{cfg.repeats} epoch {epoch+1}/{cfg.epochs} val",
            )
            for j, (images, masks) in val_bar:
                if cfg.max_val_steps is not None and j >= cfg.max_val_steps:
                    break
                images = images.to(cfg.device, non_blocking=True)
                masks = masks.to(cfg.device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_sum += float(loss.item())
                val_batches += 1
                metrics.update(outputs, masks)

        scores = metrics.compute()
        avg_val_loss = val_loss_sum / max(1, val_batches)
        best_miou = max(best_miou, float(scores["miou"]))
        best_macro_f1 = max(best_macro_f1, float(scores["macro_f1"]))
        best_rare_miou = max(best_rare_miou, float(scores["rare_miou"]))

        row = {
            "experiment_name": experiment_name,
            "loss_mode": loss_mode,
            "repeat_idx": repeat_idx,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **loss_params,
            **scores,
        }
        history_rows.append(row)
        print(
            f"[{experiment_name}] repeat {repeat_idx+1}/{cfg.repeats} epoch {epoch+1}/{cfg.epochs} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"miou={scores['miou']:.4f} macro_f1={scores['macro_f1']:.4f} rare_miou={scores['rare_miou']:.4f}"
        )

    history_df = pd.DataFrame(history_rows)
    last_row = history_df.iloc[-1].to_dict()
    summary = {
        "experiment_name": experiment_name,
        "loss_mode": loss_mode,
        "repeat_idx": repeat_idx,
        "seed": seed,
        **loss_params,
        "best_miou": float(best_miou),
        "last_miou": float(last_row["miou"]),
        "best_macro_f1": float(best_macro_f1),
        "last_macro_f1": float(last_row["macro_f1"]),
        "best_rare_miou": float(best_rare_miou),
        "last_rare_miou": float(last_row["rare_miou"]),
        "epochs": int(cfg.epochs),
    }
    return summary, history_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated hyperparameter sensitivity experiments for top loss families")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--class-weights", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--rare-class-ids", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out-dir", type=str, default=os.path.join("runs", "loss_sensitivity"))
    parser.add_argument("--unconstrained-tversky", action="store_true", help="Search alpha and beta independently instead of beta = 1 - alpha")
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
        repeats=args.repeats,
        base_seed=args.base_seed,
        rare_class_ids=args.rare_class_ids,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    experiments = build_loss_search_spaces(constrained_tversky=not args.unconstrained_tversky)
    pd.DataFrame(experiments).drop(columns=["loss_params"]).to_csv(os.path.join(args.out_dir, "experiment_plan.csv"), index=False)

    all_repeat_summaries: List[Dict[str, Any]] = []
    all_histories: List[pd.DataFrame] = []

    for exp in experiments:
        for repeat_idx in range(cfg.repeats):
            seed = cfg.base_seed + repeat_idx
            summary, history_df = train_one_repeat(
                cfg=cfg,
                experiment_name=exp["experiment_name"],
                loss_mode=exp["loss_mode"],
                loss_params=exp["loss_params"],
                repeat_idx=repeat_idx,
                seed=seed,
            )
            all_repeat_summaries.append(summary)
            all_histories.append(history_df)
            history_path = os.path.join(args.out_dir, f"history__{exp['experiment_name']}__repeat-{repeat_idx+1}.csv")
            history_df.to_csv(history_path, index=False)

    repeats_df = pd.DataFrame(all_repeat_summaries)
    repeats_df.to_csv(os.path.join(args.out_dir, "summary_repeats.csv"), index=False)

    agg_df = aggregate_repeat_summary(repeats_df)
    agg_df.to_csv(os.path.join(args.out_dir, "summary_aggregated.csv"), index=False)

    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(os.path.join(args.out_dir, "history_all.csv"), index=False)

    print("\n=== Loss sensitivity benchmark completed ===")
    print(agg_df[[
        "experiment_name",
        "loss_mode",
        "best_miou_mean",
        "best_miou_std",
        "best_macro_f1_mean",
        "best_rare_miou_mean",
        "repeats",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
