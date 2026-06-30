#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.datamodules.agri_vision import build_dataloaders, ANOMALY_CLASSES
from src.losses.losses_sensitivity import MultiLoss
from src.train import build_model, init_first_conv_nir

try:
    from src.utils.metrics import SegmentationMetrics
except Exception:
    from src.utils.metrics import IoUMetric as SegmentationMetrics

TARGET_METRIC = "best_miou"


# -----------------------------
# Small utilities
# -----------------------------
@dataclass
class StageSpec:
    name: str
    parameter: str
    values: List[float]
    epochs: int
    initial_repeats: int = 1
    final_repeats: int = 3
    shortlist_k: int | None = None
    shortlist_frac: float | None = None

    def validate(self) -> None:
        if self.initial_repeats < 1:
            raise ValueError("initial_repeats must be >= 1")
        if self.final_repeats < self.initial_repeats:
            raise ValueError("final_repeats must be >= initial_repeats")
        if self.shortlist_k is not None and self.shortlist_k < 1:
            raise ValueError("shortlist_k must be >= 1")


def unique_sorted(values: Iterable[float], ndigits: int = 8) -> List[float]:
    return sorted({round(float(v), ndigits) for v in values})


def local_grid_around(
    center: float,
    deltas: List[float],
    lo: float | None = None,
    hi: float | None = None,
) -> List[float]:
    vals = [center + d for d in deltas]
    if lo is not None:
        vals = [max(lo, v) for v in vals]
    if hi is not None:
        vals = [min(hi, v) for v in vals]
    return unique_sorted(vals)


def candidate_name(loss_name: str, params: Dict[str, Any]) -> str:
    parts = [loss_name]
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}-{v:g}")
        else:
            parts.append(f"{k}-{v}")
    return "__".join(parts)


def robust_score(mean: float, std: float, stability_weight: float = 0.25) -> float:
    std = 0.0 if pd.isna(std) else float(std)
    return float(mean) - stability_weight * std


def summarize_repeats(
    df: pd.DataFrame,
    target: str,
    stability_weight: float = 0.25,
) -> pd.DataFrame:
    grouped = (
        df.groupby("candidate_name", dropna=False)
        .agg(
            target_mean=(target, "mean"),
            target_std=(target, "std"),
            repeats=(target, "count"),
            stage_name=("stage_name", "first"),
            loss_name=("loss_name", "first"),
            params_json=("params_json", "first"),
        )
        .reset_index()
    )
    grouped["robust_score"] = grouped.apply(
        lambda r: robust_score(
            r["target_mean"], r["target_std"], stability_weight=stability_weight
        ),
        axis=1,
    )
    return grouped.sort_values(
        ["robust_score", "target_mean"], ascending=False
    ).reset_index(drop=True)


def decode_params(params_json: str) -> Dict[str, Any]:
    return json.loads(params_json)


def infer_next_primary_refinement(
    best_value: float,
    typical_step: float,
    lo: float | None = None,
    hi: float | None = None,
) -> List[float]:
    return local_grid_around(
        best_value,
        [-typical_step, -typical_step / 2, 0.0, typical_step / 2, typical_step],
        lo=lo,
        hi=hi,
    )


def choose_shortlist(agg_df: pd.DataFrame, stage: StageSpec) -> List[str]:
    if len(agg_df) == 0:
        return []

    if stage.shortlist_k is not None:
        k = min(stage.shortlist_k, len(agg_df))
        return agg_df.head(k)["candidate_name"].tolist()

    if stage.shortlist_frac is not None:
        k = max(1, int(math.ceil(len(agg_df) * stage.shortlist_frac)))
        return agg_df.head(k)["candidate_name"].tolist()

    # sane default
    k = min(len(agg_df), max(2, int(math.ceil(len(agg_df) * 0.5))))
    return agg_df.head(k)["candidate_name"].tolist()


def expected_runs_for_stage(stage: StageSpec, n_values: int) -> int:
    stage.validate()
    shortlist = stage.shortlist_k
    if shortlist is None:
        if stage.shortlist_frac is not None:
            shortlist = max(1, int(math.ceil(n_values * stage.shortlist_frac)))
        else:
            shortlist = min(n_values, max(2, int(math.ceil(n_values * 0.5))))
    shortlist = min(shortlist, n_values)
    extra_repeats = max(0, stage.final_repeats - stage.initial_repeats)
    return n_values * stage.initial_repeats + shortlist * extra_repeats


# -----------------------------
# Argparse
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast sequential search for weak-interaction losses"
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/loss_sequential_search_fast")
    parser.add_argument(
        "--losses",
        nargs="*",
        default=["soft_bce_dice", "bce_dice", "focal_tversky_mix", "dice_focal", "bce_tversky"],
    )

    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--initial-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shortlist-k", type=int, default=2)
    parser.add_argument("--final-shortlist-k", type=int, default=2)

    parser.add_argument("--coarse-epochs", type=int, default=3)
    parser.add_argument("--secondary-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)  # final refine

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")

    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--use-nir", action="store_true")
    parser.add_argument(
        "--nir-fusion",
        type=str,
        default="early",
        choices=["early", "mid", "adapter"],
    )
    parser.add_argument(
        "--nir-init-mode",
        type=str,
        default="random",
        choices=["random", "copy-r", "copy-g", "copy-b", "copy-mean"],
    )
    parser.add_argument("--norm", type=str, default="bn", choices=["bn", "ibn"])

    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--stability-weight",
        type=float,
        default=0.25,
        help="robust_score = mean - stability_weight * std",
    )

    # I/O
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save per-run history CSVs. Disable for less disk I/O.",
    )
    parser.add_argument(
        "--save-screen-histories",
        action="store_true",
        help="Also save histories for screened-out configs in early stages.",
    )
    return parser.parse_args()


# -----------------------------
# Search-space logic
# -----------------------------
def get_loss_search_plan(loss_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    init_rep = max(1, args.initial_repeats)
    final_rep = max(init_rep, args.repeats)

    if loss_name == "bce_dice":
        return {
            "base_params": {
                "mode": "bce_dice",
                "bce_weight": 0.2,
                "dice_weight": 0.8,
                "dice_smooth": 1e-5,
            },
            "stages": [
                StageSpec(
                    "stage1_mix_coarse",
                    "mix_weight",
                    [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.35],
                    epochs=args.coarse-epochs if hasattr(args, "coarse-epochs") else args.coarse_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.shortlist_k,
                ),
                StageSpec(
                    "stage2_smooth",
                    "dice_smooth",
                    [3e-6, 1e-5, 3e-5, 1e-4],
                    epochs=args.secondary_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.final_shortlist_k,
                ),
            ],
            "final_refine_primary": True,
            "primary_param": "mix_weight",
            "primary_step": 0.03,
            "primary_bounds": (0.05, 0.40),
        }

    if loss_name == "bce_tversky":
        return {
            "base_params": {
                "mode": "bce_tversky",
                "bce_weight": 0.2,
                "tversky_weight": 0.8,
                "tversky_alpha": 0.8,
                "tversky_beta": 0.2,
            },
            "stages": [
                StageSpec(
                    "stage1_mix_coarse",
                    "mix_weight",
                    [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.35],
                    epochs=args.coarse_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.shortlist_k,
                ),
                StageSpec(
                    "stage2_alpha",
                    "alpha",
                    [0.60, 0.75, 0.85, 0.92],
                    epochs=args.secondary_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.final_shortlist_k,
                ),
            ],
            "final_refine_primary": True,
            "primary_param": "mix_weight",
            "primary_step": 0.03,
            "primary_bounds": (0.05, 0.40),
        }

    if loss_name == "dice_focal":
        return {
            "base_params": {
                "mode": "dice_focal",
                "dice_weight": 0.8,
                "focal_weight": 0.2,
                "focal_gamma": 1.0,
            },
            "stages": [
                StageSpec(
                    "stage1_mix_coarse",
                    "mix_weight",
                    [0.65, 0.72, 0.78, 0.84, 0.90, 0.95],
                    epochs=args.coarse_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.shortlist_k,
                ),
                StageSpec(
                    "stage2_gamma",
                    "gamma",
                    [0.75, 1.0, 1.25, 1.5],
                    epochs=args.secondary_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.final_shortlist_k,
                ),
            ],
            "final_refine_primary": True,
            "primary_param": "mix_weight",
            "primary_step": 0.04,
            "primary_bounds": (0.60, 0.98),
        }


    if loss_name == "soft_bce_dice":
        return {
            "base_params": {
                "mode": "soft_bce_dice",
                "bce_weight": 0.2,
                "dice_weight": 0.8,
                "dice_smooth": 1e-5,
                "soft_bce_smooth": 0.05,
            },
            "stages": [
                StageSpec(
                    "stage1_mix_coarse",
                    "mix_weight",
                    [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.35],
                    epochs=args.coarse_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.shortlist_k,
                ),
                StageSpec(
                    "stage2_label_smooth",
                    "soft_bce_smooth",
                    [0.0, 0.05, 0.10, 0.15],
                    epochs=args.secondary_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.final_shortlist_k,
                ),
            ],
            "final_refine_primary": True,
            "primary_param": "mix_weight",
            "primary_step": 0.03,
            "primary_bounds": (0.05, 0.40),
        }

    if loss_name == "focal_tversky_mix":
        return {
            "base_params": {
                "mode": "focal_tversky_mix",
                "focal_weight": 0.2,
                "tversky_weight": 0.8,
                "tversky_alpha": 0.8,
                "tversky_beta": 0.2,
                "focal_gamma": 1.0,
            },
            "stages": [
                StageSpec(
                    "stage1_mix_coarse",
                    "mix_weight",
                    [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.35],
                    epochs=args.coarse_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.shortlist_k,
                ),
                StageSpec(
                    "stage2_alpha",
                    "alpha",
                    [0.60, 0.75, 0.85, 0.92],
                    epochs=args.secondary_epochs,
                    initial_repeats=init_rep,
                    final_repeats=final_rep,
                    shortlist_k=args.final_shortlist_k,
                ),
            ],
            "final_refine_primary": True,
            "primary_param": "mix_weight",
            "primary_step": 0.03,
            "primary_bounds": (0.05, 0.40),
        }

    raise ValueError(f"Unsupported loss for sequential weak-search: {loss_name}")


def apply_search_value(
    loss_name: str,
    params: Dict[str, Any],
    parameter: str,
    value: float,
) -> Dict[str, Any]:
    p = deepcopy(params)

    if loss_name == "bce_dice":
        if parameter == "mix_weight":
            p["bce_weight"] = float(value)
            p["dice_weight"] = 1.0 - float(value)
        elif parameter == "dice_smooth":
            p["dice_smooth"] = float(value)
        else:
            raise ValueError(parameter)
        return p

    if loss_name == "bce_tversky":
        if parameter == "mix_weight":
            p["bce_weight"] = float(value)
            p["tversky_weight"] = 1.0 - float(value)
        elif parameter == "alpha":
            p["tversky_alpha"] = float(value)
            p["tversky_beta"] = 1.0 - float(value)
        else:
            raise ValueError(parameter)
        return p

    if loss_name == "dice_focal":
        if parameter == "mix_weight":
            p["dice_weight"] = float(value)
            p["focal_weight"] = 1.0 - float(value)
        elif parameter == "gamma":
            p["focal_gamma"] = float(value)
        else:
            raise ValueError(parameter)
        return p

    if loss_name == "soft_bce_dice":
        if parameter == "mix_weight":
            p["bce_weight"] = float(value)
            p["dice_weight"] = 1.0 - float(value)
        elif parameter == "soft_bce_smooth":
            p["soft_bce_smooth"] = float(value)
        else:
            raise ValueError(parameter)
        return p

    if loss_name == "focal_tversky_mix":
        if parameter == "mix_weight":
            p["focal_weight"] = float(value)
            p["tversky_weight"] = 1.0 - float(value)
        elif parameter == "alpha":
            p["tversky_alpha"] = float(value)
            p["tversky_beta"] = 1.0 - float(value)
        else:
            raise ValueError(parameter)
        return p

    raise ValueError(loss_name)


def build_name_params(loss_name: str, params: Dict[str, Any]) -> Dict[str, float]:
    if loss_name == "bce_dice":
        return {
            "mix_weight": params["bce_weight"],
            "dice_smooth": params["dice_smooth"],
        }
    if loss_name == "bce_tversky":
        return {
            "mix_weight": params["bce_weight"],
            "alpha": params["tversky_alpha"],
            "beta": params["tversky_beta"],
        }
    if loss_name == "dice_focal":
        return {
            "mix_weight": params["dice_weight"],
            "gamma": params["focal_gamma"],
        }
    if loss_name == "soft_bce_dice":
        return {
            "mix_weight": params["bce_weight"],
            "dice_smooth": params["dice_smooth"],
            "soft_bce_smooth": params["soft_bce_smooth"],
        }
    if loss_name == "focal_tversky_mix":
        return {
            "mix_weight": params["focal_weight"],
            "alpha": params["tversky_alpha"],
            "beta": params["tversky_beta"],
        }
    raise ValueError(loss_name)


# -----------------------------
# Training
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_metric(num_classes: int):
    try:
        return SegmentationMetrics(
            num_classes=num_classes,
            rare_class_ids=list(range(1, num_classes)),
        )
    except TypeError:
        return SegmentationMetrics(num_classes=num_classes)


def metric_compute(metric) -> Dict[str, float]:
    out = metric.compute()
    if isinstance(out, dict):
        return out
    raise TypeError("Metric compute() must return dict")


def make_trainers(args: argparse.Namespace):
    return build_dataloaders(
        data_root=args.data_root,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        use_nir=args.use_nir,
        oversample_rare=False,
        class_weights=None,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
    )


def build_optimizer(model, lr, weight_decay):
    try:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    except TypeError:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_once(
    args: argparse.Namespace,
    loss_params: Dict[str, Any],
    seed: int,
    train_loader,
    val_loader,
    epochs: int,
    run_label: str | None = None,
) -> Dict[str, Any]:
    set_seed(seed)

    num_classes = len(ANOMALY_CLASSES)
    in_channels = 4 if args.use_nir else 3

    model = build_model(
        args.model,
        args.encoder,
        args.encoder_weights if args.encoder_weights != "none" else None,
        num_classes,
        in_channels,
        nir_fusion=args.nir_fusion,
        norm_type=args.norm,
        use_nir=args.use_nir,
    )
    if args.use_nir and args.nir_fusion == "early" and in_channels == 4:
        init_first_conv_nir(model, args.nir_init_mode)

    model = model.to(args.device)

    if args.device.startswith("cuda"):
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    criterion = MultiLoss(**loss_params).to(args.device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = args.device.startswith("cuda") and not args.disable_amp
    amp_dtype = torch.float16
    if use_amp:
        try:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
        except Exception:
            pass
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_metrics = {
        "best_miou": -1.0,
        "best_macro_f1": -1.0,
        "best_rare_miou": -1.0,
    }
    best_epoch = -1

    epoch_bar = tqdm(
        range(epochs),
        desc=run_label or f"seed={seed}",
        leave=False,
        position=2,
    )

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            if args.device.startswith("cuda"):
                images = images.to(args.device, non_blocking=True, memory_format=torch.channels_last)
                masks = masks.to(args.device, non_blocking=True)
            else:
                images = images.to(args.device, non_blocking=True)
                masks = masks.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

            if args.max_train_steps is not None and (batch_idx + 1) >= args.max_train_steps:
                break

        train_loss /= max(train_batches, 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        metric = build_metric(num_classes)

        with torch.inference_mode():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if args.device.startswith("cuda"):
                    images = images.to(args.device, non_blocking=True, memory_format=torch.channels_last)
                    masks = masks.to(args.device, non_blocking=True)
                else:
                    images = images.to(args.device, non_blocking=True)
                    masks = masks.to(args.device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += float(loss.item())
                val_batches += 1
                metric.update(outputs, masks)

                if args.max_val_steps is not None and (batch_idx + 1) >= args.max_val_steps:
                    break

        val_loss /= max(val_batches, 1)
        metrics = metric_compute(metric)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }
        history.append(row)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            miou=f"{metrics.get('miou', float('nan')):.4f}",
        )

        if metrics.get("miou", -1.0) > best_metrics["best_miou"]:
            best_metrics["best_miou"] = float(metrics.get("miou", -1.0))
            best_metrics["best_macro_f1"] = float(metrics.get("macro_f1", np.nan))
            best_metrics["best_rare_miou"] = float(metrics.get("rare_miou", np.nan))
            best_epoch = epoch + 1

    epoch_bar.close()
    final_metrics = history[-1]

    # explicit cleanup
    del model, criterion, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_miou": best_metrics["best_miou"],
        "best_macro_f1": best_metrics["best_macro_f1"],
        "best_rare_miou": best_metrics["best_rare_miou"],
        "last_miou": float(final_metrics.get("miou", np.nan)),
        "last_macro_f1": float(final_metrics.get("macro_f1", np.nan)),
        "last_rare_miou": float(final_metrics.get("rare_miou", np.nan)),
    }


# -----------------------------
# Stage runner
# -----------------------------
def run_candidate_repeat(
    args: argparse.Namespace,
    loss_name: str,
    stage: StageSpec,
    params: Dict[str, Any],
    repeat_idx: int,
    out_dir: Path,
    train_loader,
    val_loader,
):
    seed = args.base_seed + repeat_idx
    name_params = build_name_params(loss_name, params)
    cand_name = candidate_name(loss_name, name_params)
    run_label = f"{loss_name}|{stage.name}|{cand_name}|r{repeat_idx+1}/{stage.final_repeats}"

    result = train_once(
        args=args,
        loss_params=params,
        seed=seed,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=stage.epochs,
        run_label=run_label,
    )

    row = {
        "loss_name": loss_name,
        "stage_name": stage.name,
        "candidate_name": cand_name,
        "repeat_idx": repeat_idx,
        "seed": seed,
        "params_json": json.dumps(name_params, sort_keys=True),
        **name_params,
        **{k: v for k, v in params.items() if k != "mode"},
        "best_miou": result["best_miou"],
        "best_macro_f1": result["best_macro_f1"],
        "best_rare_miou": result["best_rare_miou"],
        "last_miou": result["last_miou"],
        "last_macro_f1": result["last_macro_f1"],
        "last_rare_miou": result["last_rare_miou"],
        "best_epoch": result["best_epoch"],
        "epochs": stage.epochs,
    }

    return cand_name, row, result


def run_stage(
    args: argparse.Namespace,
    loss_name: str,
    stage: StageSpec,
    current_params: Dict[str, Any],
    out_dir: Path,
    train_loader,
    val_loader,
    global_pbar: tqdm | None = None,
) -> Dict[str, Any]:
    print(f"\n=== {loss_name} | {stage.name} | varying {stage.parameter} ===")
    stage.validate()

    candidate_params: Dict[str, Dict[str, Any]] = {}
    for value in stage.values:
        params = apply_search_value(loss_name, current_params, stage.parameter, value)
        cand_name = candidate_name(loss_name, build_name_params(loss_name, params))
        candidate_params[cand_name] = params

    stage_total = expected_runs_for_stage(stage, len(stage.values))
    stage_pbar = tqdm(
        total=stage_total,
        desc=f"{loss_name}:{stage.name}",
        leave=True,
        position=1,
    )

    all_repeat_rows = []

    # Phase 1: screen everyone with few repeats
    for cand_name, params in candidate_params.items():
        for repeat_idx in range(stage.initial_repeats):
            _, row, result = run_candidate_repeat(
                args=args,
                loss_name=loss_name,
                stage=stage,
                params=params,
                repeat_idx=repeat_idx,
                out_dir=out_dir,
                train_loader=train_loader,
                val_loader=val_loader,
            )
            all_repeat_rows.append(row)

            if args.save_history and args.save_screen_histories:
                hist_df = pd.DataFrame(result["history"])
                hist_df.to_csv(
                    out_dir / f"{stage.name}__{cand_name}__repeat-{repeat_idx}.csv",
                    index=False,
                )

            stage_pbar.update(1)
            if global_pbar is not None:
                global_pbar.update(1)

            stage_pbar.set_postfix(
                phase="screen",
                candidate=cand_name[:34],
                repeat=f"{repeat_idx+1}/{stage.final_repeats}",
                best=f"{result['best_miou']:.4f}",
            )

    repeats_df = pd.DataFrame(all_repeat_rows)
    agg_df = summarize_repeats(
        repeats_df,
        target=TARGET_METRIC,
        stability_weight=args.stability_weight,
    )

    shortlist = choose_shortlist(agg_df, stage)

    # Phase 2: give extra repeats only to shortlist
    extra_needed = max(0, stage.final_repeats - stage.initial_repeats)
    if extra_needed > 0:
        for cand_name in shortlist:
            params = candidate_params[cand_name]
            for repeat_idx in range(stage.initial_repeats, stage.final_repeats):
                _, row, result = run_candidate_repeat(
                    args=args,
                    loss_name=loss_name,
                    stage=stage,
                    params=params,
                    repeat_idx=repeat_idx,
                    out_dir=out_dir,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
                all_repeat_rows.append(row)

                if args.save_history:
                    hist_df = pd.DataFrame(result["history"])
                    hist_df.to_csv(
                        out_dir / f"{stage.name}__{cand_name}__repeat-{repeat_idx}.csv",
                        index=False,
                    )

                stage_pbar.update(1)
                if global_pbar is not None:
                    global_pbar.update(1)

                stage_pbar.set_postfix(
                    phase="refine",
                    candidate=cand_name[:34],
                    repeat=f"{repeat_idx+1}/{stage.final_repeats}",
                    best=f"{result['best_miou']:.4f}",
                )

    repeats_df = pd.DataFrame(all_repeat_rows)
    repeats_df.to_csv(out_dir / f"{stage.name}__summary_repeats.csv", index=False)

    agg_df = summarize_repeats(
        repeats_df,
        target=TARGET_METRIC,
        stability_weight=args.stability_weight,
    )
    agg_df.to_csv(out_dir / f"{stage.name}__summary_aggregated.csv", index=False)

    best = agg_df.iloc[0]
    best_params = decode_params(best["params_json"])
    std = 0.0 if pd.isna(best["target_std"]) else best["target_std"]

    print(f"Shortlist after screening: {shortlist}")
    print(f"Best candidate for {loss_name} / {stage.name}:")
    print(f"  {best['candidate_name']}")
    print(
        f"  mean={best['target_mean']:.6f}, "
        f"std={std:.6f}, "
        f"robust={best['robust_score']:.6f}"
    )

    stage_pbar.close()
    return {"best_params": best_params}


# -----------------------------
# Main
# -----------------------------
def estimate_total_single_runs(args: argparse.Namespace) -> int:
    total = 0
    for loss_name in args.losses:
        plan = get_loss_search_plan(loss_name, args)
        for stage in plan["stages"]:
            total += expected_runs_for_stage(stage, len(stage.values))
        if plan.get("final_refine_primary", False):
            refine_stage = StageSpec(
                "stage3_primary_refine",
                plan["primary_param"],
                [0, 1, 2, 3, 4],  # length only
                epochs=args.epochs,
                initial_repeats=max(1, args.initial_repeats),
                final_repeats=max(max(1, args.initial_repeats), args.repeats),
                shortlist_k=args.final_shortlist_k,
            )
            total += expected_runs_for_stage(refine_stage, 5)
    return total


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # BIGGEST SPEEDUP: loaders are built once
    train_loader, val_loader = make_trainers(args)

    total_single_runs = estimate_total_single_runs(args)
    global_pbar = tqdm(
        total=total_single_runs,
        desc="total search progress",
        position=0,
    )

    final_rows = []

    for loss_name in args.losses:
        plan = get_loss_search_plan(loss_name, args)
        loss_dir = Path(args.output_dir) / loss_name
        loss_dir.mkdir(parents=True, exist_ok=True)

        current_params = deepcopy(plan["base_params"])

        for stage in plan["stages"]:
            res = run_stage(
                args=args,
                loss_name=loss_name,
                stage=stage,
                current_params=current_params,
                out_dir=loss_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                global_pbar=global_pbar,
            )
            best_params = res["best_params"]

            if loss_name == "bce_dice":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["dice_weight"] = 1.0 - best_params["mix_weight"]
                current_params["dice_smooth"] = best_params["dice_smooth"]

            elif loss_name == "bce_tversky":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["tversky_weight"] = 1.0 - best_params["mix_weight"]
                current_params["tversky_alpha"] = best_params["alpha"]
                current_params["tversky_beta"] = best_params["beta"]

            elif loss_name == "dice_focal":
                current_params["dice_weight"] = best_params["mix_weight"]
                current_params["focal_weight"] = 1.0 - best_params["mix_weight"]
                current_params["focal_gamma"] = best_params["gamma"]

            elif loss_name == "soft_bce_dice":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["dice_weight"] = 1.0 - best_params["mix_weight"]
                current_params["dice_smooth"] = best_params["dice_smooth"]
                current_params["soft_bce_smooth"] = best_params["soft_bce_smooth"]

            elif loss_name == "focal_tversky_mix":
                current_params["focal_weight"] = best_params["mix_weight"]
                current_params["tversky_weight"] = 1.0 - best_params["mix_weight"]
                current_params["tversky_alpha"] = best_params["alpha"]
                current_params["tversky_beta"] = best_params["beta"]

        if plan.get("final_refine_primary", False):
            primary = plan["primary_param"]
            step = float(plan["primary_step"])
            lo, hi = plan["primary_bounds"]

            if loss_name == "bce_dice":
                center = current_params["bce_weight"]
            elif loss_name == "bce_tversky":
                center = current_params["bce_weight"]
            elif loss_name == "dice_focal":
                center = current_params["dice_weight"]
            elif loss_name == "soft_bce_dice":
                center = current_params["bce_weight"]
            elif loss_name == "focal_tversky_mix":
                center = current_params["focal_weight"]
            else:
                center = 0.5

            refine_vals = infer_next_primary_refinement(
                center, typical_step=step, lo=lo, hi=hi
            )
            refine_stage = StageSpec(
                "stage3_primary_refine",
                primary,
                refine_vals,
                epochs=args.epochs,
                initial_repeats=max(1, args.initial_repeats),
                final_repeats=max(max(1, args.initial_repeats), args.repeats),
                shortlist_k=args.final_shortlist_k,
            )

            res = run_stage(
                args=args,
                loss_name=loss_name,
                stage=refine_stage,
                current_params=current_params,
                out_dir=loss_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                global_pbar=global_pbar,
            )
            best_params = res["best_params"]

            if loss_name == "bce_dice":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["dice_weight"] = 1.0 - best_params["mix_weight"]
                current_params["dice_smooth"] = best_params["dice_smooth"]

            elif loss_name == "bce_tversky":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["tversky_weight"] = 1.0 - best_params["mix_weight"]
                current_params["tversky_alpha"] = best_params["alpha"]
                current_params["tversky_beta"] = best_params["beta"]

            elif loss_name == "dice_focal":
                current_params["dice_weight"] = best_params["mix_weight"]
                current_params["focal_weight"] = 1.0 - best_params["mix_weight"]
                current_params["focal_gamma"] = best_params["gamma"]

            elif loss_name == "soft_bce_dice":
                current_params["bce_weight"] = best_params["mix_weight"]
                current_params["dice_weight"] = 1.0 - best_params["mix_weight"]
                current_params["dice_smooth"] = best_params["dice_smooth"]
                current_params["soft_bce_smooth"] = best_params["soft_bce_smooth"]

            elif loss_name == "focal_tversky_mix":
                current_params["focal_weight"] = best_params["mix_weight"]
                current_params["tversky_weight"] = 1.0 - best_params["mix_weight"]
                current_params["tversky_alpha"] = best_params["alpha"]
                current_params["tversky_beta"] = best_params["beta"]

        with open(loss_dir / "final_selected_params.json", "w", encoding="utf-8") as f:
            json.dump(current_params, f, indent=2)

        final_rows.append(
            {
                "loss_name": loss_name,
                "final_params_json": json.dumps(current_params, sort_keys=True),
                **current_params,
            }
        )

    global_pbar.close()

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(Path(args.output_dir) / "final_selected_params.csv", index=False)

    print("\n=== Fast sequential weak-interaction search complete ===")
    print(final_df[["loss_name", "final_params_json"]].to_string(index=False))


if __name__ == "__main__":
    main()