#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOSS = "soft_bce_dice"
BCE_WEIGHT = 0.12
DICE_WEIGHT = 0.88
DICE_SMOOTH = 1e-5
LABEL_SMOOTHING = 0.0
LEVEL_INDICES = "0,1,2,3,4"
LEVEL_NAME = "8_16_32_64_128"
ALIGN_NORM_MODE = "rgb_gvi_ibn_features"


@dataclass
class Experiment:
    name: str
    gvi_channels: int
    gvi_init_indices: str
    notes: str


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="mid_rgb_multigvi2_ibn",
        gvi_channels=2,
        gvi_init_indices="ndvi,gndvi",
        notes="RGB + Multi-GVI two-branch mid fusion with IBN; GVI channels initialized as NDVI and GNDVI.",
    ),
    Experiment(
        name="mid_rgb_multigvi3_ibn",
        gvi_channels=3,
        gvi_init_indices="ndvi,gndvi,ndwi",
        notes="RGB + Multi-GVI two-branch mid fusion with IBN; GVI channels initialized as NDVI, GNDVI and NDWI.",
    ),
    Experiment(
        name="mid_rgb_multigvi4_ibn",
        gvi_channels=4,
        gvi_init_indices="ndvi,gndvi,ndwi,random",
        notes="RGB + Multi-GVI two-branch mid fusion with IBN; GVI channels initialized as NDVI, GNDVI, NDWI and random.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Multi-GVI experiments with RGB + Multi-GVI branches and IBN alignment")
    p.add_argument("--processed-root", required=True)
    p.add_argument("--folds", type=int, default=1)
    p.add_argument("--encoder", type=str, default="timm-efficientnet-b4")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument("--model", type=str, default="fpn", choices=["unet", "fpn", "deeplabv3p"])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs/multi_gvi_wave")
    p.add_argument("--rgb-source-runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--rgb-checkpoint", type=str, default=None)
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run_base(args: argparse.Namespace) -> str:
    return f"{args.model}_{args.encoder}"


def external_rgb_checkpoint(args: argparse.Namespace, fold: int) -> Path:
    if args.rgb_checkpoint:
        return Path(args.rgb_checkpoint)
    return Path(args.rgb_source_runs_dir) / f"fold_{fold}" / f"{run_base(args)}_rgb_early_concat_copyr_{LOSS}" / "best.pth"


def expected_run_name(exp: Experiment, args: argparse.Namespace) -> str:
    base = run_base(args)
    return (
        f"{base}_mid_progressive_concat_rgb_gvi_copyr_{LOSS}_"
        f"{LEVEL_NAME}_{ALIGN_NORM_MODE}_gvi{exp.gvi_channels}"
    )


def run_dir_for(exp: Experiment, fold_runs: Path, args: argparse.Namespace) -> Path:
    exact = fold_runs / expected_run_name(exp, args)
    if exact.exists():
        return exact

    pattern = f"*_mid_progressive_concat_rgb_gvi*_{ALIGN_NORM_MODE}_gvi{exp.gvi_channels}"
    candidates = [p for p in fold_runs.glob(pattern) if p.is_dir()]
    candidates = [p for p in candidates if (p / "best.pth").exists() or (p / "last.pth").exists()]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return exact


def ckpt_from_run(run_dir: Path) -> Path:
    for name in ("best.pth", "last.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise RuntimeError(f"No best.pth/last.pth found in {run_dir}")


def build_cmd(exp: Experiment, *, data_root: Path, fold_runs: Path, args: argparse.Namespace, rgb_ckpt: Path) -> List[str]:
    cmd = [
        sys.executable, "-m", "src.train",
        "--data-root", str(data_root),
        "--model", args.model,
        "--encoder", args.encoder,
        "--encoder-weights", args.encoder_weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--img-size", str(args.img_size[0]), str(args.img_size[1]),
        "--seed", str(args.seed),
        "--runs-dir", str(fold_runs),
        "--loss", LOSS,
        "--bce-weight", str(BCE_WEIGHT),
        "--dice-weight", str(DICE_WEIGHT),
        "--dice-smooth", str(DICE_SMOOTH),
        "--label-smoothing", str(LABEL_SMOOTHING),
        "--fusion-family", "mid",
        "--fusion-method", "progressive_concat_rgb_gvi",
        "--nir-init-mode", "copy-r",
        "--freeze-rgb-encoder", "none",
        "--freeze-rgb-stages", "3",
        "--partial-unfreeze-last-n", "2",
        "--nir-branch-width", "0.5",
        "--fusion-hidden-dim", "128",
        "--phase1-freeze-epochs", "0",
        "--train-augment-mode", "full",
        "--save-last-every", "0",
        "--use-nir",
        "--nir-fusion", "early",
        "--use-gvi",
        "--gvi-channels", str(exp.gvi_channels),
        "--gvi-init-indices", exp.gvi_init_indices,
        "--progressive-level-indices", LEVEL_INDICES,
        "--progressive-level-name", LEVEL_NAME,
        "--align-norm-mode", ALIGN_NORM_MODE,
        "--rgb-pretrained-path", str(rgb_ckpt),
    ]
    if args.extra_train_flags:
        cmd += shlex.split(args.extra_train_flags)
    return cmd


def read_summary(run_dir: Path) -> Dict:
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    aggregate = []

    for fold in range(args.folds):
        fold_root = processed_root / f"fold_{fold}"
        if not fold_root.exists():
            raise FileNotFoundError(f"Missing fold directory: {fold_root}")
        fold_runs = runs_root / f"fold_{fold}"
        fold_runs.mkdir(parents=True, exist_ok=True)

        rgb_ckpt = external_rgb_checkpoint(args, fold)
        if not rgb_ckpt.exists() and not args.dry_run:
            raise FileNotFoundError(f"Missing required RGB checkpoint: {rgb_ckpt}")

        print(f"\n{'=' * 80}")
        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[RGB SOURCE] {rgb_ckpt}")
        print(f"[RUNS] {fold_runs}")
        print("[EXPERIMENTS] " + ", ".join(exp.name for exp in EXPERIMENTS))
        print(f"{'=' * 80}")

        for exp in EXPERIMENTS:
            run_dir = run_dir_for(exp, fold_runs, args)
            if args.skip_existing and (run_dir / "best.pth").exists():
                print(f"\n[{exp.name}] SKIP existing: {run_dir / 'best.pth'}")
                aggregate.append({
                    "fold": fold,
                    "experiment": exp.name,
                    "gvi_channels": exp.gvi_channels,
                    "gvi_init_indices": exp.gvi_init_indices,
                    "run_dir": str(run_dir),
                    "best_ckpt": str(ckpt_from_run(run_dir)),
                    **read_summary(run_dir),
                })
                continue

            cmd = build_cmd(exp, data_root=fold_root, fold_runs=fold_runs, args=args, rgb_ckpt=rgb_ckpt)
            print(f"\n[{exp.name}]")
            print(f"  Notes: {exp.notes}")
            print("  Command:")
            print("    " + " ".join(cmd))
            if args.dry_run:
                continue

            subprocess.run(cmd, check=True)
            aggregate.append({
                "fold": fold,
                "experiment": exp.name,
                "gvi_channels": exp.gvi_channels,
                "gvi_init_indices": exp.gvi_init_indices,
                "run_dir": str(run_dir),
                "best_ckpt": str(ckpt_from_run(run_dir)),
                **read_summary(run_dir),
            })

    out_json = runs_root / "multi_gvi_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print("[INFO] Multi-GVI wave: RGB + Multi-GVI branches, IBN alignment, K=2/3/4 spectral-index channels.")


if __name__ == "__main__":
    main()
