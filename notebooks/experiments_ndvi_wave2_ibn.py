from __future__ import annotations

import argparse
import json
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
DEFAULT_IBN_MODE = "rgb_ndvi_ibn_features"


@dataclass
class Experiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_ndvi: bool = True
    progressive_level_indices: Optional[str] = "0,1,2,3,4"
    progressive_level_name: Optional[str] = "8_16_32_64_128"
    notes: str = ""


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="mid_rgb_ndvi_ibn",
        fusion_family="mid",
        fusion_method="progressive_concat_rgb_ndvi",
        notes="Two-branch mid fusion RGB+NDVI with IBN feature/fusion alignment; RGB and NDVI encoders are initialized from previous waves.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run IBN experiment for NDVI wave2 mid_rgb_ndvi")
    p.add_argument("--processed-root", required=True)
    p.add_argument("--folds", type=int, default=1)
    p.add_argument("--encoder", type=str, default="timm-efficientnet-b4")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument("--model", type=str, default="fpn", choices=["unet", "fpn", "deeplabv3p"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs/ndvi_wave2_ibn")
    p.add_argument("--rgb-source-runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--ndvi-source-runs-dir", type=str, default="runs/ndvi_wave2")
    p.add_argument("--rgb-checkpoint", type=str, default=None)
    p.add_argument("--ndvi-checkpoint", type=str, default=None)
    p.add_argument("--align-norm-mode", type=str, default=DEFAULT_IBN_MODE)
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def external_rgb_ckpt(args: argparse.Namespace, fold: int) -> Path:
    if args.rgb_checkpoint:
        return Path(args.rgb_checkpoint)
    return (
            Path(args.rgb_source_runs_dir)
            / f"fold_{fold}"
            / f"{args.model}_{args.encoder}_rgb_early_concat_copyr_{LOSS}"
            / "best.pth"
    )


def external_ndvi_ckpt(args: argparse.Namespace, fold: int) -> Path:
    if args.ndvi_checkpoint:
        return Path(args.ndvi_checkpoint)
    return (
            Path(args.ndvi_source_runs_dir)
            / f"fold_{fold}"
            / f"{args.model}_{args.encoder}_ndvi_early_concat_copyr_{LOSS}_ndvi"
            / "best.pth"
    )


def expected_run_dir(exp: Experiment, fold_runs: Path, args: argparse.Namespace) -> Path:
    base = f"{args.model}_{args.encoder}"
    return fold_runs / f"{base}_mid_progressive_concat_rgb_ndvi_{LOSS}_8_16_32_64_128_{args.align_norm_mode}_ndvi"


def build_cmd(exp: Experiment, *, data_root: Path, fold_runs: Path, fold: int, args: argparse.Namespace) -> List[str]:
    rgb_ckpt = external_rgb_ckpt(args, fold)
    ndvi_ckpt = external_ndvi_ckpt(args, fold)
    if not args.dry_run:
        if not rgb_ckpt.exists():
            raise FileNotFoundError(f"Missing RGB checkpoint: {rgb_ckpt}")
        if not ndvi_ckpt.exists():
            raise FileNotFoundError(f"Missing NDVI checkpoint from previous wave: {ndvi_ckpt}")

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
        "--fusion-family", exp.fusion_family,
        "--fusion-method", exp.fusion_method,
        "--nir-init-mode", "random",
        "--freeze-rgb-encoder", "none",
        "--freeze-rgb-stages", "3",
        "--partial-unfreeze-last-n", "2",
        "--nir-branch-width", "0.5",
        "--fusion-hidden-dim", "128",
        "--phase1-freeze-epochs", "0",
        "--train-augment-mode", "full",
        "--save-last-every", "0",
        "--use-ndvi",
        "--progressive-level-indices", str(exp.progressive_level_indices),
        "--progressive-level-name", str(exp.progressive_level_name),
        "--align-norm-mode", args.align_norm_mode,
        "--rgb-pretrained-path", str(rgb_ckpt),
        "--ndvi-pretrained-path", str(ndvi_ckpt),
    ]
    if args.extra_train_flags:
        cmd += args.extra_train_flags.split()
    return cmd


def read_summary(run_dir: Path) -> Dict:
    summary = run_dir / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())
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

        print(f"\n{'=' * 80}")
        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[RGB SOURCE] {external_rgb_ckpt(args, fold)}")
        print(f"[NDVI SOURCE] {external_ndvi_ckpt(args, fold)}")
        print(f"[IBN MODE] {args.align_norm_mode}")
        print(f"{'=' * 80}")

        for exp in EXPERIMENTS:
            cmd = build_cmd(exp, data_root=fold_root, fold_runs=fold_runs, fold=fold, args=args)
            print(f"\n[{exp.name}]")
            print(f"  Notes: {exp.notes}")
            print("  Command:")
            print("    " + " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)
            run_dir = expected_run_dir(exp, fold_runs, args)
            best = run_dir / "best.pth"
            last = run_dir / "last.pth"
            aggregate.append({
                "fold": fold,
                "experiment": exp.name,
                "run_dir": str(run_dir),
                "best_ckpt": str(best if best.exists() else last),
                **read_summary(run_dir),
            })

    out_json = runs_root / "ndvi_wave2_ibn_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print(
        "[INFO] Experiment used full augmentations, one fold by default, and RGB/NDVI pretrained branches from previous waves.")


if __name__ == "__main__":
    main()
