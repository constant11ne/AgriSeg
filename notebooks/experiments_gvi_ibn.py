#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

LOSS = "soft_bce_dice"
BCE_WEIGHT = 0.12
DICE_WEIGHT = 0.88
DICE_SMOOTH = 1e-5
LABEL_SMOOTHING = 0.0
LEVEL_INDICES = "0,1,2,3,4"
LEVEL_NAME = "8_16_32_64_128"
DEFAULT_ALIGN = "rgb_gvi_ibn_features"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run GVI+IBN experiment initialized from the previous two-branch "
            "mid_rgb_gvi checkpoint"
        )
    )
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
    p.add_argument("--runs-dir", type=str, default="runs/gvi_wave_ibn")
    p.add_argument(
        "--gvi-source-runs-dir",
        type=str,
        default="runs/gvi_wave",
        help="Directory containing the previous two-branch mid_rgb_gvi run.",
    )
    p.add_argument(
        "--mid-rgb-gvi-checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for previous mid_rgb_gvi model.",
    )
    p.add_argument("--align-norm-mode", type=str, default=DEFAULT_ALIGN)
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def base_name(args: argparse.Namespace) -> str:
    return f"{args.model}_{args.encoder}"


def mid_rgb_gvi_checkpoint(args: argparse.Namespace, fold: int) -> Path:
    if args.mid_rgb_gvi_checkpoint:
        return Path(args.mid_rgb_gvi_checkpoint)
    return (
        Path(args.gvi_source_runs_dir)
        / f"fold_{fold}"
        / f"{base_name(args)}_mid_progressive_concat_rgb_gvi_copyr_{LOSS}_{LEVEL_NAME}_gvi"
        / "best.pth"
    )


def expected_run_name(args: argparse.Namespace) -> str:
    return (
        f"{base_name(args)}_mid_progressive_concat_rgb_gvi_copyr_"
        f"{LOSS}_{LEVEL_NAME}_{args.align_norm_mode}_gvi"
    )


def find_run_dir(fold_runs: Path, before: set[str], args: argparse.Namespace) -> Optional[Path]:
    expected = fold_runs / expected_run_name(args)
    if expected.exists():
        return expected

    created = [p for p in fold_runs.iterdir() if p.is_dir() and p.name not in before]
    if created:
        created.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return created[0]

    candidates = [p for p in fold_runs.glob("*progressive_concat_rgb_gvi*ibn*gvi*") if p.is_dir()]
    candidates += [p for p in fold_runs.glob("*progressive_concat_rgb_gvi*gvi*") if p.is_dir()]
    candidates = list(dict.fromkeys(candidates))
    candidates = [p for p in candidates if (p / "best.pth").exists() or (p / "last.pth").exists()]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def checkpoint_from_run(run_dir: Path) -> Path:
    for name in ["best.pth", "last.pth"]:
        p = run_dir / name
        if p.exists():
            return p
    raise RuntimeError(f"No best.pth/last.pth found in {run_dir}")


def build_cmd(args: argparse.Namespace, *, data_root: Path, fold_runs: Path, fold: int) -> List[str]:
    prev_two_branch_ckpt = mid_rgb_gvi_checkpoint(args, fold)
    if not args.dry_run and not prev_two_branch_ckpt.exists():
        raise FileNotFoundError(f"Missing previous mid_rgb_gvi checkpoint: {prev_two_branch_ckpt}")

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
        "--progressive-level-indices", LEVEL_INDICES,
        "--progressive-level-name", LEVEL_NAME,
        "--align-norm-mode", args.align_norm_mode,
        # Both branches are initialized from the previous two-branch checkpoint.
        # train.py will extract rgb_encoder.*, gvi_encoder.* and gvi_module.* by prefix.
        "--rgb-pretrained-path", str(prev_two_branch_ckpt),
        "--gvi-pretrained-path", str(prev_two_branch_ckpt),
    ]
    if args.extra_train_flags:
        cmd += args.extra_train_flags.split()
    return cmd


def read_summary(run_dir: Path) -> Dict:
    p = run_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
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

        prev_ckpt = mid_rgb_gvi_checkpoint(args, fold)
        print(f"\n{'=' * 80}")
        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[PRETRAINED TWO-BRANCH SOURCE] {prev_ckpt}")
        print(f"[IBN MODE] {args.align_norm_mode}")
        print(f"[RUNS] {fold_runs}")
        print(f"{'=' * 80}")

        cmd = build_cmd(args, data_root=fold_root, fold_runs=fold_runs, fold=fold)
        print("\n[mid_rgb_gvi_ibn]")
        print("  Notes: RGB+GVI mid fusion with IBN, initialized from previous two-branch mid_rgb_gvi checkpoint.")
        print("  Command:")
        print("    " + " ".join(cmd))

        before = {p.name for p in fold_runs.iterdir() if p.is_dir()}
        existing = fold_runs / expected_run_name(args)
        if args.skip_existing and (existing / "best.pth").exists():
            run_dir = existing
            print(f"[SKIP] Existing checkpoint: {existing / 'best.pth'}")
        elif args.dry_run:
            continue
        else:
            subprocess.run(cmd, check=True)
            run_dir = find_run_dir(fold_runs, before, args)
            if run_dir is None:
                raise RuntimeError("Could not identify run directory for mid_rgb_gvi_ibn")

        aggregate.append({
            "fold": fold,
            "experiment": "mid_rgb_gvi_ibn",
            "pretrained_two_branch_ckpt": str(prev_ckpt),
            "run_dir": str(run_dir),
            "best_ckpt": str(checkpoint_from_run(run_dir)),
            **read_summary(run_dir),
        })

    out_json = runs_root / "gvi_ibn_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print("[INFO] GVI+IBN initializes RGB encoder, GVI encoder and GVI module from the previous two-branch mid_rgb_gvi model.")


if __name__ == "__main__":
    main()
