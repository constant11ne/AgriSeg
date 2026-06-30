#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

LOSS = "soft_bce_dice"
BCE_WEIGHT = 0.12
DICE_WEIGHT = 0.88
DICE_SMOOTH = 1e-5
LABEL_SMOOTHING = 0.0
LEVEL_INDICES = "0,1,2,3,4"
LEVEL_NAME = "8_16_32_64_128"
GVI_IBN_MODE = "rgb_gvi_ibn_features"


@dataclass
class Experiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_nir: bool = True
    use_gvi: bool = True
    rgb_source: Optional[str] = None
    gvi_source: Optional[str] = None
    align_norm_mode: str = "none"
    notes: str = ""
    fallback_patterns: List[str] = field(default_factory=list)


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="gvi_only",
        fusion_family="gvi",
        fusion_method="gvi_ratio",
        gvi_source="ndvi_only_external",
        notes=(
            "GVI-only segmentation: encoder initialized from ndvi_only; "
            "GVI module keeps NDVI-like initialization."
        ),
        fallback_patterns=["*_gvi_gvi_ratio*_*gvi*"],
    ),
    Experiment(
        name="early_rgb_gvi",
        fusion_family="early",
        fusion_method="early_rgb_gvi",
        rgb_source="rgb_only_external",
        notes=(
            "Original-paper-style early fusion: compute learnable GVI from RGB+NIR, "
            "then segment from RGB+GVI. RGB encoder starts from rgb_only; "
            "GVI module starts as NDVI-like."
        ),
        fallback_patterns=["*_early_early_rgb_gvi*_*gvi*"],
    ),
    Experiment(
        name="mid_rgb_gvi",
        fusion_family="mid",
        fusion_method="progressive_concat_rgb_gvi",
        rgb_source="rgb_only_external",
        gvi_source="gvi_only",
        notes="Two-branch mid fusion: RGB encoder from rgb_only, GVI module+encoder from gvi_only.",
        fallback_patterns=["*_mid_progressive_concat_rgb_gvi*_*gvi*"],
    ),
    Experiment(
        name="mid_rgb_gvi_ibn",
        fusion_family="mid",
        fusion_method="progressive_concat_rgb_gvi",
        rgb_source="mid_rgb_gvi",
        gvi_source="mid_rgb_gvi",
        align_norm_mode=GVI_IBN_MODE,
        notes=(
            "Two-branch RGB+GVI mid fusion with IBN feature alignment, initialized "
            "from the previous mid_rgb_gvi two-branch checkpoint."
        ),
        fallback_patterns=["*_mid_progressive_concat_rgb_gvi*ibn*_*gvi*"],
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full GVI experiment wave: gvi_only, original-style early "
            "RGB+GVI, two-branch RGB+GVI, and RGB+GVI+IBN."
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
    p.add_argument("--runs-dir", type=str, default="runs/gvi_wave")
    p.add_argument("--rgb-source-runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--ndvi-source-runs-dir", type=str, default="runs/ndvi_wave2")
    p.add_argument("--rgb-checkpoint", type=str, default=None)
    p.add_argument("--ndvi-checkpoint", type=str, default=None)
    p.add_argument("--gvi-only-checkpoint", type=str, default=None)
    p.add_argument("--mid-rgb-gvi-checkpoint", type=str, default=None)
    p.add_argument(
        "--experiments",
        type=str,
        default=",".join(exp.name for exp in EXPERIMENTS),
        help="Comma-separated subset/order of experiments to run.",
    )
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


def external_ndvi_checkpoint(args: argparse.Namespace, fold: int) -> Path:
    if args.ndvi_checkpoint:
        return Path(args.ndvi_checkpoint)
    return Path(args.ndvi_source_runs_dir) / f"fold_{fold}" / f"{run_base(args)}_ndvi_early_concat_copyr_{LOSS}_ndvi" / "best.pth"


def expected_run_name(exp_name: str, args: argparse.Namespace) -> str:
    base = run_base(args)
    names = {
        "gvi_only": f"{base}_gvi_gvi_ratio_copyr_{LOSS}_gvi",
        "early_rgb_gvi": f"{base}_early_early_rgb_gvi_copyr_{LOSS}_gvi",
        "mid_rgb_gvi": f"{base}_mid_progressive_concat_rgb_gvi_copyr_{LOSS}_{LEVEL_NAME}_gvi",
        "mid_rgb_gvi_ibn": (
            f"{base}_mid_progressive_concat_rgb_gvi_copyr_{LOSS}_"
            f"{LEVEL_NAME}_{GVI_IBN_MODE}_gvi"
        ),
    }
    return names[exp_name]


def selected_experiments(args: argparse.Namespace) -> List[Experiment]:
    by_name = {exp.name: exp for exp in EXPERIMENTS}
    names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown experiment(s): {unknown}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def checkpoint_from_run(run_dir: Path) -> Path:
    for name in ("best.pth", "last.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise RuntimeError(f"No best.pth/last.pth found in {run_dir}")


def run_dir_for(exp: Experiment, fold_runs: Path, args: argparse.Namespace, before: Optional[Set[str]] = None) -> Path:
    exact = fold_runs / expected_run_name(exp.name, args)
    if exact.exists():
        return exact

    if before is not None and fold_runs.exists():
        created = [
            p for p in fold_runs.iterdir()
            if p.is_dir()
            and p.name not in before
            and ((p / "best.pth").exists() or (p / "last.pth").exists())
        ]
        if created:
            created.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return created[0]

    candidates: List[Path] = []
    for pattern in exp.fallback_patterns:
        candidates.extend([p for p in fold_runs.glob(pattern) if p.is_dir()])
    candidates = list(dict.fromkeys(candidates))
    candidates = [p for p in candidates if (p / "best.pth").exists() or (p / "last.pth").exists()]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return exact


def ensure_checkpoint(label: str, path: Path, dry_run: bool) -> None:
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"Missing required checkpoint {label}: {path}")


def build_cmd(
    exp: Experiment,
    *,
    data_root: Path,
    fold_runs: Path,
    args: argparse.Namespace,
    ckpts: Dict[str, Path],
) -> List[str]:
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
        "--nir-init-mode", "copy-r",
        "--freeze-rgb-encoder", "none",
        "--freeze-rgb-stages", "3",
        "--partial-unfreeze-last-n", "2",
        "--nir-branch-width", "0.5",
        "--fusion-hidden-dim", "128",
        "--phase1-freeze-epochs", "0",
        "--train-augment-mode", "full",
        "--save-last-every", "0",
    ]
    if exp.use_nir:
        cmd += ["--use-nir", "--nir-fusion", "early"]
    if exp.use_gvi:
        cmd += ["--use-gvi"]
    if exp.fusion_method.startswith("progressive_concat"):
        cmd += ["--progressive-level-indices", LEVEL_INDICES, "--progressive-level-name", LEVEL_NAME]
    if exp.align_norm_mode != "none":
        cmd += ["--align-norm-mode", exp.align_norm_mode]
    if exp.rgb_source:
        ensure_checkpoint(exp.rgb_source, ckpts[exp.rgb_source], args.dry_run)
        cmd += ["--rgb-pretrained-path", str(ckpts[exp.rgb_source])]
    if exp.gvi_source:
        ensure_checkpoint(exp.gvi_source, ckpts[exp.gvi_source], args.dry_run)
        cmd += ["--gvi-pretrained-path", str(ckpts[exp.gvi_source])]
    if args.extra_train_flags:
        cmd += shlex.split(args.extra_train_flags)
    return cmd


def read_summary(run_dir: Path) -> Dict:
    path = run_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def aggregate_record(fold: int, exp: Experiment, run_dir: Path, ckpt: Path, status: str) -> Dict:
    return {
        "fold": fold,
        "experiment": exp.name,
        "status": status,
        "run_dir": str(run_dir),
        "best_ckpt": str(ckpt),
        **read_summary(run_dir),
    }


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    experiments = selected_experiments(args)
    aggregate = []

    for fold in range(args.folds):
        fold_root = processed_root / f"fold_{fold}"
        if not fold_root.exists():
            raise FileNotFoundError(f"Missing fold directory: {fold_root}")
        fold_runs = runs_root / f"fold_{fold}"
        fold_runs.mkdir(parents=True, exist_ok=True)

        ckpts: Dict[str, Path] = {
            "rgb_only_external": external_rgb_checkpoint(args, fold),
            "ndvi_only_external": external_ndvi_checkpoint(args, fold),
        }
        if args.gvi_only_checkpoint:
            ckpts["gvi_only"] = Path(args.gvi_only_checkpoint)
        if args.mid_rgb_gvi_checkpoint:
            ckpts["mid_rgb_gvi"] = Path(args.mid_rgb_gvi_checkpoint)

        for key in ("rgb_only_external", "ndvi_only_external"):
            ensure_checkpoint(key, ckpts[key], args.dry_run)

        print(f"\n{'=' * 80}")
        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[RGB SOURCE] {ckpts['rgb_only_external']}")
        print(f"[NDVI SOURCE -> GVI-only init] {ckpts['ndvi_only_external']}")
        print(f"[RUNS] {fold_runs}")
        print(f"[EXPERIMENTS] {', '.join(exp.name for exp in experiments)}")
        print(f"{'=' * 80}")

        for exp in experiments:
            run_dir = run_dir_for(exp, fold_runs, args)
            if args.skip_existing and (run_dir / "best.pth").exists():
                ckpt = checkpoint_from_run(run_dir)
                ckpts[exp.name] = ckpt
                print(f"\n[{exp.name}] SKIP existing: {ckpt}")
                aggregate.append(aggregate_record(fold, exp, run_dir, ckpt, "skipped"))
                continue

            before = {p.name for p in fold_runs.iterdir() if p.is_dir()}
            cmd = build_cmd(exp, data_root=fold_root, fold_runs=fold_runs, args=args, ckpts=ckpts)
            print(f"\n[{exp.name}]")
            print(f"  Notes: {exp.notes}")
            if exp.name == "early_rgb_gvi":
                print("  IBN: not used; current IBN is feature alignment for two-branch models, not early concatenation.")
            print("  Command:")
            print("    " + " ".join(cmd))

            if args.dry_run:
                ckpts[exp.name] = run_dir / "best.pth"
                continue

            subprocess.run(cmd, check=True)
            run_dir = run_dir_for(exp, fold_runs, args, before=before)
            ckpt = checkpoint_from_run(run_dir)
            ckpts[exp.name] = ckpt
            aggregate.append(aggregate_record(fold, exp, run_dir, ckpt, "completed"))

    out_json = runs_root / "gvi_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print("[INFO] Full GVI wave includes gvi_only, early RGB+GVI, mid RGB+GVI, and mid RGB+GVI+IBN.")


if __name__ == "__main__":
    main()
