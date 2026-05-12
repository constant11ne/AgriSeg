#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_LOSS = "soft_bce_dice"
_BEST_BCE_WEIGHT = 0.12
_BEST_DICE_WEIGHT = 0.88
_BEST_DICE_SMOOTH = 1e-5
_BEST_LABEL_SMOOTH = 0.0


@dataclass
class AugmentationExperiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_nir: bool = True
    nir_init_mode: str = "random"
    progressive_level_indices: Optional[str] = None
    progressive_level_name: Optional[str] = None
    rgb_pretrained_from: Optional[str] = None
    nir_pretrained_from: Optional[str] = None
    late_rgb_from: Optional[str] = None
    late_nir_from: Optional[str] = None
    notes: str = ""


AUGMENTATION_EXPERIMENTS: List[AugmentationExperiment] = [
    AugmentationExperiment(
        name="early_concat_copy-r",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-r",
        notes="Early concat baseline with copy-r init",
    ),
    AugmentationExperiment(
        name="rgb_only",
        fusion_family="rgb",
        fusion_method="early_concat",
        use_nir=False,
        nir_init_mode="copy-r",
        notes="RGB-only baseline; its weights initialize the RGB branch for mid/late fusion",
    ),
    AugmentationExperiment(
        name="nir_only",
        fusion_family="nir",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-r",
        notes="NIR-only baseline; its weights initialize the NIR branch for mid/late fusion",
    ),
    AugmentationExperiment(
        name="progressive_concat_full",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        rgb_pretrained_from="rgb_only",
        nir_pretrained_from="nir_only",
        notes="Mid-level progressive full; RGB initialized from rgb_only, NIR initialized from nir_only",
    ),
    AugmentationExperiment(
        name="late_weighted",
        fusion_family="late",
        fusion_method="late_weighted",
        use_nir=True,
        late_rgb_from="rgb_only",
        late_nir_from="nir_only",
        notes="Late weighted fusion from separately trained rgb_only and nir_only models",
    ),
]

EXPECTED_RUN_DIRS = {
    "early_concat_copy-r": "fpn_timm-efficientnet-b4_early_early_concat_copyr_soft_bce_dice",
    "rgb_only": "fpn_timm-efficientnet-b4_rgb_early_concat_copyr_soft_bce_dice",
    "nir_only": "fpn_timm-efficientnet-b4_nir_early_concat_copyr_soft_bce_dice",
    "progressive_concat_full": "fpn_timm-efficientnet-b4_mid_progressive_concat_soft_bce_dice_8_16_32_64_128",
    "late_weighted": "fpn_timm-efficientnet-b4_late_late_weighted_soft_bce_dice",
}

FALLBACK_RUN_DIRS = {
    "progressive_concat_full": [
        "fpn_timm-efficientnet-b4_mid_progressive_concat_soft_bce_dice_nirl5",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full-augmentation experiments on one processed fold"
    )
    p.add_argument(
        "--processed-root",
        required=True,
        help="Root produced by tools/prepare_baseline_dataset.py",
    )
    p.add_argument("--folds", type=int, default=1)
    p.add_argument("--encoder", type=str, default="timm-efficientnet-b4")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument(
        "--model",
        type=str,
        default="fpn",
        choices=["unet", "fpn", "deeplabv3p"],
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment if its expected run directory already contains best.pth",
    )
    return p.parse_args()


def expected_run_dirs(exp_name: str, fold_runs: Path) -> List[Path]:
    names = []
    if exp_name in EXPECTED_RUN_DIRS:
        names.append(EXPECTED_RUN_DIRS[exp_name])
    names.extend(FALLBACK_RUN_DIRS.get(exp_name, []))
    return [fold_runs / name for name in names]


def resolve_run_dir(exp_name: str, fold_runs: Path, before: set[str]) -> Optional[Path]:
    for candidate in expected_run_dirs(exp_name, fold_runs):
        if candidate.exists():
            return candidate

    now_dirs = [p for p in fold_runs.iterdir() if p.is_dir()]
    created = [p for p in now_dirs if p.name not in before]
    if created:
        created.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return created[0]

    candidates = [
        p
        for p in now_dirs
        if (p / "best.pth").exists()
           or (p / "summary.json").exists()
           or (p / "last.pth").exists()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_best_checkpoint(exp_name: str, fold_runs: Path) -> Path:
    for run_dir in expected_run_dirs(exp_name, fold_runs):
        best_path = run_dir / "best.pth"
        if best_path.exists():
            return best_path
        last_path = run_dir / "last.pth"
        if last_path.exists():
            return last_path
    raise RuntimeError(f"Checkpoint for dependency '{exp_name}' not found under {fold_runs}")


def build_train_cmd(
        exp: AugmentationExperiment,
        *,
        data_root: Path,
        runs_dir: Path,
        args: argparse.Namespace,
        ckpts: Dict[str, Path],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data-root",
        str(data_root),
        "--model",
        args.model,
        "--encoder",
        args.encoder,
        "--encoder-weights",
        args.encoder_weights,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--img-size",
        str(args.img_size[0]),
        str(args.img_size[1]),
        "--seed",
        str(args.seed),
        "--runs-dir",
        str(runs_dir),
        "--loss",
        _DEFAULT_LOSS,
        "--bce-weight",
        str(_BEST_BCE_WEIGHT),
        "--dice-weight",
        str(_BEST_DICE_WEIGHT),
        "--dice-smooth",
        str(_BEST_DICE_SMOOTH),
        "--label-smoothing",
        str(_BEST_LABEL_SMOOTH),
        "--fusion-family",
        exp.fusion_family,
        "--fusion-method",
        exp.fusion_method,
        "--nir-init-mode",
        exp.nir_init_mode,
        "--freeze-rgb-encoder",
        "none",
        "--freeze-rgb-stages",
        "3",
        "--partial-unfreeze-last-n",
        "2",
        "--nir-branch-width",
        "0.5",
        "--fusion-hidden-dim",
        "128",
        "--phase1-freeze-epochs",
        "0",
        "--train-augment-mode",
        "full",
        "--save-last-every",
        "0",
    ]

    if exp.use_nir:
        cmd += ["--use-nir", "--nir-fusion", "early"]
    if exp.progressive_level_indices:
        cmd += ["--progressive-level-indices", exp.progressive_level_indices]
    if exp.progressive_level_name:
        cmd += ["--progressive-level-name", exp.progressive_level_name]
    if exp.rgb_pretrained_from:
        cmd += ["--rgb-pretrained-path", str(ckpts[exp.rgb_pretrained_from])]
    if exp.nir_pretrained_from:
        cmd += ["--nir-pretrained-path", str(ckpts[exp.nir_pretrained_from])]
    if exp.late_rgb_from:
        cmd += ["--late-fusion-checkpoint-rgb", str(ckpts[exp.late_rgb_from])]
    if exp.late_nir_from:
        cmd += ["--late-fusion-checkpoint-nir", str(ckpts[exp.late_nir_from])]
    if args.extra_train_flags:
        cmd += args.extra_train_flags.split()

    return cmd


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

        ckpts: Dict[str, Path] = {}

        print(f"[FOLD {fold}] data_root={fold_root}")

        for exp in AUGMENTATION_EXPERIMENTS:
            for dep in [exp.rgb_pretrained_from, exp.nir_pretrained_from, exp.late_rgb_from, exp.late_nir_from]:
                if dep and dep not in ckpts:
                    ckpts[dep] = resolve_best_checkpoint(dep, fold_runs)

            cmd = build_train_cmd(
                exp,
                data_root=fold_root,
                runs_dir=fold_runs,
                args=args,
                ckpts=ckpts,
            )

            print(f"\n[{exp.name}]")
            print(f"Notes: {exp.notes}")
            print("Command:")
            print(" ".join(cmd))

            before = {p.name for p in fold_runs.iterdir() if p.is_dir()}

            existing = None
            for candidate in expected_run_dirs(exp.name, fold_runs):
                if (candidate / "best.pth").exists():
                    existing = candidate
                    break

            if args.skip_existing and existing is not None:
                print(f"[INFO] Skip existing run: {existing}")
                new_dir = existing
            else:
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
                new_dir = resolve_run_dir(exp.name, fold_runs, before)
                if new_dir is None:
                    raise RuntimeError(
                        f"Could not identify run directory for {exp.name} fold {fold}"
                    )

            best_path = new_dir / "best.pth"
            if not best_path.exists():
                fallback_last = new_dir / "last.pth"
                if fallback_last.exists():
                    best_path = fallback_last
                elif args.dry_run:
                    best_path = new_dir / "best.pth"
                else:
                    raise RuntimeError(f"Could not find checkpoint in {new_dir}")

            ckpts[exp.name] = best_path

            summary_path = new_dir / "summary.json"
            summary = {}
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())

            aggregate.append(
                {
                    "fold": fold,
                    "experiment": exp.name,
                    "run_dir": str(new_dir),
                    "best_ckpt": str(best_path),
                    **summary,
                }
            )

    out_json = runs_root / "augmentations_full_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))

    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print(
        "[INFO] Full augmentations used: random scale/crop + horizontal/vertical flips + random rotation + RGB brightness/contrast jitter + resize/normalize."
    )


if __name__ == "__main__":
    main()
