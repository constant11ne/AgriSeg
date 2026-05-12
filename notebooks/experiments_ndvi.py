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
class NDVIExperiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_nir: bool = True
    use_ndvi: bool = True
    nir_init_mode: str = "random"
    progressive_level_indices: Optional[str] = None
    progressive_level_name: Optional[str] = None
    rgb_pretrained_from: Optional[str] = None
    nir_pretrained_from: Optional[str] = None
    late_rgb_from: Optional[str] = None
    late_nir_from: Optional[str] = None
    notes: str = ""


NDVI_EXPERIMENTS: List[NDVIExperiment] = [
    NDVIExperiment(
        name="early_concat_rgb_nir_ndvi",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        use_ndvi=True,
        nir_init_mode="copy-r",
        notes="Early fusion with RGB+NIR+NDVI, five input channels",
    ),
    NDVIExperiment(
        name="nir_ndvi_only",
        fusion_family="nir",
        fusion_method="early_concat",
        use_nir=True,
        use_ndvi=True,
        nir_init_mode="copy-r",
        notes="NIR+NDVI branch pretraining, two input channels; used by mid/late NDVI fusion",
    ),
    NDVIExperiment(
        name="progressive_concat_ndvi_full",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        use_ndvi=True,
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        rgb_pretrained_from="rgb_only",
        nir_pretrained_from="nir_ndvi_only",
        notes="Mid-level progressive full; RGB initialized from rgb_only, NIR+NDVI initialized from nir_ndvi_only",
    ),
    NDVIExperiment(
        name="late_weighted_ndvi",
        fusion_family="late",
        fusion_method="late_weighted",
        use_nir=True,
        use_ndvi=True,
        late_rgb_from="rgb_only",
        late_nir_from="nir_ndvi_only",
        notes="Late weighted fusion; RGB model from rgb_only, NIR+NDVI model from nir_ndvi_only",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full-augmentation NDVI experiments on one processed fold"
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
    p.add_argument("--runs-dir", type=str, default="runs/ndvi_full_1fold")
    p.add_argument(
        "--rgb-source-runs-dir",
        type=str,
        default="runs/augmentations_full_1fold",
        help="Runs root containing the pretrained rgb_only model. Default: full-augmentation baseline runs.",
    )
    p.add_argument(
        "--rgb-checkpoint",
        type=str,
        default=None,
        help="Optional direct path to rgb_only best.pth. Overrides --rgb-source-runs-dir.",
    )
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment if its expected run directory already contains best.pth",
    )
    p.add_argument(
        "--no-flag-check",
        action="store_true",
        help="Do not check that src.train supports the required NDVI flags before running.",
    )
    return p.parse_args()


def run_name_parts(args: argparse.Namespace) -> str:
    return f"{args.model}_{args.encoder}"


def rgb_only_run_names(args: argparse.Namespace) -> List[str]:
    base = run_name_parts(args)
    return [
        f"{base}_rgb_early_concat_copyr_{_DEFAULT_LOSS}",
        f"{base}_rgb_early_concat_{_DEFAULT_LOSS}",
    ]


def expected_run_names(exp_name: str, args: argparse.Namespace) -> List[str]:
    base = run_name_parts(args)
    loss = _DEFAULT_LOSS
    names: Dict[str, List[str]] = {
        "early_concat_rgb_nir_ndvi": [
            f"{base}_early_early_concat_copyr_ndvi_{loss}",
            f"{base}_early_early_concat_copyr_{loss}_ndvi",
            f"{base}_early_early_concat_copyr_{loss}",
        ],
        "nir_ndvi_only": [
            f"{base}_nir_early_concat_copyr_ndvi_{loss}",
            f"{base}_nir_early_concat_copyr_{loss}_ndvi",
            f"{base}_nir_early_concat_copyr_{loss}",
        ],
        "progressive_concat_ndvi_full": [
            f"{base}_mid_progressive_concat_ndvi_{loss}_8_16_32_64_128",
            f"{base}_mid_progressive_concat_{loss}_8_16_32_64_128_ndvi",
            f"{base}_mid_progressive_concat_{loss}_8_16_32_64_128",
            f"{base}_mid_progressive_concat_{loss}_nirl5",
        ],
        "late_weighted_ndvi": [
            f"{base}_late_late_weighted_ndvi_{loss}",
            f"{base}_late_late_weighted_{loss}_ndvi",
            f"{base}_late_late_weighted_{loss}",
        ],
    }
    return names[exp_name]


def expected_run_dirs(exp_name: str, fold_runs: Path, args: argparse.Namespace) -> List[Path]:
    return [fold_runs / name for name in expected_run_names(exp_name, args)]


def resolve_run_dir(
        exp_name: str,
        fold_runs: Path,
        before: set[str],
        args: argparse.Namespace,
) -> Optional[Path]:
    for candidate in expected_run_dirs(exp_name, fold_runs, args):
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


def checkpoint_from_run_dir(run_dir: Path) -> Optional[Path]:
    best_path = run_dir / "best.pth"
    if best_path.exists():
        return best_path
    last_path = run_dir / "last.pth"
    if last_path.exists():
        return last_path
    return None


def resolve_best_checkpoint(exp_name: str, fold_runs: Path, args: argparse.Namespace) -> Path:
    for run_dir in expected_run_dirs(exp_name, fold_runs, args):
        ckpt = checkpoint_from_run_dir(run_dir)
        if ckpt is not None:
            return ckpt
    raise RuntimeError(f"Checkpoint for dependency '{exp_name}' not found under {fold_runs}")


def resolve_rgb_checkpoint(args: argparse.Namespace, fold: int) -> Path:
    if args.rgb_checkpoint:
        path = Path(args.rgb_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"RGB checkpoint not found: {path}")
        return path

    source_root = Path(args.rgb_source_runs_dir)
    fold_runs = source_root / f"fold_{fold}"
    search_roots = [fold_runs, source_root]

    for root in search_roots:
        if not root.exists():
            continue
        for name in rgb_only_run_names(args):
            ckpt = checkpoint_from_run_dir(root / name)
            if ckpt is not None:
                return ckpt

    for root in search_roots:
        if not root.exists():
            continue
        candidates = [
            p
            for p in root.iterdir()
            if p.is_dir()
               and "_rgb_" in p.name
               and "early_concat" in p.name
               and checkpoint_from_run_dir(p) is not None
        ]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt = checkpoint_from_run_dir(candidates[0])
            assert ckpt is not None
            return ckpt

    raise RuntimeError(
        "Could not find pretrained rgb_only checkpoint. Run experiments_augmentations first "
        "or pass --rgb-checkpoint explicitly."
    )


def check_train_flags() -> None:
    required = [
        "--use-ndvi",
        "--train-augment-mode",
        "--rgb-pretrained-path",
        "--progressive-level-indices",
        "--late-fusion-checkpoint-rgb",
        "--late-fusion-checkpoint-nir",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    help_text = result.stdout
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise RuntimeError(
            "src.train does not support required NDVI experiment flags: "
            + ", ".join(missing)
            + ". Apply the NDVI train/datamodule/model patch before running this file."
        )


def build_train_cmd(
        exp: NDVIExperiment,
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
    if exp.use_ndvi:
        cmd.append("--use-ndvi")
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

    if not args.no_flag_check and not args.dry_run:
        check_train_flags()

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
        if args.dry_run and not args.rgb_checkpoint:
            ckpts["rgb_only"] = Path(args.rgb_source_runs_dir) / f"fold_{fold}" / rgb_only_run_names(args)[
                0] / "best.pth"
        else:
            ckpts["rgb_only"] = resolve_rgb_checkpoint(args, fold)

        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[RGB SOURCE] {ckpts['rgb_only']}")

        for exp in NDVI_EXPERIMENTS:
            for dep in [exp.rgb_pretrained_from, exp.nir_pretrained_from, exp.late_rgb_from, exp.late_nir_from]:
                if dep and dep not in ckpts:
                    ckpts[dep] = resolve_best_checkpoint(dep, fold_runs, args)

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
            for candidate in expected_run_dirs(exp.name, fold_runs, args):
                if (candidate / "best.pth").exists():
                    existing = candidate
                    break

            if args.skip_existing and existing is not None:
                print(f"[INFO] Skip existing run: {existing}")
                new_dir = existing
            else:
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
                new_dir = resolve_run_dir(exp.name, fold_runs, before, args)
                if new_dir is None:
                    raise RuntimeError(f"Could not identify run directory for {exp.name} fold {fold}")

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
                    "rgb_source_ckpt": str(ckpts["rgb_only"]),
                    **summary,
                }
            )

    out_json = runs_root / "ndvi_full_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))

    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print("[INFO] NDVI experiments used full augmentations and one fold by default.")


if __name__ == "__main__":
    main()
