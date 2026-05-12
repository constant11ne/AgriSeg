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

BASE = "fpn_timm-efficientnet-b4"
LOSS = "soft_bce_dice"


@dataclass
class Experiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_nir: bool = False
    use_ndvi: bool = True
    nir_init_mode: str = "copy-r"
    progressive_level_indices: Optional[str] = None
    progressive_level_name: Optional[str] = None
    rgb_source: Optional[str] = None
    nir_source: Optional[str] = None
    ndvi_source: Optional[str] = None
    notes: str = ""


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="early_rgb_ndvi",
        fusion_family="early",
        fusion_method="early_rgb_ndvi",
        use_nir=False,
        use_ndvi=True,
        notes="Early fusion with RGB+NDVI, four input channels",
    ),
    Experiment(
        name="ndvi_only",
        fusion_family="ndvi",
        fusion_method="early_concat",
        use_nir=False,
        use_ndvi=True,
        notes="NDVI-only branch pretraining, one input channel; used by mid NDVI fusion",
    ),
    Experiment(
        name="mid_rgb_nir_ndvi",
        fusion_family="mid",
        fusion_method="progressive_concat_rgb_nir_ndvi",
        use_nir=True,
        use_ndvi=True,
        nir_init_mode="random",
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        rgb_source="rgb_only_fullaug",
        nir_source="nir_only_fullaug",
        ndvi_source="ndvi_only",
        notes="Three-branch mid fusion: RGB, NIR and NDVI full encoders",
    ),
    Experiment(
        name="mid_rgb_ndvi",
        fusion_family="mid",
        fusion_method="progressive_concat_rgb_ndvi",
        use_nir=False,
        use_ndvi=True,
        nir_init_mode="random",
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        rgb_source="rgb_only_fullaug",
        ndvi_source="ndvi_only",
        notes="Two-branch mid fusion: RGB and NDVI full encoders",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NDVI wave2 mid-fusion experiments on one fold")
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
    p.add_argument("--runs-dir", type=str, default="runs/ndvi_wave2")
    p.add_argument("--rgb-source-runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--nir-source-runs-dir", type=str, default="runs/augmentations_full_1fold")
    p.add_argument("--rgb-checkpoint", type=str, default=None)
    p.add_argument("--nir-checkpoint", type=str, default=None)
    p.add_argument("--extra-train-flags", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def expected_local_run_dir(exp_name: str, encoder: str, model: str) -> str:
    base = f"{model}_{encoder}"
    mapping = {
        "early_rgb_ndvi": f"{base}_early_early_rgb_ndvi_copyr_{LOSS}_ndvi",
        "ndvi_only": f"{base}_ndvi_early_concat_copyr_{LOSS}_ndvi",
        "mid_rgb_nir_ndvi": f"{base}_mid_progressive_concat_rgb_nir_ndvi_{LOSS}_8_16_32_64_128_ndvi",
        "mid_rgb_ndvi": f"{base}_mid_progressive_concat_rgb_ndvi_{LOSS}_8_16_32_64_128_ndvi",
    }
    return mapping[exp_name]


def external_rgb_ckpt(args: argparse.Namespace, fold: int) -> Path:
    if args.rgb_checkpoint:
        return Path(args.rgb_checkpoint)
    return Path(
        args.rgb_source_runs_dir) / f"fold_{fold}" / f"{args.model}_{args.encoder}_rgb_early_concat_copyr_{LOSS}" / "best.pth"


def external_nir_ckpt(args: argparse.Namespace, fold: int) -> Path:
    if args.nir_checkpoint:
        return Path(args.nir_checkpoint)
    return Path(
        args.nir_source_runs_dir) / f"fold_{fold}" / f"{args.model}_{args.encoder}_nir_early_concat_copyr_{LOSS}" / "best.pth"


def local_ckpt(fold_runs: Path, exp_name: str, args: argparse.Namespace) -> Path:
    run_dir = fold_runs / expected_local_run_dir(exp_name, args.encoder, args.model)
    best = run_dir / "best.pth"
    if best.exists():
        return best
    last = run_dir / "last.pth"
    if last.exists():
        return last
    raise FileNotFoundError(f"Missing checkpoint for {exp_name}: expected {best}")


def build_cmd(exp: Experiment, *, data_root: Path, fold_runs: Path, fold: int, args: argparse.Namespace) -> List[str]:
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
        "--loss", _DEFAULT_LOSS,
        "--bce-weight", str(_BEST_BCE_WEIGHT),
        "--dice-weight", str(_BEST_DICE_WEIGHT),
        "--dice-smooth", str(_BEST_DICE_SMOOTH),
        "--label-smoothing", str(_BEST_LABEL_SMOOTH),
        "--fusion-family", exp.fusion_family,
        "--fusion-method", exp.fusion_method,
        "--nir-init-mode", exp.nir_init_mode,
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
    if exp.use_ndvi:
        cmd += ["--use-ndvi"]
    if exp.progressive_level_indices:
        cmd += ["--progressive-level-indices", exp.progressive_level_indices]
    if exp.progressive_level_name:
        cmd += ["--progressive-level-name", exp.progressive_level_name]
    if exp.rgb_source == "rgb_only_fullaug":
        cmd += ["--rgb-pretrained-path", str(external_rgb_ckpt(args, fold))]
    if exp.nir_source == "nir_only_fullaug":
        cmd += ["--nir-pretrained-path", str(external_nir_ckpt(args, fold))]
    if exp.ndvi_source == "ndvi_only":
        cmd += ["--ndvi-pretrained-path", str(local_ckpt(fold_runs, "ndvi_only", args))]
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

        print(f"\n{'=' * 80}")
        print(f"[FOLD {fold}] data_root={fold_root}")
        print(f"[RGB SOURCE] {external_rgb_ckpt(args, fold)}")
        print(f"[NIR SOURCE] {external_nir_ckpt(args, fold)}")
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
            run_dir = fold_runs / expected_local_run_dir(exp.name, args.encoder, args.model)
            aggregate.append({
                "fold": fold,
                "experiment": exp.name,
                "run_dir": str(run_dir),
                "best_ckpt": str((run_dir / "best.pth") if (run_dir / "best.pth").exists() else (run_dir / "last.pth")),
                **read_summary(run_dir),
            })

    out_json = runs_root / "ndvi_wave2_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"\n[DONE] Aggregate summary saved to {out_json}")
    print("[INFO] NDVI wave2 used full augmentations and one fold by default.")


if __name__ == "__main__":
    main()
