from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

ANOMALY_CLASSES = [
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "storm_damage",
    "water",
    "waterway",
    "weed_cluster",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a rare-class-focused 3-fold subset for baseline experiments"
    )
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument(
        "--source-splits",
        nargs="+",
        default=["train", "val"],
        help="Source splits to pool before subsampling",
    )
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument(
        "--drop-top-k-classes",
        type=int,
        default=2,
        help="Ignore the K most frequent classes when scoring images",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    return p.parse_args()


def load_presence(mask_dir: Path) -> np.ndarray:
    pres = np.zeros(len(ANOMALY_CLASSES), dtype=np.uint8)
    for i, cls in enumerate(ANOMALY_CLASSES):
        p = mask_dir / f"{cls}.png"
        if not p.exists():
            continue
        with Image.open(p) as m:
            arr = np.asarray(m)
            pres[i] = 1 if (arr > 0).any() else 0
    return pres


def make_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        if src.is_dir():
            raise ValueError("hardlink not supported for directories")
        os.link(src, dst)
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    src_root = Path(args.source_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = []
    class_counts = Counter()

    for split in args.source_splits:
        img_dir = src_root / split / "images"
        mask_root = src_root / split / "masks"

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.name.startswith(".") or not img_path.is_file():
                continue

            sample_id = img_path.stem
            mask_dir = mask_root / sample_id
            if not mask_dir.exists():
                continue

            presence = load_presence(mask_dir)

            for i, v in enumerate(presence):
                if v:
                    class_counts[ANOMALY_CLASSES[i]] += 1

            unique_id = f"{split}__{sample_id}"
            samples.append(
                {
                    "uid": unique_id,
                    "split": split,
                    "sample_id": sample_id,
                    "image_path": str(img_path),
                    "mask_dir": str(mask_dir),
                    "presence": presence.tolist(),
                }
            )

    if not samples:
        raise ValueError("No labelled samples found")

    total = len(samples)
    class_freq = {cls: class_counts[cls] / total for cls in ANOMALY_CLASSES}

    common_classes = [
        cls
        for cls, _ in sorted(class_freq.items(), key=lambda kv: kv[1], reverse=True)[
            : args.drop_top_k_classes
        ]
    ]
    rare_classes = [c for c in ANOMALY_CLASSES if c not in common_classes]
    inv_freq = {c: 1.0 / max(class_freq[c], 1e-9) for c in rare_classes}

    rng = np.random.RandomState(args.seed)

    for s in samples:
        pres = np.array(s["presence"], dtype=np.uint8)
        rare_present = [c for i, c in enumerate(ANOMALY_CLASSES) if pres[i] and c in rare_classes]
        common_present = [c for i, c in enumerate(ANOMALY_CLASSES) if pres[i] and c in common_classes]
        score = float(sum(inv_freq[c] for c in rare_present))

        s["rare_classes"] = rare_present
        s["common_classes"] = common_present
        s["rare_count"] = len(rare_present)
        s["score"] = score + 1e-6 * rng.rand()

    ranked = sorted(samples, key=lambda s: (-s["rare_count"], -s["score"], s["uid"]))

    needed_for_val = args.folds * args.val_size
    if len(ranked) < max(args.train_size + args.val_size, needed_for_val):
        raise ValueError(
            f"Not enough samples ({len(ranked)}) for requested subset sizes"
        )

    val_candidates = ranked[:needed_for_val]
    val_folds = [[] for _ in range(args.folds)]
    for i, sample in enumerate(val_candidates):
        fold_idx = i % args.folds
        if len(val_folds[fold_idx]) < args.val_size:
            val_folds[fold_idx].append(sample)

    for fold_idx in range(args.folds):
        val_ids = {s["uid"] for s in val_folds[fold_idx]}
        remaining = [s for s in ranked if s["uid"] not in val_ids]
        train_fold = remaining[: args.train_size]

        fold_root = out_root / f"fold_{fold_idx}"

        for subset_name, subset_samples in [("train", train_fold), ("val", val_folds[fold_idx])]:
            img_out = fold_root / subset_name / "images"
            mask_out = fold_root / subset_name / "masks"
            img_out.mkdir(parents=True, exist_ok=True)
            mask_out.mkdir(parents=True, exist_ok=True)

            ids_txt = []
            for s in subset_samples:
                src_img = Path(s["image_path"])
                src_mask = Path(s["mask_dir"])
                dst_img = img_out / f"{s['uid']}{src_img.suffix}"
                dst_mask = mask_out / s["uid"]

                make_link(src_img, dst_img, args.link_mode)
                make_link(src_mask, dst_mask, args.link_mode)
                ids_txt.append(s["uid"])

            (fold_root / f"{subset_name}_ids.txt").write_text(
                "\n".join(ids_txt) + "\n",
                encoding="utf-8",
            )

        meta = {
            "fold": fold_idx,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "common_classes_ignored_for_sampling": common_classes,
            "rare_classes_used_for_sampling": rare_classes,
            "source_splits": args.source_splits,
        }
        (fold_root / "subset_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    summary = {
        "total_source_samples": total,
        "folds": args.folds,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "common_classes_ignored_for_sampling": common_classes,
        "class_frequency": class_freq,
        "note": "Labels are preserved. Most frequent classes are ignored only during subset scoring / selection.",
    }
    (out_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] Prepared {args.folds} folds under {out_root}")
    print(f"[INFO] Common classes ignored for sampling: {common_classes}")


if __name__ == "__main__":
    main()
