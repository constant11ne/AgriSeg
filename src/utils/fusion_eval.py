from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

_CONFIG_KEYS = [
    "model", "encoder", "use_nir",
    "fusion_family", "fusion_method",
    "nir_init_mode", "nir_fusion",
    "freeze_rgb_encoder", "freeze_rgb_stages",
    "partial_unfreeze_last_n",
    "nir_branch_width", "fusion_hidden_dim",
    "freeze_encoder",
    "lora", "lora_rank", "lora_alpha",
    "bitfit",
    "loss", "lr", "epochs", "batch_size", "seed",
]


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def collect_run_metrics(runs_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return records

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists():
            continue

        config = _read_json(str(config_path)) or {}
        metrics_list = _read_json(str(metrics_path)) or []

        cfg_row: Dict[str, Any] = {"run_name": run_dir.name}
        for key in _CONFIG_KEYS:
            cfg_row[key] = config.get(key, None)

        for epoch_metrics in metrics_list:
            row = {**cfg_row, **epoch_metrics}
            records.append(row)

    return records


def build_summary_table(records: List[Dict[str, Any]]) -> "Any":
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r["run_name"]].append(r)

    rows = []
    for run_name, epoch_records in sorted(groups.items()):
        epoch_records = sorted(epoch_records, key=lambda x: x.get("epoch", 0))
        first = epoch_records[0]
        last = epoch_records[-1]

        mious = [r.get("miou", 0.0) for r in epoch_records]
        macro_f1s = [r.get("macro_f1", 0.0) for r in epoch_records]
        rare_mious = [r.get("rare_miou", 0.0) for r in epoch_records]

        best_idx = int(max(range(len(mious)), key=lambda i: mious[i]))

        best = epoch_records[best_idx]
        row: Dict[str, Any] = {
            "run_name": run_name,
            "fusion_family": first.get("fusion_family") or first.get("nir_fusion", "?"),
            "fusion_method": first.get("fusion_method", "?"),
            "encoder": first.get("encoder", "?"),
            "use_nir": first.get("use_nir", False),
            "nir_init_mode": first.get("nir_init_mode", "random"),
            "freeze_rgb_encoder": first.get("freeze_rgb_encoder", "none"),
            "lora": bool(first.get("lora", False)),
            "bitfit": bool(first.get("bitfit", False)),
            "loss": first.get("loss", "?"),
            "best_miou": round(best.get("miou", 0.0), 5),
            "best_macro_f1": round(best.get("macro_f1", 0.0), 5),
            "best_rare_miou": round(best.get("rare_miou", 0.0), 5),
            "best_epoch": best.get("epoch", best_idx + 1),
            "final_miou": round(last.get("miou", 0.0), 5),
            "final_macro_f1": round(last.get("macro_f1", 0.0), 5),
            "final_rare_miou": round(last.get("rare_miou", 0.0), 5),
            "total_epochs": len(epoch_records),
        }
        rows.append(row)

    if HAS_PANDAS:
        import pandas as pd
        return pd.DataFrame(rows)
    return rows


def rank_experiments(
        summary: "Any",
        metric: str = "best_miou",
        ascending: bool = False,
) -> "Any":
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(summary, pd.DataFrame):
            return summary.sort_values(metric, ascending=ascending).reset_index(drop=True)
    return sorted(summary, key=lambda r: r.get(metric, 0.0), reverse=not ascending)


def compare_fusion_families(summary: "Any") -> "Any":
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(summary, pd.DataFrame):
            cols = [c for c in ["best_miou", "best_macro_f1", "best_rare_miou"] if c in summary.columns]
            return (
                summary.groupby("fusion_family")[cols]
                .agg(["mean", "max", "count"])
                .round(5)
            )

    from collections import defaultdict
    family_scores: Dict[str, List[float]] = defaultdict(list)
    for r in summary:
        family_scores[r.get("fusion_family", "?")].append(r.get("best_miou", 0.0))
    return {
        fam: {
            "count": len(scores),
            "mean_best_miou": round(sum(scores) / len(scores), 5),
            "max_best_miou": round(max(scores), 5),
        }
        for fam, scores in sorted(family_scores.items())
    }


def save_summary_csv(summary: "Any", out_path: str) -> None:
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(summary, pd.DataFrame):
            summary.to_csv(out_path, index=False)
            return

    if not summary:
        return
    import csv
    fieldnames = list(summary[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def print_comparison_table(
        summary: "Any",
        top_n: int = 20,
        sort_by: str = "best_miou",
) -> None:
    ranked = rank_experiments(summary, metric=sort_by)
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(ranked, pd.DataFrame):
            cols = [c for c in [
                "run_name", "fusion_family", "fusion_method",
                "best_miou", "best_macro_f1", "best_rare_miou",
                "best_epoch", "total_epochs",
                "lora", "bitfit", "freeze_rgb_encoder",
            ] if c in ranked.columns]
            print(ranked[cols].head(top_n).to_string(index=False))
            return

    rows = ranked[:top_n] if not HAS_PANDAS else ranked
    header = f"{'run_name':<55} {'fam':>10} {'method':>25} {'best_mIoU':>10} {'rare_mIoU':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('run_name', ''):<55} "
            f"{r.get('fusion_family', '?'):>10} "
            f"{r.get('fusion_method', '?'):>25} "
            f"{r.get('best_miou', 0.0):>10.5f} "
            f"{r.get('best_rare_miou', 0.0):>10.5f}"
        )


_CLASS_NAMES = [
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


def per_class_comparison(
        records: List[Dict[str, Any]],
        run_names: Optional[List[str]] = None,
) -> "Any":
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if run_names is None or r["run_name"] in run_names:
            groups[r["run_name"]].append(r)

    rows = []
    for rname, epochs in sorted(groups.items()):
        best = max(epochs, key=lambda x: x.get("miou", 0.0))
        row = {
            "run_name": rname,
            "fusion_method": best.get("fusion_method", best.get("nir_fusion", "?")),
            "best_miou": round(best.get("miou", 0.0), 5),
        }
        for i, cls in enumerate(_CLASS_NAMES):
            row[f"iou_{cls}"] = round(best.get(f"iou_class_{i}", 0.0), 5)
        rows.append(row)

    if HAS_PANDAS:
        import pandas as pd
        return pd.DataFrame(rows)
    return rows
