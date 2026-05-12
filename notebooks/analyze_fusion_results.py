from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.fusion_eval import (
    collect_run_metrics,
    build_summary_table,
    rank_experiments,
    compare_fusion_families,
    per_class_comparison,
    save_summary_csv,
    print_comparison_table,
    HAS_PANDAS,
)


def _print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate and compare fusion experiment results."
    )
    p.add_argument("--runs-dir", type=str, default="runs",
                   help="Directory containing experiment run subdirectories")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write output CSVs (defaults to <runs-dir>/fusion_analysis)")
    p.add_argument("--top", type=int, default=20,
                   help="Number of top experiments to display in the console table")
    p.add_argument(
        "--filter-prefix", nargs="*", default=None,
        help="Only include runs whose name starts with one of these prefixes",
    )
    p.add_argument(
        "--sort-by", type=str, default="best_miou",
        choices=["best_miou", "best_macro_f1", "best_rare_miou", "final_miou"],
        help="Metric to sort experiments by",
    )
    p.add_argument(
        "--baseline-run", type=str, default=None,
        help="Name of baseline run to compute deltas against (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.runs_dir, "fusion_analysis")
    os.makedirs(out_dir, exist_ok=True)

    _print_section("Collecting experiment records")
    records = collect_run_metrics(args.runs_dir)
    if not records:
        print(f"[WARN] No metrics.json files found under '{args.runs_dir}'.")
        print("       Run experiments first with notebooks/experiments_fusion.py")
        return

    if args.filter_prefix:
        prefixes = tuple(args.filter_prefix)
        records = [r for r in records if r["run_name"].startswith(prefixes)]

    print(f"  Found {len(records)} epoch records across experiments")

    _print_section("Building summary table")
    summary = build_summary_table(records)

    if HAS_PANDAS:
        import pandas as pd
        n_runs = len(summary) if isinstance(summary, pd.DataFrame) else len(summary)
    else:
        n_runs = len(summary)
    print(f"  {n_runs} unique experiments")

    baseline_miou: Optional[float] = None
    if args.baseline_run and HAS_PANDAS:
        import pandas as pd
        if isinstance(summary, pd.DataFrame):
            base_rows = summary[summary["run_name"].str.contains(args.baseline_run, na=False)]
            if not base_rows.empty:
                baseline_miou = float(base_rows["best_miou"].max())
                print(f"  Baseline ({args.baseline_run}): best_mIoU = {baseline_miou:.5f}")
                summary["delta_vs_baseline"] = (summary["best_miou"] - baseline_miou).round(5)

    _print_section("Per-family comparison")
    family_cmp = compare_fusion_families(summary)
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(family_cmp, pd.DataFrame):
            print(family_cmp.to_string())
        else:
            _print_dict(family_cmp)
    else:
        _print_dict(family_cmp)

    _print_section(f"Top-{args.top} experiments by {args.sort_by}")
    print_comparison_table(summary, top_n=args.top, sort_by=args.sort_by)

    _print_section("Per-class IoU (top 10 experiments)")
    ranked = rank_experiments(summary, metric=args.sort_by)
    if HAS_PANDAS:
        import pandas as pd
        top_names = list(ranked["run_name"].head(10)) if isinstance(ranked, pd.DataFrame) else []
    else:
        top_names = [r["run_name"] for r in (ranked[:10] if isinstance(ranked, list) else [])]

    per_cls = per_class_comparison(records, run_names=top_names if top_names else None)
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(per_cls, pd.DataFrame):
            print(per_cls.to_string(index=False))
    else:
        for row in (per_cls[:10] if isinstance(per_cls, list) else []):
            print(row)

    _print_section("Saving outputs")

    summary_csv = os.path.join(out_dir, "summary.csv")
    save_summary_csv(summary, summary_csv)
    print(f"summary -> {summary_csv}")

    per_cls_csv = os.path.join(out_dir, "per_class.csv")
    all_per_cls = per_class_comparison(records)
    save_summary_csv(all_per_cls, per_cls_csv)
    print(f"per_class -> {per_cls_csv}")

    if HAS_PANDAS:
        import pandas as pd
        if isinstance(family_cmp, pd.DataFrame):
            fam_csv = os.path.join(out_dir, "family_comparison.csv")
            family_cmp.to_csv(fam_csv)
            print(f"family_comparison -> {fam_csv}")

    top_txt = os.path.join(out_dir, f"top{args.top}_by_{args.sort_by}.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison_table(summary, top_n=args.top, sort_by=args.sort_by)
    with open(top_txt, "w") as f:
        f.write(buf.getvalue())
    print(f"top{args.top} table -> {top_txt}")

    json_summary_path = os.path.join(out_dir, "summary.json")
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(summary, pd.DataFrame):
            summary.to_json(json_summary_path, orient="records", indent=2)
    else:
        with open(json_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"summary (json) -> {json_summary_path}")

    print(f"\n[DONE] All outputs written to {out_dir}")

    _print_section("Quick verdict")
    ranked_all = rank_experiments(summary, metric="best_miou")
    if HAS_PANDAS:
        import pandas as pd
        if isinstance(ranked_all, pd.DataFrame) and not ranked_all.empty:
            best = ranked_all.iloc[0]
            print(f"Best model:    {best['run_name']}")
            print(f"Fusion method: {best.get('fusion_method', '?')}")
            print(f"Best mIoU:     {best['best_miou']:.5f}")
            print(f"Rare mIoU:     {best.get('best_rare_miou', 0):.5f}")
            if baseline_miou is not None:
                delta = best["best_miou"] - baseline_miou
                print(f"Delta vs baseline ({args.baseline_run}): {delta:+.5f}")
    elif isinstance(ranked_all, list) and ranked_all:
        best = ranked_all[0]
        print(f"Best model: {best['run_name']}  mIoU={best.get('best_miou', 0):.5f}")


def _print_dict(d: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 4)
        else:
            print(f"{pad}{k}: {v}")


if __name__ == "__main__":
    main()
