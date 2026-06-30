from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception as e:
    raise SystemExit(
        "statsmodels is required for this script. "
        "Install it with: pip install statsmodels\n"
        f"Import error: {e}"
    )

TARGET_DEFAULT = "best_miou"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze repeated loss-sensitivity runs.")
    parser.add_argument("--input", type=str, default="summary_repeats.csv")
    parser.add_argument("--output-dir", type=str, default="analysis_loss_sensitivity")
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_DEFAULT,
        choices=["best_miou", "best_macro_f1", "best_rare_miou"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return x


def parse_experiment_name(name: str) -> Tuple[str, Dict[str, float]]:
    parts = str(name).split("__")
    base = parts[0]
    params: Dict[str, float] = {}
    for part in parts[1:]:
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        params[key] = safe_float(value)
    return base, params


def sanitize_column(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if re.match(r"^[0-9]", out):
        out = f"x_{out}"
    return out


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely extract a single Series even if duplicate column names exist.
    pandas returns a DataFrame when df[col] matches multiple columns.
    """
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        # take the first matching column
        return obj.iloc[:, 0]
    return obj


def standardize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.mean()) / std


def build_formula(target: str, columns: List[str]) -> str:
    if not columns:
        raise ValueError("No varying hyperparameter columns found.")
    inter = [f"{a}:{b}" for a, b in itertools.combinations(columns, 2)]
    rhs = " + ".join(columns + inter)
    return f"{target} ~ {rhs}"


def classify_interaction_strength(coeff_df: pd.DataFrame) -> str:
    if coeff_df.empty:
        return "unknown"
    inter_mask = coeff_df["term"].str.contains(":", regex=False)
    main_mask = ~inter_mask
    inter_sum = coeff_df.loc[inter_mask, "abs_coef"].sum()
    main_sum = coeff_df.loc[main_mask, "abs_coef"].sum()
    if main_sum == 0 and inter_sum == 0:
        return "unknown"
    if main_sum == 0:
        return "strong"
    ratio = inter_sum / main_sum
    if ratio < 0.20:
        return "weak"
    if ratio < 0.50:
        return "moderate"
    return "strong"


def pretty_term(term: str) -> str:
    return term.replace(":", " × ")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"experiment_name", "loss_mode", args.target}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in input CSV: {sorted(missing)}")

    parsed_rows = []
    all_param_names = set()

    for _, row in df.iterrows():
        _, params = parse_experiment_name(row["experiment_name"])
        parsed_rows.append(params)
        all_param_names.update(params.keys())

    param_df = pd.DataFrame(parsed_rows)
    merged = pd.concat([df.reset_index(drop=True), param_df.reset_index(drop=True)], axis=1)
    merged.to_csv(output_dir / "parsed_repeats.csv", index=False)

    report_lines: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    report_lines.append("LOSS SENSITIVITY ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Target metric: {args.target}")
    report_lines.append("")
    report_lines.append("How repeats are used:")
    report_lines.append("- Repeated runs are kept as separate rows and used directly in regression.")
    report_lines.append("- This preserves variance information from the triple runs.")
    report_lines.append("")

    print("\n=== LOSS SENSITIVITY ANALYSIS ===")
    print(f"Input:  {input_path}")
    print(f"Target: {args.target}")
    print("Using repeated runs as separate observations.")
    print("Writing outputs to:", output_dir)
    print()

    for loss_mode, g in merged.groupby("loss_mode", sort=False):
        param_cols = []
        for c in sorted(all_param_names):
            if c not in g.columns:
                continue
            s = get_series(g, c)
            vals = pd.to_numeric(s, errors="coerce")
            if vals.notna().any() and vals.nunique(dropna=True) > 1:
                param_cols.append(c)

        if not param_cols:
            print(f"[WARN] {loss_mode}: no varying numeric hyperparameters found, skipping.")
            continue

        local = g.copy()
        rename_map = {}
        for c in param_cols:
            new_c = sanitize_column(c)
            rename_map[c] = new_c
            local[new_c] = standardize_series(get_series(local, c))

        y_col = "target_metric"
        local[y_col] = pd.to_numeric(get_series(local, args.target), errors="coerce")

        formula = build_formula(y_col, list(rename_map.values()))
        model = smf.ols(formula=formula, data=local).fit()

        coef_rows = []
        for term, coef, pval, stderr, tval in zip(
            model.params.index,
            model.params.values,
            model.pvalues.values,
            model.bse.values,
            model.tvalues.values,
        ):
            if term == "Intercept":
                continue
            coef_rows.append(
                {
                    "loss_mode": loss_mode,
                    "term": term,
                    "pretty_term": pretty_term(term),
                    "coef": float(coef),
                    "abs_coef": float(abs(coef)),
                    "pvalue": float(pval),
                    "stderr": float(stderr),
                    "tvalue": float(tval),
                    "is_interaction": ":" in term,
                }
            )

        coef_df = pd.DataFrame(coef_rows).sort_values("abs_coef", ascending=False)
        coef_df.to_csv(output_dir / f"per_loss_{loss_mode}_coefficients.csv", index=False)

        agg = (
            g.groupby("experiment_name", dropna=False)[args.target]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        best_row = agg.iloc[0]
        best_name = str(best_row["experiment_name"])
        best_mean = float(best_row["mean"])
        best_std = float(best_row["std"]) if not pd.isna(best_row["std"]) else 0.0
        best_n = int(best_row["count"])

        interaction_strength = classify_interaction_strength(coef_df)
        strongest_interactions = coef_df.loc[coef_df["is_interaction"]]

        print(f"--- {loss_mode} ---")
        print(f"rows={len(g)}, unique_configs={g['experiment_name'].nunique()}, params={param_cols}")
        print(f"best config: {best_name}")
        print(f"best {args.target}: {best_mean:.6f} ± {best_std:.6f} (n={best_n})")
        print(f"R^2={model.rsquared:.4f}, adj_R^2={model.rsquared_adj:.4f}")
        print(f"interaction strength (heuristic): {interaction_strength}")
        print("strongest terms:")
        if coef_df.empty:
            print("  [none]")
        else:
            for _, r in coef_df.head(args.top_k).iterrows():
                print(
                    f"  {r['pretty_term']}: coef={r['coef']:+.4f}, "
                    f"|coef|={r['abs_coef']:.4f}, p={r['pvalue']:.4g}"
                )
        print()

        report_lines.append(f"LOSS MODE: {loss_mode}")
        report_lines.append("-" * 80)
        report_lines.append(f"Rows used (including repeats): {len(g)}")
        report_lines.append(f"Unique configurations: {g['experiment_name'].nunique()}")
        report_lines.append(f"Hyperparameters analyzed: {param_cols}")
        report_lines.append(f"Best configuration: {best_name}")
        report_lines.append(f"Best mean {args.target}: {best_mean:.6f} ± {best_std:.6f} (n={best_n})")
        report_lines.append(f"Model formula: {formula}")
        report_lines.append(f"R^2={model.rsquared:.6f}")
        report_lines.append(f"Adjusted R^2={model.rsquared_adj:.6f}")
        report_lines.append(f"Interaction strength (heuristic): {interaction_strength}")
        report_lines.append("Top terms by absolute standardized coefficient:")
        if coef_df.empty:
            report_lines.append("  [none]")
        else:
            for _, r in coef_df.head(args.top_k).iterrows():
                report_lines.append(
                    f"  {r['pretty_term']}: coef={r['coef']:+.6f}, "
                    f"abs_coef={r['abs_coef']:.6f}, p={r['pvalue']:.6g}, "
                    f"stderr={r['stderr']:.6f}, t={r['tvalue']:.6f}"
                )
        report_lines.append("")

        summary_rows.append(
            {
                "loss_mode": loss_mode,
                "target": args.target,
                "rows_used": len(g),
                "unique_configs": int(g["experiment_name"].nunique()),
                "n_hyperparams": len(param_cols),
                "hyperparams": json.dumps(param_cols),
                "best_experiment_name": best_name,
                f"best_{args.target}_mean": best_mean,
                f"best_{args.target}_std": best_std,
                "rsquared": float(model.rsquared),
                "rsquared_adj": float(model.rsquared_adj),
                "interaction_strength": interaction_strength,
                "strongest_term": None if coef_df.empty else str(coef_df.iloc[0]["pretty_term"]),
                "strongest_term_abs_coef": None if coef_df.empty else float(coef_df.iloc[0]["abs_coef"]),
                "strongest_interaction_term": None if strongest_interactions.empty else str(strongest_interactions.iloc[0]["pretty_term"]),
                "strongest_interaction_abs_coef": None if strongest_interactions.empty else float(strongest_interactions.iloc[0]["abs_coef"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "effects_summary.csv", index=False)
    (output_dir / "analysis_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("=== Analysis complete ===")
    print("Saved:")
    print(" -", output_dir / "analysis_report.txt")
    print(" -", output_dir / "effects_summary.csv")
    print(" -", output_dir / "parsed_repeats.csv")
    print(" - per-loss coefficient CSVs")


if __name__ == "__main__":
    main()
