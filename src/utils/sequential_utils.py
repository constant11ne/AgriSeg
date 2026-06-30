from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Iterable
import pandas as pd
import numpy as np


@dataclass
class StageSpec:
    name: str
    parameter: str
    values: List[float]
    epochs: int
    initial_repeats: int = 1
    final_repeats: int = 3
    shortlist_k: int | None = None
    shortlist_frac: float | None = None

    def validate(self) -> None:
        if self.initial_repeats < 1:
            raise ValueError("initial_repeats must be >= 1")
        if self.final_repeats < self.initial_repeats:
            raise ValueError("final_repeats must be >= initial_repeats")
        if self.shortlist_k is not None and self.shortlist_k < 1:
            raise ValueError("shortlist_k must be >= 1")


def unique_sorted(values: Iterable[float], ndigits: int = 8) -> List[float]:
    return sorted({round(float(v), ndigits) for v in values})


def local_grid_around(
    center: float,
    deltas: List[float],
    lo: float | None = None,
    hi: float | None = None,
) -> List[float]:
    vals = [center + d for d in deltas]
    if lo is not None:
        vals = [max(lo, v) for v in vals]
    if hi is not None:
        vals = [min(hi, v) for v in vals]
    return unique_sorted(vals)


def candidate_name(loss_name: str, params: Dict[str, Any]) -> str:
    parts = [loss_name]
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}-{v:g}")
        else:
            parts.append(f"{k}-{v}")
    return "__".join(parts)


def robust_score(mean: float, std: float, stability_weight: float = 0.25) -> float:
    std = 0.0 if pd.isna(std) else float(std)
    return float(mean) - stability_weight * std


def summarize_repeats(
    df: pd.DataFrame,
    target: str,
    stability_weight: float = 0.25,
) -> pd.DataFrame:
    grouped = (
        df.groupby("candidate_name", dropna=False)
        .agg(
            target_mean=(target, "mean"),
            target_std=(target, "std"),
            repeats=(target, "count"),
            stage_name=("stage_name", "first"),
            loss_name=("loss_name", "first"),
            params_json=("params_json", "first"),
        )
        .reset_index()
    )
    grouped["robust_score"] = grouped.apply(
        lambda r: robust_score(
            r["target_mean"], r["target_std"], stability_weight=stability_weight
        ),
        axis=1,
    )
    return grouped.sort_values(
        ["robust_score", "target_mean"], ascending=False
    ).reset_index(drop=True)


def decode_params(params_json: str) -> Dict[str, Any]:
    return json.loads(params_json)


def infer_next_primary_refinement(
    best_value: float,
    typical_step: float,
    lo: float | None = None,
    hi: float | None = None,
) -> List[float]:
    return local_grid_around(
        best_value,
        [-typical_step, -typical_step / 2, 0.0, typical_step / 2, typical_step],
        lo=lo,
        hi=hi,
    )


def choose_shortlist(
    agg_df: pd.DataFrame,
    stage: StageSpec,
) -> List[str]:
    if len(agg_df) == 0:
        return []
    if stage.shortlist_k is not None:
        k = min(stage.shortlist_k, len(agg_df))
        return agg_df.head(k)["candidate_name"].tolist()
    if stage.shortlist_frac is not None:
        k = max(1, int(np.ceil(len(agg_df) * stage.shortlist_frac)))
        return agg_df.head(k)["candidate_name"].tolist()
    k = min(len(agg_df), max(2, int(np.ceil(len(agg_df) * 0.5))))
    return agg_df.head(k)["candidate_name"].tolist()


def expected_runs_for_stage(stage: StageSpec, n_values: int) -> int:
    stage.validate()
    shortlist = stage.shortlist_k if stage.shortlist_k is not None else None
    if shortlist is None:
        if stage.shortlist_frac is not None:
            shortlist = max(1, int(np.ceil(n_values * stage.shortlist_frac)))
        else:
            shortlist = min(n_values, max(2, int(np.ceil(n_values * 0.5))))
    shortlist = min(shortlist, n_values)
    extra_repeats = max(0, stage.final_repeats - stage.initial_repeats)
    return n_values * stage.initial_repeats + shortlist * extra_repeats