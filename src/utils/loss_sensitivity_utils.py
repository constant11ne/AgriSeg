from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Sequence
import pandas as pd


def normalize_pair_weight(mix_weight: float) -> tuple[float, float]:
    return float(mix_weight), float(1.0 - mix_weight)


def cartesian_dict(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def add_name(prefix: str, params: Dict[str, Any]) -> str:
    parts = [prefix]
    for key, value in params.items():
        if isinstance(value, float):
            if value >= 1e-3:
                sval = f"{value:.3f}".rstrip("0").rstrip(".")
            else:
                sval = f"{value:.0e}"
        else:
            sval = str(value)
        parts.append(f"{key}-{sval}")
    return "__".join(parts)


def build_loss_search_spaces(constrained_tversky: bool = True) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []

    for params in cartesian_dict({
        "mix_weight": [0.2, 0.5, 0.8],
        "dice_smooth": [1e-7, 1e-6, 1e-5],
        "soft_bce_smooth": [0.0, 0.05, 0.1],
    }):
        bce_w, dice_w = normalize_pair_weight(params["mix_weight"])
        loss_params = {
            "bce_weight": bce_w,
            "dice_weight": dice_w,
            "dice_smooth": float(params["dice_smooth"]),
            "soft_bce_smooth": float(params["soft_bce_smooth"]),
        }
        experiments.append({
            "experiment_name": add_name("soft_bce_dice", params),
            "loss_mode": "soft_bce_dice",
            "loss_params": loss_params,
        })

    for params in cartesian_dict({
        "mix_weight": [0.2, 0.5, 0.8],
        "dice_smooth": [1e-7, 1e-6, 1e-5],
    }):
        bce_w, dice_w = normalize_pair_weight(params["mix_weight"])
        loss_params = {
            "bce_weight": bce_w,
            "dice_weight": dice_w,
            "dice_smooth": float(params["dice_smooth"]),
        }
        experiments.append({
            "experiment_name": add_name("bce_dice", params),
            "loss_mode": "bce_dice",
            "loss_params": loss_params,
        })

    if constrained_tversky:
        for params in cartesian_dict({
            "mix_weight": [0.2, 0.5, 0.8],
            "alpha": [0.2, 0.5, 0.8],
            "gamma": [1.0, 2.0, 3.0],
        }):
            beta = 1.0 - float(params["alpha"])
            focal_w, tversky_w = normalize_pair_weight(params["mix_weight"])
            loss_params = {
                "focal_weight": focal_w,
                "tversky_weight": tversky_w,
                "tversky_alpha": float(params["alpha"]),
                "tversky_beta": beta,
                "focal_gamma": float(params["gamma"]),
            }
            named_params = dict(params)
            named_params["beta"] = beta
            experiments.append({
                "experiment_name": add_name("focal_tversky_mix", named_params),
                "loss_mode": "focal_tversky_mix",
                "loss_params": loss_params,
            })
    else:
        for params in cartesian_dict({
            "mix_weight": [0.2, 0.5, 0.8],
            "alpha": [0.2, 0.5, 0.8],
            "beta": [0.2, 0.5, 0.8],
            "gamma": [1.0, 2.0, 3.0],
        }):
            focal_w, tversky_w = normalize_pair_weight(params["mix_weight"])
            loss_params = {
                "focal_weight": focal_w,
                "tversky_weight": tversky_w,
                "tversky_alpha": float(params["alpha"]),
                "tversky_beta": float(params["beta"]),
                "focal_gamma": float(params["gamma"]),
            }
            experiments.append({
                "experiment_name": add_name("focal_tversky_mix", params),
                "loss_mode": "focal_tversky_mix",
                "loss_params": loss_params,
            })

    for params in cartesian_dict({
        "mix_weight": [0.2, 0.5, 0.8],
        "gamma": [1.0, 2.0, 3.0],
    }):
        dice_w, focal_w = normalize_pair_weight(params["mix_weight"])
        loss_params = {
            "dice_weight": dice_w,
            "focal_weight": focal_w,
            "focal_gamma": float(params["gamma"]),
        }
        experiments.append({
            "experiment_name": add_name("dice_focal", params),
            "loss_mode": "dice_focal",
            "loss_params": loss_params,
        })

    if constrained_tversky:
        for params in cartesian_dict({
            "mix_weight": [0.2, 0.5, 0.8],
            "alpha": [0.2, 0.5, 0.8],
        }):
            beta = 1.0 - float(params["alpha"])
            bce_w, tversky_w = normalize_pair_weight(params["mix_weight"])
            loss_params = {
                "bce_weight": bce_w,
                "tversky_weight": tversky_w,
                "tversky_alpha": float(params["alpha"]),
                "tversky_beta": beta,
            }
            named_params = dict(params)
            named_params["beta"] = beta
            experiments.append({
                "experiment_name": add_name("bce_tversky", named_params),
                "loss_mode": "bce_tversky",
                "loss_params": loss_params,
            })
    else:
        for params in cartesian_dict({
            "mix_weight": [0.2, 0.5, 0.8],
            "alpha": [0.2, 0.5, 0.8],
            "beta": [0.2, 0.5, 0.8],
        }):
            bce_w, tversky_w = normalize_pair_weight(params["mix_weight"])
            loss_params = {
                "bce_weight": bce_w,
                "tversky_weight": tversky_w,
                "tversky_alpha": float(params["alpha"]),
                "tversky_beta": float(params["beta"]),
            }
            experiments.append({
                "experiment_name": add_name("bce_tversky", params),
                "loss_mode": "bce_tversky",
                "loss_params": loss_params,
            })

    return experiments


def aggregate_repeat_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "best_miou",
        "last_miou",
        "best_macro_f1",
        "last_macro_f1",
        "best_rare_miou",
        "last_rare_miou",
    ]
    key_cols = ["experiment_name", "loss_mode"]

    base_cols = [c for c in df.columns if c not in metric_cols + ["repeat_idx", "seed"]]
    agg_rows = []
    for _, g in df.groupby(key_cols, sort=False):
        row = {col: g.iloc[0][col] for col in base_cols}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        row["repeats"] = int(len(g))
        agg_rows.append(row)
    result = pd.DataFrame(agg_rows)
    if not result.empty:
        result = result.sort_values(["best_miou_mean", "best_macro_f1_mean"], ascending=False).reset_index(drop=True)
    return result
