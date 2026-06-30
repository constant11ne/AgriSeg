#!/usr/bin/env python3
"""
notebooks/experiments_fusion.py
=================================
Fusion-family experiment runner.

This script executes the full fusion experiment matrix as described
in the project blueprint.  It is organised into three waves:

  Wave A  – mandatory baselines (RGB, NIR, early, late)
  Wave B  – core mid / adapter / attention variants
  Wave C  – optional cross-attention and PEFT overlays

Usage
-----
    python -m notebooks.experiments_fusion \\
        --data-root /path/to/dataset \\
        --wave A \\
        --encoder timm-efficientnet-b4 \\
        --epochs 50 \\
        --batch-size 8 \\
        --dry-run        # print commands only, don't execute

To run only a specific experiment by name:
    python -m notebooks.experiments_fusion \\
        --data-root /path/to/dataset \\
        --run rgb_only

All experiments share:
  * soft_bce_dice loss (bce_weight=0.12, dice_weight=0.88, dice_smooth=1e-5, label_smoothing=0.0)
  * same optimizer / scheduler
  * same dataset splits and augmentations
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime

# -----------------------------------------------------------------------
#  Experiment specification dataclass
# -----------------------------------------------------------------------

@dataclass
class FusionExperiment:
    name: str
    fusion_family: str
    fusion_method: str
    use_nir: bool = True
    nir_init_mode: str = "random"
    freeze_rgb_encoder: str = "none"
    freeze_rgb_stages: int = 3
    partial_unfreeze_last_n: int = 2
    nir_branch_width: float = 0.5
    fusion_hidden_dim: int = 128
    nir_active_levels: int = 5
    progressive_level_indices: Optional[str] = None
    progressive_level_name: Optional[str] = None
    use_ndvi: bool = False
    align_norm_mode: str = "none"
    phase1_freeze_epochs: Optional[int] = None
    extra_flags: List[str] = field(default_factory=list)
    wave: str = "A"
    notes: str = ""


# -----------------------------------------------------------------------
#  Experiment matrix
# -----------------------------------------------------------------------

# Default loss params (best from sequential search on soft_bce_dice)
_DEFAULT_LOSS = "soft_bce_dice"
_BEST_BCE_WEIGHT    = 0.12
_BEST_DICE_WEIGHT   = 0.88
_BEST_DICE_SMOOTH   = 1e-5
_BEST_LABEL_SMOOTH  = 0.0

# --- Wave A: baselines ---
WAVE_A: List[FusionExperiment] = [
    FusionExperiment(
        name="rgb_only",
        fusion_family="rgb",
        fusion_method="early_concat",
        use_nir=False,
        wave="A",
        notes="RGB-only baseline (no NIR)",
    ),
    FusionExperiment(
        name="nir_only",
        fusion_family="nir",
        fusion_method="early_concat",
        use_nir=True,
        wave="A",
        notes="NIR-only baseline (1-channel input)",
    ),
    FusionExperiment(
        name="early_concat_random",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="random",
        wave="A",
        notes="Early concat fusion, NIR init=random",
    ),
    FusionExperiment(
        name="early_concat_copy-r",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-r",
        wave="A",
        notes="Early concat fusion, NIR init=copy-r",
    ),
    # FusionExperiment(
    #     name="early_concat_copy-g",
    #     fusion_family="early",
    #     fusion_method="early_concat",
    #     use_nir=True,
    #     nir_init_mode="copy-g",
    #     wave="A",
    #     notes="Early concat fusion, NIR init=copy-g",
    # ),
    # FusionExperiment(
    #     name="early_concat_copy-b",
    #     fusion_family="early",
    #     fusion_method="early_concat",
    #     use_nir=True,
    #     nir_init_mode="copy-b",
    #     wave="A",
    #     notes="Early concat fusion, NIR init=copy-b",
    # ),
    FusionExperiment(
        name="early_concat_copy-mean",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-mean",
        wave="A",
        notes="Early concat fusion, NIR init=copy-mean",
    ),
    FusionExperiment(
        name="late_avg",
        fusion_family="late",
        fusion_method="late_avg",
        use_nir=True,
        wave="A",
        notes="Late avg logits fusion",
    ),
    FusionExperiment(
        name="late_weighted",
        fusion_family="late",
        fusion_method="late_weighted",
        use_nir=True,
        wave="A",
        notes="Late learnable-scalar-weighted fusion",
    ),
    FusionExperiment(
        name="late_per_class",
        fusion_family="late",
        fusion_method="late_per_class",
        use_nir=True,
        wave="A",
        notes="Late per-class weighted fusion",
    ),
]


# --- Wave B: high-capacity dual-stream / adapter / attention variants only ---
# WAVE_B: List[FusionExperiment] = [
#     FusionExperiment(
#         name="mid_bottleneck_concat_h256",
#         fusion_family="mid",
#         fusion_method="bottleneck_concat",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="High-capacity bottleneck concat with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_bottleneck_sum_h256",
#         fusion_family="mid",
#         fusion_method="bottleneck_sum",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="High-capacity bottleneck sum with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_concat_h256",
#         fusion_family="mid",
#         fusion_method="multiscale_concat",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="Higher-capacity multiscale concat with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_concat_wide_h256",
#         fusion_family="mid",
#         fusion_method="multiscale_concat",
#         use_nir=True,
#         nir_branch_width=1.0,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="Wide NIR multiscale concat with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_concat_h384",
#         fusion_family="mid",
#         fusion_method="multiscale_concat",
#         use_nir=True,
#         fusion_hidden_dim=384,
#         wave="B",
#         notes="Higher-capacity multiscale concat with hidden dim 384",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_concat_wide_h384",
#         fusion_family="mid",
#         fusion_method="multiscale_concat",
#         use_nir=True,
#         nir_branch_width=1.0,
#         fusion_hidden_dim=384,
#         wave="B",
#         notes="Wide NIR multiscale concat with hidden dim 384",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_se_h256",
#         fusion_family="mid",
#         fusion_method="multiscale_se",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="SE multiscale fusion with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_se_wide_h256",
#         fusion_family="mid",
#         fusion_method="multiscale_se",
#         use_nir=True,
#         nir_branch_width=1.0,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="Wide NIR SE multiscale fusion with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="mid_multiscale_cbam_h256",
#         fusion_family="attention",
#         fusion_method="multiscale_cbam",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="CBAM multiscale fusion with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="attention_cross_attn_bottleneck_h256",
#         fusion_family="attention",
#         fusion_method="cross_attn_bottleneck",
#         use_nir=True,
#         fusion_hidden_dim=256,
#         wave="B",
#         notes="Cross-attention bottleneck with hidden dim 256",
#     ),
#     FusionExperiment(
#         name="adapter_input_wide_frozen_rgb",
#         fusion_family="adapter",
#         fusion_method="adapter_input",
#         use_nir=True,
#         nir_branch_width=1.0,
#         freeze_rgb_encoder="full",
#         wave="B",
#         notes="Input adapter with full-width NIR branch and fully frozen RGB backbone",
#     ),
#     FusionExperiment(
#         name="adapter_multilevel_wide_frozen_rgb",
#         fusion_family="adapter",
#         fusion_method="adapter_multilevel",
#         use_nir=True,
#         nir_branch_width=1.0,
#         freeze_rgb_encoder="full",
#         wave="B",
#         notes="Multilevel residual adapter with full-width NIR branch, RGB fully frozen",
#     ),
#     FusionExperiment(
#         name="adapter_residual_wide_partial_unfreeze",
#         fusion_family="adapter",
#         fusion_method="adapter_residual",
#         use_nir=True,
#         nir_branch_width=1.0,
#         freeze_rgb_encoder="partial",
#         partial_unfreeze_last_n=2,
#         wave="B",
#         notes="Residual adapter with full-width NIR branch, RGB partially unfrozen",
#     ),
#     FusionExperiment(
#         name="adapter_multilevel_wide_first3_frozen",
#         fusion_family="adapter",
#         fusion_method="adapter_multilevel",
#         use_nir=True,
#         nir_branch_width=1.0,
#         freeze_rgb_encoder="first_N",
#         freeze_rgb_stages=3,
#         wave="B",
#         notes="Multilevel adapter with full-width NIR branch, first 3 RGB stages frozen",
#     ),
# ]

# --- Wave B: sorted from most promising to least promising (theory) ---
WAVE_B: List[FusionExperiment] = [
    FusionExperiment(
        name="mid_multiscale_se_wide_h256",
        fusion_family="mid",
        fusion_method="multiscale_se",
        use_nir=True,
        nir_branch_width=1.0,
        fusion_hidden_dim=256,
        wave="B",
        notes="Wide NIR SE multiscale fusion with hidden dim 256",
    ),
    FusionExperiment(
        name="mid_multiscale_concat_wide_h256",
        fusion_family="mid",
        fusion_method="multiscale_concat",
        use_nir=True,
        nir_branch_width=1.0,
        fusion_hidden_dim=256,
        wave="B",
        notes="Wide NIR multiscale concat with hidden dim 256",
    ),
    FusionExperiment(
        name="mid_multiscale_cbam_h256",
        fusion_family="attention",
        fusion_method="multiscale_cbam",
        use_nir=True,
        fusion_hidden_dim=256,
        wave="B",
        notes="CBAM multiscale fusion with hidden dim 256",
    ),
    FusionExperiment(
        name="adapter_residual_wide_partial_unfreeze",
        fusion_family="adapter",
        fusion_method="adapter_residual",
        use_nir=True,
        nir_branch_width=1.0,
        freeze_rgb_encoder="partial",
        partial_unfreeze_last_n=2,
        wave="B",
        notes="Residual adapter with full-width NIR branch, RGB partially unfrozen",
    ),
    FusionExperiment(
        name="adapter_multilevel_wide_frozen_rgb",
        fusion_family="adapter",
        fusion_method="adapter_multilevel",
        use_nir=True,
        nir_branch_width=1.0,
        freeze_rgb_encoder="full",
        wave="B",
        notes="Multilevel residual adapter with full-width NIR branch, RGB fully frozen",
    ),
    FusionExperiment(
        name="attention_cross_attn_bottleneck_h256",
        fusion_family="attention",
        fusion_method="cross_attn_bottleneck",
        use_nir=True,
        fusion_hidden_dim=256,
        wave="B",
        notes="Cross-attention bottleneck with hidden dim 256",
    ),
    FusionExperiment(
        name="mid_bottleneck_concat_h256",
        fusion_family="mid",
        fusion_method="bottleneck_concat",
        use_nir=True,
        fusion_hidden_dim=256,
        wave="B",
        notes="High-capacity bottleneck concat with hidden dim 256",
    ),
    FusionExperiment(
        name="mid_bottleneck_sum_h256",
        fusion_family="mid",
        fusion_method="bottleneck_sum",
        use_nir=True,
        fusion_hidden_dim=256,
        wave="B",
        notes="High-capacity bottleneck sum with hidden dim 256",
    ),
]


# --- Wave C: optional / expensive ---
WAVE_C: List[FusionExperiment] = [
    # Cross-attention at bottleneck
    FusionExperiment(
        name="attention_cross_attn_bottleneck",
        fusion_family="attention",
        fusion_method="cross_attn_bottleneck",
        use_nir=True,
        wave="C",
        notes="Cross-attention at bottleneck only",
    ),
    # LoRA overlays on best early fusion model
    FusionExperiment(
        name="early_concat_copymean_lora",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-mean",
        extra_flags=["--lora", "--lora-rank", "4", "--lora-alpha", "8"],
        wave="C",
        notes="Early fusion copy-mean + LoRA on all conv layers",
    ),
    # LoRA overlay on best adapter model
    FusionExperiment(
        name="adapter_multilevel_frozen_rgb_lora",
        fusion_family="adapter",
        fusion_method="adapter_multilevel",
        use_nir=True,
        freeze_rgb_encoder="full",
        extra_flags=["--lora", "--lora-rank", "4", "--lora-alpha", "8"],
        wave="C",
        notes="Multilevel adapter + LoRA, RGB fully frozen",
    ),
    # BitFit overlay
    FusionExperiment(
        name="early_concat_copymean_bitfit",
        fusion_family="early",
        fusion_method="early_concat",
        use_nir=True,
        nir_init_mode="copy-mean",
        extra_flags=["--bitfit"],
        wave="C",
        notes="Early fusion copy-mean + BitFit",
    ),
]


WAVE_F: List[FusionExperiment] = [
    FusionExperiment(
        name="progressive_concat_ibn_8_16_32",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        align_norm_mode="rgb_nir_ibn_stem_s1",
        progressive_level_indices="2,3,4",
        progressive_level_name="8_16_32",
        phase1_freeze_epochs=0,
        wave="F",
        notes="Progressive concat with RGB+NIR IBN alignment on stem+s1, levels 8,16,32",
    ),
    FusionExperiment(
        name="progressive_concat_ibn_8_16_32_64",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        align_norm_mode="rgb_nir_ibn_stem_s1",
        progressive_level_indices="1,2,3,4",
        progressive_level_name="8_16_32_64",
        phase1_freeze_epochs=0,
        wave="F",
        notes="Progressive concat with RGB+NIR IBN alignment on stem+s1, levels 8,16,32,64",
    ),
    FusionExperiment(
        name="progressive_concat_ibn_full",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        align_norm_mode="rgb_nir_ibn_stem_s1",
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        phase1_freeze_epochs=0,
        wave="F",
        notes="Progressive concat with RGB+NIR IBN alignment on stem+s1, full encoder",
    ),
]

WAVE_C_SE: List[FusionExperiment] = []
WAVE_D: List[FusionExperiment] = []
WAVE_E: List[FusionExperiment] = []

WAVE_G: List[FusionExperiment] = [
    FusionExperiment(
        name="progressive_concat_ibn_ndvi_8_16_32",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        use_ndvi=True,
        align_norm_mode="rgb_nir_ibn_stem_s1",
        progressive_level_indices="2,3,4",
        progressive_level_name="8_16_32",
        phase1_freeze_epochs=0,
        wave="G",
        notes="Progressive concat with RGB+NIR IBN alignment and NDVI on levels 8,16,32",
    ),
    FusionExperiment(
        name="progressive_concat_ibn_ndvi_full",
        fusion_family="mid",
        fusion_method="progressive_concat",
        use_nir=True,
        use_ndvi=True,
        align_norm_mode="rgb_nir_ibn_stem_s1",
        progressive_level_indices="0,1,2,3,4",
        progressive_level_name="8_16_32_64_128",
        phase1_freeze_epochs=0,
        wave="G",
        notes="Progressive concat with RGB+NIR IBN alignment and NDVI on the full encoder",
    ),
]

ALL_EXPERIMENTS: List[FusionExperiment] = WAVE_A + WAVE_B + WAVE_C + WAVE_C_SE + WAVE_D + WAVE_E + WAVE_F + WAVE_G
DUAL_STREAM_FAMILIES = {"mid", "adapter", "attention"}


def expected_run_dir_name(exp: FusionExperiment, model: str, encoder: str, loss_name: str = _DEFAULT_LOSS) -> str:
    parts = [model, encoder]
    if exp.fusion_family:
        parts.append(exp.fusion_family)
        if exp.fusion_method:
            parts.append(exp.fusion_method)
    else:
        parts.append(f"nir{int(exp.use_nir)}")
        parts.append("early")
    nir_init = getattr(exp, "nir_init_mode", "random")
    if nir_init and nir_init != "random":
        parts.append(nir_init.replace("-", ""))
    freeze_rgb = getattr(exp, "freeze_rgb_encoder", "none")
    if freeze_rgb and freeze_rgb != "none":
        parts.append(f"freeze_{freeze_rgb}")
        if freeze_rgb == "first_N":
            parts.append(f"s{exp.freeze_rgb_stages}")
        elif freeze_rgb == "partial":
            parts.append(f"last{exp.partial_unfreeze_last_n}")
    nir_branch_width = float(getattr(exp, "nir_branch_width", 0.5))
    if nir_branch_width != 0.5:
        parts.append(f"nirw{str(nir_branch_width).replace('.', 'p')}")
    fusion_hidden_dim = int(getattr(exp, "fusion_hidden_dim", 128))
    if fusion_hidden_dim != 128:
        parts.append(f"h{fusion_hidden_dim}")
    parts.append(loss_name)
    for i, flag in enumerate(exp.extra_flags):
        if flag == "--lora":
            try:
                rank = exp.extra_flags[i + 2]
                alpha = exp.extra_flags[i + 4]
                parts.append(f"lora{rank}a{alpha}")
            except Exception:
                parts.append("lora")
        elif flag == "--bitfit":
            parts.append("bitfit")
    return "_".join(parts)


def find_best_checkpoint_for_experiment(exp: FusionExperiment, runs_dir: str, model: str, encoder: str) -> Optional[str]:
    exp_dir = os.path.join(runs_dir, expected_run_dir_name(exp, model=model, encoder=encoder))
    best_path = os.path.join(exp_dir, "best.pth")
    return best_path if os.path.exists(best_path) else None


# -----------------------------------------------------------------------
#  CLI builder
# -----------------------------------------------------------------------

def build_command(
    exp: FusionExperiment,
    data_root: str,
    encoder: str,
    encoder_weights: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    img_size: List[int],
    seed: int,
    runs_dir: str,
    late_rgb_ckpt: Optional[str] = None,
    late_nir_ckpt: Optional[str] = None,
    nir_pretrained_lite_ckpt: Optional[str] = None,
    nir_pretrained_lite_wide_ckpt: Optional[str] = None,
    nir_pretrained_smp_ckpt: Optional[str] = None,
    phase1_freeze_epochs: Optional[int] = 3,
    cache_root: Optional[str] = None,
    cache_mmap: bool = False,
    extra_train_flags: Optional[List[str]] = None,
) -> List[str]:
    """Build the subprocess command list for a given experiment."""
    cmd = [
        sys.executable, "-m", "src.train",
        "--data-root", data_root,
        "--model", model,
        "--encoder", encoder,
        "--encoder-weights", encoder_weights,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--img-size", str(img_size[0]), str(img_size[1]),
        "--seed", str(seed),
        "--runs-dir", runs_dir,
        # Default best loss config (soft_bce_dice, tuned params from sequential search)
        "--loss", _DEFAULT_LOSS,
        "--bce-weight", str(_BEST_BCE_WEIGHT),
        "--dice-weight", str(_BEST_DICE_WEIGHT),
        "--dice-smooth", str(_BEST_DICE_SMOOTH),
        "--label-smoothing", str(_BEST_LABEL_SMOOTH),
        # Fusion flags
        "--fusion-family", exp.fusion_family,
        "--fusion-method", exp.fusion_method,
        "--nir-init-mode", exp.nir_init_mode,
        "--freeze-rgb-encoder", exp.freeze_rgb_encoder,
        "--freeze-rgb-stages", str(exp.freeze_rgb_stages),
        "--partial-unfreeze-last-n", str(exp.partial_unfreeze_last_n),
        "--nir-branch-width", str(exp.nir_branch_width),
        "--fusion-hidden-dim", str(exp.fusion_hidden_dim),
        "--nir-active-levels", str(exp.nir_active_levels),
    ]
    if exp.progressive_level_indices:
        cmd += ["--progressive-level-indices", exp.progressive_level_indices]
    if exp.progressive_level_name:
        cmd += ["--progressive-level-name", exp.progressive_level_name]
    if getattr(exp, "align_norm_mode", "none") not in (None, "", "none"):
        cmd += ["--align-norm-mode", exp.align_norm_mode]
    effective_phase1_freeze_epochs = exp.phase1_freeze_epochs if getattr(exp, "phase1_freeze_epochs", None) is not None else phase1_freeze_epochs
    if effective_phase1_freeze_epochs is not None:
        cmd += ["--phase1-freeze-epochs", str(effective_phase1_freeze_epochs)]
    if exp.use_nir:
        cmd.append("--use-nir")
    if getattr(exp, "use_ndvi", False):
        cmd.append("--use-ndvi")
    if exp.use_nir:
        # Legacy flag (kept for compatibility)
        cmd += ["--nir-fusion", "early"]

    # Late fusion checkpoints
    if exp.fusion_method in ("late_avg", "late_weighted", "late_per_class"):
        if not late_rgb_ckpt or not late_nir_ckpt:
            raise ValueError(
                f"Late-fusion experiment '{exp.name}' requires both RGB and NIR checkpoints. "
                "Pass --late-fusion-checkpoint-rgb and --late-fusion-checkpoint-nir."
            )
        cmd += ["--late-fusion-checkpoint-rgb", late_rgb_ckpt]
        cmd += ["--late-fusion-checkpoint-nir", late_nir_ckpt]

    if exp.fusion_family in DUAL_STREAM_FAMILIES:
        lite_ckpt = nir_pretrained_lite_wide_ckpt if float(getattr(exp, "nir_branch_width", 0.5)) == 1.0 else nir_pretrained_lite_ckpt
        if lite_ckpt:
            cmd += ["--nir-pretrained-path", lite_ckpt]
    elif exp.fusion_family in {"nir", "late"} and nir_pretrained_smp_ckpt:
        cmd += ["--nir-pretrained-path", nir_pretrained_smp_ckpt]

    if cache_root:
        cmd += ["--cache-root", cache_root]
    if cache_mmap:
        cmd.append("--cache-mmap")

    # Experiment-specific extra flags (e.g. --lora)
    cmd.extend(exp.extra_flags)

    # Additional flags passed from the CLI
    if extra_train_flags:
        cmd.extend(extra_train_flags)

    return cmd


# -----------------------------------------------------------------------
#  Runner
# -----------------------------------------------------------------------

def run_experiments(
    experiments: List[FusionExperiment],
    data_root: str,
    encoder: str,
    encoder_weights: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    img_size: List[int],
    seed: int,
    runs_dir: str,
    dry_run: bool = False,
    late_rgb_ckpt: Optional[str] = None,
    late_nir_ckpt: Optional[str] = None,
    nir_pretrained_lite_ckpt: Optional[str] = None,
    nir_pretrained_lite_wide_ckpt: Optional[str] = None,
    nir_pretrained_smp_ckpt: Optional[str] = None,
    phase1_freeze_epochs: Optional[int] = 3,
    cache_root: Optional[str] = None,
    cache_mmap: bool = False,
    extra_train_flags: Optional[List[str]] = None,
) -> None:
    results_log: list = []
    total = len(experiments)

    if late_rgb_ckpt is None:
        late_rgb_ckpt = find_best_checkpoint_for_experiment(
            next((e for e in WAVE_A if e.name == "rgb_only"), None) or FusionExperiment("rgb_only","rgb","early_concat",use_nir=False),
            runs_dir=runs_dir, model=model, encoder=encoder,
        )
    if late_nir_ckpt is None:
        late_nir_ckpt = find_best_checkpoint_for_experiment(
            next((e for e in WAVE_A if e.name == "nir_only"), None) or FusionExperiment("nir_only","nir","early_concat",use_nir=True),
            runs_dir=runs_dir, model=model, encoder=encoder,
        )

    for idx, exp in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{total}] EXPERIMENT: {exp.name}  (Wave {exp.wave})")
        if exp.notes:
            print(f"  Notes: {exp.notes}")
        print(f"{'='*70}")

        actual_epochs = epochs

        if exp.fusion_family == "late" and (not late_rgb_ckpt or not late_nir_ckpt):
            print(f"[WARN] Skipping late-fusion experiment {exp.name!r}: both late-fusion checkpoints are required.")
            results_log.append({"name": exp.name, "wave": exp.wave, "status": "SKIPPED_MISSING_LATE_CKPTS"})
            continue

        cmd = build_command(
            exp,
            data_root=data_root,
            encoder=encoder,
            encoder_weights=encoder_weights,
            model=model,
            epochs=actual_epochs,
            batch_size=batch_size,
            lr=lr,
            img_size=img_size,
            seed=seed,
            runs_dir=runs_dir,
            late_rgb_ckpt=late_rgb_ckpt,
            late_nir_ckpt=late_nir_ckpt,
            nir_pretrained_lite_ckpt=nir_pretrained_lite_ckpt,
            nir_pretrained_lite_wide_ckpt=nir_pretrained_lite_wide_ckpt,
            nir_pretrained_smp_ckpt=nir_pretrained_smp_ckpt,
            phase1_freeze_epochs=phase1_freeze_epochs,
            cache_root=cache_root,
            cache_mmap=cache_mmap,
            extra_train_flags=extra_train_flags,
        )

        print("Command:\n  " + " ".join(cmd))

        if dry_run:
            results_log.append({"name": exp.name, "status": "dry_run", "cmd": cmd})
            continue

        start_t = datetime.now()
        try:
            ret = subprocess.run(cmd, check=True)
            status = "success"
        except subprocess.CalledProcessError as exc:
            status = f"FAILED (returncode={exc.returncode})"
            print(f"[ERROR] Experiment {exp.name} failed: {status}")

        elapsed = (datetime.now() - start_t).total_seconds()
        summary_payload = {}
        if status == "success":
            try:
                candidate_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                candidate_dirs = sorted(
                    candidate_dirs,
                    key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)),
                    reverse=True,
                )
                for d in candidate_dirs:
                    summary_file = os.path.join(runs_dir, d, "summary.json")
                    if os.path.exists(summary_file):
                        with open(summary_file, "r") as f:
                            summary_payload = json.load(f)
                        break
            except Exception as exc:
                print(f"[WARN] Failed to read summary for {exp.name}: {exc}")

        exp_dir = summary_payload.get("exp_dir") or os.path.join(runs_dir, expected_run_dir_name(exp, model=model, encoder=encoder))
        best_ckpt_candidate = os.path.join(exp_dir, "best.pth")
        if status == "success" and os.path.exists(best_ckpt_candidate):
            if exp.name == "rgb_only":
                late_rgb_ckpt = best_ckpt_candidate
                print(f"[INFO] Auto-registered RGB late-fusion checkpoint: {late_rgb_ckpt}")
            elif exp.name == "nir_only":
                late_nir_ckpt = best_ckpt_candidate
                print(f"[INFO] Auto-registered NIR late-fusion checkpoint: {late_nir_ckpt}")

        results_log.append({
            "name": exp.name,
            "wave": exp.wave,
            "status": status,
            "elapsed_s": elapsed,
            "fusion_family": exp.fusion_family,
            "fusion_method": exp.fusion_method,
            "epochs": actual_epochs,
            "notes": exp.notes,
            **summary_payload,
        })

    # Save summary
    summary_path = os.path.join(runs_dir, "fusion_experiments_summary.json")
    os.makedirs(runs_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\n[DONE] Results summary saved to {summary_path}")

    # Print quick table
    print("\n" + "="*70)
    print(f"{'Name':<45} {'Wave':>6} {'Status':<20}")
    print("-"*70)
    for r in results_log:
        print(f"{r['name']:<45} {r.get('wave','?'):>6} {r.get('status','?'):<20}")


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion experiment runner")
    p.add_argument("--data-root", required=True, help="Path to dataset root")
    p.add_argument(
        "--wave", type=str, default="A",
        choices=["A", "B", "C", "C_SE", "D", "E", "F", "G", "AB", "ABC", "ABD", "ABCD", "all"],
        help="Which wave(s) to run (A=baselines, B=core, C=optional)",
    )
    p.add_argument("--run", type=str, default=None,
                   help="Run a single experiment by name (overrides --wave)")
    p.add_argument("--encoder", type=str, default="timm-efficientnet-b4",
                   help="Backbone encoder name")
    p.add_argument("--encoder-weights", type=str, default="imagenet")
    p.add_argument("--model", type=str, default="fpn",
                   choices=["unet", "fpn", "deeplabv3p"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--late-fusion-checkpoint-rgb", type=str, default=None,
                   help="Checkpoint for RGB branch (late fusion experiments)")
    p.add_argument("--late-fusion-checkpoint-nir", type=str, default=None,
                   help="Checkpoint for NIR branch (late fusion experiments)")
    p.add_argument("--nir-pretrained-lite-checkpoint", type=str, default=None,
                   help="Pretrained encoder checkpoint for NIREncoderLite width=0.5 (mid/adapter/attention)")
    p.add_argument("--nir-pretrained-lite-wide-checkpoint", type=str, default=None,
                   help="Pretrained encoder checkpoint for NIREncoderLite width=1.0 (wide NIR experiments)")
    p.add_argument("--nir-pretrained-smp-checkpoint", type=str, default=None,
                   help="Pretrained encoder checkpoint for SMP-based NIR branch (nir/late)")
    p.add_argument("--phase1-freeze-epochs", type=int, default=3,
                   help="Epochs with frozen RGB encoder for dual-stream models; passed through to src.train")
    p.add_argument("--cache-root", type=str, default=None,
                   help="Optional dataset cache root passed through to src.train")
    p.add_argument("--cache-mmap", action="store_true",
                   help="Enable memory-mapped loading for cached arrays")
    p.add_argument("--extra-train-flags", type=str, default=None,
                   help="Extra flags passed verbatim to src.train (space-separated)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Select experiments
    if args.run is not None:
        exps = [e for e in ALL_EXPERIMENTS if e.name == args.run]
        if not exps:
            names = [e.name for e in ALL_EXPERIMENTS]
            print(f"[ERROR] Unknown experiment '{args.run}'. Available:\n  " + "\n  ".join(names))
            sys.exit(1)
    elif args.wave in ("A", "AB", "ABC", "all"):
        exps = WAVE_A
        if args.wave in ("AB", "ABC", "all"):
            exps = exps + WAVE_B
        if args.wave in ("ABC", "all"):
            exps = exps + WAVE_C
    elif args.wave == "B":
        exps = WAVE_B
    elif args.wave == "C":
        exps = WAVE_C
    elif args.wave == "C_SE":
        exps = WAVE_C_SE
    elif args.wave == "D":
        exps = WAVE_D
    elif args.wave == "E":
        exps = WAVE_E
    elif args.wave == "F":
        exps = WAVE_F
    elif args.wave == "G":
        exps = WAVE_G
    else:
        exps = ALL_EXPERIMENTS

    extra = args.extra_train_flags.split() if args.extra_train_flags else None

    print(f"[INFO] Running {len(exps)} experiments (wave={args.wave})")
    print(f"[INFO] Encoder: {args.encoder}  Model: {args.model}")
    print(f"[INFO] Epochs: {args.epochs}  Batch: {args.batch_size}")
    if args.dry_run:
        print("[INFO] DRY RUN – no training will be performed")

    run_experiments(
        experiments=exps,
        data_root=args.data_root,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        seed=args.seed,
        runs_dir=args.runs_dir,
        dry_run=args.dry_run,
        late_rgb_ckpt=args.late_fusion_checkpoint_rgb,
        late_nir_ckpt=args.late_fusion_checkpoint_nir,
        nir_pretrained_lite_ckpt=args.nir_pretrained_lite_checkpoint,
        nir_pretrained_lite_wide_ckpt=args.nir_pretrained_lite_wide_checkpoint,
        nir_pretrained_smp_ckpt=args.nir_pretrained_smp_checkpoint,
        phase1_freeze_epochs=args.phase1_freeze_epochs,
        cache_root=args.cache_root,
        cache_mmap=args.cache_mmap,
        extra_train_flags=extra,
    )


if __name__ == "__main__":
    main()