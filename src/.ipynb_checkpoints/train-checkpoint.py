from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

from src.datamodules.agri_vision import build_dataloaders, ANOMALY_CLASSES
from src.losses.losses import MultiLoss
from src.utils.metrics import IoUMetric
from src.models.lora import inject_lora, freeze_model
from src.models.bitfit import freeze_except_biases
from src.models import CustomUNet, UNetMidFusion, UNetAdapter
from src.models.late_fusion import build_late_fusion, LateFusionWrapper
from src.models.backbone_utils import count_trainable_params, freeze_encoder_full, freeze_encoder_stages, unfreeze_last_n_stages, adapt_first_conv

try:
    from src.models.dual_stream_fpn import DualStreamFPN
    from src.models.multi_stream_fpn import MultiStreamFPN
    HAS_DUAL_STREAM = True
except ImportError:
    HAS_DUAL_STREAM = False

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    smp = None
    HAS_SMP = False

import types


def _replace_first_conv(model, new_in_channels: int, init_mode: str = "random"):
    return adapt_first_conv(model, new_in_channels=new_in_channels, init_mode=init_mode)


from typing import Iterable, Callable


def _set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def _apply_rgb_freeze_policy_from_args(model: nn.Module, args: argparse.Namespace) -> None:
    if not hasattr(model, "rgb_encoder"):
        return
    rgb_encoder = model.rgb_encoder
    freeze_mode = getattr(args, "freeze_rgb_encoder", "none")
    freeze_stages = int(getattr(args, "freeze_rgb_stages", 3))
    partial_last_n = int(getattr(args, "partial_unfreeze_last_n", 2))

    for p in rgb_encoder.parameters():
        p.requires_grad = True

    if freeze_mode == "none":
        return
    if freeze_mode == "full":
        freeze_encoder_full(rgb_encoder)
    elif freeze_mode == "stem":
        freeze_encoder_stages(rgb_encoder, freeze_n_stages=1)
    elif freeze_mode == "first_N":
        freeze_encoder_stages(rgb_encoder, freeze_n_stages=freeze_stages)
    elif freeze_mode == "partial":
        freeze_encoder_full(rgb_encoder)
        unfreeze_last_n_stages(rgb_encoder, n=partial_last_n)
    else:
        raise ValueError(f"Unknown freeze_rgb_encoder mode: {freeze_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model on Agriculture‑Vision")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "fpn", "deeplabv3p"],
        help=(
            "Model architecture.  When segmentation_models_pytorch is installed, "
            "'unet', 'fpn' and 'deeplabv3p' are available.  Otherwise only "
            "'unet' is supported via the custom implementation."
        ),
    )
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder name (e.g. resnet34, efficientnet-b0)")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Pretrained weights (e.g. imagenet)")
    parser.add_argument("--use-nir", action="store_true", help="Whether input images include a fourth NIR channel")
    parser.add_argument("--use-ndvi", action="store_true", help="Append fixed NDVI channel computed from NIR and red band")
    parser.add_argument("--train-augment-mode", type=str, default="full", choices=["full", "minimal", "none"], help="Training augmentation policy")
    parser.add_argument("--oversample-rare", action="store_true", help="Use WeightedRandomSampler to oversample rare classes in the training set")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA injection on convolutional layers")
    parser.add_argument("--lora-rank", type=int, default=4, help="Rank of LoRA matrices")
    parser.add_argument("--lora-alpha", type=int, default=8, help="Alpha scaling for LoRA")
    parser.add_argument("--bitfit", action="store_true", help="Enable BitFit (train only biases)")
    parser.add_argument(
        "--loss",
        type=str,
        default="bce_dice",
        choices=[
            "bce",
            "dice",
            "bce_dice",
            "soft_bce_dice",
            "tversky",
            "bce_tversky",
            "focal",
            "bce_focal",
            "dice_focal",
            "focal_tversky_mix",
        ],
        help=(
            "Loss function.  Options:"
            " 'bce' (binary cross entropy),"
            " 'dice' (Dice loss),"
            " 'bce_dice' (equal mix of BCE and Dice),"
            " 'tversky' (Tversky loss),"
            " 'bce_tversky' (equal mix of BCE and Tversky),"
            " 'focal' (focal loss),"
            " 'bce_focal' (equal mix of BCE and focal),"
            " 'dice_focal' (equal mix of Dice and focal)."
        ),
    )
    parser.add_argument("--class-weights", type=str, default=None, help="JSON list of class weights, e.g. '[1.0, 2.0, ...]'")
    parser.add_argument("--tversky-alpha", type=float, default=0.5, help="Alpha for Tversky loss")
    parser.add_argument("--tversky-beta", type=float, default=0.5, help="Beta for Tversky loss")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight of BCE component in mixed losses")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Weight of Dice component in mixed losses")
    parser.add_argument("--dice-smooth", type=float, default=1e-6, help="Smoothing constant for Dice loss")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for soft_bce_dice")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512], help="Image size (H W)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for DataLoader (when num_workers>0)")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable persistent DataLoader workers")
    parser.add_argument("--cache-root", type=str, default=None, help="Optional path to compact precomputed image/mask cache")
    parser.add_argument("--cache-mmap", action="store_true", help="Use memory-mapped reads for cache files when possible")
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile if available")
    parser.add_argument("--disable-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP) even when using CUDA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--save-last-every", type=int, default=1, help="Save last.pth every N epochs (0=disable)")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Update tqdm postfix every N steps")

    parser.add_argument(
        "--max-train-steps-per-epoch",
        type=int,
        default=None,
        help="Maximum number of batches to process in each training epoch (None = use full loader)",
    )
    parser.add_argument(
        "--max-val-steps-per-epoch",
        type=int,
        default=None,
        help="Maximum number of batches to process in each validation epoch (None = use full loader)",
    )

    parser.add_argument(
        "--nir-fusion", type=str, default="early", choices=["early", "mid", "adapter"],
        help=(
            "Strategy to integrate NIR channel: "
            "'early' concatenates NIR with RGB at input (default), "
            "'mid' uses separate RGB/NIR branches fused before encoding, "
            "'adapter' inserts lightweight adapters into the backbone."
        ),
    )
    parser.add_argument(
        "--nir-init-mode", type=str, default="random", choices=["random", "copy-r", "copy-g", "copy-b", "copy-mean"],
        help=(
            "How to initialise weights for the first convolution's NIR channel when using early fusion. "
            "'random' leaves the weights randomly initialised, while 'copy-r/g/b' copies the weights from the corresponding RGB channel, "
            "and 'copy-mean' averages RGB channel weights to initialise NIR."
        ),
    )
    parser.add_argument(
        "--norm", type=str, default="bn", choices=["bn", "ibn"],
        help="Normalisation layer to use: BatchNorm ('bn') or IBN ('ibn').",
    )
    parser.add_argument(
        "--logit-scales", type=str, default=None,
        help=(
            "Optional JSON list of scaling factors to apply to class logits during evaluation. "
            "Useful for boosting rare classes and suppressing the background. "
            "Example: '[1.0, 1.2, 1.2, 1.5, 1.5, 1.3]' for 6 classes."
        ),
    )

    parser.add_argument(
        "--fusion-family",
        type=str,
        default=None,
        choices=["rgb", "nir", "ndvi", "early", "mid", "late", "adapter", "attention"],
        help=(
            "High-level fusion family.  When set, overrides --nir-fusion. "
            "Choices: rgb (RGB-only baseline), nir (NIR-only baseline), "
            "early, mid, late, adapter, attention."
        ),
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default=None,
        choices=[
            "early_concat", "early_rgb_ndvi",
            "bottleneck_concat", "bottleneck_sum",
            "multiscale_concat", "multiscale_sum",
            "multiscale_gated", "multiscale_se", "multiscale_cbam", "progressive_concat", "progressive_se",
            "progressive_concat_rgb_nir_ndvi", "progressive_concat_rgb_ndvi",
            "adapter_input", "adapter_multilevel", "adapter_residual",
            "late_avg", "late_weighted", "late_per_class",
            "cross_attn_bottleneck",
        ],
        help="Specific fusion method within the chosen family.",
    )
    parser.add_argument(
        "--freeze-rgb-encoder",
        type=str,
        default="none",
        choices=["none", "full", "stem", "first_N", "partial"],
        help="Freeze policy for the RGB encoder branch in dual-stream models.",
    )
    parser.add_argument(
        "--freeze-rgb-stages", type=int, default=3,
        help="Number of RGB encoder stages to freeze when --freeze-rgb-encoder=first_N.",
    )
    parser.add_argument(
        "--partial-unfreeze-last-n", type=int, default=2,
        help="Number of last RGB encoder stages to keep trainable when --freeze-rgb-encoder=partial.",
    )
    parser.add_argument(
        "--nir-branch-width", type=float, default=0.5,
        help="Width multiplier for the lightweight NIR branch in dual-stream models.",
    )
    parser.add_argument(
        "--fusion-hidden-dim", type=int, default=128,
        help="Channel width inside the FPN decoder for dual-stream models.",
    )
    parser.add_argument(
        "--nir-active-levels", type=int, default=5,
        help="Number of deepest NIR branch levels to fuse in progressive Wave C/C_SE experiments (1..5).",
    )
    parser.add_argument(
        "--progressive-level-indices", type=str, default=None,
        help="Optional comma-separated level indices for progressive fusion in shallow-to-deep order, e.g. '0,1' or '1,2,3'.",
    )
    parser.add_argument(
        "--progressive-level-name", type=str, default=None,
        help="Optional label used in experiment directory naming for progressive fusion, e.g. '64_32_16'.",
    )
    parser.add_argument(
        "--align-norm-mode", type=str, default="none",
        choices=["none", "nir_ibn_stem", "nir_ibn_stem_s1", "rgb_nir_ibn_stem", "rgb_nir_ibn_stem_s1", "rgb_ndvi_ibn_features", "rgb_nir_ndvi_ibn_features"],
        help="Optional early alignment normalization for dual/multi-stream models.",
    )
    parser.add_argument(
        "--late-fusion-checkpoint-rgb", type=str, default=None,
        help="Path to checkpoint for the RGB branch of a late-fusion model.",
    )
    parser.add_argument(
        "--late-fusion-checkpoint-nir", type=str, default=None,
        help="Path to checkpoint for the NIR branch of a late-fusion model.",
    )
    parser.add_argument(
        "--nir-pretrained-path", type=str, default=None,
        help="Path to a pretrained NIR encoder checkpoint (best_encoder_only.pt or full checkpoint).",
    )
    parser.add_argument(
        "--rgb-pretrained-path", type=str, default=None,
        help="Optional checkpoint used to initialize RGB encoder branch in dual-stream models.",
    )
    parser.add_argument(
        "--ndvi-pretrained-path", type=str, default=None,
        help="Optional checkpoint used to initialize NDVI encoder branch in multi-stream models.",
    )
    parser.add_argument(
        "--phase1-freeze-epochs", type=int, default=3,
        help="For dual-stream: epochs to freeze encoder before differential-LR phase. 3=default, 0=disable, -1=auto (35%%).",
    )
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory where experiment folders will be created")
    args = parser.parse_args()
    return args


def build_model(
    model_name: str,
    encoder: str,
    encoder_weights: Optional[str],
    num_classes: int,
    in_channels: int,
    *,
    nir_fusion: str = "early",
    norm_type: str = "bn",
    use_nir: bool = False,
    use_ndvi: bool = False,
    nir_init_mode: str = "random",
    fusion_family: Optional[str] = None,
    fusion_method: Optional[str] = None,
    freeze_rgb: str = "none",
    freeze_rgb_stages: int = 3,
    partial_unfreeze_last_n: int = 2,
    nir_branch_width: float = 0.5,
    fusion_hidden_dim: int = 128,
    align_norm_mode: str = "none",
    nir_active_levels: int = 5,
    progressive_level_indices: Optional[str] = None,
    output_size: tuple = (512, 512),
    late_rgb_model: Optional[nn.Module] = None,
    late_nir_model: Optional[nn.Module] = None,
) -> nn.Module:
    model_name = model_name.lower()

    if fusion_family is not None:
        return _build_model_fusion_family(
            fusion_family=fusion_family,
            fusion_method=fusion_method or "early_concat",
            model_name=model_name,
            encoder=encoder,
            encoder_weights=encoder_weights,
            num_classes=num_classes,
            norm_type=norm_type,
            freeze_rgb=freeze_rgb,
            freeze_rgb_stages=freeze_rgb_stages,
            partial_unfreeze_last_n=partial_unfreeze_last_n,
            nir_branch_width=nir_branch_width,
            fusion_hidden_dim=fusion_hidden_dim,
            nir_init_mode=nir_init_mode,
            use_nir=use_nir,
            use_ndvi=use_ndvi,
            align_norm_mode=align_norm_mode,
            nir_active_levels=nir_active_levels,
            progressive_level_indices=progressive_level_indices,
            output_size=output_size,
            late_rgb_model=late_rgb_model,
            late_nir_model=late_nir_model,
        )

    use_smp = HAS_SMP and model_name in {"unet", "fpn", "deeplabv3p"}
    if use_smp and (nir_fusion == "early" or not use_nir or in_channels == 3):
        if model_name == "unet":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=3,
                activation=None,
            )
        elif model_name == "fpn":
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=3,
                activation=None,
            )
        elif model_name == "deeplabv3p":
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                in_channels=3,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported SMP model: {model_name}")
        if in_channels != 3:
            _replace_first_conv(model, new_in_channels=in_channels, init_mode="random")
        return model
    else:
        if model_name != "unet":
            raise ValueError(f"Unsupported model '{model_name}' for custom implementation")
        if nir_fusion == "early" or not use_nir or in_channels == 3:
            return CustomUNet(n_channels=in_channels, n_classes=num_classes, norm_type=norm_type)
        elif nir_fusion == "mid":
            return UNetMidFusion(n_channels=in_channels, n_classes=num_classes, norm_type=norm_type)
        elif nir_fusion == "adapter":
            return UNetAdapter(n_channels=in_channels, n_classes=num_classes, norm_type=norm_type)
        else:
            raise ValueError(f"Unsupported NIR fusion strategy: {nir_fusion}")


_DUAL_STREAM_METHODS = {
    "bottleneck_concat", "bottleneck_sum",
    "multiscale_concat", "multiscale_sum",
    "multiscale_gated", "multiscale_se", "multiscale_cbam",
    "adapter_input", "adapter_multilevel", "adapter_residual",
    "cross_attn_bottleneck",
}
_LATE_METHODS = {"late_avg", "late_weighted", "late_per_class"}


def _build_smp_model(
    model_name: str,
    encoder: str,
    encoder_weights: Optional[str],
    num_classes: int,
    in_channels: int = 3,
) -> nn.Module:
    if not HAS_SMP:
        raise ImportError("segmentation_models_pytorch is required. "
                          "Install with: pip install segmentation-models-pytorch")
    if model_name == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                        classes=num_classes, in_channels=in_channels, activation=None)
    elif model_name == "fpn":
        return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights,
                       classes=num_classes, in_channels=in_channels, activation=None)
    elif model_name == "deeplabv3p":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights,
                                 classes=num_classes, in_channels=in_channels, activation=None)
    raise ValueError(f"Unknown model_name: {model_name}")


def _build_model_fusion_family(
    fusion_family: str,
    fusion_method: str,
    model_name: str,
    encoder: str,
    encoder_weights: Optional[str],
    num_classes: int,
    norm_type: str = "bn",
    nir_init_mode: str = "random",
    use_nir: bool = False,
    use_ndvi: bool = False,
    freeze_rgb: str = "none",
    freeze_rgb_stages: int = 3,
    partial_unfreeze_last_n: int = 2,
    nir_branch_width: float = 0.5,
    fusion_hidden_dim: int = 128,
    align_norm_mode: str = "none",
    nir_active_levels: int = 5,
    progressive_level_indices: Optional[str] = None,
    output_size: tuple = (512, 512),
    late_rgb_model: Optional[nn.Module] = None,
    late_nir_model: Optional[nn.Module] = None,
) -> nn.Module:
    if fusion_family == "rgb":
        return _build_smp_model(model_name, encoder, encoder_weights, num_classes, in_channels=3)

    if fusion_family == "nir":
        nir_model_in_channels = 2 if use_ndvi else 1
        if HAS_SMP:
            m = _build_smp_model(model_name, encoder, encoder_weights, num_classes, in_channels=3)
            _replace_first_conv(m, new_in_channels=nir_model_in_channels, init_mode=nir_init_mode)
            return m
        return CustomUNet(n_channels=nir_model_in_channels, n_classes=num_classes, norm_type=norm_type)

    if fusion_family == "ndvi":
        if HAS_SMP:
            m = _build_smp_model(model_name, encoder, encoder_weights, num_classes, in_channels=3)
            _replace_first_conv(m, new_in_channels=1, init_mode=nir_init_mode)
            return m
        return CustomUNet(n_channels=1, n_classes=num_classes, norm_type=norm_type)

    if fusion_family == "early" or fusion_method in {"early_concat", "early_rgb_ndvi"}:
        early_in_channels = 3 + (1 if use_nir else 0) + (1 if use_ndvi else 0)
        if HAS_SMP:
            m = _build_smp_model(model_name, encoder, encoder_weights, num_classes, in_channels=3)
            _replace_first_conv(m, new_in_channels=early_in_channels, init_mode=nir_init_mode)
            return m
        return CustomUNet(n_channels=early_in_channels, n_classes=num_classes, norm_type=norm_type)

    if fusion_family == "late" or fusion_method in _LATE_METHODS:
        if late_rgb_model is None or late_nir_model is None:
            raise ValueError(
                "late_rgb_model and late_nir_model must be provided for late fusion. "
                "Build each separately and pass them in."
            )
        return build_late_fusion(
            fusion_method=fusion_method,
            rgb_model=late_rgb_model,
            nir_model=late_nir_model,
            num_classes=num_classes,
            freeze_branches=True,
        )

    if fusion_method in {"progressive_concat_rgb_nir_ndvi", "progressive_concat_rgb_ndvi"}:
        if not HAS_DUAL_STREAM:
            raise ImportError("segmentation_models_pytorch is required for MultiStreamFPN.")
        branches = ("nir", "ndvi") if fusion_method == "progressive_concat_rgb_nir_ndvi" else ("ndvi",)
        return MultiStreamFPN(
            num_classes=num_classes,
            branches=branches,
            fusion_method=fusion_method,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            fpn_channels=fusion_hidden_dim,
            freeze_rgb=freeze_rgb,
            freeze_rgb_stages=freeze_rgb_stages,
            partial_unfreeze_last_n=partial_unfreeze_last_n,
            output_size=output_size,
            depth=5,
            progressive_level_indices=progressive_level_indices,
            align_norm_mode=align_norm_mode,
        )

    if fusion_family in ("mid", "adapter", "attention") or fusion_method in _DUAL_STREAM_METHODS:
        if not HAS_DUAL_STREAM:
            raise ImportError(
                "segmentation_models_pytorch is required for DualStreamFPN. "
                "Install with: pip install segmentation-models-pytorch"
            )
        smp_method = fusion_method
        return DualStreamFPN(
            num_classes=num_classes,
            fusion_method=smp_method,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            nir_width_mult=nir_branch_width,
            fpn_channels=fusion_hidden_dim,
            freeze_rgb=freeze_rgb,
            freeze_rgb_stages=freeze_rgb_stages,
            partial_unfreeze_last_n=partial_unfreeze_last_n,
            output_size=output_size,
            align_norm_mode=align_norm_mode,
            nir_active_levels=nir_active_levels,
            progressive_level_indices=progressive_level_indices,
            nir_in_channels=2 if use_ndvi else 1,
        )

    raise ValueError(
        f"Unknown fusion_family='{fusion_family}' / fusion_method='{fusion_method}'."
    )


def init_first_conv_nir(model: nn.Module, mode: str) -> None:
    if mode == "random":
        return
    channel_map = {
        "copy-r": 0,
        "copy-g": 1,
        "copy-b": 2,
    }
    copy_idx = channel_map.get(mode, None)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels >= 4:
            w = m.weight.data
            if mode == "copy-mean":
                rgb_mean = w[:, :3, :, :].mean(dim=1, keepdim=True)
                w[:, 3:4, :, :] = rgb_mean
            elif copy_idx is not None:
                w[:, 3:4, :, :] = w[:, copy_idx:copy_idx + 1, :, :]
            m.weight.data = w
            break




def _load_state_dict_flexible(path: str) -> Dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("encoder_state_dict", "model_state_dict", "state_dict"):
            sub = state.get(key)
            if isinstance(sub, dict):
                return sub
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if any(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict


def _load_encoder_compatible(target_module: nn.Module, ckpt_path: str, label: str) -> None:
    state = _load_state_dict_flexible(ckpt_path)
    for prefix in ("encoder.", "model.encoder.", f"{label.lower()}_encoder.", ""):
        if prefix and any(k.startswith(prefix) for k in state.keys()):
            cleaned = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break
    else:
        cleaned = state

    target_state = target_module.state_dict()
    compatible = {}
    skipped = []
    for k, v in cleaned.items():
        if k in target_state and getattr(target_state[k], "shape", None) == getattr(v, "shape", None):
            compatible[k] = v
        else:
            skipped.append(k)
    result = target_module.load_state_dict(compatible, strict=False)
    missing = list(getattr(result, "missing_keys", [])) if result is not None else []
    unexpected = list(getattr(result, "unexpected_keys", [])) if result is not None else []
    print(f"[INFO] Loaded pretrained {label} weights from {ckpt_path} | compatible={len(compatible)} skipped={len(skipped)}")
    if missing:
        print(f"[WARN] Missing {label} keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected {label} keys: {len(unexpected)}")


def _load_pretrained_nir_branch(model: nn.Module, fusion_family: Optional[str], ckpt_path: Optional[str], fusion_method: Optional[str] = None) -> None:
    if not ckpt_path:
        return
    if hasattr(model, "nir_encoder"):
        _load_encoder_compatible(model.nir_encoder, ckpt_path, "NIR")
    elif fusion_family == "nir" and hasattr(model, "encoder"):
        _load_encoder_compatible(model.encoder, ckpt_path, "NIR")
    elif fusion_family == "late" and hasattr(model, "nir_model") and hasattr(model.nir_model, "encoder"):
        _load_encoder_compatible(model.nir_model.encoder, ckpt_path, "NIR")
    else:
        print(f"[WARN] --nir-pretrained-path ignored for fusion_family={fusion_family}")


def _load_pretrained_rgb_branch(model: nn.Module, fusion_family: Optional[str], ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    if hasattr(model, "rgb_encoder"):
        _load_encoder_compatible(model.rgb_encoder, ckpt_path, "RGB")
    elif fusion_family == "late" and hasattr(model, "rgb_model") and hasattr(model.rgb_model, "encoder"):
        _load_encoder_compatible(model.rgb_model.encoder, ckpt_path, "RGB")
    else:
        print(f"[WARN] --rgb-pretrained-path ignored for fusion_family={fusion_family}")


def _load_pretrained_ndvi_branch(model: nn.Module, fusion_family: Optional[str], ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    if hasattr(model, "ndvi_encoder"):
        _load_encoder_compatible(model.ndvi_encoder, ckpt_path, "NDVI")
    elif fusion_family == "ndvi" and hasattr(model, "encoder"):
        _load_encoder_compatible(model.encoder, ckpt_path, "NDVI")
    else:
        print(f"[WARN] --ndvi-pretrained-path ignored for fusion_family={fusion_family}")

def prepare_experiment_name(args: argparse.Namespace) -> str:
    parts = [args.model, args.encoder]
    if getattr(args, "fusion_family", None):
        parts.append(args.fusion_family)
        fm = getattr(args, "fusion_method", None)
        if fm:
            parts.append(fm)
    else:
        parts.append(f"nir{int(args.use_nir)}")
        parts.append(getattr(args, "nir_fusion", "early"))
    nir_init = getattr(args, "nir_init_mode", "random")
    if nir_init and nir_init != "random":
        parts.append(nir_init.replace("-", ""))
    freeze_rgb = getattr(args, "freeze_rgb_encoder", "none")
    if freeze_rgb and freeze_rgb != "none":
        parts.append(f"freeze_{freeze_rgb}")
        if freeze_rgb == "first_N":
            parts.append(f"s{args.freeze_rgb_stages}")
        elif freeze_rgb == "partial":
            parts.append(f"last{args.partial_unfreeze_last_n}")
    parts.append(args.loss)
    if args.lora:
        parts.append(f"lora{args.lora_rank}a{args.lora_alpha}")
    if args.bitfit:
        parts.append("bitfit")
    if args.freeze_encoder:
        parts.append("freezeenc")
    if args.class_weights:
        parts.append("cw")
    nir_branch_width = float(getattr(args, "nir_branch_width", 0.5))
    if nir_branch_width != 0.5:
        parts.append(f"nirw{str(nir_branch_width).replace('.', 'p')}")
    fusion_hidden_dim = int(getattr(args, "fusion_hidden_dim", 128))
    if fusion_hidden_dim != 128:
        parts.append(f"h{fusion_hidden_dim}")
    if str(getattr(args, "fusion_method", "")).startswith("progressive_concat") or getattr(args, "fusion_method", None) == "progressive_se":
        prog_name = getattr(args, "progressive_level_name", None)
        if prog_name:
            parts.append(prog_name)
        else:
            parts.append(f"nirl{int(getattr(args, 'nir_active_levels', 5))}")
    align_norm_mode = getattr(args, "align_norm_mode", "none")
    if align_norm_mode and align_norm_mode != "none":
        parts.append(align_norm_mode)
    if getattr(args, "use_ndvi", False):
        parts.append("ndvi")
    return "_".join(parts)


def _select_model_input(images: torch.Tensor, fusion_family: Optional[str], use_ndvi: bool) -> torch.Tensor:
    if fusion_family == "rgb":
        return images[:, :3]

    if fusion_family == "ndvi":
        if images.shape[1] >= 5:
            return images[:, 4:5]
        if images.shape[1] == 4:
            return images[:, 3:4]
        if images.shape[1] == 1:
            return images
        raise RuntimeError(f"NDVI-only model expected NDVI channel, got {images.shape[1]} input channels")

    if fusion_family == "nir":
        if use_ndvi:
            if images.shape[1] >= 5:
                return images[:, 3:5]
            if images.shape[1] == 2:
                return images
            if images.shape[1] == 4:
                red01 = (images[:, 0:1] + 1.0) * 0.5
                nir01 = (images[:, 3:4] + 1.0) * 0.5
                ndvi = (nir01 - red01) / (nir01 + red01 + 1e-6)
                ndvi01 = (torch.clamp(ndvi, -1.0, 1.0) + 1.0) * 0.5
                ndvi_norm = (ndvi01 - 0.5) / 0.5
                return torch.cat([images[:, 3:4], ndvi_norm], dim=1)
            raise RuntimeError(
                f"NIR+NDVI model expected 5 channels from dataloader or 2 selected channels, got {images.shape[1]}"
            )
        return images[:, 3:4] if images.shape[1] >= 4 else images[:, :1]

    return images

from tqdm.auto import tqdm

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    class_weights: Optional[List[float]] = None
    if args.class_weights:
        class_weights = json.loads(args.class_weights)
        assert len(class_weights) == len(ANOMALY_CLASSES), (
            "class_weights must have length equal to number of classes"
        )

    dl_kwargs: Dict[str, Any] = {}
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
        dl_kwargs["persistent_workers"] = not args.no_persistent_workers

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=(args.train_augment_mode != "none"),
        augment_mode=getattr(args, "train_augment_mode", "full"),
        use_nir=args.use_nir,
        use_ndvi=getattr(args, "use_ndvi", False),
        oversample_rare=args.oversample_rare,
        class_weights=class_weights,
        cache_root=getattr(args, "cache_root", None),
        cache_mmap=getattr(args, "cache_mmap", False),
        **dl_kwargs,
    )


    num_classes = len(ANOMALY_CLASSES)
    in_channels = 3 + (1 if args.use_nir else 0) + (1 if getattr(args, "use_ndvi", False) else 0)
    encoder_weights_val = args.encoder_weights if args.encoder_weights != "none" else None

    fusion_family = getattr(args, "fusion_family", None)
    fusion_method = getattr(args, "fusion_method", None)

    late_rgb_model: Optional[nn.Module] = None
    late_nir_model: Optional[nn.Module] = None
    if fusion_family == "late" or (fusion_method and fusion_method.startswith("late_")):
        print("[INFO] Building late-fusion branch models...")
        late_rgb_model = build_model(
            args.model, args.encoder, encoder_weights_val, num_classes, 3,
            fusion_family="rgb",
        )
        late_nir_channels = 2 if getattr(args, "use_ndvi", False) else 1
        late_nir_model = build_model(
            args.model, args.encoder, encoder_weights_val, num_classes, late_nir_channels,
            fusion_family="nir",
            use_ndvi=getattr(args, "use_ndvi", False),
        )
        rgb_ckpt = getattr(args, "late_fusion_checkpoint_rgb", None)
        nir_ckpt = getattr(args, "late_fusion_checkpoint_nir", None)
        if rgb_ckpt:
            state = torch.load(rgb_ckpt, map_location="cpu")
            late_rgb_model.load_state_dict(state.get("model_state_dict", state), strict=True)
            print(f"[INFO] Loaded RGB branch from {rgb_ckpt}")
        if nir_ckpt:
            state = torch.load(nir_ckpt, map_location="cpu")
            late_nir_model.load_state_dict(state.get("model_state_dict", state), strict=True)
            print(f"[INFO] Loaded NIR branch from {nir_ckpt}")

    model = build_model(
        args.model,
        args.encoder,
        encoder_weights_val,
        num_classes,
        in_channels,
        nir_fusion=args.nir_fusion,
        norm_type=args.norm,
        use_nir=args.use_nir,
        use_ndvi=getattr(args, "use_ndvi", False),
        fusion_family=fusion_family,
        fusion_method=fusion_method,
        freeze_rgb=getattr(args, "freeze_rgb_encoder", "none"),
        freeze_rgb_stages=getattr(args, "freeze_rgb_stages", 3),
        partial_unfreeze_last_n=getattr(args, "partial_unfreeze_last_n", 2),
        nir_branch_width=getattr(args, "nir_branch_width", 0.5),
        fusion_hidden_dim=getattr(args, "fusion_hidden_dim", 128),
        align_norm_mode=getattr(args, "align_norm_mode", "none"),
        nir_active_levels=getattr(args, "nir_active_levels", 5),
        progressive_level_indices=getattr(args, "progressive_level_indices", None),
        output_size=tuple(args.img_size),
        late_rgb_model=late_rgb_model,
        late_nir_model=late_nir_model,
    )
    _load_pretrained_rgb_branch(model, fusion_family, getattr(args, "rgb_pretrained_path", None))
    _load_pretrained_nir_branch(model, fusion_family, getattr(args, "nir_pretrained_path", None), fusion_method)
    _load_pretrained_ndvi_branch(model, fusion_family, getattr(args, "ndvi_pretrained_path", None))

    is_early = (fusion_family in (None, "early")) and (
        fusion_method in (None, "early_concat")
    ) and args.nir_fusion == "early"
    if args.use_nir and is_early and in_channels >= 4:
        init_first_conv_nir(model, args.nir_init_mode)

    trainable_p, total_p = count_trainable_params(model)
    print(f"[INFO] Parameters: trainable={trainable_p:,}  total={total_p:,}")

    if args.freeze_encoder:
        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False
        else:
            for name, p in model.named_parameters():
                if not name.startswith("decoder"):
                    p.requires_grad = False

    if args.lora:
        freeze_model(model)
        inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    if args.bitfit:
        freeze_except_biases(model)

    model = model.to(args.device)

    if args.device.startswith("cuda"):
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if args.compile:
        try:
            model = torch.compile(model)
            print("[INFO] Model compiled with torch.compile")
        except Exception as compile_err:
            print(f"[WARN] torch.compile failed: {compile_err}. Proceeding without compilation.")

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    use_amp = args.device.startswith("cuda") and not args.disable_amp
    scaler: Optional[Any] = None
    device_type = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
        except Exception:
            amp_dtype = torch.float16
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and amp_dtype == torch.float16))
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and amp_dtype == torch.float16))

    criterion = MultiLoss(
        mode=args.loss,
        class_weights=class_weights,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        label_smoothing=args.label_smoothing,
    )
    criterion = criterion.to(args.device)

    logit_scales: Optional[List[float]] = None
    if args.logit_scales:
        try:
            scales = json.loads(args.logit_scales)
            assert isinstance(scales, list) and len(scales) == num_classes
            logit_scales = [float(x) for x in scales]
        except Exception as e:
            print(f"[WARN] Could not parse logit_scales '{args.logit_scales}': {e}. Ignoring.")

    
    _DUAL_STREAM_FAMILIES = {"mid", "adapter", "attention"}
    _is_dual_stream = fusion_family in _DUAL_STREAM_FAMILIES and hasattr(model, "rgb_encoder")

    if _is_dual_stream:
        if args.phase1_freeze_epochs == 0:
            phase1_epochs = 0
        elif args.phase1_freeze_epochs > 0:
            phase1_epochs = args.phase1_freeze_epochs
        else:
            phase1_epochs = max(4, round(args.epochs * 0.35))  # auto: 35%
    else:
        phase1_epochs = 0

    if _is_dual_stream and phase1_epochs > 0:
        for p in model.rgb_encoder.parameters():
            p.requires_grad = False
        phase1_params = [p for p in model.parameters() if p.requires_grad]

        phase1_max_lr = args.lr * 0.3
        optimizer = optim.AdamW(phase1_params, lr=phase1_max_lr, weight_decay=args.weight_decay)

        phase1_warmup = max(1, phase1_epochs // 2)
        phase1_cosine = max(1, phase1_epochs - phase1_warmup)
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=phase1_warmup)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=phase1_cosine, eta_min=phase1_max_lr * 0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                                  milestones=[phase1_warmup])

        print(f"[INFO] TWO-PHASE training: phase1={phase1_epochs} ep (encoder FROZEN, "
              f"max_lr={phase1_max_lr:.1e}), phase2={args.epochs - phase1_epochs} ep (differential LR)")
    else:
        _is_dual_stream = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            optimizer = None
            scheduler = None
            args.epochs = 1
            print("[INFO] No trainable parameters detected. Running eval-only single-epoch mode.")
        else:
            optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            warmup_epochs = max(1, round(args.epochs * 0.2))
            cosine_epochs = max(1, args.epochs - warmup_epochs)
            warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                                      milestones=[warmup_epochs])
            print(f"[INFO] Scheduler: warmup {warmup_epochs} ep → cosine {cosine_epochs} ep")

    exp_name = prepare_experiment_name(args)
    exp_dir = os.path.join(args.runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    _ff = getattr(args, "fusion_family", None)

    best_miou = float("-inf")
    best_epoch = 0
    best_ckpt_path = os.path.join(exp_dir, "best.pth")
    last_ckpt_path = os.path.join(exp_dir, "last.pth")
    _phase2_started = False
    all_metrics = []

    for epoch in range(args.epochs):
        current_epoch = epoch + 1
        print(f"\n=== Epoch {current_epoch}/{args.epochs} ===")

        if _is_dual_stream and phase1_epochs > 0 and epoch == phase1_epochs and not _phase2_started:
            _phase2_started = True
            print("[INFO] *** PHASE 2 START: applying requested RGB freeze policy with differential LR ***")
            _apply_rgb_freeze_policy_from_args(model, args)
            enc_params   = [p for n, p in model.named_parameters()
                             if p.requires_grad and n.startswith("rgb_encoder")]
            other_params = [p for n, p in model.named_parameters()
                             if p.requires_grad and not n.startswith("rgb_encoder")]
            p2_decoder_lr = args.lr * 0.20
            p2_encoder_lr = args.lr * 0.05
            param_groups = []
            if enc_params:
                param_groups.append({"params": enc_params, "lr": p2_encoder_lr})
            if other_params:
                param_groups.append({"params": other_params, "lr": p2_decoder_lr})
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
            remaining = args.epochs - phase1_epochs
            p2_warmup = min(2, remaining) if remaining > 0 else 1
            p2_cosine = max(1, remaining - p2_warmup)
            p2_warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                        total_iters=p2_warmup)
            p2_cosine_sched = CosineAnnealingLR(optimizer, T_max=p2_cosine)
            scheduler = SequentialLR(optimizer,
                                     schedulers=[p2_warmup_sched, p2_cosine_sched],
                                     milestones=[p2_warmup])
            print(f"[INFO] Phase 2: encoder lr={p2_encoder_lr:.1e}, decoder lr={p2_decoder_lr:.1e} "
                  f"(capped at 20% of base), warmup={p2_warmup} ep + cosine={p2_cosine} ep")

        model.train()
        if _is_dual_stream and (not _phase2_started or getattr(args, "freeze_rgb_encoder", "none") == "full"):
            _set_bn_eval(model.rgb_encoder)

        train_loss = torch.zeros((), device=args.device if args.device.startswith("cuda") else "cpu", dtype=torch.float64)
        num_batches = 0
        _dbg_gnorm_sum = torch.zeros_like(train_loss)

        if optimizer is not None:
            _lr0 = optimizer.param_groups[0]["lr"]
            _lr1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else _lr0
        else:
            _lr0 = 0.0
            _lr1 = 0.0
        _phase_str = "1 (enc frozen)" if (_is_dual_stream and not _phase2_started) else "2 (enc unfrozen)"
        print(f"  LR this epoch: group0={_lr0:.2e}, group1={_lr1:.2e} | phase={_phase_str}")

        train_steps = len(train_loader)
        if args.max_train_steps_per_epoch is not None:
            train_steps = min(train_steps, args.max_train_steps_per_epoch)
        
        train_bar = tqdm(
            train_loader,
            desc=f"Train {current_epoch}/{args.epochs}",
            ncols=120,
            disable=(optimizer is None),
            total=train_steps
        )

        for batch_idx, (images, masks) in enumerate(train_bar):
            if args.device.startswith("cuda"):
                images = images.to(args.device, non_blocking=True, memory_format=torch.channels_last)
                masks = masks.to(args.device, non_blocking=True)
            else:
                images = images.to(args.device, non_blocking=True)
                masks = masks.to(args.device, non_blocking=True)

            if masks.ndim == 4 and masks.shape[1] != num_classes and masks.shape[-1] == num_classes:
                masks = masks.permute(0, 3, 1, 2).contiguous()

            current_max_grad_norm = args.max_grad_norm
            if _is_dual_stream:
                current_max_grad_norm = min(current_max_grad_norm, 0.5)

            if optimizer is None:
                gnorm = torch.zeros((), device=train_loss.device)
                with torch.inference_mode():
                    if use_amp:
                        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                            outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                        loss = criterion(outputs, masks)
            else:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        (p for pg in optimizer.param_groups for p in pg["params"]),
                        max_norm=current_max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                    loss = criterion(outputs, masks)
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        (p for pg in optimizer.param_groups for p in pg["params"]),
                        max_norm=current_max_grad_norm)
                    optimizer.step()

            train_loss += loss.detach().to(train_loss.dtype)
            num_batches += 1
            _dbg_gnorm_sum += gnorm.detach().to(_dbg_gnorm_sum.dtype)
            if args.log_every_n_steps > 0 and (((batch_idx + 1) % args.log_every_n_steps == 0) or (batch_idx + 1 == len(train_loader))):
                train_bar.set_postfix(loss=f"{float(loss.detach().cpu()):.3f}", gnorm=f"{float(gnorm.detach().cpu()):.2f}")
            if args.max_train_steps_per_epoch is not None and (batch_idx + 1) >= args.max_train_steps_per_epoch:
                break

        train_loss = float((train_loss / max(num_batches, 1)).detach().cpu())
        avg_gnorm = float((_dbg_gnorm_sum / max(num_batches, 1)).detach().cpu())
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = torch.zeros((), device=args.device if args.device.startswith("cuda") else "cpu", dtype=torch.float64)
        num_val_batches = 0
        metric = IoUMetric(num_classes=num_classes, device=args.device)
        _dbg_logit_max_t = torch.zeros((), device=val_loss.device, dtype=torch.float32)
        _dbg_pos_rate_sum_t = torch.zeros((), device=val_loss.device, dtype=torch.float64)

        scales_tensor: Optional[torch.Tensor] = None
        if logit_scales is not None:
            try:
                scales_tensor = torch.tensor(logit_scales, device=args.device).view(1, -1, 1, 1)
            except Exception:
                scales_tensor = None

        with torch.inference_mode():
            val_steps = len(val_loader)
            if args.max_val_steps_per_epoch is not None:
                val_steps = min(val_steps, args.max_val_steps_per_epoch)
            
            val_bar = tqdm(
                val_loader,
                desc=f"Val   {current_epoch}/{args.epochs}",
                ncols=120,
                total=val_steps,
            )
            for batch_idx, (images, masks) in enumerate(val_bar):
                if args.device.startswith("cuda"):
                    images = images.to(args.device, non_blocking=True, memory_format=torch.channels_last)
                    masks = masks.to(args.device, non_blocking=True)
                else:
                    images = images.to(args.device, non_blocking=True)
                    masks = masks.to(args.device, non_blocking=True)

                if masks.ndim == 4 and masks.shape[1] != num_classes and masks.shape[-1] == num_classes:
                    masks = masks.permute(0, 3, 1, 2).contiguous()

                if use_amp:
                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(_select_model_input(images, fusion_family, getattr(args, "use_ndvi", False)))
                    loss = criterion(outputs, masks)

                val_loss += loss.detach().to(val_loss.dtype)
                num_val_batches += 1

                if scales_tensor is not None:
                    scaled_outputs = outputs * scales_tensor
                else:
                    scaled_outputs = outputs

                metric.update(scaled_outputs, masks)
                _dbg_logit_max_t = torch.maximum(_dbg_logit_max_t, outputs.detach().abs().amax())
                _dbg_pos_rate_sum_t += torch.sigmoid(outputs.detach().float()).mean().to(_dbg_pos_rate_sum_t.dtype)

                if args.max_val_steps_per_epoch is not None and (batch_idx + 1) >= args.max_val_steps_per_epoch:
                    break

        val_loss = float((val_loss / max(num_val_batches, 1)).detach().cpu())
        _dbg_logit_max = float(_dbg_logit_max_t.detach().cpu())
        _dbg_pos_rate = float((_dbg_pos_rate_sum_t / max(num_val_batches, 1)).detach().cpu())
        results = metric.compute()
        miou = results["miou"]

        if optimizer is not None:
            lr_group_0 = optimizer.param_groups[0]["lr"]
            lr_group_1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_group_0
        else:
            lr_group_0 = 0.0
            lr_group_1 = 0.0
        collapse_flag = best_miou > 0.05 and miou < best_miou * 0.7
        collapse_str  = f"  ⚠ COLLAPSE: {best_miou:.4f}→{miou:.4f}" if collapse_flag else ""
        print(f"Epoch [{current_epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  mIoU: {miou:.4f}{collapse_str}")
        print(f"  DEBUG | avg_gnorm={avg_gnorm:.3f}  "
              f"logit_abs_max={_dbg_logit_max:.1f}  pos_rate={_dbg_pos_rate:.4f}  "
              f"LR_next=[{lr_group_0:.2e}, {lr_group_1:.2e}]")

        metrics_path = os.path.join(exp_dir, "metrics.json")
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr_group_0": lr_group_0,
            "lr_group_1": lr_group_1,
            **results,
        }
        all_metrics.append(epoch_metrics)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        recover_flag = (
            _is_dual_stream
            and _phase2_started
            and best_miou > 0.05
            and (
                (miou < best_miou * 0.90 and _dbg_logit_max > 60.0)
                or (miou < best_miou * 0.85)
                or (avg_gnorm > 0.35 and miou < best_miou * 0.95)
            )
        )
        if recover_flag and os.path.exists(best_ckpt_path):
            print("[WARN] Real instability detected -> restoring best checkpoint and shrinking LR")
            best_ckpt = torch.load(best_ckpt_path, map_location=args.device)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
            if optimizer is not None and best_ckpt.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(best_ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            if scheduler is not None and best_ckpt.get("scheduler_state_dict") is not None:
                try:
                    scheduler.load_state_dict(best_ckpt["scheduler_state_dict"])
                except Exception:
                    pass
            for pg in optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, 1e-6)
            continue

        improved = miou > best_miou
        if improved:
            best_miou = miou
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_miou": best_miou,
                "best_epoch": best_epoch,
            }, best_ckpt_path)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(exp_dir, f"model_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_miou": best_miou,
                "best_epoch": best_epoch,
            }, ckpt_path)

        if args.save_last_every > 0 and ((epoch + 1) % args.save_last_every == 0 or (epoch + 1) == args.epochs):
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_miou": best_miou,
                "best_epoch": best_epoch,
            }, last_ckpt_path)

    final_summary = {
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "last_miou": all_metrics[-1]["miou"] if all_metrics else None,
        "epoch_of_best": best_epoch,
        "collapse_flag": bool(all_metrics and all_metrics[-1]["miou"] < 0.5 * best_miou),
        "epochs": args.epochs,
        "exp_dir": exp_dir,
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"Training finished. Best mIoU: {best_miou:.4f} at epoch {best_epoch}. Checkpoints saved in {exp_dir}")


if __name__ == "__main__":
    main()