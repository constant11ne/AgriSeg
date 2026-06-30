from .unet_custom import CustomUNet, UNetMidFusion, UNetAdapter
from .ibn import IBN

from .fusion_blocks import (
    ConcatFusion,
    SumFusion,
    WeightedSumFusion,
    SEFusion,
    CBAMFusion,
    GatedFusion,
    CrossAttentionFusion,
    ResidualAdapterFusion,
    build_fusion_block,
    FUSION_BLOCK_REGISTRY,
)

from .nir_branches import NIRStem, NIREncoderLite, NIRPyramidLite

from .backbone_utils import (
    build_rgb_encoder,
    adapt_first_conv,
    freeze_encoder_full,
    freeze_encoder_stages,
    unfreeze_last_n_stages,
    count_trainable_params,
)

try:
    from .dual_stream_fpn import DualStreamFPN
    HAS_DUAL_STREAM = True
except ImportError:
    HAS_DUAL_STREAM = False

from .late_fusion import (
    LateFusionWrapper,
    WeightedLogitsFusion,
    PerClassWeightedFusion,
    build_late_fusion,
)

from .lora import LoRAConv2d, inject_lora, freeze_model
from .bitfit import freeze_except_biases

__all__ = [
    "CustomUNet",
    "UNetMidFusion",
    "UNetAdapter",
    "IBN",

    "ConcatFusion",
    "SumFusion",
    "WeightedSumFusion",
    "SEFusion",
    "CBAMFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "ResidualAdapterFusion",
    "build_fusion_block",
    "FUSION_BLOCK_REGISTRY",

    "NIRStem",
    "NIREncoderLite",
    "NIRPyramidLite",

    "build_rgb_encoder",
    "adapt_first_conv",
    "freeze_encoder_full",
    "freeze_encoder_stages",
    "unfreeze_last_n_stages",
    "count_trainable_params",

    "DualStreamFPN",
    "HAS_DUAL_STREAM",

    "LateFusionWrapper",
    "WeightedLogitsFusion",
    "PerClassWeightedFusion",
    "build_late_fusion",

    "LoRAConv2d",
    "inject_lora",
    "freeze_model",
    "freeze_except_biases",
]
try:
    from .multi_stream_fpn import MultiStreamFPN
except Exception:
    MultiStreamFPN = None
