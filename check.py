from src.models.backbone_utils import build_rgb_encoder

encoder, rgb_stage_channels = build_rgb_encoder(
    encoder_name="timm-efficientnet-b4",
    encoder_weights="imagenet",
    depth=5,
)

print(rgb_stage_channels)
print(rgb_stage_channels[1:])