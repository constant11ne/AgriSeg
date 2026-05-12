from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodules.agri_vision import ANOMALY_CLASSES, _load_image
from src.train import build_model


def load_image_for_inference(
        path: str,
        img_size: Tuple[int, int],
        use_nir: bool,
) -> torch.Tensor:
    img_arr = _load_image(path, use_nir=use_nir)

    from PIL import Image as PILImage
    arr_u8 = (img_arr * 255).clip(0, 255).astype(np.uint8)
    h, w = img_size
    if arr_u8.ndim == 2:
        pil = PILImage.fromarray(arr_u8).resize((w, h), PILImage.BILINEAR)
        resized = np.array(pil, dtype=np.float32) / 255.0
    elif arr_u8.shape[2] in (1, 2, 3, 4):
        channels = []
        for c in range(arr_u8.shape[2]):
            pil = PILImage.fromarray(arr_u8[:, :, c]).resize((w, h), PILImage.BILINEAR)
            channels.append(np.array(pil, dtype=np.float32) / 255.0)
        resized = np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unexpected channel count: {arr_u8.shape}")

    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return tensor


def load_model_from_config(
        config: Dict,
        checkpoint_path: str,
        device: str,
) -> nn.Module:
    num_classes = len(ANOMALY_CLASSES)
    in_channels = 4 if config.get("use_nir", False) else 3

    model = build_model(
        model_name=config.get("model", "fpn"),
        encoder=config.get("encoder", "timm-efficientnet-b4"),
        encoder_weights=None,
        num_classes=num_classes,
        in_channels=in_channels,
        nir_fusion=config.get("nir_fusion", "early"),
        norm_type=config.get("norm", "bn"),
        use_nir=config.get("use_nir", False),
        fusion_family=config.get("fusion_family", None),
        fusion_method=config.get("fusion_method", None),
        freeze_rgb=config.get("freeze_rgb_encoder", "none"),
        freeze_rgb_stages=config.get("freeze_rgb_stages", 3),
        partial_unfreeze_last_n=config.get("partial_unfreeze_last_n", 2),
        nir_branch_width=config.get("nir_branch_width", 0.5),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 128),
        output_size=tuple(config.get("img_size", [512, 512])),
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


def run_inference(
        model: nn.Module,
        images_dir: str,
        out_dir: str,
        img_size: Tuple[int, int],
        use_nir: bool,
        threshold: float,
        device: str,
        save_probs: bool = False,
        save_json_summary: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    img_paths = sorted([
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in img_extensions
    ])
    if not img_paths:
        print(f"[WARN] No images found in '{images_dir}'")
        return

    print(f"[INFO] Running inference on {len(img_paths)} images -> {out_dir}")

    all_summaries = []
    with torch.inference_mode():
        for idx, img_path in enumerate(img_paths):
            stem = img_path.stem

            try:
                tensor = load_image_for_inference(str(img_path), img_size, use_nir)
            except Exception as e:
                print(f"[WARN] Could not load {img_path.name}: {e}")
                continue

            tensor = tensor.to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).squeeze(0).cpu().numpy().astype(np.uint8) * 255

            img_out_dir = os.path.join(out_dir, stem)
            os.makedirs(img_out_dir, exist_ok=True)
            for c, cls_name in enumerate(ANOMALY_CLASSES):
                mask = preds[c]
                mask_pil = Image.fromarray(mask, mode="L")
                mask_pil.save(os.path.join(img_out_dir, f"{cls_name}.png"))

            if save_probs:
                probs_np = probs.squeeze(0).cpu().numpy()
                for c, cls_name in enumerate(ANOMALY_CLASSES):
                    prob_u8 = (probs_np[c] * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(prob_u8, mode="L").save(
                        os.path.join(img_out_dir, f"{cls_name}_prob.png")
                    )

            if save_json_summary:
                mean_probs = probs.squeeze(0).mean(dim=(1, 2)).cpu().tolist()
                summary = {
                    "image": img_path.name,
                    "predictions": {
                        cls: {
                            "mean_prob": round(mean_probs[c], 5),
                            "any_positive": bool((preds[c] > 0).any()),
                        }
                        for c, cls in enumerate(ANOMALY_CLASSES)
                    },
                }
                all_summaries.append(summary)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
                print(f"  {idx + 1}/{len(img_paths)} done")

    if save_json_summary and all_summaries:
        summary_path = os.path.join(out_dir, "predictions_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"[INFO] Summary written to {summary_path}")

    print(f"[DONE] Predictions saved to {out_dir}")


def evaluate_on_directory(
        model: nn.Module,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int],
        use_nir: bool,
        threshold: float,
        device: str,
        num_classes: int,
) -> Dict:
    from src.utils.metrics import SegmentationMetrics
    from src.datamodules.agri_vision import _load_mask

    metric = SegmentationMetrics(num_classes=num_classes)
    img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    img_paths = sorted([
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in img_extensions
    ])

    with torch.inference_mode():
        for img_path in img_paths:
            stem = img_path.stem
            mask_sub = os.path.join(masks_dir, stem)
            if not os.path.isdir(mask_sub):
                continue

            try:
                tensor = load_image_for_inference(str(img_path), img_size, use_nir)
                mask_np = _load_mask(mask_sub)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            except Exception as e:
                print(f"[WARN] Skipping {stem}: {e}")
                continue

            tensor = tensor.to(device)
            logits = model(tensor)
            if logits.shape[2:] != mask_t.shape[2:]:
                mask_t = F.interpolate(
                    mask_t.float(), size=logits.shape[2:], mode="nearest"
                ).to(mask_t.dtype)
            metric.update(logits.cpu(), mask_t)

    return metric.compute()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a fusion model")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--config", required=True, help="Path to config.json from the same run")
    p.add_argument("--images-dir", required=True, help="Directory of input images")
    p.add_argument("--out-dir", required=True, help="Output directory for predictions")
    p.add_argument("--masks-dir", type=str, default=None,
                   help="Optional: ground-truth masks dir for evaluation")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary threshold for prediction masks (default 0.5)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-probs", action="store_true",
                   help="Also save raw probability maps as uint8 PNGs")
    p.add_argument("--no-json", action="store_true",
                   help="Skip writing per-image JSON summaries")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    img_size = tuple(config.get("img_size", [512, 512]))
    use_nir = config.get("use_nir", False)
    num_classes = len(ANOMALY_CLASSES)

    print(f"[INFO] Loading model from {args.checkpoint}")
    print(f"[INFO] Fusion family : {config.get('fusion_family', 'N/A')}")
    print(f"[INFO] Fusion method : {config.get('fusion_method', 'N/A')}")
    print(f"[INFO] Use NIR       : {use_nir}")
    print(f"[INFO] Image size    : {img_size}")

    model = load_model_from_config(config, args.checkpoint, args.device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Params: total={total:,}  trainable={trainable:,}")

    run_inference(
        model=model,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        img_size=img_size,
        use_nir=use_nir,
        threshold=args.threshold,
        device=args.device,
        save_probs=args.save_probs,
        save_json_summary=not args.no_json,
    )

    if args.masks_dir:
        print(f"\n[INFO] Evaluating against ground truth in {args.masks_dir}")
        results = evaluate_on_directory(
            model=model,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            img_size=img_size,
            use_nir=use_nir,
            threshold=args.threshold,
            device=args.device,
            num_classes=num_classes,
        )
        print("\nEvaluation results:")
        print(f"mIoU       : {results['miou']:.5f}")
        print(f"macro_F1   : {results['macro_f1']:.5f}")
        print(f"rare_mIoU  : {results['rare_miou']:.5f}")
        for i, cls in enumerate(ANOMALY_CLASSES):
            print(f"  {cls:<20}: {results.get(f'iou_class_{i}', 0):.5f}")

        eval_path = os.path.join(args.out_dir, "eval_results.json")
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
