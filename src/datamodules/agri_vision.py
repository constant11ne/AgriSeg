from __future__ import annotations

import os
from typing import Callable, Optional, Tuple, List
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance


try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

import torch
from torch.utils.data import Dataset, DataLoader
import random


ANOMALY_CLASSES = [
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


def _compute_ndvi_channel(img_arr: np.ndarray) -> np.ndarray:
    if img_arr.shape[-1] < 4:
        raise ValueError("NDVI requires an image with at least 4 channels: RGB+NIR")
    red = img_arr[..., 0]
    nir = img_arr[..., 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndvi01 = (ndvi + 1.0) * 0.5
    return ndvi01[..., None].astype(np.float32)


def _load_image(path: str, use_nir: bool = False, use_ndvi: bool = False) -> np.ndarray:

    with Image.open(path) as img:
        img_arr = np.array(img)

        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        
        if img_arr.dtype == np.uint8:
            img_arr = img_arr.astype(np.float32) / 255.0
        elif img_arr.dtype == np.uint16:
            img_arr = img_arr.astype(np.float32) / 65535.0
        else:
            img_arr = img_arr.astype(np.float32)
            vmax = float(img_arr.max()) if img_arr.size > 0 else 1.0
            if vmax > 1.0:
                img_arr /= vmax
        
        if use_ndvi:
            if img_arr.shape[-1] < 4:
                raise ValueError(f"Cannot compute NDVI for {path}: expected RGB+NIR image")
            ndvi = _compute_ndvi_channel(img_arr)
            if use_nir:
                img_arr = np.concatenate([img_arr[..., :4], ndvi], axis=-1)
            else:
                img_arr = np.concatenate([img_arr[..., :3], ndvi], axis=-1)
        elif not use_nir and img_arr.shape[-1] > 3:
            img_arr = img_arr[..., :3]
        elif use_nir and img_arr.shape[-1] > 4:
            img_arr = img_arr[..., :4]
        return img_arr



def _load_mask(mask_dir: str, classes: List[str] = ANOMALY_CLASSES) -> np.ndarray:
    
    masks = []

    for cls in classes:
        mask_path = os.path.join(mask_dir, f"{cls}.png")
        with Image.open(mask_path) as m:
            mask_arr = np.array(m)

            mask_arr = (mask_arr > 0).astype(np.uint8)
            masks.append(mask_arr)
    masks = np.stack(masks, axis=0)  # shape (C, H, W)
    return masks


def _find_image_path(img_dir: str, sample_id: str) -> str:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        path = os.path.join(img_dir, sample_id + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image for sample '{sample_id}' not found in {img_dir}")


def _cache_paths(cache_root: str, split: str, sample_id: str) -> Tuple[str, str]:
    split_root = Path(cache_root) / split
    return str(split_root / 'images' / f'{sample_id}.npz'), str(split_root / 'masks' / f'{sample_id}.npz')


class AgricultureVisionDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (512, 512),
        use_nir: bool = False,
        use_ndvi: bool = False,
        cache_root: Optional[str] = None,
        cache_mmap: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.transform = transform
        self.img_size = img_size
        self.use_nir = use_nir
        self.use_ndvi = use_ndvi
        self.cache_root = cache_root
        self.cache_mmap = cache_mmap

        self.ids = []
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.startswith("."):
                continue
            base, _ = os.path.splitext(filename)
            self.ids.append(base)
        if not self.ids:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[idx]

        img = None
        mask = None
        if self.cache_root:
            cache_img_path, cache_mask_path = _cache_paths(self.cache_root, self.split, sample_id)
            if os.path.exists(cache_img_path) and os.path.exists(cache_mask_path):
                img_obj = np.load(cache_img_path, mmap_mode='r' if self.cache_mmap else None)
                mask_obj = np.load(cache_mask_path, mmap_mode='r' if self.cache_mmap else None)
                img = img_obj['image']
                mask = mask_obj['mask']
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                elif img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                else:
                    img = img.astype(np.float32)
                    vmax = float(img.max()) if img.size > 0 else 1.0
                    if vmax > 1.0:
                        img /= vmax
                if self.use_ndvi:
                    if img.shape[-1] < 4:
                        raise ValueError(f"Cannot compute NDVI for cached sample {sample_id}: expected RGB+NIR image")
                    ndvi = _compute_ndvi_channel(img)
                    if self.use_nir:
                        img = np.concatenate([img[..., :4], ndvi], axis=-1)
                    else:
                        img = np.concatenate([img[..., :3], ndvi], axis=-1)
                elif not self.use_nir and img.shape[-1] > 3:
                    img = img[..., :3]
                elif self.use_nir and img.shape[-1] > 4:
                    img = img[..., :4]
                mask = (mask > 0).astype(np.uint8)

        if img is None or mask is None:
            img_path = _find_image_path(self.img_dir, sample_id)
            img = _load_image(img_path, use_nir=self.use_nir, use_ndvi=self.use_ndvi)
            mask_path = os.path.join(self.mask_dir, sample_id)
            mask = _load_mask(mask_path)
    
        if mask.ndim == 3:
            mask = np.transpose(mask, (1, 2, 0))
    
        if self.transform is not None:
            augmented = self.transform(img, mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
            return img_tensor, mask_tensor

        else:
            img_chw = np.transpose(img, (2, 0, 1))
            img_tensor = torch.from_numpy(img_chw.astype(np.float32))
            
            n_channels = 3 + (1 if self.use_nir else 0) + (1 if self.use_ndvi else 0)
            mean = torch.tensor([0.5] * n_channels).view(n_channels, 1, 1)
            std = torch.tensor([0.5] * n_channels).view(n_channels, 1, 1)

            if img_tensor.shape[0] > n_channels:
                img_tensor = img_tensor[:n_channels]
            img_tensor = (img_tensor - mean) / std

            mask_arr = mask
            if mask_arr.ndim == 3 and mask_arr.shape[-1] == len(ANOMALY_CLASSES):
                mask_arr = np.transpose(mask_arr, (2, 0, 1))
            elif mask_arr.ndim == 2:
                mask_arr = mask_arr[None, ...]
            mask_tensor = torch.from_numpy(mask_arr.astype(np.float32))
            mask_tensor = (mask_tensor > 0.5).float()
            return img_tensor, mask_tensor



class BasicTransforms:

    def __init__(self, img_size: Tuple[int, int] = (512, 512), augment: bool = True, use_nir: bool = False, use_ndvi: bool = False, augment_mode: str = "full") -> None:
        self.img_size = img_size
        self.augment = augment
        self.use_nir = use_nir
        self.use_ndvi = use_ndvi
        self.augment_mode = augment_mode
        n_channels = 3 + (1 if use_nir else 0) + (1 if use_ndvi else 0)
        self.mean = np.array([0.5] * n_channels, dtype=np.float32)
        self.std = np.array([0.5] * n_channels, dtype=np.float32)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:

        H, W = image.shape[:2]

        if self.augment:
            if self.augment_mode == "minimal":
                if random.random() < 0.5:
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)
                if random.random() < 0.5:
                    image = np.flip(image, axis=0)
                    mask = np.flip(mask, axis=0)
            elif self.augment_mode == "full":
                scale = random.uniform(0.8, 1.0)
                new_h = max(int(H * scale), 1)
                new_w = max(int(W * scale), 1)
                top = random.randint(0, H - new_h) if H - new_h > 0 else 0
                left = random.randint(0, W - new_w) if W - new_w > 0 else 0
                image = image[top : top + new_h, left : left + new_w, :]
                mask = mask[top : top + new_h, left : left + new_w, ...]
                H, W = new_h, new_w
                if random.random() < 0.5:
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)
                if random.random() < 0.5:
                    image = np.flip(image, axis=0)
                    mask = np.flip(mask, axis=0)
                if random.random() < 0.5:
                    k = random.randint(1, 3)
                    image = np.rot90(image, k, axes=(0, 1))
                    mask = np.rot90(mask, k, axes=(0, 1))
                if random.random() < 0.3:
                    rgb = image[..., :3]
                    extra = image[..., 3:] if image.shape[-1] > 3 else None
                    factor_b = 1.0 + 0.2 * (random.random() - 0.5) * 2
                    factor_c = 1.0 + 0.2 * (random.random() - 0.5) * 2
                    rgb_aug = ((rgb - 0.5) * factor_c + 0.5) * factor_b
                    rgb_aug = np.clip(rgb_aug, 0.0, 1.0)
                    image = np.concatenate([rgb_aug, extra], axis=-1) if extra is not None else rgb_aug
            elif self.augment_mode == "none":
                pass

        target_h, target_w = self.img_size

        if mask.ndim == 3:
            mask_hwc = mask
        else:
            mask_hwc = np.transpose(mask, (1, 2, 0))
        mask_channels = mask_hwc.shape[-1]
        if _HAS_CV2:
            img_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            mask_resized_hwc = cv2.resize(mask_hwc, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            if mask_resized_hwc.ndim == 2:
                mask_resized_hwc = mask_resized_hwc[..., None]
            mask_resized = np.transpose(mask_resized_hwc, (2, 0, 1))
        else:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_resized = np.array(img_pil.resize((target_w, target_h), Image.BILINEAR)).astype(np.float32) / 255.0
            masks_resized = []
            for c in range(mask_channels):
                m = mask_hwc[..., c]
                m_pil = Image.fromarray((m * 255).astype(np.uint8))
                m_resized = np.array(m_pil.resize((target_w, target_h), Image.NEAREST)).astype(np.float32) / 255.0
                masks_resized.append(m_resized)
            mask_resized = np.stack(masks_resized, axis=0)
        if img_resized.ndim == 2:
            img_resized = np.stack([img_resized] * (3 + (1 if self.use_nir else 0) + (1 if self.use_ndvi else 0)), axis=-1)
        n_channels = 3 + (1 if self.use_nir else 0) + (1 if self.use_ndvi else 0)
        if img_resized.shape[-1] > n_channels:
            img_resized = img_resized[..., :n_channels]
        img_norm = (img_resized - self.mean) / self.std
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img_norm, (2, 0, 1)))).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_resized)).float()
        mask_tensor = (mask_tensor > 0.5).float()
        return {"image": img_tensor, "mask": mask_tensor}


def get_transforms(
    img_size: Tuple[int, int] = (512, 512),
    augment: bool = True,
    use_nir: bool = False,
    use_ndvi: bool = False,
    augment_mode: str = "full",
) -> BasicTransforms:
    return BasicTransforms(img_size=img_size, augment=augment, use_nir=use_nir, use_ndvi=use_ndvi, augment_mode=augment_mode)


def build_dataloaders(
    data_root: str,
    img_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = True,
    augment_mode: str = "full",
    use_nir: bool = False,
    use_ndvi: bool = False,
    oversample_rare: bool = False,
    class_weights: Optional[List[float]] = None,
    *,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    cache_root: Optional[str] = None,
    cache_mmap: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    
    train_transforms = get_transforms(img_size, augment=augment, use_nir=use_nir, use_ndvi=use_ndvi, augment_mode=augment_mode)
    val_transforms = get_transforms(img_size, augment=False, use_nir=use_nir, use_ndvi=use_ndvi, augment_mode="none")

    train_dataset = AgricultureVisionDataset(
        root=data_root, split="train", transform=train_transforms, img_size=img_size, use_nir=use_nir, use_ndvi=use_ndvi,
        cache_root=cache_root, cache_mmap=cache_mmap,
    )
    val_dataset = AgricultureVisionDataset(
        root=data_root, split="val", transform=val_transforms, img_size=img_size, use_nir=use_nir, use_ndvi=use_ndvi,
        cache_root=cache_root, cache_mmap=cache_mmap,
    )

    if oversample_rare:
        num_samples = len(train_dataset)
        num_classes = len(ANOMALY_CLASSES)

        class_counts = np.zeros(num_classes, dtype=np.int64)

        presence = np.zeros((num_samples, num_classes), dtype=np.bool_)
        for idx, sample_id in enumerate(train_dataset.ids):

            mask_path = os.path.join(train_dataset.mask_dir, sample_id)
            for cls_idx, cls_name in enumerate(ANOMALY_CLASSES):
                mask_file = os.path.join(mask_path, f"{cls_name}.png")
                with Image.open(mask_file) as m:
                    mask_arr = np.array(m)

                    present = (mask_arr > 0).any()
                    presence[idx, cls_idx] = present
                    if present:
                        class_counts[cls_idx] += 1

        epsilon = 1e-6
        class_freq = class_counts.astype(np.float64) / (num_samples + epsilon)

        if class_weights is not None:

            cw = np.array(class_weights, dtype=np.float64)
            if cw.sum() > 0:
                cw = cw / cw.sum() * num_classes
        else:
            cw = 1.0 / (class_freq + epsilon)

            cw = cw / cw.mean()

        sample_weights = (presence * cw[np.newaxis, :]).sum(axis=1)

        sample_weights = torch.from_numpy(sample_weights).float()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dl_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dl_kwargs,
    )
    return train_loader, val_loader
