from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.datamodules.agri_vision import ANOMALY_CLASSES, _find_image_path, _load_mask
from PIL import Image


def _load_raw_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr


def _process_one(data_root: str, cache_root: str, split: str, sample_id: str, overwrite: bool) -> None:
    img_out = Path(cache_root) / split / 'images' / f'{sample_id}.npz'
    mask_out = Path(cache_root) / split / 'masks' / f'{sample_id}.npz'
    if not overwrite and img_out.exists() and mask_out.exists():
        return
    img_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    img = _load_raw_image(_find_image_path(os.path.join(data_root, split, 'images'), sample_id))
    mask = _load_mask(os.path.join(data_root, split, 'masks', sample_id)).astype(np.uint8)

    np.savez_compressed(img_out, image=img)
    np.savez_compressed(mask_out, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    tasks = []
    for split in args.splits:
        img_dir = Path(args.data_root) / split / 'images'
        for filename in sorted(os.listdir(img_dir)):
            if filename.startswith('.'):
                continue
            sample_id = Path(filename).stem
            tasks.append((split, sample_id))

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = [
            ex.submit(_process_one, args.data_root, args.cache_root, split, sample_id, args.overwrite)
            for split, sample_id in tasks
        ]
        for fut in futures:
            fut.result()


if __name__ == '__main__':
    main()
