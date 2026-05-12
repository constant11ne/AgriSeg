import os, sys, glob
import numpy as np
import imageio.v3 as iio

root = sys.argv[1] if len(sys.argv) > 1 else "standard/train"
img_dir = os.path.join(root, "images")
msk_dir = os.path.join(root, "masks")

imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
msks = sorted(glob.glob(os.path.join(msk_dir, "*")))
print(f"images: {len(imgs)}, masks: {len(msks)}")

sample_img = imgs[0]
sample_msk = msks[0]

im = iio.imread(sample_img)
mk = iio.imread(sample_msk)

print("img:", sample_img, "shape:", im.shape, "dtype:", im.dtype)
print("msk:", sample_msk, "shape:", np.array(mk).shape, "dtype:", mk.dtype)

if im.ndim == 3:
    print("channels:", im.shape[2], "(expect 4 if using --use-nir)")

if mk.ndim == 2:
    vals = np.unique(mk)
    print("mask unique vals:", vals[:20], "count:", len(vals))
elif mk.ndim == 3:
    print("mask channels:", mk.shape[2], "(expect 6 for 6 anomalies)")
