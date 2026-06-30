from pathlib import Path
import argparse
import sys
from typing import Dict, Optional, Tuple
from PIL import Image

IMG_EXTS  = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MASK_EXTS = {".png", ".tif", ".tiff"}

CLASSES = [
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
VARIANTS = {
    "doubleplant": "double_plant", "double_plant": "double_plant",
    "double-plant": "double_plant", "double plant": "double_plant",
    "drydown": "drydown", "dry_down": "drydown", "dry-down": "drydown", "dry down": "drydown",
    "endrow": "endrow", "end_row": "endrow", "end-row": "endrow", "end row": "endrow",
    "nutrientdeficiency": "nutrient_deficiency", "nutrient_deficiency": "nutrient_deficiency",
    "nutrient-deficiency": "nutrient_deficiency", "nutrient deficiency": "nutrient_deficiency",
    "planterskip": "planter_skip", "planter_skip": "planter_skip",
    "planter-skip": "planter_skip", "planter skip": "planter_skip",
    "stormdamage": "storm_damage", "storm_damage": "storm_damage",
    "storm-damage": "storm_damage", "storm damage": "storm_damage",
    "water": "water",
    "waterway": "waterway", "water_way": "waterway",
    "water-way": "waterway",
    "weedcluster": "weed_cluster", "weed_cluster": "weed_cluster",
    "weed-cluster": "weed_cluster", "weed cluster": "weed_cluster",
}
def norm_name(s:str)->str: return s.lower().replace("-","_").replace(" ","")

def ensure_dir(p:Path)->Path:
    p.mkdir(parents=True, exist_ok=True); return p

def list_files_by_stem(d:Path, exts)->Dict[str,Path]:
    out={}
    if not d.exists(): return out
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in exts: out[f.stem]=f
    return out

def image_size(p:Path)->Tuple[int,int]:
    with Image.open(p) as im: return im.size

def make_zero_mask(dst:Path, size:Tuple[int,int]):
    ensure_dir(dst.parent); Image.new("L", size, 0).save(dst)

def save_1ch(dst:Path, src:Path, size:Tuple[int,int]):
    ensure_dir(dst.parent)
    with Image.open(src) as im:
        im=im.convert("L")
        if im.size!=size: im=im.resize(size, resample=Image.NEAREST)
        im.save(dst)

def build_4ch_png(dst_png:Path, rgb_path:Path, nir_path:Optional[Path]):
    with Image.open(rgb_path) as rgb:
        rgb = rgb.convert("RGB"); W,H = rgb.size
        if nir_path is None:
            nir = Image.new("L",(W,H),0)
        else:
            with Image.open(nir_path) as im:
                nir = im.convert("L")
                if im.size!=(W,H): nir = nir.resize((W,H), resample=Image.BILINEAR)
        rgba = Image.merge("RGBA", (*rgb.split(), nir))
        ensure_dir(dst_png.parent); rgba.save(dst_png)

def normalize_split(src_split:Path, dst_split:Path, verbose:bool=False, fake_masks:bool=False):
    print(f"[INFO] Processing split: {src_split.name}")
    rgb_dir = src_split/"images"/"rgb"
    nir_dir = src_split/"images"/"nir"
    if not rgb_dir.exists():
        raise RuntimeError(f"[ERR] missing images/rgb: {rgb_dir}")
    rgb = list_files_by_stem(rgb_dir, IMG_EXTS)
    nir = list_files_by_stem(nir_dir, IMG_EXTS)
    if verbose: print(f"[DBG] RGB files: {len(rgb)} | NIR files: {len(nir)}")

    out_images = ensure_dir(dst_split/"images")
    out_masks_root = ensure_dir(dst_split/"masks")

    labels_dir = src_split/"labels"
    have_labels = labels_dir.exists()
    if verbose:
        print(f"[DBG] labels dir: {'present' if have_labels else 'ABSENT'}")

    cls_maps = {}
    if have_labels:
        for d in labels_dir.iterdir():
            if not d.is_dir(): continue
            cname = VARIANTS.get(norm_name(d.name))
            if cname not in CLASSES: continue
            cls_maps[cname] = list_files_by_stem(d, MASK_EXTS)
        if verbose:
            for c in CLASSES: print(f"[DBG] class {c}: {len(cls_maps.get(c,{}))} files")

    tiles = sorted(rgb.keys())
    print(f"[INFO] Found {len(tiles)} tiles with RGB")

    for tid in tiles:
        rgb_path = rgb[tid]; nir_path = nir.get(tid)
        build_4ch_png(out_images/f"{tid}.png", rgb_path, nir_path)

        W,H = image_size(rgb_path)
        tmask_dir = ensure_dir(out_masks_root/tid)
        for cls in CLASSES:
            dst_mask = tmask_dir/f"{cls}.png"
            if have_labels:
                src_mask = cls_maps.get(cls,{}).get(tid)
                if src_mask is None:
                    if fake_masks: make_zero_mask(dst_mask,(W,H))
                    else:
                        make_zero_mask(dst_mask,(W,H))
                    if verbose: print(f"[MISS] {tid}/{cls} -> ZERO")
                else:
                    save_1ch(dst_mask, src_mask, (W,H))
                    if verbose: print(f"[HIT ] {tid}/{cls}")
            else:
                if fake_masks:
                    make_zero_mask(dst_mask,(W,H))
                if verbose: print(f"[UNLABELED] {tid}/{cls}: {'ZERO' if fake_masks else 'SKIP'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fake-masks", action="store_true", help="Create zero masks when labels are missing (e.g., test split)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    ensure_dir(dst)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src_split = src/split
        if not src_split.exists():
            print(f"[WARN] split missing: {src_split}, skipping"); continue
        normalize_split(src_split, ensure_dir(dst/split), verbose=args.verbose, fake_masks=args.fake_masks)

    print(f"[OK] Normalized dataset at: {dst}")

if __name__=="__main__":
    main()
