import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.restoration import inpaint
from tqdm import tqdm


def load_split_records(root: Path, split: str):
    ann_path = root / "Annotations" / f"apricot_annotations_{split}.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_map = {img["id"]: img for img in data["images"]}
    records = []
    for ann in data["annotations"]:
        img = image_map[ann["image_id"]]
        records.append(
            {
                "split": split,
                "file_name": img["file_name"],
                "orig_w": img["width"],
                "orig_h": img["height"],
                "bbox": ann["bbox"],
            }
        )
    return records


def clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def make_clean_with_inpaint(img_arr, bbox_xywh, pad):
    h, w, _ = img_arr.shape
    x, y, bw, bh = bbox_xywh
    x0 = int(np.floor(x - pad))
    y0 = int(np.floor(y - pad))
    x1 = int(np.ceil(x + bw + pad))
    y1 = int(np.ceil(y + bh + pad))
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    clean = inpaint.inpaint_biharmonic(img_arr, mask, channel_axis=2)
    return clean


def main():
    parser = argparse.ArgumentParser(description="Prepare APRICOT for Saliuitl object-detection runs.")
    parser.add_argument("--apricot_root", default="APRICOTv1.0", type=str)
    parser.add_argument("--out_root", default="data/apricot_saliuitl", type=str)
    parser.add_argument("--size", default=416, type=int)
    parser.add_argument("--pad", default=6, type=int, help="extra mask padding around patch bbox after resize")
    parser.add_argument("--limit", default=0, type=int, help="optional limit of images to process (0 = all)")
    args = parser.parse_args()

    src_root = Path(args.apricot_root)
    out_root = Path(args.out_root)
    out_clean = out_root / "clean"
    out_patch = out_root / "1p"
    out_clean.mkdir(parents=True, exist_ok=True)
    out_patch.mkdir(parents=True, exist_ok=True)

    records = []
    for split in ("dev", "test"):
        records.extend(load_split_records(src_root, split))
    if args.limit > 0:
        records = records[: args.limit]

    effective_files = []
    for rec in tqdm(records, desc="Preparing APRICOT"):
        src_img = src_root / "Images" / rec["split"] / rec["file_name"]
        if not src_img.exists():
            continue

        stem = f"{rec['split']}_{Path(rec['file_name']).stem}"
        out_name = f"{stem}.png"

        with Image.open(src_img).convert("RGB") as im:
            im = im.resize((args.size, args.size), Image.BILINEAR)
            arr = np.asarray(im).astype(np.float32) / 255.0

        # Scale bbox from original resolution to resized image.
        x, y, bw, bh = rec["bbox"]
        sx = args.size / float(rec["orig_w"])
        sy = args.size / float(rec["orig_h"])
        bbox_resized = [x * sx, y * sy, bw * sx, bh * sy]

        clean = make_clean_with_inpaint(arr, bbox_resized, pad=args.pad)
        clean_uint8 = np.clip(clean * 255.0, 0, 255).astype(np.uint8)
        patch_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(clean_uint8).save(out_clean / out_name)
        Image.fromarray(patch_uint8).save(out_patch / out_name)
        effective_files.append(out_name)

    np.save(out_patch / "effective_1p.npy", np.array(effective_files))
    print(f"Prepared {len(effective_files)} images at: {out_root}")
    print(f"effective_1p.npy: {out_patch / 'effective_1p.npy'}")


if __name__ == "__main__":
    main()
