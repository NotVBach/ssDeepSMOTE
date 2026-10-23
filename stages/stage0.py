import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.config import build_parser, load_config, save_config, stage_dir
from utils.data import MANIFEST_COLUMNS


# Load a COCO annotation file; return images by id and annotations grouped by image id
def load_coco(path):
    with open(path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {image["id"]: image for image in coco["images"]}
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return images, by_image


# Round a COCO [x, y, w, h] box to integer pixel bounds inside a width x height image
def clamp_box(bbox, width, height):
    x, y, w, h = bbox
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(width, int(round(x + w)))
    bottom = min(height, int(round(y + h)))
    return left, top, right, bottom


# Crop every annotated cell of one split into its own PNG and return the manifest rows and the number of skipped boxes
def crop_split(cfg, split, crop_root):
    data = cfg["data"]
    root = cfg["paths"]["dataset_root"]
    classes = data["classes"]
    label_of = {int(cid): index for index, cid in enumerate(data["category_ids"])}
    limit = data.get("limit_per_class")
    overwrite = bool(cfg["stage0"].get("overwrite", False))
    for name in classes:
        os.makedirs(os.path.join(crop_root, split, name), exist_ok=True)
    images, by_image = load_coco(os.path.join(root, "annotations", f"{split}.json"))
    taken = [0] * len(classes)
    rows, skipped = [], 0
    for image_id in tqdm(sorted(by_image), desc=f"cropping {split}"):
        info = images[image_id]
        wanted = []
        for ann in by_image[image_id]:
            label = label_of.get(int(ann["category_id"]))
            if label is not None and (not limit or taken[label] < int(limit)):
                wanted.append((ann, label))
                taken[label] += 1
        if not wanted:
            continue
        stem = os.path.splitext(info["file_name"])[0]
        with Image.open(os.path.join(root, split, info["file_name"])) as photo:
            photo = photo.convert("RGB")
            for ann, label in wanted:
                left, top, right, bottom = clamp_box(ann["bbox"], photo.width, photo.height)
                if right - left < 1 or bottom - top < 1:
                    skipped += 1
                    taken[label] -= 1
                    continue
                path = os.path.join(crop_root, split, classes[label], f"{stem}_a{ann['id']}.png").replace(os.sep, "/")
                if overwrite or not os.path.exists(path):
                    photo.crop((left, top, right, bottom)).save(path)
                rows.append({
                    "path": path,
                    "label": label,
                    "class_name": classes[label],
                    "width": right - left,
                    "height": bottom - top,
                    "source_image": info["file_name"],
                    "ann_id": ann["id"],
                    "synthetic": 0,
                })
    return rows, skipped


def main():
    args = build_parser("Stage 0: crop annotated cells").parse_args()
    cfg = load_config(args)
    out_dir = stage_dir(cfg, "data")
    save_config(cfg, out_dir)
    classes = cfg["data"]["classes"]
    expected = cfg["stage0"].get("expected_counts") or {}
    crop_root = os.path.join(out_dir, "crops")
    for split in cfg["stage0"]["splits"]:
        rows, skipped = crop_split(cfg, split, crop_root)
        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        counts = df["label"].value_counts().reindex(range(len(classes)), fill_value=0).astype(int).tolist()
        print(f"{split}: {len(df)} crops, {skipped} empty boxes skipped")
        for name, count in zip(classes, counts):
            print(f"  {name:<5} {count}")
        target = expected.get(split)
        if not cfg["data"].get("limit_per_class") and target is not None and [int(v) for v in target] != counts:
            print(f"WARNING: {split} counts {counts} differ from the thesis table {list(target)}")


if __name__ == "__main__":
    main()
