"""Convert written YOLO labels into COCO JSON (one file per split).

Run after all YOLO labels are written. Reads the labels back from disk so the
COCO files are a faithful reflection of what training will actually consume.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

CATEGORIES = [{"id": 1, "name": "fish", "supercategory": "none"}]


def _yolo_line_to_coco_bbox(parts: list[str]) -> tuple[float, float, float, float]:
    """YOLO normalized (cls xc yc w h) -> COCO pixel bbox [x_left, y_top, w, h]."""
    _, xc, yc, w, h = parts[0], *map(float, parts[1:])
    x_left = (xc - w / 2.0) * config.IMG_WIDTH
    y_top = (yc - h / 2.0) * config.IMG_HEIGHT
    width_px = w * config.IMG_WIDTH
    height_px = h * config.IMG_HEIGHT
    return x_left, y_top, width_px, height_px


def yolo_labels_to_coco_json(split: str, output_path: Path) -> Path:
    """Build coco_{split}.json from output_path/{split}/{images,labels}.

    Returns the written JSON path. Images with no label lines (negatives) are
    still included (with no annotations) so the detector sees true negatives.
    """
    split_dir = output_path / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images: list[dict] = []
    annotations: list[dict] = []
    img_id = 0
    ann_id = 0

    for img_path in sorted(images_dir.glob("*.png")):
        img_id += 1
        rel_name = f"{split}/images/{img_path.name}"
        images.append({
            "id": img_id,
            "file_name": rel_name,
            "width": config.IMG_WIDTH,
            "height": config.IMG_HEIGHT,
        })
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            x_left, y_top, w_px, h_px = _yolo_line_to_coco_bbox(parts)
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_left, y_top, w_px, h_px],
                "area": w_px * h_px,
                "iscrowd": 0,
            })

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    out = output_path / f"coco_{split}.json"
    out.write_text(json.dumps(coco), encoding="utf-8")
    return out
