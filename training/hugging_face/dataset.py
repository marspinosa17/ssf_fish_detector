"""COCO-backed torch Dataset for the existing spectrogram tiles.

Wraps pycocotools.coco.COCO directly against data/spectrograms/coco_{split}.json
— no HF `datasets` library / hub machinery needed, since the data is already
local and fully materialized as PNG tiles.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

try:
    from . import config
    from .augmentations import AugmentationConfig, apply_augmentations
except ImportError:
    import config
    from augmentations import AugmentationConfig, apply_augmentations

logger = logging.getLogger("rtdetr_hf.dataset")


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


class SpectrogramCocoDataset(Dataset):
    """One item per image (spectrogram tile), COCO-format annotations.

    Returns dicts of the form HF's ``RTDetrImageProcessor`` expects for
    object detection: ``{"image_id": int, "image": PIL.Image, "annotations":
    [{"image_id", "category_id", "bbox": [x,y,w,h], "area", "iscrowd"}, ...]}``.
    Negative tiles (no boxes) yield an empty ``annotations`` list — RT-DETR's
    matcher/loss handles empty targets natively (all queries assigned "no
    object").
    """

    def __init__(
        self,
        coco_json_path: Path,
        images_root: Path,
        image_processor,
        augmentation_config: AugmentationConfig | None = None,
        seed: int = config.RANDOM_SEED,
        image_ids: list[int] | None = None,
    ) -> None:
        self.coco = COCO(str(coco_json_path))
        self.images_root = Path(images_root)
        self.image_processor = image_processor
        self.aug_cfg = augmentation_config or AugmentationConfig()
        self.image_ids = image_ids if image_ids is not None else sorted(self.coco.imgs.keys())
        # Deterministic per-item augmentation draw (independent of DataLoader
        # worker/shuffle order), matching preprocess.py's tile_rng() pattern.
        self._seed = seed

    def __len__(self) -> int:
        return len(self.image_ids)

    def _item_rng(self, index: int) -> random.Random:
        return random.Random(self._seed ^ (index * 2654435761 & 0xFFFFFFFF))

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        img_info = self.coco.imgs[image_id]
        img_path = self.images_root / img_info["file_name"]
        # coco_{split}.json's file_name is already "{split}/images/xxx.png"
        # relative to the spectrograms root — images_root is that root.
        image = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes_xywh = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
        # coco_{split}.json uses COCO's 1-indexed category_id (fish=1, see
        # preprocessing/coco_converter.py CATEGORIES); the model's classification
        # head is 0-indexed (config.LABEL2ID = {"fish": 0}). Remap here so
        # class_labels handed to the loss/matcher never exceeds num_labels-1.
        category_ids = np.array(
            [config.COCO_FISH_CATEGORY_ID_TO_MODEL_ID[a["category_id"]] for a in anns], dtype=np.int64,
        )

        if self.aug_cfg.hflip_prob > 0 or self.aug_cfg.scale_jitter_prob > 0:
            boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
            rng = self._item_rng(index)
            image, boxes_xyxy, keep = apply_augmentations(image, boxes_xyxy, self.aug_cfg, rng)
            category_ids = category_ids[keep]
            boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

        annotations = [
            {
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [float(v) for v in box],
                "area": float(box[2] * box[3]),
                "iscrowd": 0,
            }
            for box, cat_id in zip(boxes_xywh, category_ids)
        ]

        target = {"image_id": image_id, "annotations": annotations}
        encoded = self.image_processor(images=image, annotations=target, return_tensors="pt")
        # image_processor batches a singleton dim; squeeze it back out here so
        # collate_fn controls batching explicitly.
        pixel_values = encoded["pixel_values"][0]
        labels = encoded["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch: list[dict]) -> dict:
    """Stack pixel_values, leave labels as a list (variable boxes/image)."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def build_dataset(
    split: str,
    image_processor,
    augmentation_config: AugmentationConfig | None = None,
    smoke: bool = False,
    smoke_per_source: int = 4,
) -> SpectrogramCocoDataset:
    """Construct a SpectrogramCocoDataset for one split.

    smoke=True restricts to a tiny, per-source-balanced subset (using
    dataset_manifest.csv's `source` column to stratify) for fast pipeline
    sanity checks — mirrors preprocess.py's --smoke behavior.
    """
    coco_json = config.SPECTROGRAM_ROOT / f"coco_{split}.json"
    if not coco_json.exists():
        raise FileNotFoundError(
            f"{coco_json} not found — run preprocessing/preprocess.py first."
        )

    image_ids = None
    if smoke:
        image_ids = _smoke_image_ids(coco_json, split, smoke_per_source)

    ds = SpectrogramCocoDataset(
        coco_json_path=coco_json,
        images_root=config.SPECTROGRAM_ROOT,
        image_processor=image_processor,
        augmentation_config=augmentation_config,
        image_ids=image_ids,
    )
    logger.info("Loaded %s split: %d images (smoke=%s)", split, len(ds), smoke)
    return ds


def _smoke_image_ids(coco_json: Path, split: str, per_source: int) -> list[int]:
    """Pick `per_source` images per source (xavier/seth/tagus) for --smoke.

    Uses dataset_manifest.csv's `source`/`image_path`/`split` columns to map
    COCO image file_names back to source, then samples deterministically —
    same purpose as preprocess.py's --smoke, applied post-hoc since this
    module only reads already-written tiles.
    """
    import pandas as pd

    coco = COCO(str(coco_json))
    manifest = pd.read_csv(config.MANIFEST_PATH)
    manifest = manifest[manifest["split"] == split]
    path_to_source = dict(zip(manifest["image_path"], manifest["source"]))

    file_name_to_id = {img["file_name"]: img["id"] for img in coco.dataset["images"]}

    selected: list[int] = []
    for source in config.SOURCES:
        candidates = sorted(
            img_id
            for file_name, img_id in file_name_to_id.items()
            if path_to_source.get(file_name) == source
        )
        selected.extend(candidates[:per_source])

    if not selected:
        logger.warning("Smoke subset for %s found 0 matching images; falling back to first %d",
                       split, per_source * len(config.SOURCES))
        selected = sorted(file_name_to_id.values())[: per_source * len(config.SOURCES)]
    return selected
