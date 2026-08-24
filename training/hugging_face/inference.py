"""Shared inference helpers for eval.py and predict_samples.py.

Runs the model over a split at a very low confidence floor so downstream
code can sweep confidence thresholds for PR/F1 curves without re-running
inference per threshold.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger("rtdetr_hf.inference")

# Floor kept low (not 0) purely to bound memory/time on very confident
# background predictions; every threshold used in PR/F1 sweeps is well above this.
SCORE_FLOOR = 0.01


@torch.no_grad()
def run_inference(
    model, image_processor, dataset, device: str, batch_size: int = 16,
) -> dict[int, dict]:
    """Run the model over every image in `dataset`.

    Returns {image_id: {"boxes": (N,4) xyxy pixel tensor, "scores": (N,), "labels": (N,)}}.
    """
    try:
        from .dataset import collate_fn as _collate_fn
    except ImportError:
        from dataset import collate_fn as _collate_fn

    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    results: dict[int, dict] = {}
    idx = 0
    for batch in tqdm(loader, desc="inference", unit="batch"):
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)

        # RTDetrImageProcessor needs each image's *original* (pre-resize) size
        # to rescale predicted boxes back to pixel coordinates.
        batch_ids = dataset.image_ids[idx: idx + pixel_values.shape[0]]
        target_sizes = torch.tensor(
            [[config.IMG_HEIGHT, config.IMG_WIDTH]] * pixel_values.shape[0]
        ).to(device)
        processed = image_processor.post_process_object_detection(
            outputs, threshold=SCORE_FLOOR, target_sizes=target_sizes,
        )
        for image_id, pred in zip(batch_ids, processed):
            results[image_id] = {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            }
        idx += pixel_values.shape[0]

    return results


def load_image_id_to_source(split: str) -> dict[str, str]:
    """Map a COCO file_name (as stored in coco_{split}.json) to its source.

    dataset_manifest.csv's image_path column already matches coco json's
    file_name format ("{split}/images/{stem}.png") — see
    preprocessing/preprocess.py:_manifest_row and coco_converter.py.
    """
    manifest = pd.read_csv(config.MANIFEST_PATH)
    manifest = manifest[manifest["split"] == split]
    return dict(zip(manifest["image_path"], manifest["source"]))
