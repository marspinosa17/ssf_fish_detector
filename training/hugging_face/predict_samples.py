"""Render GT-vs-prediction visualizations for a sample of val/test tiles.

Equivalent of Ultralytics' val_batch*_labels.jpg / val_batch*_pred.jpg: GT
boxes and predicted boxes overlaid on the spectrogram in different colors,
predictions labeled with confidence.

Coordinate system note (see preprocessing/PREPROCESSING.md "Coordinate
system"): the PNG pixels already encode the frequency flip — row 0 (top of
image) is F_MAX (2000 Hz), row IMG_HEIGHT-1 (bottom) is 0 Hz (see
preprocessing/window_utils.py:render_image, which does np.flipud before
writing). Both GT boxes (from coco_{split}.json, derived from the same YOLO
labels written against that flipped image) and predicted boxes (from the
model, operating on that same PNG) are already in this flipped pixel space.
Drawing them with standard top-left-origin image coordinates is therefore
correct as-is — no additional flip at draw time. This script only asserts
that invariant; see `_sanity_check_orientation`.

Run:
    python -m training.rtdetr_hf.predict_samples --model-dir models/rtdetr_hf/final --split val
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

try:
    from . import config
    from .dataset import build_dataset
    from .inference import load_image_id_to_source, run_inference
    from .logging_utils import setup_logging
    from .pr_curves import box_iou
except ImportError:
    import config
    from dataset import build_dataset
    from inference import load_image_id_to_source, run_inference
    from logging_utils import setup_logging
    from pr_curves import box_iou

logger = logging.getLogger("rtdetr_hf.predict_samples")

GT_COLOR = (0, 200, 0)        # green
TP_COLOR = (255, 165, 0)      # orange — predicted box that matches a GT box
FP_COLOR = (255, 0, 0)        # red — predicted box with no matching GT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GT-vs-prediction sample tiles")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--split", choices=config.SPLITS, default="val")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="defaults to --model-dir/predictions_sample")
    p.add_argument("--n-per-source", type=int, default=8,
                   help="number of sample tiles per source")
    p.add_argument("--conf-threshold", type=float, default=0.3,
                   help="confidence threshold for boxes drawn in the visualization")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for TP/FP coloring of predicted boxes")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def _sanity_check_orientation(image_array: np.ndarray) -> None:
    """Cheap runtime guard: spectrogram tiles should have more energy in
    lower rows (near-DC / low-frequency content dominates fish calls and
    ambient noise) than in the very top rows (near F_MAX). This doesn't
    prove box orientation is correct, but a violation would flag that the
    image itself isn't in the expected flipped-y layout before we trust box
    placement against it.
    """
    top_band = image_array[: image_array.shape[0] // 8].astype(np.float32).mean()
    bottom_band = image_array[-image_array.shape[0] // 8:].astype(np.float32).mean()
    if top_band > bottom_band * 1.5:
        logger.warning(
            "Orientation sanity check: top band mean (%.1f) notably brighter than "
            "bottom band (%.1f) — expected low-frequency energy to dominate near the "
            "bottom of the image per PREPROCESSING.md. Verify image orientation.",
            top_band, bottom_band,
        )


def _select_samples(
    dataset, predictions: dict[int, dict], ids_by_source: dict[str, list[int]],
    n_per_source: int, conf_threshold: float, iou_threshold: float, rng: random.Random,
) -> dict[str, list[tuple[int, str]]]:
    """Pick a mix of TP/FP/FN-containing tiles per source, not just random ones.

    Categorizes each image by whether, at conf_threshold, it has at least one
    false positive, at least one false negative (missed GT), or is a clean
    match — then samples across those buckets so the grid isn't dominated by
    easy true negatives.
    """
    selected: dict[str, list[tuple[int, str]]] = {}
    for source, ids in ids_by_source.items():
        buckets: dict[str, list[int]] = {"fp": [], "fn": [], "clean_positive": [], "other": []}
        for image_id in ids:
            pred = predictions[image_id]
            mask = pred["scores"] >= conf_threshold
            pred_boxes = pred["boxes"][mask].numpy()
            gt_boxes = _gt_boxes(dataset, image_id)

            if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
                buckets["other"].append(image_id)
                continue

            ious = box_iou(pred_boxes, gt_boxes) if pred_boxes.shape[0] and gt_boxes.shape[0] else \
                np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))
            gt_hit = (ious >= iou_threshold).any(axis=0) if ious.size else np.zeros(gt_boxes.shape[0], dtype=bool)
            pred_hit = (ious >= iou_threshold).any(axis=1) if ious.size else np.zeros(pred_boxes.shape[0], dtype=bool)

            has_fn = gt_boxes.shape[0] > 0 and not gt_hit.all()
            has_fp = pred_boxes.shape[0] > 0 and not pred_hit.all()

            if has_fp:
                buckets["fp"].append(image_id)
            elif has_fn:
                buckets["fn"].append(image_id)
            elif gt_boxes.shape[0] > 0:
                buckets["clean_positive"].append(image_id)
            else:
                buckets["other"].append(image_id)

        # spread the sample budget across buckets, favoring the interesting ones
        picks: list[tuple[int, str]] = []
        quota = {"fp": max(1, n_per_source // 3), "fn": max(1, n_per_source // 3),
                 "clean_positive": max(1, n_per_source // 4)}
        for bucket_name, k in quota.items():
            pool = buckets[bucket_name]
            rng.shuffle(pool)
            picks.extend((iid, bucket_name) for iid in pool[:k])
        if len(picks) < n_per_source:
            remaining = [iid for iid in buckets["other"] if iid not in {p[0] for p in picks}]
            rng.shuffle(remaining)
            picks.extend((iid, "other") for iid in remaining[: n_per_source - len(picks)])
        selected[source] = picks[:n_per_source]
    return selected


def _gt_boxes(dataset, image_id: int) -> np.ndarray:
    ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
    anns = dataset.coco.loadAnns(ann_ids)
    boxes = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0]:
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
    return boxes


def _draw_tile(
    image_path: Path, gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray,
    iou_threshold: float, title: str,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in gt_boxes:
        draw.rectangle(list(box), outline=GT_COLOR, width=2)

    if pred_boxes.shape[0] and gt_boxes.shape[0]:
        ious = box_iou(pred_boxes, gt_boxes)
        pred_hit = (ious >= iou_threshold).any(axis=1)
    else:
        pred_hit = np.zeros(pred_boxes.shape[0], dtype=bool)

    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    for box, score, hit in zip(pred_boxes, pred_scores, pred_hit):
        color = TP_COLOR if hit else FP_COLOR
        draw.rectangle(list(box), outline=color, width=2)
        label = f"{score:.2f}"
        text_y = max(0, box[1] - 12)
        draw.text((box[0], text_y), label, fill=color, font=font)

    banner_h = 16
    banner = Image.new("RGB", (img.width, img.height + banner_h), (30, 30, 30))
    banner.paste(img, (0, banner_h))
    d2 = ImageDraw.Draw(banner)
    d2.text((4, 2), title, fill=(255, 255, 255), font=font)
    return banner


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.model_dir / "predictions_sample")
    logger_local = setup_logging(output_dir.parent, "predict_samples.log", "rtdetr_hf.predict_samples")
    logger_local.info("Args: %s", vars(args))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(args.model_dir))
    model.to(device)

    dataset = build_dataset(args.split, image_processor, augmentation_config=None)
    predictions = run_inference(model, image_processor, dataset, device, batch_size=args.batch_size)

    file_name_to_id = {img["file_name"]: img["id"] for img in dataset.coco.dataset["images"]}
    id_to_file_name = {v: k for k, v in file_name_to_id.items()}
    path_to_source = load_image_id_to_source(args.split)

    ids_by_source: dict[str, list[int]] = {s: [] for s in config.SOURCES}
    for image_id in dataset.image_ids:
        source = path_to_source.get(id_to_file_name[image_id])
        if source in ids_by_source:
            ids_by_source[source].append(image_id)

    rng = random.Random(args.seed)
    selection = _select_samples(dataset, predictions, ids_by_source, args.n_per_source,
                                args.conf_threshold, args.iou_threshold, rng)

    orientation_checked = False
    n_written = 0
    for source, picks in selection.items():
        source_dir = output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        for image_id, bucket in picks:
            file_name = id_to_file_name[image_id]
            image_path = config.SPECTROGRAM_ROOT / file_name

            if not orientation_checked:
                _sanity_check_orientation(np.array(Image.open(image_path).convert("L")))
                orientation_checked = True

            pred = predictions[image_id]
            mask = pred["scores"] >= args.conf_threshold
            pred_boxes = pred["boxes"][mask].numpy()
            pred_scores = pred["scores"][mask].numpy()
            gt_boxes = _gt_boxes(dataset, image_id)

            title = f"{Path(file_name).stem} [{bucket}] GT={gt_boxes.shape[0]} pred={pred_boxes.shape[0]}"
            tile = _draw_tile(image_path, gt_boxes, pred_boxes, pred_scores, args.iou_threshold, title)
            out_path = source_dir / f"{Path(file_name).stem}_{bucket}.png"
            tile.save(out_path)
            n_written += 1

    logger_local.info("Wrote %d prediction sample tiles to %s", n_written, output_dir)
    logger_local.info("Legend: green=GT, orange=TP prediction, red=FP prediction "
                      "(conf>=%.2f, IoU>=%.2f)", args.conf_threshold, args.iou_threshold)


if __name__ == "__main__":
    main()
