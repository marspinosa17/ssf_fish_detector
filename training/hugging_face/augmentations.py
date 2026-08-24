"""Spectrogram-appropriate augmentations for RT-DETR fine-tuning.

Deliberately excludes anything that assumes natural-image semantics:

- No mosaic (mixes unrelated time/frequency contexts across tiles, which
  doesn't correspond to anything physical for a spectrogram).
- No hue/saturation/color jitter (the RGB channels are a grayscale-derived
  rendering of a single power spectrum, not independent color information —
  see preprocessing/window_utils.py:render_image, cv2.COLOR_GRAY2RGB).
- No vertical flip, ever. The y-axis is frequency (top = F_MAX, bottom =
  0 Hz per preprocessing/PREPROCESSING.md). Flipping it would invert the
  frequency axis, which is not a symmetry a fish call detector should be
  taught to expect.

Horizontal flip (time axis) is plausibly a valid symmetry — a fish call
doesn't inherently "read" left-to-right — but it is UNVALIDATED for this
data, so it defaults to off. Enable explicitly via --hflip-prob once
checked against real examples.

Scale/resize jitter is a standard detection augmentation (random crop-and-
resize around the existing 640x512 canvas) and is kept, since it doesn't
touch axis semantics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    """Toggles/probabilities for spectrogram-safe augmentations. All off by default."""

    hflip_prob: float = 0.0          # time-axis flip; UNVALIDATED, off by default
    scale_jitter_prob: float = 0.0   # random crop-and-resize scale jitter
    scale_jitter_range: tuple[float, float] = (0.8, 1.2)


def hflip(image: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip image + boxes along the time (x) axis. image: (H, W, C) uint8."""
    w = image.shape[1]
    flipped = image[:, ::-1, :].copy()
    out = boxes_xyxy.copy()
    out[:, 0] = w - boxes_xyxy[:, 2]
    out[:, 2] = w - boxes_xyxy[:, 0]
    return flipped, out


def scale_jitter(
    image: np.ndarray, boxes_xyxy: np.ndarray, scale_range: tuple[float, float],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly crop-and-resize back to the original canvas size.

    scale > 1 crops a smaller region and upsamples (zoom in); scale < 1 pads
    with zero (the recording's noise floor, per image_writer.py) and the
    content shrinks (zoom out). Boxes are transformed and clipped to bounds;
    boxes fully cropped out are dropped by the caller (empty-array-safe).
    """
    import cv2

    h, w = image.shape[:2]
    scale = rng.uniform(*scale_range)

    if scale >= 1.0:
        # zoom in: crop a smaller region, then resize up to (w, h)
        crop_w, crop_h = w / scale, h / scale
        x0 = rng.uniform(0, w - crop_w)
        y0 = rng.uniform(0, h - crop_h)
        x1, y1 = x0 + crop_w, y0 + crop_h
        cropped = image[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
        out_img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        out_boxes = boxes_xyxy.copy()
        out_boxes[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - x0) * (w / crop_w)
        out_boxes[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - y0) * (h / crop_h)
    else:
        # zoom out: resize down, paste onto a zero-padded canvas at a random offset
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out_img = np.zeros_like(image)
        x0 = rng.randint(0, w - new_w)
        y0 = rng.randint(0, h - new_h)
        out_img[y0:y0 + new_h, x0:x0 + new_w] = resized
        out_boxes = boxes_xyxy.copy()
        out_boxes[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * scale + x0
        out_boxes[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * scale + y0

    out_boxes[:, [0, 2]] = np.clip(out_boxes[:, [0, 2]], 0, w)
    out_boxes[:, [1, 3]] = np.clip(out_boxes[:, [1, 3]], 0, h)
    return out_img, out_boxes


def apply_augmentations(
    image: np.ndarray, boxes_xyxy: np.ndarray, cfg: AugmentationConfig, rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the configured augmentations in sequence. No-op if all probs are 0.

    boxes_xyxy: (N, 4) float array of [x1, y1, x2, y2] in pixel coordinates.
    Returns (image, boxes_xyxy_kept, keep_mask) — keep_mask is a boolean (N,)
    array the caller uses to filter parallel per-box arrays (labels, ids)
    since boxes degenerating to <1px in either dimension after a transform
    are dropped here.
    """
    if boxes_xyxy.size == 0:
        boxes_xyxy = boxes_xyxy.reshape(0, 4)
    keep = np.ones(boxes_xyxy.shape[0], dtype=bool)

    if cfg.hflip_prob > 0 and rng.random() < cfg.hflip_prob:
        image, boxes_xyxy = hflip(image, boxes_xyxy)

    if cfg.scale_jitter_prob > 0 and rng.random() < cfg.scale_jitter_prob:
        image, boxes_xyxy = scale_jitter(image, boxes_xyxy, cfg.scale_jitter_range, rng)

    if boxes_xyxy.shape[0] > 0:
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        keep = (w >= 1) & (h >= 1)
        boxes_xyxy = boxes_xyxy[keep]

    return image, boxes_xyxy, keep
