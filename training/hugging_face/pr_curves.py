"""Single-class PR/F1 curve computation and plotting, matching Ultralytics'
BoxPR_curve.png / BoxF1_curve.png visual style so the two implementations'
outputs are directly comparable.

Since there is exactly one class ("fish"), IoU-based greedy matching per
image at each confidence threshold gives precision/recall directly, without
needing COCOeval's multi-class machinery.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("rtdetr_hf.pr_curves")


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) / (M,4) xyxy pixel-coordinate arrays -> (N,M)."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_predictions(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy highest-score-first matching of predictions to GT boxes for one image.

    Returns (tp_flags, scores, n_gt) where tp_flags[i] is True if pred i (in
    the same order as pred_scores/pred_boxes) is a true positive. Each GT box
    can be matched at most once, so extra detections of the same fish call
    are counted as false positives (standard COCO/Ultralytics convention).
    """
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    tp = np.zeros(n_pred, dtype=bool)
    if n_pred == 0:
        return tp, pred_scores, n_gt
    if n_gt == 0:
        return tp, pred_scores, n_gt

    order = np.argsort(-pred_scores)
    ious = box_iou(pred_boxes[order], gt_boxes)
    matched_gt = np.zeros(n_gt, dtype=bool)
    for i, row in enumerate(ious):
        best_gt = np.argmax(row)
        if row[best_gt] >= iou_threshold and not matched_gt[best_gt]:
            matched_gt[best_gt] = True
            tp[order[i]] = True
    return tp, pred_scores, n_gt


def compute_pr_f1(
    all_tp: np.ndarray, all_scores: np.ndarray, n_gt_total: int, n_points: int = 200,
) -> dict:
    """Sweep confidence thresholds -> precision/recall/F1 arrays + AP (area under PR).

    all_tp / all_scores are flat arrays pooling every prediction across every
    image in the split (or the per-source subset). Standard COCO-style AP:
    area under the precision-recall curve built by sorting all detections by
    score descending and accumulating TP/FP.
    """
    if all_scores.size == 0 or n_gt_total == 0:
        thresholds = np.linspace(0, 1, n_points)
        zeros = np.zeros(n_points)
        return {
            "thresholds": thresholds, "precision": zeros, "recall": zeros, "f1": zeros,
            "ap": 0.0, "best_f1": 0.0, "best_f1_threshold": 0.0,
        }

    order = np.argsort(-all_scores)
    tp_sorted = all_tp[order]
    scores_sorted = all_scores[order]
    fp_sorted = ~tp_sorted

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recall_curve = tp_cum / n_gt_total
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # AP via 101-point interpolation (COCO convention): precision envelope
    # made monotonically non-increasing from the right, then integrated.
    precision_envelope = np.maximum.accumulate(precision_curve[::-1])[::-1]
    recall_levels = np.linspace(0, 1, 101)
    interp_precision = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        idxs = np.where(recall_curve >= r)[0]
        interp_precision[i] = precision_envelope[idxs[0]] if idxs.size else 0.0
    ap = float(np.mean(interp_precision))

    # Threshold-indexed curves for plotting: for each confidence threshold,
    # precision/recall considering only detections scoring >= threshold.
    thresholds = np.linspace(0, 1, n_points)
    precision = np.zeros(n_points)
    recall = np.zeros(n_points)
    for i, t in enumerate(thresholds):
        mask = scores_sorted >= t
        n_kept = mask.sum()
        if n_kept == 0:
            precision[i] = 1.0  # Ultralytics convention: no detections -> precision defined as 1
            recall[i] = 0.0
            continue
        tp_at_t = tp_sorted[mask].sum()
        precision[i] = tp_at_t / n_kept
        recall[i] = tp_at_t / n_gt_total

    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-16), 0.0)
    best_idx = int(np.argmax(f1))

    return {
        "thresholds": thresholds, "precision": precision, "recall": recall, "f1": f1,
        "ap": ap, "best_f1": float(f1[best_idx]), "best_f1_threshold": float(thresholds[best_idx]),
    }


def plot_pr_curve(curve: dict, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    # sort by recall for a clean monotone-ish line (thresholds sweep high->low score)
    order = np.argsort(curve["recall"])
    ax.plot(curve["recall"][order], curve["precision"][order], linewidth=2, color="#2971D6",
            label=f"fish (AP={curve['ap']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_f1_curve(curve: dict, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(curve["thresholds"], curve["f1"], linewidth=2, color="#2971D6",
            label=f"fish (best F1={curve['best_f1']:.3f} @ conf={curve['best_f1_threshold']:.2f})")
    ax.axvline(curve["best_f1_threshold"], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
