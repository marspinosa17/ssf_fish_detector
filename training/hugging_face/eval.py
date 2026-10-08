"""Evaluate a fine-tuned RT-DETRv2 checkpoint: COCO mAP + PR/F1 curves.

mAP is computed with torchmetrics.detection.MeanAveragePrecision (in-process,
pure-torch) rather than a pycocotools.cocoeval wrapper: torchmetrics avoids a
JSON-serialize-then-reparse round trip and gives per-call access to the
underlying tensors, which the per-source breakdown and PR/F1 curves below
both need anyway. The tradeoff is torchmetrics' mAP is a reimplementation of
the COCO metric rather than the reference COCOeval code itself — values are
expected to match pycocotools closely but are not guaranteed bit-identical;
if that matters, cross-check against a COCOeval run on the same predictions.

Run:
    python -m training.hugging_face.eval --model-dir models/rtdetr_hf/final --split val
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoImageProcessor, AutoModelForObjectDetection

try:
    from . import config
    from .dataset import build_dataset
    from .inference import load_image_id_to_source, run_inference
    from .logging_utils import setup_logging
    from .pr_curves import compute_pr_f1, match_predictions, plot_f1_curve, plot_pr_curve
except ImportError:
    import config
    from dataset import build_dataset
    from inference import load_image_id_to_source, run_inference
    from logging_utils import setup_logging
    from pr_curves import compute_pr_f1, match_predictions, plot_f1_curve, plot_pr_curve

logger = logging.getLogger("rtdetr_hf.eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RT-DETRv2 (HuggingFace) on a split")
    p.add_argument("--model-dir", type=Path, required=True, help="fine-tuned model dir (from train.py)")
    p.add_argument("--split", choices=config.SPLITS, default="val")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="defaults to --model-dir/eval_{split}")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for TP matching in PR/F1 curves")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-per-source", type=int, default=4)
    return p.parse_args()


def to_xyxy_gt(coco, image_id: int) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    boxes = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0]:
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
    return boxes


def compute_map(dataset, predictions: dict[int, dict], image_ids: list[int] | None = None) -> dict:
    """torchmetrics MeanAveragePrecision over a set of image_ids (default: the whole dataset)."""
    ids = image_ids if image_ids is not None else dataset.image_ids
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    preds, targets = [], []
    for image_id in ids:
        pred = predictions[image_id]
        preds.append({
            "boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"],
        })
        gt_boxes = to_xyxy_gt(dataset.coco, image_id)
        targets.append({
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.zeros(gt_boxes.shape[0], dtype=torch.int64),
        })
    metric.update(preds, targets)
    result = metric.compute()
    return {k: (v.item() if hasattr(v, "item") and v.numel() == 1 else v.tolist())
            for k, v in result.items() if k != "classes"}


def pooled_tp_scores(
    dataset, predictions: dict[int, dict], iou_threshold: float, image_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool TP flags + scores across a set of images (all, or a source subset)."""
    ids = image_ids if image_ids is not None else dataset.image_ids
    all_tp, all_scores = [], []
    n_gt_total = 0
    for image_id in ids:
        pred = predictions[image_id]
        pred_boxes = pred["boxes"].numpy()
        pred_scores = pred["scores"].numpy()
        gt_boxes = to_xyxy_gt(dataset.coco, image_id)
        n_gt_total += gt_boxes.shape[0]
        tp, scores, _ = match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        all_tp.append(tp)
        all_scores.append(scores)
    return (
        np.concatenate(all_tp) if all_tp else np.array([], dtype=bool),
        np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32),
        n_gt_total,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.model_dir / f"eval_{args.split}")
    logger_local = setup_logging(output_dir, "eval.log", "rtdetr_hf.eval")
    logger_local.info("Args: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(args.model_dir))
    model.to(device)

    dataset = build_dataset(
        args.split, image_processor, augmentation_config=None,
        smoke=args.smoke, smoke_per_source=args.smoke_per_source,
    )
    predictions = run_inference(model, image_processor, dataset, device, batch_size=args.batch_size)

    # ── Overall mAP ──
    logger_local.info("Computing overall mAP ...")
    overall_map = compute_map(dataset, predictions)
    logger_local.info("Overall mAP: %s", overall_map)

    # ── Per-source mAP (mirrors the per-source guardrail in data/build_manifest.py) ──
    file_name_to_id = {img["file_name"]: img["id"] for img in dataset.coco.dataset["images"]}
    id_to_file_name = {v: k for k, v in file_name_to_id.items()}
    path_to_source = load_image_id_to_source(args.split)

    ids_by_source: dict[str, list[int]] = {s: [] for s in config.SOURCES}
    for image_id in dataset.image_ids:
        source = path_to_source.get(id_to_file_name[image_id])
        if source in ids_by_source:
            ids_by_source[source].append(image_id)

    per_source_map = {}
    for source, ids in ids_by_source.items():
        if not ids:
            logger_local.warning("No %s images in this %s subset (smoke=%s?) — skipping",
                                 source, args.split, args.smoke)
            continue
        per_source_map[source] = compute_map(dataset, predictions, ids)
        logger_local.info("Source %-6s (n=%d) mAP: map=%.4f map_50=%.4f", source, len(ids),
                          per_source_map[source]["map"], per_source_map[source]["map_50"])

    # ── PR / F1 curves: overall ──
    tp, scores, n_gt = pooled_tp_scores(dataset, predictions, args.iou_threshold)
    overall_curve = compute_pr_f1(tp, scores, n_gt)
    plot_pr_curve(overall_curve, output_dir / "PR_curve.png",
                  f"PR Curve — fish — {args.split} (IoU={args.iou_threshold})")
    plot_f1_curve(overall_curve, output_dir / "F1_curve.png",
                  f"F1 Curve — fish — {args.split} (IoU={args.iou_threshold})")
    logger_local.info("Overall AP@%.2f=%.4f best_F1=%.4f @ conf=%.2f",
                      args.iou_threshold, overall_curve["ap"], overall_curve["best_f1"],
                      overall_curve["best_f1_threshold"])

    # ── PR / F1 curves: per source ──
    per_source_curves = {}
    for source, ids in ids_by_source.items():
        if not ids:
            continue
        tp_s, scores_s, n_gt_s = pooled_tp_scores(dataset, predictions, args.iou_threshold, ids)
        curve = compute_pr_f1(tp_s, scores_s, n_gt_s)
        per_source_curves[source] = {"ap": curve["ap"], "best_f1": curve["best_f1"],
                                     "best_f1_threshold": curve["best_f1_threshold"]}
        plot_pr_curve(curve, output_dir / f"PR_curve_{source}.png",
                      f"PR Curve — fish — {args.split}/{source} (IoU={args.iou_threshold})")
        plot_f1_curve(curve, output_dir / f"F1_curve_{source}.png",
                      f"F1 Curve — fish — {args.split}/{source} (IoU={args.iou_threshold})")

    # ── Summary JSON ──
    summary = {
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "overall_map": overall_map,
        "overall_pr_ap": overall_curve["ap"],
        "overall_best_f1": overall_curve["best_f1"],
        "overall_best_f1_threshold": overall_curve["best_f1_threshold"],
        "per_source_map": per_source_map,
        "per_source_pr_f1": per_source_curves,
    }
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger_local.info("Wrote %s", summary_path)
    logger_local.info("Evaluation complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
