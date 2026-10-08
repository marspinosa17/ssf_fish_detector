#!/usr/bin/env python
"""
Overlaid PR and F1 curves for one or more Ultralytics detection models
(YOLO and/or RT-DETR checkpoints) across the val and test splits.

Unlike `model.val()`, this script keeps the raw (confidence, TP/FP)
stream from its own greedy IoU matching, so every curve is saved as a
plain array (CSV) alongside the plots -- fully re-plottable / editable
without rerunning inference. Figures are also saved as .svg (vector,
edit in Illustrator/Inkscape) and as a pickled matplotlib Figure
(reopen with `pickle.load` -> fig -> tweak -> `fig.savefig(...)`).

NOTE ON THE "AP" PRINTED HERE: it's computed directly from this script's
own matching + confidence-sorted curve, as a diagnostic that's self-
consistent with the plotted curve. It is not guaranteed to exactly match
Ultralytics' own reported mAP@0.5 (different matching/interpolation
implementation details) -- treat the two as complementary, not
interchangeable, same caveat as the torchmetrics-vs-pycocotools note in
your eval.py.

BEFORE RUNNING: check runs/<name>/args.yaml for the imgsz each model was
actually trained at and pass it via --imgsz. A mismatch silently
degrades both models' curves without erroring.

Usage:
    python pr_f1_curves.py \
        --model yolo   yolo   runs/fish_yolo26s/weights/best.pt \
        --model rtdetr rtdetr runs/ul_rtdetr_l_v1/weights/best.pt \
        --data-root data/spectrograms \
        --splits val test \
        --iou-thres 0.5 \
        --imgsz 640 \
        --out-dir analysis/pr_f1_curves

Each --model flag takes three tokens: NAME, TYPE (yolo|rtdetr), WEIGHTS_PATH.

Outputs in --out-dir:
    curve_<model>_<split>.csv   one row per prediction: conf, precision,
                                 recall, f1, precision_interp
    curves_all.csv               the above, all combos, long format
    summary.csv                  AP + best-F1(+threshold) per combo
    PR_curve.{svg,png,fig.pkl}
    F1_curve.{svg,png,fig.pkl}
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless HPC node -- must be set before importing pyplot
import matplotlib.pyplot as plt


# ───────────────────────────── CLI ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlaid PR/F1 curves, YOLO vs RT-DETR (Ultralytics)")
    p.add_argument("--model", nargs=3, action="append", required=True,
                   metavar=("NAME", "TYPE", "WEIGHTS"),
                   help="e.g. --model yolo yolo runs/fish_yolo26s/weights/best.pt "
                        "(TYPE is 'yolo' or 'rtdetr')")
    p.add_argument("--data-root", type=Path, default=Path("data/spectrograms"))
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--iou-thres", type=float, default=0.5,
                   help="IoU threshold for TP matching (this is NOT the model's own NMS IoU)")
    p.add_argument("--conf-thres", type=float, default=0.001,
                   help="prediction confidence floor -- keep low to see the full curve")
    p.add_argument("--nms-iou", type=float, default=0.7, help="model's own NMS IoU (YOLO only)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="MUST match training imgsz -- check runs/<name>/args.yaml")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="fp16 inference (faster, default off)")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/pr_f1_curves"))
    return p.parse_args()


# ───────────────────────────── matching ─────────────────────────────────

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between a (N,4) and b (M,4) xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b.reshape(1, -1) - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray, pred_conf: np.ndarray,
                 iou_thres: float) -> np.ndarray:
    """Greedy single-class matching (nc=1, so no class check needed).
    Returns a bool TP array aligned with pred_xyxy/pred_conf in their
    ORIGINAL order (not confidence-sorted)."""
    n_pred = pred_xyxy.shape[0]
    tp = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or gt_xyxy.shape[0] == 0:
        return tp  # nothing to match -> everything (if any) is FP
    order = np.argsort(-pred_conf)
    matched_gt = np.zeros(gt_xyxy.shape[0], dtype=bool)
    ious = box_iou(pred_xyxy, gt_xyxy)
    for i in order:
        row = ious[i].copy()
        row[matched_gt] = -1  # a GT box can only be claimed once
        j = int(np.argmax(row))
        if row[j] >= iou_thres:
            tp[i] = True
            matched_gt[j] = True
    return tp


# ───────────────────────────── GT loading ────────────────────────────────

def load_yolo_gt(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4))
    rows = np.loadtxt(label_path, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 4))
    xc, yc, w, h = rows[:, 1] * img_w, rows[:, 2] * img_h, rows[:, 3] * img_w, rows[:, 4] * img_h
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


# ───────────────────────────── inference + collection ───────────────────

def load_model(model_type: str, weights: Path):
    if model_type == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(str(weights))
    elif model_type == "yolo":
        from ultralytics import YOLO
        return YOLO(str(weights))
    raise ValueError(f"unknown model type {model_type!r} (use 'yolo' or 'rtdetr')")


def collect_scores(model, split_dir: Path, args) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Run inference over every image in split_dir/images, match against
    split_dir/labels, return (scores, tp_flags, total_gt, counts)."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    all_scores, all_tp = [], []
    total_gt = 0
    n_images = n_pred_boxes = n_gt_images = 0

    results_gen = model.predict(
        source=str(images_dir), stream=True, conf=args.conf_thres, iou=args.nms_iou,
        imgsz=args.imgsz, batch=args.batch, device=args.device, half=args.half, verbose=False,
    )
    for res in results_gen:
        n_images += 1
        stem = Path(res.path).stem
        h, w = res.orig_shape  # size labels are normalized against
        gt = load_yolo_gt(labels_dir / f"{stem}.txt", img_w=w, img_h=h)
        total_gt += gt.shape[0]
        if gt.shape[0]:
            n_gt_images += 1

        if res.boxes is None or len(res.boxes) == 0:
            continue
        pred_xyxy = res.boxes.xyxy.cpu().numpy()
        pred_conf = res.boxes.conf.cpu().numpy()
        n_pred_boxes += pred_xyxy.shape[0]

        tp = match_image(gt, pred_xyxy, pred_conf, args.iou_thres)
        all_scores.append(pred_conf)
        all_tp.append(tp)

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    counts = dict(n_images=n_images, n_gt_images=n_gt_images,
                  n_pred_boxes=n_pred_boxes, n_gt_boxes=total_gt)
    return scores, tp, total_gt, counts


# ───────────────────────────── curve math ────────────────────────────────

def compute_curve(scores: np.ndarray, tp: np.ndarray, total_gt: int) -> pd.DataFrame:
    """Sort all predictions by confidence descending; walk the cumulative
    precision/recall/F1 curve. One row per prediction."""
    if scores.size == 0:
        return pd.DataFrame(columns=["conf", "precision", "recall", "f1", "precision_interp"])
    order = np.argsort(-scores)
    s, t = scores[order], tp[order]
    cum_tp = np.cumsum(t)
    cum_fp = np.cumsum(~t)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / max(total_gt, 1)
    denom = np.maximum(precision + recall, 1e-9)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / denom, 0.0)
    # monotone envelope (used for the AP estimate only; raw `precision` is
    # kept too if you want to re-plot the un-smoothed curve)
    precision_interp = np.maximum.accumulate(precision[::-1])[::-1]
    return pd.DataFrame({"conf": s, "precision": precision, "recall": recall,
                          "f1": f1, "precision_interp": precision_interp})


def average_precision(curve: pd.DataFrame) -> float:
    if curve.empty:
        return 0.0
    r = np.concatenate(([0.0], curve["recall"].values))
    p = np.concatenate(([curve["precision_interp"].iloc[0]], curve["precision_interp"].values))
    return float(np.sum(np.diff(r) * p[1:]))


def best_f1(curve: pd.DataFrame) -> tuple[float, float]:
    if curve.empty:
        return 0.0, 0.0
    i = int(curve["f1"].values.argmax())
    return float(curve["f1"].iloc[i]), float(curve["conf"].iloc[i])


# ───────────────────────────── plotting ──────────────────────────────────

STYLE = {"val": "-", "test": "--"}

def plot_curves(curves: dict, out_dir: Path) -> None:
    models = sorted({m for m, _ in curves})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))
    color_of = {m: colors[i] for i, m in enumerate(models)}

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_f1, ax_f1 = plt.subplots(figsize=(6, 5))

    for (model, split), curve in curves.items():
        if curve.empty:
            continue
        label = f"{model} / {split}"
        style = STYLE.get(split, ":")
        ax_pr.plot(curve["recall"], curve["precision_interp"], style,
                   color=color_of[model], label=label, linewidth=1.8)
        ax_f1.plot(curve["conf"], curve["f1"], style,
                   color=color_of[model], label=label, linewidth=1.8)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.set_title("Precision–Recall"); ax_pr.legend(fontsize=8); ax_pr.grid(alpha=0.3)

    ax_f1.set_xlabel("Confidence threshold"); ax_f1.set_ylabel("F1")
    ax_f1.set_xlim(0, 1); ax_f1.set_ylim(0, 1.02)
    ax_f1.set_title("F1 vs. confidence"); ax_f1.legend(fontsize=8); ax_f1.grid(alpha=0.3)

    for fig, name in [(fig_pr, "PR_curve"), (fig_f1, "F1_curve")]:
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.svg")           # vector, editable
        fig.savefig(out_dir / f"{name}.png", dpi=200)  # quick-look raster
        with open(out_dir / f"{name}.fig.pkl", "wb") as fh:
            pickle.dump(fig, fh)  # reopen: pickle.load(open(...,'rb')) -> fig
        plt.close(fig)


# ───────────────────────────── main ──────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows, long_rows = [], []

    for name, mtype, weights in args.model:
        print(f"\n=== loading {name} ({mtype}) from {weights} ===")
        model = load_model(mtype, Path(weights))

        for split in args.splits:
            split_dir = args.data_root / split
            print(f"--- {name} / {split} ---")
            scores, tp, total_gt, counts = collect_scores(model, split_dir, args)
            print(f"    images={counts['n_images']} gt_boxes={counts['n_gt_boxes']} "
                  f"pred_boxes={counts['n_pred_boxes']}")

            curve = compute_curve(scores, tp, total_gt)
            curves[(name, split)] = curve
            ap = average_precision(curve)
            f1, f1_conf = best_f1(curve)
            summary_rows.append(dict(model=name, split=split, ap=ap, best_f1=f1,
                                      best_f1_conf=f1_conf, n_gt=total_gt, **counts))
            print(f"    AP={ap:.4f}  best F1={f1:.4f} @ conf={f1_conf:.3f}")

            curve.to_csv(args.out_dir / f"curve_{name}_{split}.csv", index=False)
            curve_out = curve.copy()
            curve_out.insert(0, "split", split)
            curve_out.insert(0, "model", name)
            long_rows.append(curve_out)

    pd.concat(long_rows, ignore_index=True).to_csv(args.out_dir / "curves_all.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    plot_curves(curves, args.out_dir)
    print(f"\nWrote curves + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
