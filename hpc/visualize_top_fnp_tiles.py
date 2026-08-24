"""
Visualize the top-N tiles (spectrogram images) with the most false negatives
OR false positives, depending on --mode.

For each of the top-N stems, reruns inference on that single image and draws:
  - green solid box  = GT box the model caught (correct detection)
  - red solid box    = GT box the model missed entirely (missed detection)
  - blue dashed box  = every model prediction (labeled with confidence) --
                        in --mode fp this includes the unmatched ones driving
                        the ranking, since they're still just "a prediction"
                        visually; compare against the GT boxes to see which
                        blue boxes don't overlap any green/red box

Saves one annotated PNG per tile plus a single contact-sheet grid image.

Usage:
    # top 10 tiles by false negatives
    python visualize_top_fn_tiles.py \
        --weights runs/fish_yolo26s/weights/best.pt \
        --data-root data/spectrograms \
        --miss-csv miss_analysis_v2.csv \
        --split val --n 10 --mode fn --out-dir top_fn_tiles/

    # top 10 tiles by false positives
    python visualize_top_fn_tiles.py \
        --weights runs/fish_yolo26s/weights/best.pt \
        --data-root data/spectrograms \
        --miss-csv miss_analysis_v2.csv \
        --split val --n 10 --mode fp --out-dir top_fp_tiles/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(xc, yc, w, h):
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def load_gt(label_path: Path):
    boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().splitlines():
            _, xc, yc, w, h = line.split()
            boxes.append(yolo_to_xyxy(float(xc), float(yc), float(w), float(h)))
    return boxes


def draw_tile(ax, img_path, gt_boxes, pred_boxes, pred_confs, conf_thr, iou_thr, title):
    img = Image.open(img_path)
    w, h = img.size
    ax.imshow(img)

    # Match GT against ALL predictions (any confidence) -- mirrors
    # analyze_misses_v2.py, which also matches unconditionally and only
    # uses conf_thr afterward to split TP_confident vs TP_low_conf. Matching
    # only against >=conf_thr predictions (the old behavior) threw away
    # correct low-confidence hits and drew them as red/missed, which
    # disagreed with the CSV's TP_low_conf label for the same box.
    used = set()
    for gx1, gy1, gx2, gy2 in gt_boxes:
        best_iou, best_j, best_conf = 0.0, -1, 0.0
        for j, pb in enumerate(pred_boxes):
            if j in used:
                continue
            iou = box_iou([gx1, gy1, gx2, gy2], pb)
            if iou > best_iou:
                best_iou, best_j, best_conf = iou, j, pred_confs[j]
        is_match = best_iou >= iou_thr
        if is_match:
            used.add(best_j)
        # Only a match AND at/above the operating threshold counts as a real
        # (green) detection; a low-confidence match still shows red since it
        # wouldn't fire at deployment -- but note it's TP_low_conf, not a
        # true FN, in the underlying miss-analysis CSV.
        color = "green" if (is_match and best_conf >= conf_thr) else "red"
        rect = patches.Rectangle((gx1 * w, gy1 * h), (gx2 - gx1) * w, (gy2 - gy1) * h,
                                  linewidth=2.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Only render predictions that clear the operating threshold -- these are
    # the only ones analyze_misses_v2.py counts as FP, and the only ones that
    # would actually fire at deployment. Below-threshold predictions are
    # deliberately not drawn to avoid cluttering the plot with boxes that
    # aren't part of the FP count in the title.
    for j, (px1, py1, px2, py2) in enumerate(pred_boxes):
        if pred_confs[j] < conf_thr:
            continue
        rect = patches.Rectangle((px1 * w, py1 * h), (px2 - px1) * w, (py2 - py1) * h,
                                  edgecolor="blue", facecolor="none",
                                  linestyle="--", linewidth=2.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(px1 * w, max(py1 * h - 3, 0), f"{pred_confs[j]:.2f}",
                 color="blue", fontsize=8, fontweight="bold", va="bottom")

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--miss-csv", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--mode", default="fn", choices=["fn", "fp"],
                     help="'fn' = top tiles by missed-fish count, "
                          "'fp' = top tiles by false-alarm count (needs v2 miss-csv)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conf-thr", type=float, default=0.223)
    ap.add_argument("--low-conf-floor", type=float, default=0.05)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default=None,
                     help="default: top_fn_tiles or top_fp_tiles based on --mode")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / args.split / "images"
    labels_dir = data_root / args.split / "labels"
    out_dir = Path(args.out_dir or f"top_{args.mode}_tiles")
    (out_dir / "individual").mkdir(parents=True, exist_ok=True)

    miss = pd.read_csv(args.miss_csv)
    target_status = "FN" if args.mode == "fn" else "FP"
    if args.mode == "fp" and "box_type" not in miss.columns:
        raise SystemExit("--mode fp requires a miss-csv produced by analyze_misses_v2.py "
                          "(needs the box_type/FP columns) -- v1 output only has GT rows.")
    counts = miss[miss["status"] == target_status].groupby("stem").size().sort_values(ascending=False)
    top_stems = counts.head(args.n)
    print(f"Top {args.n} tiles by {target_status} count:\n{top_stems}")

    model = YOLO(args.weights)

    ncols = 5
    nrows = int(np.ceil(len(top_stems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (stem, n_hits) in zip(axes, top_stems.items()):
        img_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        gt_boxes = load_gt(label_path)

        res = model.predict(str(img_path), conf=args.low_conf_floor, iou=0.6, verbose=False)[0]
        pred_boxes = res.boxes.xyxyn.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        pred_confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,))

        source = miss.loc[miss["stem"] == stem, "source"].iloc[0]
        title = f"{stem}\n{source} | {n_hits} {target_status} / {len(gt_boxes)} GT boxes"
        draw_tile(ax, img_path, gt_boxes, pred_boxes, pred_confs,
                  args.conf_thr, args.iou_thr, title)

        # also save a standalone full-res version
        fig_i, ax_i = plt.subplots(figsize=(8, 6))
        draw_tile(ax_i, img_path, gt_boxes, pred_boxes, pred_confs,
                  args.conf_thr, args.iou_thr, title)
        fig_i.tight_layout()
        fig_i.savefig(out_dir / "individual" / f"{stem}.png", dpi=150)
        plt.close(fig_i)

    for ax in axes[len(top_stems):]:
        ax.axis("off")

    metric_desc = "false-negative (missed fish)" if args.mode == "fn" else "false-positive (false alarm)"
    fig.suptitle(f"Top {args.n} tiles by {metric_desc} count ({args.split} split)\n"
                 f"green=correct detection (GT)   red=missed detection (GT)   blue dashed=all model predictions",
                 fontsize=11)
    fig.tight_layout()
    grid_path = out_dir / f"top_{args.mode}_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {grid_path}")
    print(f"Individual tiles in {out_dir / 'individual'}/")


if __name__ == "__main__":
    main()