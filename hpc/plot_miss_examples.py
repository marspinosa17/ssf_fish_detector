#!/usr/bin/env python
"""
Render GT / YOLO / RT-DETR box overlays on spectrogram tiles -- either the
top N rows of a ranked CSV (delta_*.csv or top_discrepancies_*.csv from
miss_delta_analysis.py) or one specific tile by stem.

Box source: the per-model miss-analysis CSVs from analyze_misses_v3.py.
Every 'gt' row carries the matched (or nearest-candidate) prediction's
coords in pred_x1..pred_y2; every standalone 'pred'/FP row is an extra
unmatched prediction. This script unions both sources per model and
de-duplicates identical coordinates, so a near-miss candidate referenced
by more than one FN row isn't drawn twice.

Usage -- top N from a ranked CSV:
    python plot_miss_examples.py \
        --yolo-csv   runs/fish_yolo26s/miss_analysis_val_184463.csv \
        --rtdetr-csv runs/ul_rtdetr_l_v1/miss_analysis_val_184461.csv \
        --images-dir data/spectrograms/val/images \
        --rank-csv runs/detect/miss_delta/delta_rtdetr_vs_yolo_val.csv \
        --top-n 20 \
        --out-dir runs/detect/miss_delta/plots

Usage -- one specific tile:
    python plot_miss_examples.py \
        --yolo-csv   runs/fish_yolo26s/miss_analysis_val_184463.csv \
        --rtdetr-csv runs/ul_rtdetr_l_v1/miss_analysis_val_184461.csv \
        --images-dir data/spectrograms/val/images \
        --stem seth_CL_170420010502_000030000 \
        --out-dir runs/detect/miss_delta/plots --svg

--rank-csv accepts EITHER delta_<a>_vs_<b>_<split>.csv (mixed, sorted by
|delta|) or top_discrepancies_<a>_vs_<b>_<split>.csv (both tails, has a
'favors' column) -- only a 'stem' column is required; --top-n truncates
whatever order the file is already in.

Output: one PNG per tile (rank-prefixed in top-N mode so they sort in
delta order in a file browser), plus a .svg per tile if --svg is passed.
Each plot has real Time (s) / Frequency (Hz) axis ticks (using --window-dur
/ --f-max, matching PREPROCESSING.md's WINDOW_DUR/F_MAX) and, for YOLO/
RT-DETR boxes, a small confidence value printed at the box's corner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Distinguishable, colorblind-friendlier trio. GT and YOLO dashed per spec;
# RT-DETR solid so the eye can always tell it apart even in gray-scale prints.
GT_COLOR, GT_STYLE = "green", "-"
YOLO_COLOR, YOLO_STYLE = "red", "--"
RTDETR_COLOR, RTDETR_STYLE = "blue", "--"
LINEWIDTH = 2.0


# ───────────────────────────── CLI ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay GT/YOLO/RT-DETR boxes on spectrogram tiles")
    p.add_argument("--yolo-csv", required=True, type=Path,
                   help="YOLO's miss-analysis CSV (from analyze_misses_v3.py)")
    p.add_argument("--rtdetr-csv", required=True, type=Path,
                   help="RT-DETR's miss-analysis CSV (from analyze_misses_v3.py)")
    p.add_argument("--images-dir", required=True, type=Path,
                   help="e.g. data/spectrograms/val/images -- must match the split "
                        "the two CSVs above were generated from")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rank-csv", type=Path,
                       help="delta_*.csv or top_discrepancies_*.csv from miss_delta_analysis.py "
                            "(needs a 'stem' column)")
    mode.add_argument("--stem", nargs="+", help="plot one or more specific tiles by stem")
    p.add_argument("--top-n", type=int, default=None,
                   help="only with --rank-csv; default plots every row in the file")
    p.add_argument("--out-dir", type=Path, default=Path("miss_plots"))
    p.add_argument("--svg", action="store_true",
                   help="also save a .svg per tile (vector, editable) -- default off "
                        "since top-N mode can mean dozens of tiles")
    p.add_argument("--window-dur", type=float, default=3.0,
                   help="tile duration in seconds, for the time axis (match PREPROCESSING.md WINDOW_DUR)")
    p.add_argument("--f-max", type=float, default=2000.0,
                   help="tile's top frequency in Hz, for the frequency axis (match PREPROCESSING.md F_MAX)")
    p.add_argument("--no-conf-labels", dest="conf_labels", action="store_false", default=True,
                   help="hide the confidence value printed at each YOLO/RT-DETR box")
    return p.parse_args()


# ───────────────────────────── CSV loading ───────────────────────────────

def read_miss_csv(path: Path) -> pd.DataFrame:
    """Tab first (in case of a hand-edited/exported copy), else whatever
    pandas' own to_csv() default (comma) produced -- which is what
    analyze_misses_v3.py actually writes."""
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


def boxes_for_stem(df: pd.DataFrame, stem: str) -> tuple[list[tuple], list[tuple]]:
    """Return (gt_boxes, pred_boxes) for one stem from one model's miss CSV.

    gt_boxes are plain (x1,y1,x2,y2) tuples. pred_boxes are (x1,y1,x2,y2,conf)
    -- conf is carried through so it can be printed on the plot.

    pred_x1..pred_y2 show up on BOTH 'gt' rows (the matched TP, or the
    nearest FN candidate) and standalone 'pred'/FP rows (extra unmatched
    predictions) -- unioned and de-duplicated here (by coordinate only, so a
    near-miss candidate referenced by more than one FN row isn't drawn twice).
    """
    sub = df[df["stem"] == stem]
    gt_boxes, pred_boxes = [], []
    for row in sub.itertuples(index=False):
        row = row._asdict()
        if row.get("box_type") == "gt" and pd.notna(row.get("gt_x1")):
            gt_boxes.append(tuple(float(row[c]) for c in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")))
        if pd.notna(row.get("pred_x1")):
            conf = float(row["match_conf"]) if pd.notna(row.get("match_conf")) else None
            pred_boxes.append((float(row["pred_x1"]), float(row["pred_y1"]),
                               float(row["pred_x2"]), float(row["pred_y2"]), conf))

    def dedupe(boxes: list[tuple]) -> list[tuple]:
        seen, out = set(), []
        for b in boxes:
            key = tuple(round(v, 4) for v in b[:4])  # ignore conf in the de-dup key
            if key not in seen:
                seen.add(key)
                out.append(b)
        return out

    return dedupe(gt_boxes), dedupe(pred_boxes)


def gt_boxes_union(yolo_df: pd.DataFrame, rtdetr_df: pd.DataFrame, stem: str) -> list[tuple]:
    """GT should agree between the two models' CSVs (same underlying label
    file) -- union+dedupe defensively in case one is missing a row."""
    gt_a, _ = boxes_for_stem(yolo_df, stem)
    gt_b, _ = boxes_for_stem(rtdetr_df, stem)
    seen, out = set(), []
    for b in gt_a + gt_b:
        key = tuple(round(v, 4) for v in b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


# ───────────────────────────── image + plotting ──────────────────────────

def find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image found for stem {stem!r} in {images_dir}")


def plot_tile(stem: str, images_dir: Path, gt_boxes: list[tuple], yolo_boxes: list[tuple],
              rtdetr_boxes: list[tuple], title: str, out_path: Path, also_svg: bool,
              window_dur: float, f_max: float, conf_labels: bool) -> None:
    img_path = find_image(images_dir, stem)
    img = plt.imread(img_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Normalized box coords are [0,1] with y=0 at the TOP of the tile, which
    # PREPROCESSING.md renders as F_MAX Hz (the spectrogram is frequency-
    # flipped: top=F_MAX, bottom=0Hz). This extent plots boxes directly in
    # that space with no pixel conversion, and gives real Hz/second ticks.
    ax.imshow(img, extent=[0, 1, 1, 0], aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def draw(boxes: list[tuple], color: str, style: str, label: str) -> None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linestyle=style, linewidth=LINEWIDTH,
                label=label if i == 0 else None,
            ))
            conf = box[4] if len(box) > 4 else None
            if conf_labels and conf is not None:
                ax.text(x1, max(y1 - 0.015, 0.0), f"{conf:.2f}", color=color, fontsize=7,
                        va="bottom", ha="left",
                        bbox=dict(facecolor="black", alpha=0.55, pad=0.5, linewidth=0))

    draw(gt_boxes, GT_COLOR, GT_STYLE, "GT")
    draw(yolo_boxes, YOLO_COLOR, YOLO_STYLE, "YOLO")
    draw(rtdetr_boxes, RTDETR_COLOR, RTDETR_STYLE, "RT-DETR")

    xt = np.linspace(0, 1, 7)
    ax.set_xticks(xt); ax.set_xticklabels([f"{t * window_dur:.1f}" for t in xt])
    yt = np.linspace(0, 1, 5)
    ax.set_yticks(yt); ax.set_yticklabels([f"{(1 - t) * f_max:.0f}" for t in yt])
    ax.set_xlabel("Time in tile (s)"); ax.set_ylabel("Frequency (Hz)")

    ax.set_title(title, fontsize=9)
    if gt_boxes or yolo_boxes or rtdetr_boxes:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if also_svg:
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def build_rank_title(rank: int, total: int, stem: str, row: dict) -> str:
    parts = [f"[{rank + 1}/{total}]", stem]
    if pd.notna(row.get("source")):
        parts.append(f"source={row['source']}")
    if pd.notna(row.get("favors")):
        parts.append(f"favors={row['favors']}")
    if pd.notna(row.get("delta_score")):
        parts.append(f"\u0394score={row['delta_score']:+.0f}")
    return "  |  ".join(parts)


# ───────────────────────────── main ──────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    yolo_df = read_miss_csv(args.yolo_csv)
    rtdetr_df = read_miss_csv(args.rtdetr_csv)

    if args.stem:
        total = len(args.stem)
        for i, stem in enumerate(args.stem):
            gt_boxes = gt_boxes_union(yolo_df, rtdetr_df, stem)
            _, yolo_boxes = boxes_for_stem(yolo_df, stem)
            _, rtdetr_boxes = boxes_for_stem(rtdetr_df, stem)
            out_path = args.out_dir / f"{stem}.png"
            try:
                plot_tile(stem, args.images_dir, gt_boxes, yolo_boxes, rtdetr_boxes,
                          stem, out_path, also_svg=args.svg, window_dur=args.window_dur,
                          f_max=args.f_max, conf_labels=args.conf_labels)
            except FileNotFoundError as e:
                print(f"  skipping {stem}: {e}")
                continue
            print(f"[{i+1}/{total}] wrote {out_path.name}"
                  f"{' + .svg' if args.svg else ''}  "
                  f"(gt={len(gt_boxes)} yolo={len(yolo_boxes)} rtdetr={len(rtdetr_boxes)})")
        print(f"\nWrote plots to {args.out_dir}/")
        return

    rank_df = pd.read_csv(args.rank_csv)
    if "stem" not in rank_df.columns:
        raise ValueError(f"{args.rank_csv} has no 'stem' column (found: {list(rank_df.columns)})")
    if args.top_n:
        rank_df = rank_df.head(args.top_n)
    total = len(rank_df)

    for i, row in enumerate(rank_df.to_dict("records")):
        stem = row["stem"]
        gt_boxes = gt_boxes_union(yolo_df, rtdetr_df, stem)
        _, yolo_boxes = boxes_for_stem(yolo_df, stem)
        _, rtdetr_boxes = boxes_for_stem(rtdetr_df, stem)
        title = build_rank_title(i, total, stem, row)
        out_path = args.out_dir / f"{i:03d}_{stem}.png"
        try:
            plot_tile(stem, args.images_dir, gt_boxes, yolo_boxes, rtdetr_boxes,
                      title, out_path, also_svg=args.svg, window_dur=args.window_dur,
                      f_max=args.f_max, conf_labels=args.conf_labels)
        except FileNotFoundError as e:
            print(f"  skipping {stem}: {e}")
            continue
        print(f"[{i+1}/{total}] wrote {out_path.name} "
              f"(gt={len(gt_boxes)} yolo={len(yolo_boxes)} rtdetr={len(rtdetr_boxes)})")

    print(f"\nWrote plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
