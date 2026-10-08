#!/usr/bin/env python
"""
Overlaid PR and F1 curves for one or more Ultralytics detection models
(YOLO and/or RT-DETR checkpoints), across ANY number of named eval
directories -- not hardcoded to val/test. Adding a second test set (e.g.
Hawaii) later is just another --split flag, no code changes.

Unlike a hand-rolled matcher, this pulls the curve arrays straight out of
Ultralytics' own `model.val()` (via `metrics.box.curves_results`), so
AP/F1/precision/recall here are numerically IDENTICAL to what Ultralytics
already reported for your training runs -- this is the same code path,
not a second implementation that might drift from it. Verified directly
against ultralytics==8.4.115 source (utils/metrics.py: Metric.curves_results,
ap_per_class) -- if your cluster has a very different version installed,
run `python -c "import ultralytics; print(ultralytics.__version__)"` and
sanity-check the shapes printed by this script on first run.

Each named directory just needs the standard YOLO layout:
    <dir>/images/*.png (or .jpg)
    <dir>/labels/*.txt

BEFORE RUNNING: check runs/<name>/args.yaml for the imgsz each model was
actually trained at and pass it via --imgsz. Ultralytics' val() uses
rect=True (aspect-preserving batching) by default regardless, but the
imgsz value itself still needs to match training.

NOTE FOR AIR-GAPPED COMPUTE NODES: Ultralytics may try to download
Arial.ttf on first use even with plots=False. If your GPU nodes have no
internet access, run this once on the login node first (or anywhere with
outbound access) so it lands in ~/.config/Ultralytics/, then it's cached
for subsequent runs.

Usage:
    python pr_f1_curves.py \
        --model yolo   yolo   runs/fish_yolo26s/weights/best.pt \
        --model rtdetr rtdetr runs/ul_rtdetr_l_v1/weights/best.pt \
        --split val          data/spectrograms/val \
        --split test         data/spectrograms/test \
        --split test_hawaii  data/hawaii/test \
        --imgsz 640 \
        --out-dir runs/detect/pr_f1_curves

Each --model flag takes: NAME, TYPE (yolo|rtdetr), WEIGHTS_PATH.
Each --split flag takes: NAME, DIRECTORY (must contain images/ + labels/).

Outputs in --out-dir:
    curve_<model>_<split>.csv   one row per confidence-axis point (1000 pts):
                                 conf, f1, precision_at_conf, recall_at_conf,
                                 recall_axis, precision_at_recall
    curves_all.csv               the above, all combos, long format
    summary.csv                  map50, map, map75, best F1(+conf), per combo
    PR_curve.{svg,png,fig.pkl}
    F1_curve.{svg,png,fig.pkl}
    _tmp_yaml/                   the small per-split dataset.yaml files this
                                  script generates for model.val() -- kept
                                  around so you can rerun `yolo val data=...`
                                  yourself if you want to double check anything
    _val_artifacts/               Ultralytics' own val() run folders (plots=False,
                                  so these are mostly just logs)
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
    p = argparse.ArgumentParser(description="Overlaid PR/F1 curves, YOLO vs RT-DETR (Ultralytics native metrics)")
    p.add_argument("--model", nargs=3, action="append", required=True,
                   metavar=("NAME", "TYPE", "WEIGHTS"),
                   help="e.g. --model yolo yolo runs/fish_yolo26s/weights/best.pt "
                        "(TYPE is 'yolo' or 'rtdetr')")
    p.add_argument("--split", nargs=2, action="append", required=True,
                   metavar=("NAME", "DIRECTORY"),
                   help="e.g. --split test_hawaii data/hawaii/test "
                        "(DIRECTORY must contain images/ and labels/ subfolders)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="MUST match training imgsz -- check runs/<name>/args.yaml")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001,
                   help="confidence floor passed to val() -- keep low for the full curve")
    p.add_argument("--nms-iou", type=float, default=0.7, help="model's own NMS IoU (YOLO only)")
    p.add_argument("--half", action="store_true", help="fp16 inference (faster, default off)")
    p.add_argument("--color", nargs=2, action="append", metavar=("MODEL", "COLOR"),
                   help="override a model's line color (any matplotlib color name/hex), e.g. "
                        "--color yolo red --color rtdetr blue -- these are already the defaults, "
                        "matching plot_miss_examples.py's GT=green/YOLO=red/RT-DETR=blue scheme "
                        "(GT has no curve here, so only yolo/rtdetr matter)")
    p.add_argument("--out-dir", type=Path, default=Path("runs/detect/pr_f1_curves"))
    return p.parse_args()


# ───────────────────────────── model / dataset plumbing ─────────────────

def load_model(model_type: str, weights: Path):
    if model_type == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(str(weights))
    elif model_type == "yolo":
        from ultralytics import YOLO
        return YOLO(str(weights))
    raise ValueError(f"unknown model type {model_type!r} (use 'yolo' or 'rtdetr')")


def build_temp_yaml(name: str, split_dir: Path, tmp_dir: Path) -> Path:
    """One tiny dataset.yaml per named split, so model.val() can point at
    an arbitrary directory instead of a fixed project-wide dataset.yaml.

    `train:` is required by Ultralytics' YAML schema check even though
    we never train from it -- pointed at the same images/ folder, unused.
    """
    split_dir = split_dir.resolve()
    if not (split_dir / "images").is_dir():
        raise FileNotFoundError(f"{split_dir} has no images/ subdirectory")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = tmp_dir / f"{name}.yaml"
    yaml_path.write_text(
        f"path: {split_dir.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['fish']\n"
    )
    return yaml_path


def run_val(model, yaml_path: Path, model_name: str, split_name: str, args):
    return model.val(
        data=str(yaml_path), split="val",
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=args.conf, iou=args.nms_iou, half=args.half,
        plots=False, save_json=False, verbose=True,
        project=str(args.out_dir / "_val_artifacts"),
        name=f"{model_name}_{split_name}", exist_ok=True,
    )


# ───────────────────────────── curve extraction ──────────────────────────

def extract_curve_df(metrics) -> pd.DataFrame:
    """Pull the 4 curves Ultralytics already computed for this val() call.

    metrics.box.curves_results == [
        [px, prec_values, "Recall",     "Precision"],  # PR curve
        [px, f1_curve,    "Confidence", "F1"],
        [px, p_curve,     "Confidence", "Precision"],
        [px, r_curve,     "Confidence", "Recall"],
    ]
    px is the same 0..1, 1000-point grid reused as two different physical
    axes (confidence for 3 of the 4 curves, recall for the PR curve) --
    that's Ultralytics' own convention, not a bug here. Each py array is
    shape (nc, 1000); nc=1 (fish only) so we take row 0.
    """
    curves = metrics.box.curves_results
    px_pr, prec_values = np.asarray(curves[0][0]), np.asarray(curves[0][1])
    px_f1, f1_curve = np.asarray(curves[1][0]), np.asarray(curves[1][1])
    _, p_curve = np.asarray(curves[2][0]), np.asarray(curves[2][1])
    _, r_curve = np.asarray(curves[3][0]), np.asarray(curves[3][1])

    def row0(arr: np.ndarray, like: np.ndarray) -> np.ndarray:
        return arr[0] if arr.ndim == 2 and arr.shape[0] else np.zeros_like(like)

    return pd.DataFrame({
        "conf": px_f1,
        "f1": row0(f1_curve, px_f1),
        "precision_at_conf": row0(p_curve, px_f1),
        "recall_at_conf": row0(r_curve, px_f1),
        "recall_axis": px_pr,                       # same grid, read as recall for the next column
        "precision_at_recall": row0(prec_values, px_pr),
    })


def best_f1_conf(curve: pd.DataFrame) -> tuple[float, float]:
    """Replicates Ultralytics' own best-F1 selection (smoothed argmax) so
    this number matches what you'd see in a normal training/val log."""
    from ultralytics.utils.metrics import smooth
    f1 = curve["f1"].to_numpy()
    if f1.size == 0:
        return 0.0, 0.0
    i = int(smooth(f1, 0.1).argmax())
    return float(f1[i]), float(curve["conf"].iloc[i])


def scalar_summary(metrics) -> dict:
    box = metrics.box
    return dict(
        map50=float(box.map50), map=float(box.map),
        map75=float(getattr(box, "map75", float("nan"))),
        mp=float(box.mp), mr=float(box.mr),
    )


# ───────────────────────────── plotting ──────────────────────────────────

LINESTYLES = ["-", "--", "-.", ":"]

# Cross-figure-consistent defaults: same red=YOLO/blue=RT-DETR mapping used
# in plot_miss_examples.py (GT doesn't apply here -- a PR/F1 curve has no
# GT line, only model lines). Override via --color, e.g. --color yolo darkred.
DEFAULT_MODEL_COLORS = {"yolo": "firebrick", "rtdetr": "mediumslateblue"}
FALLBACK_PALETTE = [plt.cm.tab10(i) for i in range(10)]  # used only for model names not above


def resolve_model_colors(models: list[str], overrides: dict[str, str] | None) -> dict[str, str]:
    """Explicit per-model color map: DEFAULT_MODEL_COLORS, then --color
    overrides, then a tab10 fallback for any model name neither covers (so
    adding a third model later doesn't crash this)."""
    color_of = dict(DEFAULT_MODEL_COLORS)
    if overrides:
        color_of.update(overrides)
    fallback_i = 0
    for m in models:
        if m not in color_of:
            color_of[m] = FALLBACK_PALETTE[fallback_i % len(FALLBACK_PALETTE)]
            fallback_i += 1
    return color_of


def plot_curves(curves: dict, split_order: list[str], out_dir: Path,
                color_of: dict[str, str]) -> None:
    style_of = {s: LINESTYLES[i % len(LINESTYLES)] for i, s in enumerate(split_order)}

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_f1, ax_f1 = plt.subplots(figsize=(6, 5))

    for (model, split), curve in curves.items():
        if curve.empty:
            continue
        label = f"{model} / {split}"
        style = style_of.get(split, ":")
        ax_pr.plot(curve["recall_axis"], curve["precision_at_recall"], style,
                   color=color_of[model], label=label, linewidth=1.8)
        ax_f1.plot(curve["conf"], curve["f1"], style,
                   color=color_of[model], label=label, linewidth=1.8)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.set_title("Precision–Recall (IoU=0.5)"); ax_pr.legend(fontsize=8); ax_pr.grid(alpha=0.3)

    ax_f1.set_xlabel("Confidence threshold"); ax_f1.set_ylabel("F1")
    ax_f1.set_xlim(0, 1); ax_f1.set_ylim(0, 1.02)
    ax_f1.set_title("F1 vs. confidence (IoU=0.5)"); ax_f1.legend(fontsize=8); ax_f1.grid(alpha=0.3)

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
    tmp_yaml_dir = args.out_dir / "_tmp_yaml"

    split_order = [name for name, _ in args.split]  # preserves CLI order for linestyle assignment
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows, long_rows = [], []

    for model_name, mtype, weights in args.model:
        print(f"\n=== loading {model_name} ({mtype}) from {weights} ===")
        model = load_model(mtype, Path(weights))

        for split_name, split_dir in args.split:
            print(f"--- {model_name} / {split_name} ({split_dir}) ---")
            yaml_path = build_temp_yaml(split_name, Path(split_dir), tmp_yaml_dir)
            metrics = run_val(model, yaml_path, model_name, split_name, args)

            curve = extract_curve_df(metrics)
            curves[(model_name, split_name)] = curve
            f1, f1_conf = best_f1_conf(curve)
            summary_rows.append(dict(model=model_name, split=split_name,
                                     best_f1=f1, best_f1_conf=f1_conf,
                                     **scalar_summary(metrics)))
            print(f"    map50={summary_rows[-1]['map50']:.4f}  "
                  f"best F1={f1:.4f} @ conf={f1_conf:.3f}")

            curve.to_csv(args.out_dir / f"curve_{model_name}_{split_name}.csv", index=False)
            curve_out = curve.copy()
            curve_out.insert(0, "split", split_name)
            curve_out.insert(0, "model", model_name)
            long_rows.append(curve_out)

    pd.concat(long_rows, ignore_index=True).to_csv(args.out_dir / "curves_all.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    color_overrides = {name: color for name, color in args.color} if args.color else None
    color_of = resolve_model_colors([m for m, _, _ in args.model], color_overrides)
    plot_curves(curves, split_order, args.out_dir, color_of)
    print(f"\nWrote curves + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()