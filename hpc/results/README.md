# HPC results: YOLO26s vs RT-DETR-L

Comparison of the two Ultralytics models (`models/yolo/fish_yolo26s`,
`models/rtdetr/ul_rtdetr_l_v1`) on the val and test splits, plus the
held-out Hawaii set for the PR/F1 curves.

## `metrics.txt`
Ultralytics `model.val()` mAP50, mAP50-95, precision and recall per model and
split, the model sizes (YOLO26s: 9.9M params / 22.5 GFLOPs; RT-DETR-L: 32.8M
params / 108.0 GFLOPs), and a copy of `pr_f1_curves/summary.csv`.

## `pr_f1_curves/`
Produced by [`hpc/pr_f1_curves.py`](../pr_f1_curves.py) (slurm: `run_pr_f1_curves`).
Curves come straight from Ultralytics' `model.val()`, so they match the
training-run metrics.

- `summary.csv`: mAP50 / mAP / mAP75, mean P/R, and best F1 with its confidence
  threshold per model and split. The miss analysis below runs each model near
  its val best-F1 threshold (YOLO about 0.22, RT-DETR about 0.45).
- `curve_<model>_<split>.csv`, `curves_all.csv`: the raw curve arrays.
- `PR_curve.{svg,png}`, `F1_curve.{svg,png}`: overlaid curves for all
  model/split combinations.

## `miss_delta/`
Per-tile comparison of where the two models disagree.

1. [`hpc/analyze_misses.py`](../analyze_misses.py) runs each model at its
   operating threshold and labels every box TP / FN / FP (match IoU >= 0.5).
2. [`hpc/miss_delta.py`](../miss_delta.py) aggregates those per tile
   (score = TP - FN - FP), joins the two models, and writes:
   - `per_source_metrics.csv`: P/R/F1 by data source (seth, tagus, xavier).
   - `recall_by_duration.csv`, `recall_by_bandwidth.csv`: recall in 5 quantile
     bins of call duration and frequency bandwidth.
   - `top_discrepancies_rtdetr_vs_yolo_<split>.csv`: the 40 tiles favoring each
     model most (`favors` column).

   The full per-tile tables (`tile_scores_*`, `delta_*`, `*_detail.csv`, about
   23 MB) are not committed; rerun `miss_delta.py` on the step-1 CSVs to
   regenerate them.
3. `plots/`: spectrogram tiles with GT, YOLO and RT-DETR boxes, from
   [`hpc/plot_miss_examples.py`](../plot_miss_examples.py). `000`–`019` are the
   top 20 val tiles ranked by |score difference|; `seth_YA_170411233002_000005000.png`
   is a single hand-picked tile.
