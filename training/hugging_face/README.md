# RT-DETRv2 fine-tuning (HuggingFace transformers)

A second, independent RT-DETR implementation for the PAM fish-call detection
project, alongside the existing YOLO / Ultralytics-RT-DETR training. This one
fine-tunes `RTDetrV2ForObjectDetection` via HuggingFace `transformers`
directly, so the comparison doesn't depend on Ultralytics' training
internals or its AGPL license.

Reads the tiles already produced by `preprocessing/preprocess.py`
(`data/spectrograms/{train,val,test}/images/*.png` +
`coco_{train,val,test}.json` + `dataset_manifest.csv`). This module is
**read-only** with respect to `preprocessing/` and `data/` — it never
modifies the cached tiles, labels, or manifest.

Single class throughout: `fish` (id 0).

## Setup

```bash
conda activate fd_framework
cd ssf_fish_detector

pip install -r requirements.txt
# match torchvision to your installed torch/CUDA build, e.g. on the H200 node:
pip install torchvision --index-url https://download.pytorch.org/whl/cu130
```

Requires `data/spectrograms/` to already exist (run
`preprocessing/preprocess.py` first — see `preprocessing/PREPROCESSING.md`).

## Smoke test (a few minutes)

Trains ~20 steps on a tiny per-source subset, then evaluates it, to catch
pipeline breakage fast:

```bash
python -m training.hugging_face.train --smoke --output-dir models/rtdetr_hf/smoke
python -m training.hugging_face.eval --model-dir models/rtdetr_hf/smoke/final --split val --smoke
python -m training.hugging_face.predict_samples --model-dir models/rtdetr_hf/smoke/final --split val --n-per-source 2
```

## Full training run

```bash
python -m training.hugging_face.train \
    --epochs 50 --batch-size 16 --lr 1e-4 --backbone-lr-mult 0.1 \
    --output-dir models/rtdetr_hf/run1
```

- Loads `PekingU/rtdetr_v2_r50vd` (override with `--checkpoint`) via
  `AutoModelForObjectDetection`, remapped to 1 class
  (`ignore_mismatched_sizes=True` drops COCO's 80-class head).
- Backbone parameters use `lr * --backbone-lr-mult` (default 0.1x); the rest
  of the model (encoder/decoder/detection heads) use the full `--lr` —
  standard DETR-family practice since the backbone starts from stronger
  pretraining than the head.
- Resume: `--resume-from-checkpoint auto` picks up the latest checkpoint
  under `--output-dir`, or pass an explicit checkpoint path.
- Logs to stdout and `--output-dir/train.log` (same pattern as
  `preprocessing/preprocess.py`'s `setup_logging()`).
- Final model + image processor saved to `--output-dir/final/`.

### Augmentations

Off by default; enable explicitly via CLI flags. See `augmentations.py` for
the full rationale.

| Flag | Default | Notes |
|---|---|---|
| `--hflip-prob` | 0.0 | Time-axis flip. Plausibly valid but **unvalidated** — check it doesn't hurt val mAP before trusting it. |
| `--scale-jitter-prob` | 0.0 | Random crop-and-resize scale jitter. |

Vertical flip is **not implemented and must never be added** — it would
invert the frequency axis (top = 2000 Hz, bottom = 0 Hz per
`preprocessing/PREPROCESSING.md`), which is not a valid symmetry for this
data. Mosaic and hue/saturation/color jitter are also intentionally absent —
they're photo-oriented augmentations that don't map to spectrogram
semantics (the RGB channels here are a grayscale power-spectrum rendering,
not independent color information).

## Evaluation

```bash
python -m training.hugging_face.eval \
    --model-dir models/rtdetr_hf/run1/final --split test --iou-threshold 0.5
```

Produces, under `--model-dir/eval_{split}/`:

- `eval_summary.json` — overall mAP (torchmetrics COCO-style: mAP, mAP@50,
  mAP@75, ...) and per-source mAP (xavier/seth/tagus), plus PR/F1 AP and
  best-F1 operating point.
- `PR_curve.png` / `F1_curve.png` — overall, single-class ("fish").
- `PR_curve_{source}.png` / `F1_curve_{source}.png` — per source.

Per-source metrics use `dataset_manifest.csv`'s `source` column to map each
image back to its source, mirroring the per-source guardrail check in
`data/build_manifest.py` — this is specifically to catch a source-specific
weakness that an overall blended mAP could hide.

**mAP implementation choice:** uses `torchmetrics.detection.MeanAveragePrecision`
rather than a `pycocotools.cocoeval` wrapper. This avoids a JSON
serialize/reparse round trip and gives direct tensor access for the
per-source split and the PR/F1 sweep (which reuses the same predictions).
Tradeoff: torchmetrics is a reimplementation of the COCO metric, not the
reference `COCOeval` code — expect close but not necessarily bit-identical
numbers to a strict `pycocotools` run; cross-check with `COCOeval` directly
if that distinction matters for a specific comparison.

**PR/F1 curves:** built from scratch in `pr_curves.py` via greedy
IoU-threshold matching per image (single class, so this is simpler than
COCOeval's multi-class machinery), matching Ultralytics'
`BoxPR_curve.png` / `BoxF1_curve.png` visual style. `--iou-threshold`
(default 0.5) controls the TP/FP cutoff for both the curves and the
`predictions_sample` coloring.

## Prediction sample visualization

```bash
python -m training.hugging_face.predict_samples \
    --model-dir models/rtdetr_hf/run1/final --split val --n-per-source 8
```

Writes `--model-dir/predictions_sample/{source}/*.png` — GT boxes in green,
true-positive predictions in orange, false-positive predictions in red
(labeled with confidence). Samples are drawn across true-positive,
false-positive, and false-negative buckets per source (not purely random),
so the grid actually surfaces failure modes.

Coordinate system: the PNG pixels already encode the frequency-axis flip
(`preprocessing/window_utils.py:render_image` does `np.flipud` before
writing, so row 0 = 2000 Hz, bottom row = 0 Hz). Both GT boxes (from
`coco_{split}.json`) and predicted boxes (from the model, run against that
same PNG) live in this already-flipped pixel space, so drawing with
standard top-left-origin image coordinates is correct without any further
transform. `predict_samples.py` also runs a cheap runtime sanity check
(`_sanity_check_orientation`) that warns if a tile's brightness distribution
looks inverted from the expected low-frequency-heavy pattern.

## What this module does manually that Ultralytics would have handled automatically

- **COCO-format data loading and batching**: Ultralytics' `YOLODataset` /
  `RTDETRDataset` handle image loading, label parsing, and collation
  internally. Here, `dataset.py` wraps `pycocotools.coco.COCO` directly and
  `collate_fn` explicitly handles the variable-length box lists HF's
  `RTDetrImageProcessor` expects.
- **LR scheduling / warmup / param groups**: Ultralytics configures
  optimizer param groups (including differential LR for the backbone) and
  warmup internally from `args.yaml`. Here, `train.py` builds the backbone
  vs. head param groups and passes a `torch.optim.AdamW` + HF
  `TrainingArguments`' cosine scheduler explicitly.
- **Checkpointing / resume**: Ultralytics' `--resume` auto-detects
  `last.pt`. Here, `--resume-from-checkpoint auto` uses
  `transformers.trainer_utils.get_last_checkpoint` to do the equivalent.
- **Validation metrics (mAP, PR curve, F1 curve, confusion matrix)**:
  Ultralytics computes and plots these automatically after every training
  run. Here, `eval.py` and `pr_curves.py` reimplement COCO-style mAP
  (via torchmetrics) and single-class PR/F1 curves from scratch, styled to
  match Ultralytics' output filenames/format for visual comparability. Note:
  there is no confusion-matrix or `BoxP_curve.png`/`BoxR_curve.png`
  equivalent implemented here — only `PR_curve.png`/`F1_curve.png`, per the
  original task scope.
- **Prediction visualization grids**: Ultralytics writes
  `val_batch*_labels.jpg` / `val_batch*_pred.jpg` automatically. Here,
  `predict_samples.py` reimplements GT-vs-prediction overlay rendering from
  scratch, including the bucketed (TP/FP/FN) sampling strategy and the
  explicit frequency-axis orientation check.
- **Multi-GPU / mixed precision plumbing**: handled here via HF `Trainer`
  (which wraps `Accelerate` internally) rather than `Accelerate`'s
  lower-level API directly — a single-node H200 run doesn't need the extra
  manual control Accelerate's raw API would add over what `Trainer` already
  provides.
