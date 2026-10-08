# Phase 1 Preprocessing Pipeline

Converts raw underwater audio + `data/manifest.csv` into cached spectrogram images and YOLO/COCO labels for fish-detection training.

## Run

```bash
conda activate fd_framework
cd ssf_fish_detector

python -m preprocessing.preprocess            # full run (resume-safe)
python -m preprocessing.preprocess --smoke    # 2 files/source for a quick check
python -m preprocessing.preprocess --limit-files 30  # larger sample
python -m preprocessing.preprocess --cpu      # force CPU
python -m preprocessing.preprocess --splits hawaii   # only process one split's files
```

## What it produces

```
data/spectrograms/
├── train/images/   val/images/   test/images/   hawaii/images/   ← 640×512 RGB PNG spectrograms
├── train/labels/   val/labels/   test/labels/   hawaii/labels/   ← YOLO .txt (empty = negative)
├── dataset_manifest.csv   ← one row per output image with full metadata
├── dataset.yaml           ← YOLO training config
├── coco_train/val/test/hawaii.json  ← COCO format for DETR
└── processed_files.log    ← resume checkpoint (one audio_path per line)
```

Every file is tiled end-to-end at a fixed 1.0 s stride (whole-file tiling) —
see "Key design decisions" below. train's negative tiles are subsampled
per source to a 3:1 neg:pos ratio; val/test/hawaii keep every tile.

**hawaii split:** a standalone eval-only subset (Olowalu, HI), never divided
into train/val/test — every hawaii row in the manifest already carries
`split=hawaii` (see `data/build_manifest.py`). It's processed like val/test
(no negative subsampling, every tile kept). Use `--splits hawaii` to
(re)process just this split without touching train/val/test's cached output.

## Spectrogram parameters (`config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `TARGET_SR` | 4 000 Hz | All sources resampled to this |
| `N_FFT` / `HOP_LENGTH` | 256 / 64 | 16 ms hop at 4 kHz |
| `F_MAX` | 2 000 Hz | Nyquist |
| `WINDOW_DUR` | 3.0 s | Context window per tile |
| `IMG_WIDTH` × `IMG_HEIGHT` | 640 × 512 px | Time × frequency |

Output shape: 129 frequency bins × 188 time frames per raw slice, resized to 640×512.

**Changing these parameters invalidates any already-written images** — delete `data/spectrograms/` and rerun.

## Coordinate system

- x-axis (width): time — left = `window_start_s`, right = `window_end_s`
- y-axis (height): frequency — **top = 2000 Hz, bottom = 0 Hz** (image is flipped)
- YOLO class `0` = fish; normalized `[0, 1]` coords

## Key design decisions

**Normalization:** Per-recording 5th–95th percentile dB normalization → uint8. A `< 10 s` short-file fallback exists (fixed bounds) but never triggers — all sources have long continuous recordings (Tagus 240–1816 s, Seth ~63 s, Xavier 300–1800 s).

**Whole-file tiling:** Every file is tiled from t=0 to end-of-file at a fixed `STRIDE_S` (1.0 s) stride, plus one final tile ending exactly at file duration so the tail of the file isn't left uncovered by the stride grid. No jitter, no per-annotation centering, no window deduplication — the stride grid itself already gives dense, uniform coverage, including of densely-annotated regions (e.g. Seth chorus clusters), so a call is never over- or under-represented relative to how often it's actually heard.

**Box inclusion per tile:** A fish call gets a box in a tile only if the visible slice covers >= 50% of the call's own duration AND >= 150 ms of it absolutely. Calls meeting neither are left out of that tile's labels entirely (never written as a truncated sliver). A tile can and does carry multiple boxes when multiple calls each clear the threshold.

**NA (unresolved `is_fish`) exclusion:** Any tile overlapping a span whose `is_fish` is NA is dropped from the dataset entirely — fish presence there is unknown, so it's neither a safe positive nor a safe negative. This currently applies to two labels: Xavier's `UN` (uncertain) and Seth's `chorus`. Chorus was added to this exclusion after a held-out miss-analysis showed it's a poor fit for box-based detection (~19% recall vs. 74–88% for every other label, lower confidence and IoU on the true positives it does get), consistent with chorus being diffuse, overlapping, continuous vocalization rather than a discrete single-fish call that fits a tight bounding box.

**Negative subsampling:** All positive tiles are kept. Negative tiles are subsampled per source, **train split only**, down to `NEG_POS_TRAIN_RATIO` (3:1) via a deterministic per-tile Bernoulli draw (seeded off the tile's own stem — order-independent, resume-safe). val/test keep every negative tile, since that's what makes their class balance deployment-realistic.

**Resume:** `processed_files.log` tracks completed audio files. Rerunning skips them. Changing `WINDOW_DUR`, `STRIDE_S`, the overlap thresholds, or `NEG_POS_TRAIN_RATIO` invalidates the cache — delete `data/spectrograms/` and rerun rather than resuming.

## Test set realism

`test/images/`, like train and val, is whole-file tiles at the same fixed stride used everywhere else — the same thing a sliding-window inference pass over a full recording would produce. There is no separate "centered extraction" test path and no optimism gap to correct for; test metrics computed directly from `test/images/` already reflect deployment-realistic class balance and call framing.

## Environment

```
conda activate fd_framework
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python-headless soundfile
# pandas, numpy, tqdm already present
```

Tested: torch 2.6.0+cu124, torchaudio 2.6.0, soundfile 0.14.0
`soundfile` is required — torchaudio 2.6 ships no default audio backend on Windows.
