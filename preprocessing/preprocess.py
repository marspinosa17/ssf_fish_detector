"""preprocessing entry point.

Converts raw underwater audio + manifest.csv into cached spectrogram tiles and
YOLO/COCO labels for fish-detection training.

Run:
    python -m preprocessing.preprocess            # full pipeline
    python preprocessing/preprocess.py            # same (script-style import)
    python -m preprocessing.preprocess --smoke    # tiny subset for validation

TILING: every file is tiled at a fixed stride (config.STRIDE_S) from t=0 to
end-of-file — this is deployment-realistic by construction, since it's the
same thing a sliding-window inference pass would see, for every split
including test (see write_dataset_yaml). A tile is "positive" if >=1 fish
call clears the overlap threshold in it, "negative" otherwise. Tiles
overlapping an annotation whose is_fish is unresolved/NA (currently Xavier's
UN label and Seth's chorus calls) are dropped entirely — fish presence there
is unknown, so they're neither a safe positive nor a safe negative.
Negative tiles are subsampled per source, TRAIN SPLIT ONLY, down to
config.NEG_POS_TRAIN_RATIO; val/test keep every tile so their metrics reflect
the real, natural class balance of a deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config, audio_utils, window_utils, annotation_utils, image_writer, coco_converter
except ImportError:
    import config, audio_utils, window_utils, annotation_utils, image_writer, coco_converter

logger = logging.getLogger("preprocess")
SOURCES = ("xavier", "seth", "tagus", "hawaii")
SPLITS = ("train", "val", "test", "hawaii")


# ── Helpers ──────────────────────────────────────────────────────────────────
def setup_logging(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_path / "preprocess.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # surface audio_utils warnings (e.g. short-file path) through the same handlers
    logging.getLogger("preprocessing.audio_utils").setLevel(logging.INFO)
    logging.getLogger("preprocessing.audio_utils").addHandler(fh)
    logging.getLogger("preprocessing.audio_utils").addHandler(sh)


def _seed_from(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ config.RANDOM_SEED) & 0xFFFFFFFF


def tile_rng(stem: str) -> np.random.Generator:
    """Deterministic, order-independent RNG per tile (resume-safe).

    Used only for the train-split negative-subsampling keep/skip draw — a
    pure function of the tile's own identity, so it doesn't matter which
    order files are (re)processed in.
    """
    return np.random.default_rng(_seed_from(stem))


def make_dirs(output_path: Path) -> None:
    for split in SPLITS:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)


def stem_of(audio_path: str) -> str:
    return Path(audio_path).stem


def window_stem(source: str, audio_stem: str, window_start_s: float, neg: bool) -> str:
    ms = int(round(window_start_s * 1000))
    suffix = "_neg" if neg else ""
    # zero-pad to 9 digits; negative (boundary) windows keep their sign
    body = f"{ms:09d}" if ms >= 0 else f"-{abs(ms):08d}"
    return f"{source}_{audio_stem}_{body}{suffix}"


def _overlaps_any(intervals: list[tuple], start_s: float, end_s: float) -> bool:
    return any(iv_begin < end_s and iv_end > start_s for iv_begin, iv_end in intervals)


# ── Per-file index structures ────────────────────────────────────────────────
class Indexes:
    """Pre-built per-file lookup tables."""

    def __init__(self, df: pd.DataFrame):
        self.fish_index = annotation_utils.build_annotation_index(df)

        # Spans where is_fish is NA (unresolved) — currently Xavier's UN label
        # and Seth's chorus calls — neither a safe positive nor a safe negative.
        self.ignore: dict[str, list[tuple]] = defaultdict(list)
        for row in df.itertuples(index=False):
            if pd.isna(row.is_fish):
                self.ignore[row.audio_path].append((row.begin_s, row.end_s))

        # file -> split / source (first row wins; consistent per file by construction)
        self.file_split: dict[str, str] = {}
        self.file_source: dict[str, str] = {}
        for ap, grp in df.groupby("audio_path"):
            self.file_split[ap] = grp["split"].iloc[0]
            self.file_source[ap] = grp["source"].iloc[0]


# ── Pre-pass: durations + tile classification (no audio decode needed) ──────
def prepass(idx: Indexes, files: list[str]) -> dict:
    """Classify every tile of every file as kept-positive / kept-negative / dropped.

    Classification only needs file duration + annotation timestamps, so this
    runs over the whole dataset up front, cheaply, without decoding any
    audio. The result is reused verbatim in process_file so the actual write
    pass never re-derives it.
    """
    durations: dict[str, float] = {}
    # ap -> list of (window_start_s, is_positive) for tiles that survive NA exclusion
    tile_class: dict[str, list[tuple[float, bool]]] = {}
    n_raw_tiles = 0

    logger.info("Pre-pass: reading durations and classifying tiles for %d files", len(files))
    for ap in tqdm(files, desc="prepass", unit="file"):
        try:
            dur = audio_utils.get_file_duration(ap)
        except Exception as e:  # noqa: BLE001
            logger.warning("Pre-pass duration failed for %s: %s", ap, e)
            durations[ap] = None
            continue
        durations[ap] = dur

        starts = window_utils.tile_starts(dur)
        n_raw_tiles += len(starts)
        ignore_ivals = idx.ignore.get(ap, [])
        fish_ivals = idx.fish_index.get(ap, [])

        kept: list[tuple[float, bool]] = []
        for ws in starts:
            we = ws + config.WINDOW_DUR
            if _overlaps_any(ignore_ivals, ws, we):
                continue  # fish presence unknown here — drop the tile entirely
            anns = annotation_utils.find_overlapping_annotations(fish_ivals, ws, we)
            kept.append((ws, len(anns) > 0))
        tile_class[ap] = kept

    # Per-source negative keep-probability for train (largest ratio we can
    # afford while keeping ALL positives; val/test are never subsampled).
    neg_keep_prob: dict[str, float] = {}
    for source in SOURCES:
        train_files = [f for f in files
                       if idx.file_source.get(f) == source and idx.file_split.get(f) == "train"]
        n_pos = sum(1 for f in train_files for _, is_pos in tile_class.get(f, []) if is_pos)
        n_neg = sum(1 for f in train_files for _, is_pos in tile_class.get(f, []) if not is_pos)
        target_neg = round(n_pos * config.NEG_POS_TRAIN_RATIO)
        keep_prob = 1.0 if n_neg == 0 or n_neg <= target_neg else target_neg / n_neg
        neg_keep_prob[source] = keep_prob
        logger.info("Source %-6s train: pos=%d neg_available=%d target_neg=%d keep_prob=%.4f",
                    source, n_pos, n_neg, target_neg, keep_prob)

    n_pos_total = sum(1 for f in files for _, is_pos in tile_class.get(f, []) if is_pos)
    n_kept_total = sum(len(v) for v in tile_class.values())
    logger.info("Pre-pass: %d raw stride tiles -> %d survive NA exclusion (%d positive)",
                n_raw_tiles, n_kept_total, n_pos_total)

    return {
        "durations": durations, "tile_class": tile_class,
        "neg_keep_prob": neg_keep_prob, "n_pos_total": n_pos_total,
        "n_raw_tiles": n_raw_tiles, "n_kept_total": n_kept_total,
    }


def project_final_tile_count(idx: Indexes, pre: dict, files: list[str]) -> int:
    """Expected final written-tile count: all positives + subsampled train
    negatives (by keep_prob) + full val/test negatives."""
    n_pos = pre["n_pos_total"]
    n_neg_final = 0.0
    for source in SOURCES:
        keep_prob = pre["neg_keep_prob"].get(source, 1.0)
        for f in files:
            if idx.file_source.get(f) != source:
                continue
            is_train = idx.file_split.get(f) == "train"
            for _, is_pos in pre["tile_class"].get(f, []):
                if is_pos:
                    continue
                n_neg_final += keep_prob if is_train else 1.0
    return int(round(n_pos + n_neg_final))


def check_storage(output_path: Path, n_total_projected: int, no_prompt: bool) -> None:
    est_gb = n_total_projected * config.EST_BYTES_PER_PNG / 1e9
    free_gb = shutil.disk_usage(output_path).free / 1e9
    logger.info("Storage estimate: ~%d images, ~%.1f GB needed; %.1f GB free",
                n_total_projected, est_gb, free_gb)
    if free_gb < est_gb * 1.5 and not no_prompt:
        logger.warning("Low free space (need ~%.1f GB, have %.1f GB). Continue? [y/N]",
                       est_gb, free_gb)
        if input().strip().lower() != "y":
            logger.info("Aborted by user.")
            sys.exit(0)


# ── Main per-file processing ─────────────────────────────────────────────────
def process_file(ap: str, idx: Indexes, pre: dict, device) -> list[dict]:
    """Process one audio file; returns dataset_manifest rows for it."""
    dur = pre["durations"].get(ap)
    if dur is None:
        return []
    source = idx.file_source[ap]
    split = idx.file_split[ap]
    audio_stem = stem_of(ap)
    images_dir = config.OUTPUT_PATH / split / "images"
    labels_dir = config.OUTPUT_PATH / split / "labels"

    try:
        waveform, _ = audio_utils.load_and_resample(ap)
    except Exception as e:  # noqa: BLE001
        logger.warning("Load failed for %s: %s", ap, e)
        return []
    S_power = audio_utils.compute_spectrogram(waveform, device)
    S_uint8, p_low, p_high = audio_utils.normalize_spectrogram(S_power, dur, source)

    file_index = idx.fish_index.get(ap, [])
    keep_prob = pre["neg_keep_prob"].get(source, 1.0)
    rows: list[dict] = []
    n_pos_img = n_neg_img = n_neg_skipped = n_boxes = 0

    for ws, is_positive in pre["tile_class"].get(ap, []):
        we = ws + config.WINDOW_DUR
        stem = window_stem(source, audio_stem, ws, neg=not is_positive)

        if not is_positive and split == "train":
            if tile_rng(stem).random() >= keep_prob:
                n_neg_skipped += 1
                continue

        sl, pad_l, pad_r = window_utils.extract_slice(S_uint8, ws, dur)
        if is_positive:
            anns = annotation_utils.find_overlapping_annotations(file_index, ws, we)
            lines = [annotation_utils.annotation_to_yolo_line(a, ws) for a in anns]
            n_pos_img += 1
            n_boxes += len(lines)
        else:
            lines = []
            n_neg_img += 1

        image_writer.write_image_and_label(sl, lines, images_dir, labels_dir, stem)
        rows.append(_manifest_row(stem, split, source, ap, ws, we, not is_positive,
                                  len(lines), pad_l, pad_r, p_low, p_high))

    logger.info("%s [%s/%s] pos=%d neg=%d neg_skipped=%d boxes=%d", audio_stem, source, split,
                n_pos_img, n_neg_img, n_neg_skipped, n_boxes)
    return rows


def _manifest_row(stem, split, source, ap, ws, we, is_neg, n_boxes, pad_l, pad_r,
                  p_low, p_high) -> dict:
    return {
        "image_path": f"{split}/images/{stem}.png",
        "label_path": f"{split}/labels/{stem}.txt",
        "split": split, "source": source, "audio_path": ap,
        "window_start_s": ws, "window_end_s": we,
        "is_negative": is_neg, "n_fish_boxes": n_boxes,
        "has_padding": (pad_l > 0 or pad_r > 0),
        "pad_left_s": pad_l, "pad_right_s": pad_r,
        "norm_p5": p_low, "norm_p95": p_high,
    }


# ── Validation pass ──────────────────────────────────────────────────────────
def validate(output_path: Path, rng: np.random.Generator, splits: tuple[str, ...] = SPLITS) -> None:
    """Validate written output for `splits` (default: all).

    Callers that restrict a run to a subset of splits (--splits) should pass
    that same subset here too -- otherwise a hawaii-only run would still
    re-glob and re-check the entire, untouched train/val/test corpus every
    time, undoing the point of restricting the run in the first place.
    """
    import cv2
    logger.info("Validation pass (%s)...", ",".join(splits))
    for split in splits:
        imgs = list((output_path / split / "images").glob("*.png"))
        lbls = list((output_path / split / "labels").glob("*.txt"))
        assert len(imgs) == len(lbls), f"{split}: {len(imgs)} images != {len(lbls)} labels"
        logger.info("  %s: %d images == %d labels", split, len(imgs), len(lbls))

    all_imgs = [p for s in splits for p in (output_path / s / "images").glob("*.png")]
    sample = rng.choice(len(all_imgs), size=min(100, len(all_imgs)), replace=False) if all_imgs else []
    for i in sample:
        im = cv2.imread(str(all_imgs[int(i)]))
        assert im.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, 3), (all_imgs[int(i)], im.shape)
        assert im.dtype == np.uint8

    # positive label bbox sanity (also exercises the multi-box-per-tile path)
    pos_labels = []
    max_boxes_seen = 0
    for s in splits:
        for p in (output_path / s / "labels").glob("*.txt"):
            if p.stat().st_size > 0:
                pos_labels.append(p)
    psample = rng.choice(len(pos_labels), size=min(100, len(pos_labels)), replace=False) if pos_labels else []
    for i in psample:
        lines = pos_labels[int(i)].read_text().splitlines()
        max_boxes_seen = max(max_boxes_seen, len(lines))
        for line in lines:
            _, xc, yc, w, h = line.split()
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            for v in (xc, yc, w, h):
                assert 0.0 <= v <= 1.0, (pos_labels[int(i)], line)
            assert -1e-6 <= xc - w / 2 and xc + w / 2 <= 1 + 1e-6, line
            assert -1e-6 <= yc - h / 2 and yc + h / 2 <= 1 + 1e-6, line
    logger.info("  max boxes seen in one sampled label (of %d sampled): %d",
                len(psample), max_boxes_seen)

    # negatives must be empty
    for s in splits:
        for p in (output_path / s / "labels").glob("*_neg.txt"):
            assert p.stat().st_size == 0, f"non-empty negative label: {p}"
    logger.info("Validation pass OK.")


# ── Outputs ──────────────────────────────────────────────────────────────────
def write_dataset_yaml(output_path: Path) -> None:
    # hawaii is a standalone eval-only split (see build_manifest.py) -- not
    # part of the train/val/test rotation, so it's listed separately below
    # rather than as train/val/test's `test:` key. Ultralytics ignores
    # unrecognized top-level keys, so this is safe for YOLO training configs
    # while still documenting where the hawaii images live for eval scripts.
    extra_splits = "\n".join(
        f"{s}: {s}/images" for s in SPLITS if s not in ("train", "val", "test")
        and (output_path / s / "images").exists()
    )
    yaml = f"""path: {output_path.as_posix()}
train: train/images
val:   val/images
test:  test/images
{extra_splits}

nc: 1
names: ['fish']

# Spectrogram parameters (reference)
img_width:   {config.IMG_WIDTH}
img_height:  {config.IMG_HEIGHT}
sample_rate: {config.TARGET_SR}
n_fft:       {config.N_FFT}
hop_length:  {config.HOP_LENGTH}
f_max:       {int(config.F_MAX)}

# All three splits (including test) are whole-file tiles at a fixed
# {config.STRIDE_S}s stride — the same thing a sliding-window inference pass
# produces, so test metrics computed directly from test/images are already
# deployment-realistic. No separate sliding-window eval pass is needed to
# de-optimism-correct them.
"""
    (output_path / "dataset.yaml").write_text(yaml, encoding="utf-8")


def append_log(output_path: Path, ap: str) -> None:
    with (output_path / "processed_files.log").open("a", encoding="utf-8") as f:
        f.write(ap + "\n")


def load_processed(output_path: Path) -> set[str]:
    p = output_path / "processed_files.log"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1e6


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap_parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    ap_parser.add_argument("--smoke", action="store_true",
                           help="process a tiny subset (2 files/source) for validation")
    ap_parser.add_argument("--limit-files", type=int, default=0,
                           help="limit number of audio files per source (0 = all)")
    ap_parser.add_argument("--splits", type=str, default="",
                           help="comma-separated split(s) to process, e.g. 'hawaii' or "
                                "'train,val' (default: all splits in the manifest). "
                                "Other splits' existing output is left untouched.")
    ap_parser.add_argument("--no-prompt", action="store_true",
                           help="skip interactive disk-space prompt")
    ap_parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap_parser.parse_args()

    setup_logging(config.OUTPUT_PATH)
    make_dirs(config.OUTPUT_PATH)
    logger.info("Loading manifest %s", config.MANIFEST_PATH)
    df = pd.read_csv(config.MANIFEST_PATH, dtype={"is_fish": "boolean"})
    logger.info("Manifest: %d rows, %d unique audio files", len(df), df["audio_path"].nunique())

    idx = Indexes(df)
    device = audio_utils.get_device(use_gpu=not args.cpu)
    logger.info("Device: %s", device)

    # File set (optionally subset)
    files_all = list(df["audio_path"].unique())
    target_splits: tuple[str, ...] = SPLITS
    if args.splits:
        requested = {s.strip() for s in args.splits.split(",") if s.strip()}
        unknown = requested - set(df["split"].unique())
        if unknown:
            logger.warning("--splits requested split(s) not in manifest: %s", sorted(unknown))
        target_splits = tuple(sorted(requested))
        files_all = [f for f in files_all if idx.file_split.get(f) in requested]
        logger.info("--splits filter (%s): %d files", ",".join(target_splits), len(files_all))
    if args.smoke or args.limit_files:
        lim = 2 if args.smoke else args.limit_files
        subset: list[str] = []
        for source in SOURCES:
            src_files = [f for f in files_all if idx.file_source.get(f) == source]
            subset.extend(src_files[:lim])
        files = subset
        logger.info("Subset mode: %d files (%d/source)", len(files), lim)
    else:
        files = files_all

    pre = prepass(idx, files)
    n_projected = project_final_tile_count(idx, pre, files)
    check_storage(config.OUTPUT_PATH, n_projected, args.no_prompt or args.smoke)

    processed = load_processed(config.OUTPUT_PATH)
    todo = [f for f in files if f not in processed]
    logger.info("%d files to process (%d already done)", len(todo), len(files) - len(todo))

    all_rows: list[dict] = []
    by_source = defaultdict(list)
    for f in todo:
        by_source[idx.file_source[f]].append(f)

    for source in SOURCES:
        sfiles = by_source.get(source, [])
        if not sfiles:
            continue
        for ap in tqdm(sfiles, desc=source, unit="file"):
            rows = process_file(ap, idx, pre, device)
            all_rows.extend(rows)
            append_log(config.OUTPUT_PATH, ap)

    # ── dataset_manifest.csv (merge with any prior run) ──
    man_path = config.OUTPUT_PATH / "dataset_manifest.csv"
    new_df = pd.DataFrame(all_rows)
    if man_path.exists() and not new_df.empty:
        old = pd.read_csv(man_path)
        new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates("image_path")
    elif man_path.exists():
        new_df = pd.read_csv(man_path)
    if not new_df.empty:
        new_df.to_csv(man_path, index=False)

    write_dataset_yaml(config.OUTPUT_PATH)
    logger.info("Writing COCO JSON (%s)...", ",".join(target_splits))
    for split in target_splits:
        out = coco_converter.yolo_labels_to_coco_json(split, config.OUTPUT_PATH)
        logger.info("  wrote %s", out.name)

    validate(config.OUTPUT_PATH, np.random.default_rng(config.RANDOM_SEED), splits=target_splits)
    print_summary(config.OUTPUT_PATH, new_df)


def print_summary(output_path: Path, man: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    if not man.empty:
        logger.info("Images per split: %s", man["split"].value_counts().to_dict())
        logger.info("Images per source: %s", man["source"].value_counts().to_dict())
        logger.info("Positive=%d  Negative=%d",
                    int((~man["is_negative"]).sum()), int(man["is_negative"].sum()))
        logger.info("Total fish boxes: %d", int(man["n_fish_boxes"].sum()))
        logger.info("Source x Split (rows=source, cols=split):")
        for line in str(pd.crosstab(man["source"], man["split"])).split("\n"):
            logger.info("  %s", line)
        logger.info("Pos:Neg ratio by split:")
        for split, grp in man.groupby("split"):
            n_pos = int((~grp["is_negative"]).sum())
            n_neg = int(grp["is_negative"].sum())
            ratio = (n_neg / n_pos) if n_pos else float("nan")
            logger.info("  %-6s pos=%-7d neg=%-7d neg:pos=%.2f:1", split, n_pos, n_neg, ratio)
    logger.info("Disk usage of spectrograms/: %.1f MB", dir_size_mb(output_path))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
