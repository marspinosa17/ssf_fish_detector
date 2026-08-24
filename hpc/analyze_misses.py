"""
Single-threshold miss analysis.

Runs YOLO at one confidence threshold and evaluates detections at one
ground-truth matching IoU threshold.

Output rows:
    box_type='gt':
        status='TP' if matched to a prediction at IoU >= --iou-thr
        status='FN' otherwise

    box_type='pred':
        status='FP' if the prediction was not matched to any GT box

Only predictions with confidence >= --conf-thr are considered. Predictions
below that threshold are neither matched nor written to the output.

For unmatched GT rows, pred_x1/pred_y1/pred_x2/pred_y2 contain the closest
prediction that survived --conf-thr, when one exists. match_conf and
match_iou describe that diagnostic candidate.

Example:

    python analyze_misses.py \
        --weights runs/fish_yolo26s/weights/best.pt \
        --data-root data/spectrograms \
        --split val \
        --conf-thr 0.223 \
        --iou-thr 0.5 \
        --out miss_analysis.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


WINDOW_DUR = 3.0
F_MAX = 2000.0

# This is YOLO's non-maximum-suppression threshold, not the threshold used
# to match predictions to ground-truth boxes.
NMS_IOU = 0.6


OUTPUT_COLUMNS = [
    "stem",
    "audio_path",
    "source",
    "box_type",
    "status",
    "match_conf",
    "match_iou",
    "approx_duration_s",
    "approx_freq_lo_hz",
    "approx_freq_hi_hz",
    "box_start_s",
    "box_end_s",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
]


def box_iou(a, b):
    """Return IoU between two normalized xyxy boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def yolo_to_xyxy(xc, yc, width, height):
    """Convert normalized YOLO xc/yc/w/h coordinates to xyxy."""
    return [
        xc - width / 2.0,
        yc - height / 2.0,
        xc + width / 2.0,
        yc + height / 2.0,
    ]


def load_gt(label_path: Path):
    """Load normalized ground-truth boxes from a YOLO label file."""
    boxes = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes

    for line_number, line in enumerate(
        label_path.read_text().splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}: {line!r}"
            )

        # Class ID is ignored because this analysis is class-agnostic.
        _, xc, yc, width, height = parts[:5]

        boxes.append(
            yolo_to_xyxy(
                float(xc),
                float(yc),
                float(width),
                float(height),
            )
        )

    return boxes


def match_predictions_to_gt(
    gt_boxes,
    pred_boxes,
    pred_confs,
    iou_threshold,
):
    """
    Perform one-to-one matching at a fixed IoU threshold.

    Predictions are considered in descending confidence order, which follows
    the usual detector-evaluation convention. Each GT and prediction can be
    used at most once.

    Returns:
        gt_to_pred: dict mapping GT index to prediction index
        pred_to_gt: dict mapping prediction index to GT index
    """
    gt_to_pred = {}
    pred_to_gt = {}

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_to_pred, pred_to_gt

    prediction_order = np.argsort(-pred_confs, kind="stable")

    for pred_index in prediction_order:
        best_gt_index = -1
        best_iou = -1.0

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in gt_to_pred:
                continue

            iou = box_iou(gt_box, pred_boxes[pred_index])

            if iou < iou_threshold:
                continue

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0:
            gt_to_pred[best_gt_index] = int(pred_index)
            pred_to_gt[int(pred_index)] = best_gt_index

    return gt_to_pred, pred_to_gt


def find_best_candidate(gt_box, pred_boxes, pred_confs):
    """
    Find the prediction with the highest IoU to one GT box.

    This is diagnostic only. The candidate may already be matched to another
    GT box and may have IoU below the requested matching threshold.
    """
    if len(pred_boxes) == 0:
        return -1, None, None

    candidate_ious = np.asarray(
        [box_iou(gt_box, pred_box) for pred_box in pred_boxes],
        dtype=float,
    )

    best_index = int(np.argmax(candidate_ious))

    return (
        best_index,
        float(candidate_ious[best_index]),
        float(pred_confs[best_index]),
    )


def extract_manifest_metadata(manifest, stem):
    """Return source metadata for one tile."""
    if stem not in manifest.index:
        return None, None, "unknown"

    meta = manifest.loc[stem]

    audio_path = meta.get("audio_path")
    if pd.isna(audio_path):
        audio_path = None

    window_start_s = meta.get("window_start_s")
    if pd.isna(window_start_s):
        window_start_s = None
    else:
        window_start_s = float(window_start_s)

    source = meta.get("source", "unknown")
    if pd.isna(source):
        source = "unknown"

    return audio_path, window_start_s, source


def absolute_box_times(window_start_s, x1, x2):
    """Convert normalized tile x-coordinates into source-audio times."""
    if window_start_s is None:
        return None, None

    start_s = window_start_s + float(x1) * WINDOW_DUR
    end_s = window_start_s + float(x2) * WINDOW_DUR

    return round(start_s, 3), round(end_s, 3)


def box_measurements(box):
    """Convert one normalized spectrogram box into duration/frequency values."""
    x1, y1, x2, y2 = [float(value) for value in box]

    duration_s = (x2 - x1) * WINDOW_DUR
    freq_lo_hz = (1.0 - y2) * F_MAX
    freq_hi_hz = (1.0 - y1) * F_MAX

    return (
        round(duration_s, 3),
        round(freq_lo_hz, 1),
        round(freq_hi_hz, 1),
    )


def load_manifest(manifest_path, split):
    """Load and index the selected manifest split by image stem."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    required_columns = {"split", "image_path"}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    manifest = manifest.loc[manifest["split"] == split].copy()
    manifest["stem"] = manifest["image_path"].apply(
        lambda path: Path(str(path)).stem
    )

    duplicate_stems = manifest.loc[
        manifest["stem"].duplicated(keep=False),
        "stem",
    ].unique()

    if len(duplicate_stems) > 0:
        preview = ", ".join(map(str, duplicate_stems[:10]))
        raise ValueError(
            "Manifest contains duplicate image stems. "
            f"Examples: {preview}"
        )

    return manifest.set_index("stem")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze YOLO TP/FN/FP boxes at one operating threshold."
    )

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.223,
        help="Minimum prediction confidence included in the analysis.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="Minimum prediction-to-GT IoU required for a TP.",
    )
    parser.add_argument(
        "--out",
        default="miss_analysis.csv",
    )

    args = parser.parse_args()

    if not 0.0 <= args.conf_thr <= 1.0:
        parser.error("--conf-thr must be between 0 and 1.")

    if not 0.0 <= args.iou_thr <= 1.0:
        parser.error("--iou-thr must be between 0 and 1.")

    data_root = Path(args.data_root)
    images_dir = data_root / args.split / "images"
    labels_dir = data_root / args.split / "labels"
    manifest_path = data_root / "dataset_manifest.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    manifest = load_manifest(manifest_path, args.split)
    model = YOLO(args.weights)

    img_paths = sorted(images_dir.glob("*.png"))

    print(
        f"Running inference on {len(img_paths)} {args.split} images "
        f"at confidence >= {args.conf_thr:.4f}..."
    )

    rows = []
    batch_size = 64

    for batch_start in range(0, len(img_paths), batch_size):
        batch = img_paths[batch_start : batch_start + batch_size]

        results = model.predict(
            source=[str(path) for path in batch],
            conf=args.conf_thr,
            iou=NMS_IOU,
            verbose=False,
        )

        for img_path, result in zip(batch, results):
            stem = img_path.stem
            gt_boxes = load_gt(labels_dir / f"{stem}.txt")

            audio_path, window_start_s, source = extract_manifest_metadata(
                manifest,
                stem,
            )

            if result.boxes is not None and len(result.boxes) > 0:
                pred_boxes = result.boxes.xyxyn.cpu().numpy()
                pred_confs = result.boxes.conf.cpu().numpy()
            else:
                pred_boxes = np.zeros((0, 4), dtype=float)
                pred_confs = np.zeros((0,), dtype=float)

            gt_to_pred, pred_to_gt = match_predictions_to_gt(
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                pred_confs=pred_confs,
                iou_threshold=args.iou_thr,
            )

            # One output row for every ground-truth box.
            for gt_index, gt_box in enumerate(gt_boxes):
                gx1, gy1, gx2, gy2 = [
                    float(value) for value in gt_box
                ]

                if gt_index in gt_to_pred:
                    pred_index = gt_to_pred[gt_index]
                    pred_box = pred_boxes[pred_index].tolist()

                    status = "TP"
                    match_conf = float(pred_confs[pred_index])
                    match_iou = box_iou(gt_box, pred_box)
                else:
                    status = "FN"

                    pred_index, match_iou, match_conf = find_best_candidate(
                        gt_box,
                        pred_boxes,
                        pred_confs,
                    )

                    if pred_index >= 0:
                        pred_box = pred_boxes[pred_index].tolist()
                    else:
                        pred_box = [None, None, None, None]

                duration_s, freq_lo_hz, freq_hi_hz = box_measurements(
                    gt_box
                )

                box_start_s, box_end_s = absolute_box_times(
                    window_start_s,
                    gx1,
                    gx2,
                )

                rows.append(
                    {
                        "stem": stem,
                        "audio_path": audio_path,
                        "source": source,
                        "box_type": "gt",
                        "status": status,
                        "match_conf": (
                            round(float(match_conf), 4)
                            if match_conf is not None
                            else None
                        ),
                        "match_iou": (
                            round(float(match_iou), 4)
                            if match_iou is not None
                            else None
                        ),
                        "approx_duration_s": duration_s,
                        "approx_freq_lo_hz": freq_lo_hz,
                        "approx_freq_hi_hz": freq_hi_hz,
                        "box_start_s": box_start_s,
                        "box_end_s": box_end_s,
                        "gt_x1": gx1,
                        "gt_y1": gy1,
                        "gt_x2": gx2,
                        "gt_y2": gy2,
                        "pred_x1": pred_box[0],
                        "pred_y1": pred_box[1],
                        "pred_x2": pred_box[2],
                        "pred_y2": pred_box[3],
                    }
                )

            # Every unmatched prediction is an FP. All predictions here have
            # already passed --conf-thr.
            for pred_index, pred_box in enumerate(pred_boxes):
                if pred_index in pred_to_gt:
                    continue

                px1, py1, px2, py2 = [
                    float(value) for value in pred_box
                ]

                # This records how close the FP came to any GT. It may be
                # below --iou-thr, or it may be a duplicate around an
                # already-matched GT.
                if gt_boxes:
                    nearest_gt_iou = max(
                        box_iou(pred_box, gt_box)
                        for gt_box in gt_boxes
                    )
                else:
                    nearest_gt_iou = 0.0

                duration_s, freq_lo_hz, freq_hi_hz = box_measurements(
                    pred_box
                )

                box_start_s, box_end_s = absolute_box_times(
                    window_start_s,
                    px1,
                    px2,
                )

                rows.append(
                    {
                        "stem": stem,
                        "audio_path": audio_path,
                        "source": source,
                        "box_type": "pred",
                        "status": "FP",
                        "match_conf": round(
                            float(pred_confs[pred_index]),
                            4,
                        ),
                        "match_iou": round(float(nearest_gt_iou), 4),
                        "approx_duration_s": duration_s,
                        "approx_freq_lo_hz": freq_lo_hz,
                        "approx_freq_hi_hz": freq_hi_hz,
                        "box_start_s": box_start_s,
                        "box_end_s": box_end_s,
                        "gt_x1": None,
                        "gt_y1": None,
                        "gt_x2": None,
                        "gt_y2": None,
                        "pred_x1": px1,
                        "pred_y1": py1,
                        "pred_x2": px2,
                        "pred_y2": py2,
                    }
                )

    dataframe = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    tp_count = int((dataframe["status"] == "TP").sum())
    fn_count = int((dataframe["status"] == "FN").sum())
    fp_count = int((dataframe["status"] == "FP").sum())

    precision_denominator = tp_count + fp_count
    recall_denominator = tp_count + fn_count

    precision = (
        tp_count / precision_denominator
        if precision_denominator > 0
        else float("nan")
    )
    recall = (
        tp_count / recall_denominator
        if recall_denominator > 0
        else float("nan")
    )

    print(f"Wrote {len(dataframe)} box-level records to {output_path}")
    print(
        f"Operating point: conf >= {args.conf_thr:.4f}, "
        f"match IoU >= {args.iou_thr:.4f}"
    )
    print(f"TP: {tp_count}")
    print(f"FN: {fn_count}")
    print(f"FP: {fp_count}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
