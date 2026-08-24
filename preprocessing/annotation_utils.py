"""Annotation indexing, spatial join, and YOLO bbox conversion.

Coordinate system for the rendered image:
  * x-axis (width):  time  — left edge = window_start_s, right edge = window_end_s
  * y-axis (height): freq  — top = F_MAX, bottom = 0 Hz (after flipud)
"""
from __future__ import annotations

import bisect
from collections import defaultdict

import pandas as pd

try:
    from . import config
except ImportError:
    import config


def build_annotation_index(df: pd.DataFrame) -> dict[str, list[tuple]]:
    """Group fish annotations (is_fish == True) by audio_path, sorted by begin_s.

    Returns: audio_path -> list of (begin_s, end_s, row_dict).
    """
    index: dict[str, list[tuple]] = defaultdict(list)
    fish = df[df["is_fish"] == True]   # noqa: E712 — nullable-boolean compare
    for row in fish.itertuples(index=False):
        d = row._asdict()
        index[d["audio_path"]].append((d["begin_s"], d["end_s"], d))
    for path in index:
        index[path].sort(key=lambda x: x[0])
    return index


def find_overlapping_annotations(index_for_file: list[tuple],
                                 window_start_s: float,
                                 window_end_s: float) -> list[dict]:
    """All fish annotations that qualify for a box in [window_start_s, window_end_s].

    A call qualifies only if BOTH hold: the visible overlap covers at least
    MIN_OVERLAP_FRAC of the call's own duration, AND at least
    MIN_OVERLAP_ABS_S of it is visible. Calls meeting neither are left out of
    this tile entirely (not written as a truncated sliver). A tile can and
    does carry more than one box — every qualifying call in range is
    returned, not just whichever call the tile happens to be centred near.
    """
    begins = [x[0] for x in index_for_file]
    upper = bisect.bisect_left(begins, window_end_s)
    results: list[dict] = []
    for ann_begin, ann_end, row in index_for_file[:upper]:
        if ann_end <= window_start_s:
            continue   # ends before window starts
        overlap_s = min(ann_end, window_end_s) - max(ann_begin, window_start_s)
        ann_dur = ann_end - ann_begin
        if (ann_dur > 0
                and (overlap_s / ann_dur) >= config.MIN_OVERLAP_FRAC
                and overlap_s >= config.MIN_OVERLAP_ABS_S):
            results.append(row)
    return results


def annotation_to_yolo_line(ann: dict, window_start_s: float) -> str:
    """Convert one annotation to a YOLO label line (class 0 = fish).

    Annotation bounds are clipped to the window (time) and [0, F_MAX] (frequency).
    All output values are normalized to [0, 1].
    """
    win_dur = config.WINDOW_DUR
    f_max = config.F_MAX

    t_begin = max(ann["begin_s"], window_start_s)
    t_end = min(ann["end_s"], window_start_s + win_dur)

    f_low = max(ann["low_hz"], 0.0)
    f_high = min(ann["high_hz"], f_max)

    # time -> x (left to right)
    x_left = (t_begin - window_start_s) / win_dur
    x_right = (t_end - window_start_s) / win_dur
    x_center = (x_left + x_right) / 2.0
    box_w = x_right - x_left

    # frequency -> y (top = high freq = small y)
    y_top = 1.0 - (f_high / f_max)
    y_bot = 1.0 - (f_low / f_max)
    y_center = (y_top + y_bot) / 2.0
    box_h = y_bot - y_top

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    box_w = max(0.0, min(1.0, box_w))
    box_h = max(0.0, min(1.0, box_h))

    return f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


# ── Embedded sanity check ─────────────────────────────────────────────────────
def _sanity_check() -> None:
    """A call at 100-500 Hz centred at 1.5 s in a [0, 3] s window."""
    ann = {"begin_s": 1.4, "end_s": 1.6, "low_hz": 100.0, "high_hz": 500.0}
    line = annotation_to_yolo_line(ann, window_start_s=0.0)
    parts = line.split()
    cls, xc, yc, w, h = parts[0], *map(float, parts[1:])
    assert cls == "0"
    # y_top = 1 - 500/2000 = 0.75 ; y_bot = 1 - 100/2000 = 0.95
    assert abs(yc - 0.85) < 1e-6, yc
    assert abs(h - 0.20) < 1e-6, h
    # centred at 1.5 s -> x_center = 0.5 ; width 0.2 s / 3 s
    assert abs(xc - 0.5) < 1e-6, xc
    assert abs(w - (0.2 / 3.0)) < 1e-6, w
    print("annotation_to_yolo_line sanity check passed:", line)


if __name__ == "__main__":
    _sanity_check()
