"""Window placement, slice extraction, and image rendering.

The timestamp->frame conversion (``time_to_frame``) assumes the Spectrogram
transform uses ``center=True``. If that ever changes, this formula AND all bbox
math in ``annotation_utils`` must change together — keep them as a matched pair.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


def time_to_frame(time_s: float) -> int:
    """Convert a timestamp (seconds) to a spectrogram frame index.

    Uses round() (not floor/int) because center=True centres frame k at sample
    k*HOP_LENGTH.
    """
    return round(time_s * config.TARGET_SR / config.HOP_LENGTH)


def window_bounds(center_s: float) -> tuple[float, float]:
    """Return (window_start_s, window_end_s) for a given centre."""
    half = config.WINDOW_DUR / 2.0
    return center_s - half, center_s + half


def tile_starts(file_duration_s: float) -> list[float]:
    """Window start times (seconds) tiling an entire file at config.STRIDE_S.

    Starts run 0, STRIDE_S, 2*STRIDE_S, ... up to the last start whose window
    still fits (or, for files shorter than WINDOW_DUR, just start=0 — the
    window overhangs the file and extract_slice zero-pads it).

    A final tile is appended, ending exactly at file_duration_s, whenever the
    stride grid doesn't already reach it. Without this, the tail of every
    file (up to STRIDE_S seconds of it) would never be tiled at all, since
    the stride grid's last start is generally not aligned to end-of-file.
    """
    last_start = max(0.0, file_duration_s - config.WINDOW_DUR)
    starts: list[float] = []
    t = 0.0
    while t <= last_start + 1e-9:
        starts.append(t)
        t += config.STRIDE_S
    if not starts or abs(starts[-1] - last_start) > 1e-6:
        starts.append(last_start)
    return starts


def extract_slice(S_uint8: np.ndarray, window_start_s: float,
                  file_duration_s: float) -> tuple[np.ndarray, float, float]:
    """Extract a fixed-width slice from the full-recording spectrogram.

    Zero-pads when the window extends before file start or past file end. Zero in
    the normalized uint8 domain == the recording's noise floor (effective silence),
    which is the correct semantic for boundary padding.

    Returns (slice_uint8 shape (N_FREQ, n_frames_window), pad_left_s, pad_right_s).
    """
    n_frames_window = time_to_frame(config.WINDOW_DUR)   # 43
    start_frame = time_to_frame(window_start_s)
    end_frame = start_frame + n_frames_window

    total_frames = S_uint8.shape[1]
    pad_left = max(0, -start_frame)
    pad_right = max(0, end_frame - total_frames)

    slice_start = max(0, start_frame)
    slice_end = min(total_frames, end_frame)
    raw_slice = S_uint8[:, slice_start:slice_end]   # (N_FREQ, k), k <= n_frames_window

    if pad_left > 0 or pad_right > 0:
        raw_slice = np.pad(raw_slice, ((0, 0), (pad_left, pad_right)),
                           mode="constant", constant_values=0)

    pad_left_s = pad_left * config.HOP_LENGTH / config.TARGET_SR
    pad_right_s = pad_right * config.HOP_LENGTH / config.TARGET_SR
    return raw_slice, pad_left_s, pad_right_s


def render_image(spec_slice: np.ndarray) -> np.ndarray:
    """Render a spectrogram slice to a 640x512 RGB uint8 image.

    Input: uint8 (N_FREQ, n_frames), row 0 = 0 Hz (DC).
    Output: uint8 (IMG_HEIGHT, IMG_WIDTH, 3); after flipud, top row = F_MAX.
    """
    img = np.flipud(spec_slice)   # row 0 -> high frequency (top of image)
    img = cv2.resize(img.astype(np.float32), (config.IMG_WIDTH, config.IMG_HEIGHT),
                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img
