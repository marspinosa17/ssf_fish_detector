"""Image and label writers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .window_utils import render_image
except ImportError:
    import config
    from window_utils import render_image


def _png_path(images_dir: Path, stem: str) -> Path:
    return images_dir / f"{stem}.png"


def write_image_and_label(spec_slice: np.ndarray, yolo_lines: list[str],
                          images_dir: Path, labels_dir: Path,
                          stem: str) -> tuple[Path, Path]:
    """Render and write a spectrogram tile plus its YOLO label file.

    An empty ``yolo_lines`` produces an empty .txt (a negative example).
    cv2 expects BGR; the image is grayscale-derived so channel order is moot.
    Returns (image_path, label_path).
    """
    img = render_image(spec_slice)
    img_path = _png_path(images_dir, stem)
    cv2.imwrite(str(img_path), img)
    label_path = labels_dir / f"{stem}.txt"
    label_path.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "",
                          encoding="utf-8")
    return img_path, label_path
