"""Central configuration for the RT-DETR (HuggingFace transformers) module.

Mirrors preprocessing/config.py's "no magic numbers downstream" convention.
Values here describe the *existing* spectrogram dataset (see
preprocessing/config.py and preprocessing/PREPROCESSING.md) — this module
only reads that dataset, it never writes to preprocessing/ or data/.
"""
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FD_DATA_ROOT", REPO_ROOT / "data"))
SPECTROGRAM_ROOT = DATA_ROOT / "spectrograms"
MANIFEST_PATH = SPECTROGRAM_ROOT / "dataset_manifest.csv"

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("FD_RTDETR_OUTPUT_ROOT", REPO_ROOT / "models" / "rtdetr_hf")
)

# ── Dataset ──────────────────────────────────────────────────────────────────
IMG_WIDTH = 640
IMG_HEIGHT = 512
F_MAX = 2000  # Hz — top of image; see PREPROCESSING.md coordinate system note

# Single class throughout.
ID2LABEL = {0: "fish"}
LABEL2ID = {"fish": 0}
# COCO json category_id for "fish" (see preprocessing/coco_converter.py CATEGORIES) is
# 1 (COCO's convention: category ids start at 1). The model's classification head is
# 0-indexed (LABEL2ID above), so annotations read from coco_{split}.json must be
# remapped through this table before being handed to the image processor / loss —
# passing the raw category_id=1 through unchanged overflows a 1-class head's label
# range and crashes the Hungarian matcher with a CUDA index-out-of-bounds.
COCO_FISH_CATEGORY_ID = 1
COCO_FISH_CATEGORY_ID_TO_MODEL_ID = {COCO_FISH_CATEGORY_ID: LABEL2ID["fish"]}

SOURCES = ("xavier", "seth", "tagus")
SPLITS = ("train", "val", "test")

# ── Model ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "PekingU/rtdetr_v2_r50vd"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
