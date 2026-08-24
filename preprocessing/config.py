"""Central configuration for the Preprocessing pipeline.
"""
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT           = Path(os.environ.get("FD_DATA_ROOT", r"C:\Users\Marcello\whoissf\fd_framework\data"))
MANIFEST_PATH       = DATA_ROOT / "manifest.csv"
OUTPUT_PATH         = DATA_ROOT / "spectrograms"

# ── Audio / STFT ─────────────────────────────────────────────────────────────
TARGET_SR           = 4_000        # Hz — all audio resampled to this rate
N_FFT               = 256
HOP_LENGTH          = 64          
WIN_LENGTH          = 256
F_MAX               = TARGET_SR / 2   # 2000 Hz (Nyquist)

# Derived (documented, not used as magic numbers downstream)
N_FREQ              = N_FFT // 2 + 1          # 129 with N_FFT=256

# ── Context window / tiling ───────────────────────────────────────────────────
WINDOW_DUR          = 3.0          # seconds, per tile
STRIDE_S            = 1.0          # seconds between consecutive tile starts (whole-file tiling)

# ── Normalization ─────────────────────────────────────────────────────────────
NORM_LOW_PCT        = 5            # percentile floor for dB normalization
NORM_HIGH_PCT       = 95           # percentile ceiling
SHORT_FILE_THRESHOLD_S = 10.0      # files shorter than this use fixed dB bounds
TAGUS_FIXED_DB_MIN  = -80.0        # dB floor for short clips — VERIFY against real files
TAGUS_FIXED_DB_MAX  = -20.0        # dB ceiling for short clips — VERIFY against real files

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_WIDTH           = 640          # pixels (time axis)
IMG_HEIGHT          = 512          # pixels (frequency axis)

# ── Bounding boxes ────────────────────────────────────────────────────────────
# A call gets a box in a tile only if BOTH hold: the visible slice covers at
# least MIN_OVERLAP_FRAC of the call's own duration, AND at least
# MIN_OVERLAP_ABS_S of it is visible. The absolute floor matters for very
# short calls where 50% of the duration is still a sliver of a second.
MIN_OVERLAP_FRAC    = 0.50
MIN_OVERLAP_ABS_S   = 0.15         # seconds

# ── Negatives ─────────────────────────────────────────────────────────────────
# Whole-file tiling produces far more negative tiles than positive ones (most
# of a recording has no fish call in it). All positive tiles are kept;
# negative tiles are randomly subsampled per source, TRAIN SPLIT ONLY, down to
# this negative:positive ratio. 
# val/test are never subsampled — they keep the full natural tile
# distribution because that's what makes them deployment-realistic.
NEG_POS_TRAIN_RATIO = 3.0

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED         = 42

# ── Performance ───────────────────────────────────────────────────────────────
N_WORKERS           = 0            # 0 = single-threaded (GPU); >0 = multiprocessing (CPU only)
USE_GPU             = True         # use CUDA for the spectrogram transform when available
# Note: multiprocessing on Windows requires an if __name__ == '__main__' guard.
# CUDA tensors cannot be shared across processes; worker processes must run on CPU.

# ── Storage estimate ──────────────────────────────────────────────────────────
EST_BYTES_PER_PNG   = 255_000      # ~255 KB per 640x512 RGB PNG (measured)
