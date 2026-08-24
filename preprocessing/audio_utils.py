"""Audio loading, resampling, spectrogram computation, and normalization.

Uses torchaudio throughout. The Spectrogram transform is instantiated once per
process (lazily) and reused for every file. With ``center=True`` the signal is
padded so that frame ``k`` is centred at sample ``k * HOP_LENGTH``; the matching
timestamp->frame formula is ``round(t * TARGET_SR / HOP_LENGTH)`` (see
``window_utils.time_to_frame``).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torchaudio

try:
    from . import config
except ImportError:  # allow `python preprocessing/preprocess.py`
    import config

logger = logging.getLogger(__name__)

# Lazily-built, per-process singletons keyed by device. Caches the Spectrogram
# transform and resamplers so we don't rebuild kernels on every file.
_spec_transform: torchaudio.transforms.Spectrogram | None = None
_spec_device: torch.device | None = None
_resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def get_device(use_gpu: bool | None = None) -> torch.device:
    """Return the compute device, honouring config.USE_GPU and CUDA availability."""
    if use_gpu is None:
        use_gpu = config.USE_GPU
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_spec_transform(device: torch.device) -> torchaudio.transforms.Spectrogram:
    """Build (once) and return the power-spectrogram transform on ``device``."""
    global _spec_transform, _spec_device
    if _spec_transform is None or _spec_device != device:
        _spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            window_fn=torch.hann_window,
            power=2.0,          # power spectrogram (magnitude squared)
            center=True,        # frame k centred at sample k*hop_length
            pad_mode="reflect",
            normalized=False,
        ).to(device)
        _spec_device = device
    return _spec_transform


def get_file_duration(path: str) -> float:
    """Duration of an audio file in seconds via metadata only (no decode)."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def load_and_resample(path: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load an audio file, downmix to mono, and resample to ``target_sr``.

    Returns ``(waveform, target_sr)`` where waveform is a 1-D float32 tensor on CPU.
    """
    if target_sr is None:
        target_sr = config.TARGET_SR
    waveform, sr = torchaudio.load(path)
    # Downmix to mono -> shape (samples,)
    waveform = waveform.mean(0) if waveform.shape[0] > 1 else waveform[0]
    if sr != target_sr:
        key = (sr, target_sr)
        resampler = _resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            _resamplers[key] = resampler
        waveform = resampler(waveform)
    return waveform.to(torch.float32), target_sr


def compute_spectrogram(waveform: torch.Tensor, device: torch.device) -> np.ndarray:
    """Compute the power spectrogram for a 1-D waveform.

    Returns a float32 numpy array of shape (N_FREQ, n_frames).
    """
    transform = get_spec_transform(device)
    with torch.no_grad():
        S = transform(waveform.to(device))   # (n_freq, n_frames)
    return S.detach().cpu().numpy().astype(np.float32)


# Module-level flag so the short-file warning is only emitted once per run.
_short_file_warned = False


def normalize_spectrogram(
    S_power: np.ndarray,
    duration_s: float,
    source: str,
) -> tuple[np.ndarray, float, float]:
    """Convert a power spectrogram to a normalized uint8 image array.

    Parameters
    ----------
    S_power : float32 array, shape (N_FREQ, n_frames)
    duration_s : full recording duration; short files use fixed dB bounds.
    source : data source tag (xavier / seth / tagus) — used only for logging.

    Returns
    -------
    (S_uint8, p_low, p_high)
        S_uint8 : uint8 array, shape (N_FREQ, n_frames), values in [0, 255]
        p_low, p_high : dB bounds used for normalization (stored in the manifest)
    """
    global _short_file_warned
    S_db = 10.0 * np.log10(S_power + 1e-10)   # add epsilon to avoid log(0)

    if duration_s < config.SHORT_FILE_THRESHOLD_S:
        if not _short_file_warned:
            logger.warning(
                "Short-file normalization path taken (duration=%.3fs < %.1fs, "
                "source=%s): using fixed dB bounds [%.1f, %.1f]. Verify these "
                "constants against real files.",
                duration_s, config.SHORT_FILE_THRESHOLD_S, source,
                config.TAGUS_FIXED_DB_MIN, config.TAGUS_FIXED_DB_MAX,
            )
            _short_file_warned = True
        p_low = config.TAGUS_FIXED_DB_MIN
        p_high = config.TAGUS_FIXED_DB_MAX
    else:
        p_low = float(np.percentile(S_db, config.NORM_LOW_PCT))
        p_high = float(np.percentile(S_db, config.NORM_HIGH_PCT))

    S_norm = np.clip((S_db - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
    S_uint8 = (S_norm * 255).astype(np.uint8)
    return S_uint8, float(p_low), float(p_high)
