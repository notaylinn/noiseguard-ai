"""
Shared audio I/O helpers: loading arbitrary audio files/bytes into a
16kHz mono float32 waveform, and generating waveform plot data.

Centralizing this here means both the feature extractor and the YAMNet
classifier consume audio the exact same way, avoiding subtle mismatches
between what was analyzed for DSP features vs. what was classified.
"""
from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


def load_waveform_from_path(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    settings = get_settings()
    sr = target_sr or settings.sample_rate
    waveform, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    return waveform.astype(np.float32), sample_rate


def load_waveform_from_bytes(data: bytes, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    settings = get_settings()
    sr = target_sr or settings.sample_rate
    with io.BytesIO(data) as buf:
        waveform, sample_rate = librosa.load(buf, sr=sr, mono=True)
    return waveform.astype(np.float32), sample_rate


def validate_extension(filename: str) -> bool:
    settings = get_settings()
    return Path(filename).suffix.lower() in settings.allowed_audio_extensions


def downsample_for_plot(waveform: np.ndarray, max_points: int = 2000) -> list[float]:
    """Downsample a waveform to a manageable number of points for
    lightweight JSON transport / Plotly rendering."""
    if len(waveform) <= max_points:
        return waveform.tolist()
    step = len(waveform) // max_points
    return waveform[::step][:max_points].tolist()
