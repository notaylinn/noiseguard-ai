"""
Shared pytest fixtures: synthetic audio generation so tests never
depend on external sample files or network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_RATE = 16000


@pytest.fixture()
def sine_wave() -> np.ndarray:
    """2 seconds of a 440Hz sine tone at moderate amplitude."""
    t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def sine_wave_file(tmp_path: Path, sine_wave: np.ndarray) -> Path:
    path = tmp_path / "test_tone.wav"
    sf.write(str(path), sine_wave, SAMPLE_RATE)
    return path


@pytest.fixture()
def silence() -> np.ndarray:
    return np.zeros(SAMPLE_RATE, dtype=np.float32)
