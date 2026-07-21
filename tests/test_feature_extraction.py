"""
Unit tests for the DSP feature extraction module. These do NOT require
TensorFlow/YAMNet — only librosa — so they run fast and independently
of the ML classification model.
"""
from __future__ import annotations

import numpy as np

from backend.app.ml.feature_extraction import LibrosaFeatureExtractor

SAMPLE_RATE = 16000


def test_extract_returns_expected_shape(sine_wave):
    extractor = LibrosaFeatureExtractor()
    features = extractor.extract(sine_wave, SAMPLE_RATE)

    assert features.duration_sec == 2.0
    assert isinstance(features.mfcc_mean, list)
    assert len(features.mfcc_mean) == 13  # default n_mfcc
    assert features.rms_mean > 0
    assert features.spectral_centroid > 0


def test_silence_has_low_rms(silence):
    extractor = LibrosaFeatureExtractor()
    features = extractor.extract(silence, SAMPLE_RATE)

    assert features.rms_mean < 1e-4
    assert features.estimated_dbfs < -60  # very quiet in dBFS terms


def test_louder_signal_has_higher_estimated_dbfs():
    extractor = LibrosaFeatureExtractor()
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    quiet = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    loud = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    quiet_features = extractor.extract(quiet, SAMPLE_RATE)
    loud_features = extractor.extract(loud, SAMPLE_RATE)

    assert loud_features.estimated_dbfs > quiet_features.estimated_dbfs


def test_mono_conversion_from_stereo():
    extractor = LibrosaFeatureExtractor()
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([mono, mono])  # shape (2, N) -> librosa.to_mono handles this

    features = extractor.extract(stereo, SAMPLE_RATE)
    assert features.duration_sec == 1.0
