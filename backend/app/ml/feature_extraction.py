"""
DSP feature extraction using librosa.

This module implements `FeatureExtractorPort` and is the single place
where raw waveform -> numeric acoustic descriptors happens. It is
completely independent of the classification model: even if YAMNet
were swapped out, these features would still be computed the same way.

NOTE ON LOUDNESS: `estimated_dbfs` is a **relative, uncalibrated**
loudness proxy computed from the digital signal's RMS amplitude
(dBFS = decibels relative to full scale). It is NOT a certified sound
pressure level (dB SPL) measurement — smartphone microphones have
uncalibrated gain, so this value must only be interpreted as a
relative intensity indicator, never as a legally admissible reading.
"""
from __future__ import annotations

import numpy as np
import librosa

from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger
from backend.app.domain.entities import AcousticFeatures
from backend.app.domain.interfaces import FeatureExtractorPort

logger = get_logger(__name__)

_EPS = 1e-10


class LibrosaFeatureExtractor(FeatureExtractorPort):
    """Concrete DSP feature extractor built on librosa."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def extract(self, waveform: np.ndarray, sample_rate: int) -> AcousticFeatures:
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)
        waveform = waveform.astype(np.float32)

        duration_sec = float(len(waveform) / sample_rate)

        rms = librosa.feature.rms(y=waveform, hop_length=self._settings.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=waveform, hop_length=self._settings.hop_length)[0]

        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=sample_rate, n_fft=self._settings.n_fft, hop_length=self._settings.hop_length
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=sample_rate, n_fft=self._settings.n_fft, hop_length=self._settings.hop_length
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform, sr=sample_rate, n_fft=self._settings.n_fft, hop_length=self._settings.hop_length
        )[0]

        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sample_rate, n_mfcc=self._settings.n_mfcc, n_fft=self._settings.n_fft,
            hop_length=self._settings.hop_length,
        )
        mfcc_mean = mfcc.mean(axis=1).tolist()

        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sample_rate, n_mels=self._settings.n_mels, n_fft=self._settings.n_fft,
            hop_length=self._settings.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        rms_mean = float(np.mean(rms))
        estimated_dbfs = float(20.0 * np.log10(rms_mean + _EPS))

        features = AcousticFeatures(
            duration_sec=round(duration_sec, 3),
            rms_mean=round(rms_mean, 6),
            rms_std=round(float(np.std(rms)), 6),
            zero_crossing_rate=round(float(np.mean(zcr)), 6),
            spectral_centroid=round(float(np.mean(spectral_centroid)), 3),
            spectral_bandwidth=round(float(np.mean(spectral_bandwidth)), 3),
            spectral_rolloff=round(float(np.mean(spectral_rolloff)), 3),
            mfcc_mean=[round(v, 4) for v in mfcc_mean],
            mel_spectrogram_db_mean=round(float(np.mean(mel_spec_db)), 3),
            estimated_dbfs=round(estimated_dbfs, 2),
        )
        logger.info("Extracted DSP features: duration=%.2fs dbfs=%.2f", duration_sec, estimated_dbfs)
        return features
