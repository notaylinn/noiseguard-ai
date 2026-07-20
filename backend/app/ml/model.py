"""
YAMNet-based sound classifier.

Loads Google's pretrained YAMNet model from TensorFlow Hub and runs
real inference on a 16kHz mono waveform. This module implements
`SoundClassifierPort` — the rest of the application only ever talks to
that interface, never to `tensorflow_hub` or the raw model directly.

The model is loaded once per process (module-level singleton via
`YamnetClassifier.get_instance`) because loading from TF-Hub / its
local cache is expensive (seconds), while inference on a short clip is
fast (tens of milliseconds on CPU).
"""
from __future__ import annotations

import os
import threading

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger
from backend.app.domain.entities import ClassPrediction
from backend.app.domain.interfaces import SoundClassifierPort

logger = get_logger(__name__)

_lock = threading.Lock()
_instance: "YamnetClassifier | None" = None


class YamnetClassifier(SoundClassifierPort):
    """Thin, testable wrapper around the TF-Hub YAMNet model."""

    def __init__(self) -> None:
        settings = get_settings()
        os.environ.setdefault("TFHUB_CACHE_DIR", str(settings.model_cache_dir))

        # Imported lazily so importing this module (e.g. for type checking
        # or non-ML tests) never requires TensorFlow to be installed.
        import tensorflow_hub as hub
        import csv
        import io

        logger.info("Loading YAMNet from %s (this may take a while on first run)...", settings.yamnet_handle)
        self._model = hub.load(settings.yamnet_handle)
        self._settings = settings

        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        self._class_names = self._load_class_names(class_map_path)
        logger.info("YAMNet loaded successfully with %d classes.", len(self._class_names))

    @staticmethod
    def _load_class_names(class_map_csv_path: str) -> list[str]:
        import csv

        import tensorflow as tf

        with tf.io.gfile.GFile(class_map_csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # header: index, mid, display_name
            return [row[2] for row in reader]

    @classmethod
    def get_instance(cls) -> "YamnetClassifier":
        """Process-wide singleton accessor (thread-safe lazy init)."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls()
        return _instance

    def classify(self, waveform: np.ndarray, sample_rate: int) -> list[ClassPrediction]:
        settings = self._settings
        if sample_rate != settings.sample_rate:
            raise ValueError(
                f"YAMNet requires {settings.sample_rate}Hz mono audio; got {sample_rate}Hz. "
                "Resample before calling classify()."
            )

        waveform = waveform.astype(np.float32)
        scores, embeddings, spectrogram = self._model(waveform)
        scores_np = scores.numpy()  # shape: (frames, 521)

        # Mean-pool frame-level scores across the whole clip -> one
        # confidence per class for the entire recording.
        mean_scores = scores_np.mean(axis=0)

        top_k = settings.top_k_predictions
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        predictions = [
            ClassPrediction(label=self._class_names[i], confidence=float(mean_scores[i]))
            for i in top_indices
            if mean_scores[i] >= settings.min_confidence_display
        ]

        if not predictions:
            # Always return at least the single top class even if below threshold,
            # so the UI never has an empty result set.
            top_i = int(np.argmax(mean_scores))
            predictions = [ClassPrediction(label=self._class_names[top_i], confidence=float(mean_scores[top_i]))]

        logger.info("YAMNet inference complete. Top prediction: %s (%.3f)", predictions[0].label, predictions[0].confidence)
        return predictions
