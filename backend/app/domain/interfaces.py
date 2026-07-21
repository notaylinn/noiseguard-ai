"""
Abstract interfaces (ports) that decouple the service layer from
concrete implementations of the ML model and the persistence layer.

This is the core of the Dependency Inversion Principle applied here:
`AnalysisService` depends on `SoundClassifierPort` and
`AnalysisRepositoryPort`, never on TensorFlow or SQLAlchemy directly.
Swapping YAMNet for another model, or SQLite for Postgres, requires
implementing these interfaces only — no service code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.domain.entities import AcousticFeatures, AnalysisResult, ClassPrediction


class SoundClassifierPort(ABC):
    """Contract for any audio classification model."""

    @abstractmethod
    def classify(self, waveform, sample_rate: int) -> list[ClassPrediction]:
        """Return ranked class predictions for a mono waveform."""
        raise NotImplementedError


class FeatureExtractorPort(ABC):
    """Contract for DSP feature extraction."""

    @abstractmethod
    def extract(self, waveform, sample_rate: int) -> AcousticFeatures:
        raise NotImplementedError


class AnalysisRepositoryPort(ABC):
    """Contract for persistence of analysis results."""

    @abstractmethod
    def save(self, result: AnalysisResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, analysis_id: str) -> AnalysisResult | None:
        raise NotImplementedError

    @abstractmethod
    def list_all(self, limit: int = 100, offset: int = 0) -> list[AnalysisResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, analysis_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
