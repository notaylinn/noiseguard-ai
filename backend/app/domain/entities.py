"""
Domain entities — plain data structures with no framework dependencies.

These represent the core business concepts of NoiseGuard AI and are
shared between the ML layer, service layer, repositories and API
schemas. Keeping them framework-agnostic (no SQLAlchemy, no FastAPI)
is what lets the business logic stay independent of infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LoudnessCategory(str, Enum):
    QUIET = "quiet"
    MODERATE = "moderate"
    LOUD = "loud"
    VERY_LOUD = "very_loud"


@dataclass
class ClassPrediction:
    """A single YAMNet class prediction."""
    label: str
    confidence: float  # 0.0 - 1.0


@dataclass
class AcousticFeatures:
    """Deterministic DSP features extracted with librosa."""
    duration_sec: float
    rms_mean: float
    rms_std: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc_mean: list[float]
    mel_spectrogram_db_mean: float
    estimated_dbfs: float  # relative loudness estimate, NOT calibrated dB SPL


@dataclass
class LoudnessEstimate:
    dbfs: float
    category: LoudnessCategory
    explanation: str


@dataclass
class AnalysisResult:
    """Full result of running the ML pipeline on one audio sample."""
    id: str
    filename: str
    created_at: datetime
    predictions: list[ClassPrediction]
    top_prediction: ClassPrediction
    features: AcousticFeatures
    loudness: LoudnessEstimate
    environmental_interpretation: str
    recommendations: list[str] = field(default_factory=list)
    waveform_path: str | None = None
    audio_path: str | None = None
