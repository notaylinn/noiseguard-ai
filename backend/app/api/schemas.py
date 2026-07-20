"""
Pydantic request/response schemas for the REST API.

Kept separate from `domain.entities` on purpose: these are the API's
wire format (HTTP/JSON contract), while domain entities are the
internal business representation. This separation means the API shape
can evolve (versioning, renaming fields for clients) without touching
business logic.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.domain.entities import AnalysisResult


class ClassPredictionSchema(BaseModel):
    label: str
    confidence: float


class AcousticFeaturesSchema(BaseModel):
    duration_sec: float
    rms_mean: float
    rms_std: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc_mean: list[float]
    mel_spectrogram_db_mean: float
    estimated_dbfs: float


class LoudnessEstimateSchema(BaseModel):
    dbfs: float
    category: str
    explanation: str


class AnalysisResultSchema(BaseModel):
    id: str
    filename: str
    created_at: datetime
    predictions: list[ClassPredictionSchema]
    top_prediction: ClassPredictionSchema
    features: AcousticFeaturesSchema
    loudness: LoudnessEstimateSchema
    environmental_interpretation: str
    recommendations: list[str]
    waveform_preview: list[float] = Field(default_factory=list)

    @classmethod
    def from_entity(cls, result: AnalysisResult, waveform_preview: list[float] | None = None) -> "AnalysisResultSchema":
        return cls(
            id=result.id,
            filename=result.filename,
            created_at=result.created_at,
            predictions=[ClassPredictionSchema(label=p.label, confidence=p.confidence) for p in result.predictions],
            top_prediction=ClassPredictionSchema(label=result.top_prediction.label, confidence=result.top_prediction.confidence),
            features=AcousticFeaturesSchema(**result.features.__dict__),
            loudness=LoudnessEstimateSchema(
                dbfs=result.loudness.dbfs, category=result.loudness.category.value, explanation=result.loudness.explanation
            ),
            environmental_interpretation=result.environmental_interpretation,
            recommendations=result.recommendations,
            waveform_preview=waveform_preview or [],
        )


class AnalysisSummarySchema(BaseModel):
    id: str
    filename: str
    created_at: datetime
    top_label: str
    top_confidence: float
    loudness_category: str
    estimated_dbfs: float
    duration_sec: float


class StatisticsSchema(BaseModel):
    total_analyses: int
    category_distribution: dict[str, int]
    loudness_distribution: dict[str, int]
    average_dbfs: float
    average_duration_sec: float


class HealthSchema(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
