"""
Repository pattern implementation for `AnalysisResult` persistence.

`SqlAlchemyAnalysisRepository` implements `AnalysisRepositoryPort` and
is the ONLY place in the codebase that issues SQLAlchemy queries for
analysis records. Services depend on the interface, not this class,
so persistence can be swapped (e.g. to Postgres) by adding a new
implementation of the same port.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.db.models import AnalysisRecord
from backend.app.domain.entities import (
    AcousticFeatures,
    AnalysisResult,
    ClassPrediction,
    LoudnessCategory,
    LoudnessEstimate,
)
from backend.app.domain.interfaces import AnalysisRepositoryPort

logger = get_logger(__name__)


class SqlAlchemyAnalysisRepository(AnalysisRepositoryPort):
    def __init__(self, db: Session) -> None:
        self._db = db

    def save(self, result: AnalysisResult) -> None:
        record = AnalysisRecord(
            id=result.id,
            filename=result.filename,
            created_at=result.created_at,
            predictions=[{"label": p.label, "confidence": p.confidence} for p in result.predictions],
            top_label=result.top_prediction.label,
            top_confidence=result.top_prediction.confidence,
            duration_sec=result.features.duration_sec,
            rms_mean=result.features.rms_mean,
            rms_std=result.features.rms_std,
            zero_crossing_rate=result.features.zero_crossing_rate,
            spectral_centroid=result.features.spectral_centroid,
            spectral_bandwidth=result.features.spectral_bandwidth,
            spectral_rolloff=result.features.spectral_rolloff,
            mfcc_mean=result.features.mfcc_mean,
            mel_spectrogram_db_mean=result.features.mel_spectrogram_db_mean,
            estimated_dbfs=result.features.estimated_dbfs,
            loudness_category=result.loudness.category.value,
            loudness_explanation=result.loudness.explanation,
            environmental_interpretation=result.environmental_interpretation,
            recommendations=result.recommendations,
            audio_path=result.audio_path,
            waveform_path=result.waveform_path,
        )
        self._db.add(record)
        self._db.commit()
        logger.info("Saved analysis record %s (%s)", record.id, record.top_label)

    def get(self, analysis_id: str) -> AnalysisResult | None:
        record = self._db.get(AnalysisRecord, analysis_id)
        return self._to_entity(record) if record else None

    def list_all(self, limit: int = 100, offset: int = 0) -> list[AnalysisResult]:
        records = (
            self._db.query(AnalysisRecord)
            .order_by(AnalysisRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [self._to_entity(r) for r in records]

    def delete(self, analysis_id: str) -> bool:
        record = self._db.get(AnalysisRecord, analysis_id)
        if record is None:
            return False
        self._db.delete(record)
        self._db.commit()
        return True

    def count(self) -> int:
        return self._db.query(func.count(AnalysisRecord.id)).scalar() or 0

    @staticmethod
    def _to_entity(record: AnalysisRecord) -> AnalysisResult:
        predictions = [ClassPrediction(label=p["label"], confidence=p["confidence"]) for p in record.predictions]
        features = AcousticFeatures(
            duration_sec=record.duration_sec,
            rms_mean=record.rms_mean,
            rms_std=record.rms_std,
            zero_crossing_rate=record.zero_crossing_rate,
            spectral_centroid=record.spectral_centroid,
            spectral_bandwidth=record.spectral_bandwidth,
            spectral_rolloff=record.spectral_rolloff,
            mfcc_mean=record.mfcc_mean,
            mel_spectrogram_db_mean=record.mel_spectrogram_db_mean,
            estimated_dbfs=record.estimated_dbfs,
        )
        loudness = LoudnessEstimate(
            dbfs=record.estimated_dbfs,
            category=LoudnessCategory(record.loudness_category),
            explanation=record.loudness_explanation,
        )
        return AnalysisResult(
            id=record.id,
            filename=record.filename,
            created_at=record.created_at,
            predictions=predictions,
            top_prediction=ClassPrediction(label=record.top_label, confidence=record.top_confidence),
            features=features,
            loudness=loudness,
            environmental_interpretation=record.environmental_interpretation,
            recommendations=record.recommendations,
            waveform_path=record.waveform_path,
            audio_path=record.audio_path,
        )
