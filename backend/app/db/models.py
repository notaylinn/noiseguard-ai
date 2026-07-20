"""
SQLAlchemy ORM models — the persistence-layer representation of an
analysis. Kept separate from `domain.entities.AnalysisResult` (the
framework-agnostic dataclass used by business logic) so the ORM schema
can evolve without leaking into the service/domain layers, and vice versa.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    # Predictions stored as JSON list of {label, confidence}
    predictions: Mapped[list] = mapped_column(JSON)
    top_label: Mapped[str] = mapped_column(String(255), index=True)
    top_confidence: Mapped[float] = mapped_column(Float)

    # Flattened acoustic features for easy querying/statistics
    duration_sec: Mapped[float] = mapped_column(Float)
    rms_mean: Mapped[float] = mapped_column(Float)
    rms_std: Mapped[float] = mapped_column(Float)
    zero_crossing_rate: Mapped[float] = mapped_column(Float)
    spectral_centroid: Mapped[float] = mapped_column(Float)
    spectral_bandwidth: Mapped[float] = mapped_column(Float)
    spectral_rolloff: Mapped[float] = mapped_column(Float)
    mfcc_mean: Mapped[list] = mapped_column(JSON)
    mel_spectrogram_db_mean: Mapped[float] = mapped_column(Float)
    estimated_dbfs: Mapped[float] = mapped_column(Float)

    loudness_category: Mapped[str] = mapped_column(String(32))
    loudness_explanation: Mapped[str] = mapped_column(Text)
    environmental_interpretation: Mapped[str] = mapped_column(Text)
    recommendations: Mapped[list] = mapped_column(JSON)

    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    waveform_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
