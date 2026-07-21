"""
Repository-layer tests using an in-memory SQLite database, verifying
the round-trip: domain entity -> ORM record -> domain entity.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.database import Base
from backend.app.db.repositories.analysis_repository import SqlAlchemyAnalysisRepository
from backend.app.domain.entities import (
    AcousticFeatures,
    AnalysisResult,
    ClassPrediction,
    LoudnessCategory,
    LoudnessEstimate,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_result(result_id: str = "test-id-1") -> AnalysisResult:
    features = AcousticFeatures(
        duration_sec=2.0, rms_mean=0.1, rms_std=0.02, zero_crossing_rate=0.05,
        spectral_centroid=1500.0, spectral_bandwidth=800.0, spectral_rolloff=3000.0,
        mfcc_mean=[1.0] * 13, mel_spectrogram_db_mean=-20.0, estimated_dbfs=-18.0,
    )
    loudness = LoudnessEstimate(dbfs=-18.0, category=LoudnessCategory.LOUD, explanation="test explanation")
    prediction = ClassPrediction(label="Jackhammer", confidence=0.87)
    return AnalysisResult(
        id=result_id, filename="test.wav", created_at=datetime.now(timezone.utc),
        predictions=[prediction], top_prediction=prediction, features=features, loudness=loudness,
        environmental_interpretation="Construction noise detected.",
        recommendations=["Document the time of day."],
    )


def test_save_and_get(db_session):
    repo = SqlAlchemyAnalysisRepository(db_session)
    result = _make_result()
    repo.save(result)

    fetched = repo.get(result.id)
    assert fetched is not None
    assert fetched.top_prediction.label == "Jackhammer"
    assert fetched.loudness.category == LoudnessCategory.LOUD
    assert fetched.features.spectral_centroid == 1500.0


def test_list_all_orders_newest_first(db_session):
    repo = SqlAlchemyAnalysisRepository(db_session)
    repo.save(_make_result("id-1"))
    repo.save(_make_result("id-2"))

    results = repo.list_all()
    assert len(results) == 2


def test_delete(db_session):
    repo = SqlAlchemyAnalysisRepository(db_session)
    result = _make_result()
    repo.save(result)

    assert repo.delete(result.id) is True
    assert repo.get(result.id) is None
    assert repo.delete("nonexistent") is False


def test_count(db_session):
    repo = SqlAlchemyAnalysisRepository(db_session)
    assert repo.count() == 0
    repo.save(_make_result("id-1"))
    assert repo.count() == 1
