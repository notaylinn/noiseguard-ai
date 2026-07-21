"""
API-level tests using FastAPI's TestClient.

The `/analysis` endpoint depends on a loaded YAMNet model, which is
too heavy (multi-hundred-MB download) to require for a fast unit-test
suite. We therefore override `get_analysis_service` with a fake
service built from a lightweight stub classifier for these tests only
— this is standard dependency-injection testing practice and does not
affect production behavior, where `main.py` always wires the real
`YamnetClassifier`. Feature extraction (librosa) IS exercised for real.
"""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_analysis_service, get_repository
from backend.app.db.database import Base
from backend.app.domain.entities import ClassPrediction
from backend.app.domain.interfaces import SoundClassifierPort
from backend.app.main import app
from backend.app.ml.feature_extraction import LibrosaFeatureExtractor
from backend.app.services.analysis_service import AnalysisService


class StubClassifier(SoundClassifierPort):
    """Deterministic stand-in for YAMNet, used only in tests."""

    def classify(self, waveform, sample_rate):
        return [
            ClassPrediction(label="Speech", confidence=0.62),
            ClassPrediction(label="Silence", confidence=0.21),
        ]


def _override_analysis_service():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()

    from backend.app.db.repositories.analysis_repository import SqlAlchemyAnalysisRepository

    repo = SqlAlchemyAnalysisRepository(session)
    return AnalysisService(classifier=StubClassifier(), feature_extractor=LibrosaFeatureExtractor(), repository=repo), repo


_service, _repo = _override_analysis_service()
app.dependency_overrides[get_analysis_service] = lambda: _service
app.dependency_overrides[get_repository] = lambda: _repo

client = TestClient(app)


def test_root_health():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "NoiseGuard AI"


def test_analyze_endpoint_rejects_bad_extension():
    resp = client.post("/api/v1/analysis", files={"file": ("test.txt", b"not audio", "text/plain")})
    assert resp.status_code == 400


def test_analyze_endpoint_full_flow(tmp_path):
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = tmp_path / "sample.wav"
    sf.write(str(path), audio, sr)

    with open(path, "rb") as f:
        resp = client.post("/api/v1/analysis", files={"file": ("sample.wav", f, "audio/wav")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["top_prediction"]["label"] == "Speech"
    assert "features" in body
    assert body["features"]["duration_sec"] == 1.5

    analysis_id = body["id"]

    history_resp = client.get("/api/v1/history")
    assert history_resp.status_code == 200
    assert any(r["id"] == analysis_id for r in history_resp.json())

    detail_resp = client.get(f"/api/v1/history/{analysis_id}")
    assert detail_resp.status_code == 200

    stats_resp = client.get("/api/v1/statistics")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total_analyses"] >= 1


def test_history_404_for_unknown_id():
    resp = client.get("/api/v1/history/does-not-exist")
    assert resp.status_code == 404
