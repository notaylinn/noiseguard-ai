"""
FastAPI dependency-injection wiring.

This is the composition root for the API layer: it constructs concrete
implementations (YamnetClassifier, LibrosaFeatureExtractor,
SqlAlchemyAnalysisRepository) and injects them behind the interfaces
that `AnalysisService` expects. Routes never instantiate services or
repositories directly — they declare a dependency and let FastAPI
resolve it, which keeps routes thin and testable (mockable via
`app.dependency_overrides`).
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.repositories.analysis_repository import SqlAlchemyAnalysisRepository
from backend.app.domain.interfaces import AnalysisRepositoryPort
from backend.app.ml.feature_extraction import LibrosaFeatureExtractor
from backend.app.ml.model import YamnetClassifier
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.export_service import CsvExportService
from backend.app.services.report_service import PdfReportService


@lru_cache
def get_feature_extractor() -> LibrosaFeatureExtractor:
    return LibrosaFeatureExtractor()


def get_classifier() -> YamnetClassifier:
    # Singleton is managed inside YamnetClassifier itself (expensive to load).
    return YamnetClassifier.get_instance()


def get_repository(db: Session = Depends(get_db)) -> AnalysisRepositoryPort:
    return SqlAlchemyAnalysisRepository(db)


def get_analysis_service(
    repository: AnalysisRepositoryPort = Depends(get_repository),
) -> AnalysisService:
    return AnalysisService(
        classifier=get_classifier(),
        feature_extractor=get_feature_extractor(),
        repository=repository,
    )


@lru_cache
def get_report_service() -> PdfReportService:
    return PdfReportService()


@lru_cache
def get_export_service() -> CsvExportService:
    return CsvExportService()
