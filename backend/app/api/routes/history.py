"""
History, retrieval, deletion and statistics endpoints.
"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import get_repository
from backend.app.api.schemas import AnalysisResultSchema, AnalysisSummarySchema, StatisticsSchema
from backend.app.core.logging_config import get_logger
from backend.app.domain.interfaces import AnalysisRepositoryPort
from backend.app.ml.yamnet_labels import map_label_to_category

router = APIRouter(tags=["History"])
logger = get_logger(__name__)


@router.get("/history", response_model=list[AnalysisSummarySchema])
def get_history(
    limit: int = 50,
    offset: int = 0,
    repository: AnalysisRepositoryPort = Depends(get_repository),
) -> list[AnalysisSummarySchema]:
    results = repository.list_all(limit=limit, offset=offset)
    return [
        AnalysisSummarySchema(
            id=r.id,
            filename=r.filename,
            created_at=r.created_at,
            top_label=r.top_prediction.label,
            top_confidence=r.top_prediction.confidence,
            loudness_category=r.loudness.category.value,
            estimated_dbfs=r.features.estimated_dbfs,
            duration_sec=r.features.duration_sec,
        )
        for r in results
    ]


@router.get("/history/{analysis_id}", response_model=AnalysisResultSchema)
def get_analysis(analysis_id: str, repository: AnalysisRepositoryPort = Depends(get_repository)) -> AnalysisResultSchema:
    result = repository.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisResultSchema.from_entity(result)


@router.delete("/history/{analysis_id}")
def delete_analysis(analysis_id: str, repository: AnalysisRepositoryPort = Depends(get_repository)) -> dict:
    deleted = repository.delete(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"status": "deleted", "id": analysis_id}


@router.get("/statistics", response_model=StatisticsSchema)
def get_statistics(repository: AnalysisRepositoryPort = Depends(get_repository)) -> StatisticsSchema:
    results = repository.list_all(limit=10_000, offset=0)
    total = len(results)

    if total == 0:
        return StatisticsSchema(
            total_analyses=0, category_distribution={}, loudness_distribution={}, average_dbfs=0.0, average_duration_sec=0.0
        )

    category_counter: Counter[str] = Counter()
    loudness_counter: Counter[str] = Counter()
    dbfs_sum = 0.0
    duration_sum = 0.0

    for r in results:
        category = map_label_to_category(r.top_prediction.label)
        category_counter[category] += 1
        loudness_counter[r.loudness.category.value] += 1
        dbfs_sum += r.features.estimated_dbfs
        duration_sum += r.features.duration_sec

    return StatisticsSchema(
        total_analyses=total,
        category_distribution=dict(category_counter),
        loudness_distribution=dict(loudness_counter),
        average_dbfs=round(dbfs_sum / total, 2),
        average_duration_sec=round(duration_sum / total, 2),
    )
