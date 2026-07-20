"""
CSV export of analysis history using pandas.
"""
from __future__ import annotations

from io import StringIO

import pandas as pd

from backend.app.core.logging_config import get_logger
from backend.app.domain.entities import AnalysisResult

logger = get_logger(__name__)


class CsvExportService:
    """Converts a list of `AnalysisResult` into a CSV string."""

    @staticmethod
    def to_csv(results: list[AnalysisResult]) -> str:
        rows = []
        for r in results:
            rows.append(
                {
                    "id": r.id,
                    "filename": r.filename,
                    "created_at": r.created_at.isoformat(),
                    "top_label": r.top_prediction.label,
                    "top_confidence": round(r.top_prediction.confidence, 4),
                    "duration_sec": r.features.duration_sec,
                    "estimated_dbfs": r.features.estimated_dbfs,
                    "loudness_category": r.loudness.category.value,
                    "spectral_centroid": r.features.spectral_centroid,
                    "spectral_bandwidth": r.features.spectral_bandwidth,
                    "spectral_rolloff": r.features.spectral_rolloff,
                    "zero_crossing_rate": r.features.zero_crossing_rate,
                    "environmental_interpretation": r.environmental_interpretation,
                }
            )
        df = pd.DataFrame(rows)
        buf = StringIO()
        df.to_csv(buf, index=False)
        logger.info("Exported %d analysis records to CSV", len(rows))
        return buf.getvalue()
