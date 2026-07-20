"""
PDF report and CSV export endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import io

from backend.app.api.dependencies import get_export_service, get_report_service, get_repository
from backend.app.domain.interfaces import AnalysisRepositoryPort
from backend.app.services.export_service import CsvExportService
from backend.app.services.report_service import PdfReportService

router = APIRouter(tags=["Reports"])


@router.get("/reports/{analysis_id}/pdf")
def download_pdf_report(
    analysis_id: str,
    repository: AnalysisRepositoryPort = Depends(get_repository),
    report_service: PdfReportService = Depends(get_report_service),
) -> FileResponse:
    result = repository.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    pdf_path = report_service.generate(result)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"noiseguard_report_{analysis_id[:8]}.pdf",
    )


@router.get("/export/csv")
def export_csv(
    repository: AnalysisRepositoryPort = Depends(get_repository),
    export_service: CsvExportService = Depends(get_export_service),
) -> StreamingResponse:
    results = repository.list_all(limit=10_000, offset=0)
    csv_content = export_service.to_csv(results)
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=noiseguard_history.csv"},
    )
