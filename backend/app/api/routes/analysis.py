"""
Analysis endpoints: upload/record audio and run the full AI pipeline.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api.dependencies import get_analysis_service
from backend.app.api.schemas import AnalysisResultSchema
from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger
from backend.app.services.analysis_service import AnalysisService
from backend.app.utils.audio_utils import (
    downsample_for_plot,
    load_waveform_from_path,
    validate_extension,
)

router = APIRouter(prefix="/analysis", tags=["Analysis"])
logger = get_logger(__name__)


@router.post("", response_model=AnalysisResultSchema)
async def analyze_audio(
    file: UploadFile = File(...),
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResultSchema:
    settings = get_settings()

    if not file.filename or not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.allowed_audio_extensions}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max {settings.max_upload_mb}MB.")
    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    stored_name = f"{uuid.uuid4()}_{file.filename}"
    stored_path = settings.audio_storage_dir / stored_name
    with open(stored_path, "wb") as f:
        f.write(contents)

    try:
        waveform, sr = load_waveform_from_path(stored_path, target_sr=settings.sample_rate)
        if len(waveform) < sr * 0.25:
            raise HTTPException(status_code=400, detail="Audio too short — please provide at least 0.25s of audio.")

        result = service.analyze(waveform, sr, filename=file.filename, persist=True, audio_path=str(stored_path))
        preview = downsample_for_plot(waveform)
        return AnalysisResultSchema.from_entity(result, waveform_preview=preview)
    except ValueError as e:
        logger.warning("Analysis rejected for '%s': %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error analyzing '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {e}") from e
