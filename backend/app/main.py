"""
FastAPI application entry point.

Wires together middleware, routers and startup/shutdown events. Kept
deliberately thin: no business logic lives here, only application
assembly (Clean Architecture's "frameworks & drivers" outer layer).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import analysis, history, reports
from backend.app.api.schemas import HealthSchema
from backend.app.core.config import get_settings
from backend.app.core.logging_config import configure_logging, get_logger
from backend.app.db.database import init_db

settings = get_settings()
configure_logging(log_dir=settings.storage_dir / "logs")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment)
    init_db()
    logger.info("Database initialized at %s", settings.database_path)
    # Note: the YAMNet model is loaded lazily on first classification
    # request (see ml/model.py singleton) to keep API startup fast;
    # the first /analysis call will take longer while the model loads.
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered environmental noise assessment platform for Kazakhstan. "
        "Classifies environmental sounds with YAMNet, extracts DSP features with "
        "librosa, and generates objective, explainable noise-complaint reports."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix=settings.api_prefix)
app.include_router(history.router, prefix=settings.api_prefix)
app.include_router(reports.router, prefix=settings.api_prefix)


@app.get("/", response_model=HealthSchema, tags=["Health"])
def root() -> HealthSchema:
    return HealthSchema(status="ok", app_name=settings.app_name, version=settings.app_version, model_loaded=False)


@app.get(f"{settings.api_prefix}/health", response_model=HealthSchema, tags=["Health"])
def health() -> HealthSchema:
    from backend.app.ml import model as model_module

    return HealthSchema(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=model_module._instance is not None,
    )
