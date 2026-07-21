# Architecture

NoiseGuard AI follows Clean Architecture: dependencies point inward,
toward the domain layer, and outer layers (API, DB, ML frameworks) are
swappable implementations of inner-layer interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (Streamlit)              — presentation only            │
│  frontend/streamlit_app.py, frontend/pages/*, components/*        │
└───────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (REST, JSON)
┌───────────────────────────▼─────────────────────────────────────┐
│  API layer (FastAPI)               — request/response, wiring     │
│  backend/app/api/routes/*, api/schemas.py, api/dependencies.py    │
└───────────────────────────┬─────────────────────────────────────┘
                             │ depends on interfaces only
┌───────────────────────────▼─────────────────────────────────────┐
│  Service / business logic layer    — orchestration, rules         │
│  backend/app/services/analysis_service.py                         │
│  backend/app/services/report_service.py, export_service.py        │
└───────┬───────────────────────────────────────┬─────────────────┘
        │ implements                            │ implements
┌───────▼────────────────┐           ┌──────────▼──────────────────┐
│  ML layer               │           │  Persistence layer            │
│  backend/app/ml/model.py│           │  backend/app/db/*              │
│  (YAMNet, TF Hub)        │           │  (SQLAlchemy ORM + Repository)│
│  feature_extraction.py   │           │  repositories/analysis_repo.py│
│  (librosa DSP)           │           └────────────────────────────────┘
└──────────────────────────┘
                             ▲
┌────────────────────────────┴────────────────────────────────────┐
│  Domain layer               — framework-agnostic core             │
│  backend/app/domain/entities.py    (dataclasses)                  │
│  backend/app/domain/interfaces.py  (abstract ports)                │
└─────────────────────────────────────────────────────────────────┘
```

## Why this shape

- **`domain/interfaces.py`** defines `SoundClassifierPort`,
  `FeatureExtractorPort` and `AnalysisRepositoryPort`. `AnalysisService`
  (the core business logic) depends only on these abstractions —
  never on `tensorflow`, `tensorflow_hub`, `librosa`, or `sqlalchemy`
  directly. This is Dependency Inversion (the "D" in SOLID) in
  practice: the model or the database can be replaced by implementing
  the same interface, with zero changes to `analysis_service.py`.
- **Single Responsibility** is enforced per module: `model.py` only
  runs YAMNet inference, `feature_extraction.py` only computes DSP
  features, `report_service.py` only builds PDFs, `analysis_repository.py`
  only persists/retrieves records. Each has one reason to change.
- **Repository pattern** (`AnalysisRepositoryPort` /
  `SqlAlchemyAnalysisRepository`) isolates all SQL/ORM code from the
  rest of the app, and makes the service layer trivially testable with
  an in-memory SQLite database (see `tests/test_repository.py`).
- **Dependency injection** is wired in `api/dependencies.py`, the
  composition root. FastAPI's `Depends()` resolves concrete
  implementations at request time; `app.dependency_overrides` lets
  tests substitute fakes (see `tests/test_api.py`) without touching
  production code.
- **Domain entities** (`domain/entities.py`) are plain dataclasses with
  no ORM or Pydantic dependency, so business rules (like loudness
  category thresholds in `analysis_service.py`) don't leak framework
  concerns and are easy to unit test.

## Request lifecycle (analysis)

1. Streamlit page (`frontend/pages/1_Upload_Record.py`) sends multipart
   audio bytes to `POST /api/v1/analysis`.
2. `routes/analysis.py` validates size/extension, stores the raw file
   under `storage/audio/`, loads it as a 16kHz mono waveform
   (`utils/audio_utils.py`).
3. `AnalysisService.analyze()` calls, in order:
   `FeatureExtractorPort.extract()` (librosa DSP features) →
   `SoundClassifierPort.classify()` (real YAMNet inference) →
   internal loudness/category/recommendation logic.
4. The resulting `AnalysisResult` domain entity is persisted via
   `AnalysisRepositoryPort.save()` and returned as JSON
   (`AnalysisResultSchema`).
5. Streamlit renders waveform, prediction bars, confidence gauge,
   loudness badge, interpretation and recommendations from that JSON.

## Data flow for reports/exports

`PdfReportService` and `CsvExportService` both consume
`domain.entities.AnalysisResult` objects fetched through
`AnalysisRepositoryPort` — they never touch the ORM or the ML layer
directly, keeping report formatting fully decoupled from persistence
and inference.
