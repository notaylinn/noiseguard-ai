"""
Centralized application configuration.

Uses pydantic-settings so every configurable value (paths, model name,
DB URL, thresholds) lives in one place and can be overridden via
environment variables or a `.env` file without touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]  # repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NOISEGUARD_", extra="ignore")

    # --- App metadata ---
    app_name: str = "NoiseGuard AI"
    app_version: str = "1.0.0"
    environment: str = "development"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    # --- Storage ---
    base_dir: Path = BASE_DIR
    storage_dir: Path = BASE_DIR / "storage"
    audio_storage_dir: Path = BASE_DIR / "storage" / "audio"
    report_storage_dir: Path = BASE_DIR / "storage" / "reports"
    database_path: Path = BASE_DIR / "storage" / "noiseguard.db"

    # --- Database ---
    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    # --- ML model ---
    yamnet_handle: str = "https://tfhub.dev/google/yamnet/1"
    model_cache_dir: Path = BASE_DIR / ".model_cache"
    sample_rate: int = 16000  # YAMNet requires 16kHz mono input
    min_confidence_display: float = 0.05  # ignore classes below this score
    top_k_predictions: int = 5

    # --- DSP / feature extraction ---
    n_mfcc: int = 13
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512

    # --- Relative loudness thresholds (dBFS-based heuristic, NOT certified dB SPL) ---
    loudness_quiet_max: float = -40.0
    loudness_moderate_max: float = -25.0
    loudness_loud_max: float = -12.0
    # anything above loud_max => "very loud"

    # --- Upload constraints ---
    max_upload_mb: int = 25
    allowed_audio_extensions: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.audio_storage_dir, self.report_storage_dir, self.model_cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
