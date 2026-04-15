"""
Core configuration module using Pydantic BaseSettings.
All settings are env-driven and can be overridden via .env file or environment variables.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

# Resolve project root (two levels up from core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Defaults are production-safe where possible.
    """

    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "ESG Auditor"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(default="change-me-in-production-32chars!!", env="SECRET_KEY")

    # ── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = "/v1"
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        env="CORS_ORIGINS",
    )
    API_RATE_LIMIT: int = Field(default=60, env="API_RATE_LIMIT")  # req/min

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default=f"sqlite+aiosqlite:///{PROJECT_ROOT}/data/esg_auditor.db",
        env="DATABASE_URL",
    )
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")

    # ── Redis / Celery ───────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT_BACKEND")
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=600, env="CELERY_TASK_SOFT_TIME_LIMIT")
    CELERY_TASK_TIME_LIMIT: int = Field(default=900, env="CELERY_TASK_TIME_LIMIT")

    # ── ML Models ────────────────────────────────────────────────────────────
    MODEL_DIR: str = Field(
        default=str(PROJECT_ROOT / "ml" / "models"),
        env="MODEL_DIR",
    )
    MODEL_PATH: str = Field(
        default=str(PROJECT_ROOT / "ml" / "models" / "esg_xgb_v1.pkl"),
        env="MODEL_PATH",
    )
    PIPELINE_PATH: str = Field(
        default=str(PROJECT_ROOT / "ml" / "models" / "feature_pipeline.pkl"),
        env="PIPELINE_PATH",
    )
    AUTO_TRAIN_MODEL: bool = Field(default=True, env="AUTO_TRAIN_MODEL")

    # ── NLP / Embeddings ─────────────────────────────────────────────────────
    ZERO_SHOT_MODEL: str = Field(
        default="facebook/bart-large-mnli", env="ZERO_SHOT_MODEL"
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )
    EMBEDDING_DIM: int = Field(default=384, env="EMBEDDING_DIM")
    DEVICE: str = Field(default="cpu", env="DEVICE")
    MAX_TEXTS_FOR_ZERO_SHOT: int = Field(default=10, env="MAX_TEXTS_FOR_ZERO_SHOT")
    EMERGING_RISK_THRESHOLD: float = Field(default=0.35, env="EMERGING_RISK_THRESHOLD")

    # ── Vector Store ─────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = Field(
        default=str(PROJECT_ROOT / "data" / "faiss_index"),
        env="FAISS_INDEX_PATH",
    )

    # ── Agents ───────────────────────────────────────────────────────────────
    VERBOSE_AGENTS: bool = Field(default=False, env="VERBOSE_AGENTS")
    AGENT_MAX_ITER: int = Field(default=5, env="AGENT_MAX_ITER")

    # ── Reports ──────────────────────────────────────────────────────────────
    REPORTS_DIR: str = Field(
        default=str(PROJECT_ROOT / "outputs"),
        env="REPORTS_DIR",
    )
    ENABLE_PDF_REPORTS: bool = Field(default=False, env="ENABLE_PDF_REPORTS")
    TEMPLATES_DIR: str = Field(
        default=str(PROJECT_ROOT / "templates"),
        env="TEMPLATES_DIR",
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = Field(
        default=str(PROJECT_ROOT / "data"),
        env="DATA_DIR",
    )

    # ── External APIs (optional) ─────────────────────────────────────────────
    NEWS_API_KEY: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # ── Monitoring ───────────────────────────────────────────────────────────
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # "json" | "text"
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")

    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "testing"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v

    @validator("DEVICE")
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            v = "cpu"
        return v

    def get_database_url_sync(self) -> str:
        """Return synchronous SQLAlchemy database URL (strips aiosqlite)."""
        return self.DATABASE_URL.replace("+aiosqlite", "")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


# Module-level singleton for convenient import
settings = get_settings()

# Ensure critical directories exist on import
for _dir in [
    settings.DATA_DIR,
    settings.REPORTS_DIR,
    settings.MODEL_DIR,
    settings.TEMPLATES_DIR,
]:
    os.makedirs(_dir, exist_ok=True)