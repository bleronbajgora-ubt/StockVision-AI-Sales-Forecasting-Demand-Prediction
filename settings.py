"""
=============================================================================
 app/core/settings.py — Application Settings
=============================================================================
 WHY pydantic-settings:
   Validates every environment variable at startup. If DATABASE_URL is
   missing or SECRET_KEY is too short, the app refuses to start with a
   clear error — far better than a cryptic crash at the first DB call.

 WHY a Settings class instead of os.environ.get() scattered everywhere:
   - One place to see all configuration
   - Type coercion (string "true" → bool True automatically)
   - IDE autocomplete on settings.SECRET_KEY instead of magic strings
=============================================================================
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────
    APP_NAME:        str = "Amazon ML Review Analyzer API"
    APP_VERSION:     str = "1.0.0"
    DEBUG:           bool = False
    ENVIRONMENT:     str = "production"
    API_V1_PREFIX:   str = "/api/v1"

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL:       str = "postgresql+asyncpg://postgres:password@localhost:5432/amazon_ml_db"
    DATABASE_URL_SYNC:  str = "postgresql://postgres:password@localhost:5432/amazon_ml_db"

    # Pool settings — WHY: asyncpg pools connections for reuse, avoiding
    # the overhead of a new TCP connection per query (~5-20ms each).
    DB_POOL_SIZE:     int = 10
    DB_MAX_OVERFLOW:  int = 20
    DB_POOL_TIMEOUT:  int = 30

    # ── Security / JWT ────────────────────────────────────────────────────
    SECRET_KEY:                    str = "change-this-to-a-real-secret-key-min-32-chars"
    ALGORITHM:                     str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:   int = 30
    REFRESH_TOKEN_EXPIRE_DAYS:     int = 7

    # ── CORS ──────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # ── ML ────────────────────────────────────────────────────────────────
    ML_MODEL_BUNDLE_PATH: str = "ml/models/model_bundle.pkl"

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_must_be_long(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters for security.")
        return v

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def ml_model_path(self) -> Path:
        return Path(self.ML_MODEL_BUNDLE_PATH)


@lru_cache()
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    WHY @lru_cache:
      Settings reads from .env on every instantiation.
      lru_cache ensures the file is read once at startup and the same
      object is returned on every subsequent call — zero overhead.

    Used as FastAPI dependency: Depends(get_settings)
    """
    return Settings()


settings = get_settings()