"""
Application configuration — reads from environment variables via pydantic-settings.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── App ─────────────────────────────────────────────────
    app_name: str = "AI Driver Safety & Intelligence Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    # ─── Server ──────────────────────────────────────────────
    fastapi_url: str = "http://localhost:8000"

    # ─── CORS ────────────────────────────────────────────────
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ─── Supabase (server-side only — NEVER expose to frontend) ──
    supabase_url:              str = ""
    supabase_anon_key:         str = ""   # public anon key — safe for user-scoped requests
    supabase_service_role_key: str = ""   # admin key — NEVER expose outside backend

    # ─── Database ────────────────────────────────────────────
    database_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


# Convenience singleton
settings: Settings = get_settings()
