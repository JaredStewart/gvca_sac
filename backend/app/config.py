"""Application configuration settings."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "GVCA SAC API"
    debug: bool = False

    # Paths
    data_dir: Path = Path(os.environ.get("DATA_DIR", "/app/data"))
    artifacts_dir: Path = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # PocketBase
    pocketbase_url: str = "http://localhost:8090"

    # OpenAI
    openai_api_key: str = ""
    default_llm_model: str = "gpt-5-nano"

    # Tagging defaults
    default_n_samples: int = 4
    default_threshold: int = 2

    # Clustering defaults
    default_embed_model: str = "text-embedding-3-small"
    default_min_cluster_size: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
