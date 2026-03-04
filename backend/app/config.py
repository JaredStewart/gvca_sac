"""Application configuration settings."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "GVCA SAC API"
    debug: bool = False

    # Paths
    data_dir: Path = Path("/app/data")
    artifacts_dir: Path = Path("/app/artifacts")

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
