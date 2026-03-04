"""Pipeline manager for handling survey analysis pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import HTTPException

from app.config import get_settings
from app.core.survey_config import SurveyConfig
from app.core.transform import load_flattened, transform_raw


@dataclass
class PipelineState:
    """State of a pipeline for a specific year."""

    year: str
    loaded: bool = False
    data: pl.DataFrame | None = None
    statistics: dict[str, Any] | None = None
    tagging_complete: bool = False
    embeddings_complete: bool = False
    clustering_complete: bool = False
    last_updated: datetime | None = None
    config: SurveyConfig = field(default_factory=SurveyConfig)


class PipelineManager:
    """Manages pipeline instances for different survey years."""

    def __init__(self):
        self._pipelines: dict[str, PipelineState] = {}
        self.settings = get_settings()

    def get_available_years(self) -> list[str]:
        """Get list of available survey years from data directory."""
        data_dir = self.settings.data_dir
        if not data_dir.exists():
            return []

        years = []
        for f in data_dir.glob("*.csv"):
            # Check if filename is a year (4 digits)
            name = f.stem
            if name.isdigit() and len(name) == 4:
                years.append(name)

        return sorted(years, reverse=True)

    def get_pipeline(self, year: str) -> PipelineState | None:
        """Get pipeline state for a year."""
        return self._pipelines.get(year)

    def init_pipeline(self, year: str) -> PipelineState:
        """Initialize a pipeline for a year."""
        if year not in self._pipelines:
            self._pipelines[year] = PipelineState(year=year)
        return self._pipelines[year]

    async def load_data(
        self,
        year: str,
        force_reprocess: bool = False,
        weight_by_parents: bool = False,
    ) -> pl.DataFrame:
        """Load and process survey data for a year."""
        pipeline = self.init_pipeline(year)

        data_path = self.settings.data_dir / f"{year}.csv"
        processed_dir = self.settings.data_dir / "processed"
        processed_path = processed_dir / f"{year}.csv"

        # Check if we need to process raw data
        needs_reprocess = force_reprocess or not processed_path.exists()
        if not needs_reprocess and data_path.exists() and processed_path.exists():
            # Reprocess if raw file is newer than processed file (e.g. encoding fix)
            if data_path.stat().st_mtime > processed_path.stat().st_mtime:
                needs_reprocess = True

        if needs_reprocess:
            if data_path.exists():
                processed_dir.mkdir(parents=True, exist_ok=True)
                transform_raw(str(data_path), str(processed_path))

        # Load the processed data
        if processed_path.exists():
            df = load_flattened(str(processed_path))
            pipeline.data = df
            pipeline.loaded = True
            pipeline.last_updated = datetime.utcnow()
            return df
        elif data_path.exists():
            # Try loading raw data directly if no processing needed
            df = load_flattened(str(data_path))
            pipeline.data = df
            pipeline.loaded = True
            pipeline.last_updated = datetime.utcnow()
            return df
        else:
            raise FileNotFoundError(f"No data file found for year {year}")

    def get_data(self, year: str) -> pl.DataFrame | None:
        """Get loaded data for a year."""
        pipeline = self._pipelines.get(year)
        if pipeline and pipeline.loaded:
            return pipeline.data
        return None

    def _check_embeddings_on_disk(self, year: str) -> bool:
        """Check if embeddings parquet file exists on disk."""
        embeddings_path = self.settings.data_dir / "embeddings" / f"{year}.parquet"
        return embeddings_path.exists()

    def get_status(self, year: str) -> dict[str, Any]:
        """Get status of a pipeline."""
        pipeline = self._pipelines.get(year)
        # Embeddings may exist on disk even if pipeline isn't initialized
        embeddings_on_disk = self._check_embeddings_on_disk(year)

        if not pipeline:
            return {
                "year": year,
                "initialized": False,
                "loaded": False,
                "row_count": 0,
                "tagging_complete": False,
                "embeddings_complete": embeddings_on_disk,
                "clustering_complete": False,
                "last_updated": None,
            }

        return {
            "year": year,
            "initialized": True,
            "loaded": pipeline.loaded,
            "row_count": len(pipeline.data) if pipeline.data is not None else 0,
            "tagging_complete": pipeline.tagging_complete,
            "embeddings_complete": pipeline.embeddings_complete or embeddings_on_disk,
            "clustering_complete": pipeline.clustering_complete,
            "last_updated": pipeline.last_updated.isoformat()
            if pipeline.last_updated
            else None,
        }

    async def ensure_loaded(self, year: str) -> PipelineState:
        """Ensure pipeline data is loaded for a year, auto-loading if needed.

        Returns the PipelineState with data loaded. If no CSV exists,
        raises HTTPException(400).
        """
        pipeline = self.get_pipeline(year)
        if pipeline and pipeline.loaded:
            return pipeline

        try:
            await self.load_data(year)
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail=f"No data file for year {year}",
            )
        return self._pipelines[year]

    def set_tagging_complete(self, year: str, complete: bool = True):
        """Mark tagging as complete for a year."""
        if year in self._pipelines:
            self._pipelines[year].tagging_complete = complete
            self._pipelines[year].last_updated = datetime.utcnow()

    def set_embeddings_complete(self, year: str, complete: bool = True):
        """Mark embeddings as complete for a year."""
        if year in self._pipelines:
            self._pipelines[year].embeddings_complete = complete
            self._pipelines[year].last_updated = datetime.utcnow()

    def set_clustering_complete(self, year: str, complete: bool = True):
        """Mark clustering as complete for a year."""
        if year in self._pipelines:
            self._pipelines[year].clustering_complete = complete
            self._pipelines[year].last_updated = datetime.utcnow()


# Singleton instance
pipeline_manager = PipelineManager()
