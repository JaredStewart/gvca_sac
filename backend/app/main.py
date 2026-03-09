"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

# Configure logging early, before any other imports that use loggers
_settings = get_settings()
logging.basicConfig(
    level=logging.DEBUG if _settings.debug else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from app.routers import (
    artifacts,
    batch_jobs,
    charts,
    cluster_metadata,
    clustering,
    config_api,
    data,
    deck,
    jobs,
    pipeline,
    tagging,
    tags,
)
from app.services.job_queue import job_queue
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    await job_queue.shutdown()
    await pb_client.close()


app = FastAPI(
    title="GVCA SAC API",
    description="Survey Analysis Pipeline API for Golden View Classical Academy",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
_cors_origins = get_settings().cors_origins.split(",")
if os.environ.get("CODESPACES") == "true":
    _cors_origins.append("*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(tagging.router, prefix="/api/tagging", tags=["Tagging"])
app.include_router(tags.router, prefix="/api/tags", tags=["Tags"])
app.include_router(clustering.router, prefix="/api/clustering", tags=["Clustering"])
app.include_router(cluster_metadata.router, prefix="/api/clustering", tags=["Cluster Metadata"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["Artifacts"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(config_api.router, prefix="/api/config", tags=["Config"])
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"])
app.include_router(batch_jobs.router, prefix="/api/batch-jobs", tags=["Batch Jobs"])
app.include_router(deck.router, prefix="/api/deck", tags=["Deck"])

# Mount artifacts directory for static file serving
settings = get_settings()
if settings.artifacts_dir.exists():
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(settings.artifacts_dir)),
        name="artifacts",
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "GVCA SAC API",
        "version": "0.1.0",
        "docs": "/docs",
    }
