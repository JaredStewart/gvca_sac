"""Artifact serving endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings

router = APIRouter()


@router.get("/{year}/list")
async def list_artifacts(year: str):
    """List generated artifacts for a year."""
    settings = get_settings()
    artifacts_dir = settings.artifacts_dir

    if not artifacts_dir.exists():
        return {"year": year, "artifacts": []}

    # Find files matching the year
    artifacts = []
    for f in artifacts_dir.iterdir():
        if f.is_file() and year in f.name:
            artifacts.append(
                {
                    "filename": f.name,
                    "type": f.suffix.lstrip("."),
                    "size": f.stat().st_size,
                    "url": f"/api/artifacts/{year}/file/{f.name}",
                }
            )

    # Sort by filename
    artifacts.sort(key=lambda x: x["filename"])

    return {
        "year": year,
        "total": len(artifacts),
        "artifacts": artifacts,
    }


@router.get("/{year}/file/{filename}")
async def get_artifact(year: str, filename: str):
    """Serve a specific artifact file."""
    settings = get_settings()
    file_path = settings.artifacts_dir / filename

    # Security check: ensure file is in artifacts directory
    try:
        file_path = file_path.resolve()
        artifacts_dir = settings.artifacts_dir.resolve()
        if not str(file_path).startswith(str(artifacts_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".html": "text/html",
        ".csv": "text/csv",
        ".json": "application/json",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
    )


@router.get("/list")
async def list_all_artifacts():
    """List all generated artifacts across all years."""
    settings = get_settings()
    artifacts_dir = settings.artifacts_dir

    if not artifacts_dir.exists():
        return {"artifacts": []}

    artifacts = []
    for f in artifacts_dir.iterdir():
        if f.is_file():
            # Try to extract year from filename
            name = f.stem
            year = None
            for part in name.split("_"):
                if part.isdigit() and len(part) == 4:
                    year = part
                    break

            artifacts.append(
                {
                    "filename": f.name,
                    "year": year,
                    "type": f.suffix.lstrip("."),
                    "size": f.stat().st_size,
                    "url": f"/api/artifacts/{year or 'unknown'}/file/{f.name}",
                }
            )

    # Sort by filename
    artifacts.sort(key=lambda x: x["filename"])

    return {
        "total": len(artifacts),
        "artifacts": artifacts,
    }
