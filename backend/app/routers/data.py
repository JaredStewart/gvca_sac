"""Data and statistics endpoints."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.survey_config import FREE_RESPONSE_QUESTIONS, LEVELS, QUESTIONS, get_taxonomy_tags
from app.models.data import ImportResult
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class DataLoadRequest(BaseModel):
    force_reprocess: bool = False


# ========== Pydantic Models for Tagging Interface ==========

class KeywordMismatch(BaseModel):
    """Represents a keyword that was found but the corresponding tag is missing."""
    tag: str
    keywords: list[str]


class TaggableResponse(BaseModel):
    """Unified response structure combining free_response + tagging data."""
    id: str | None  # tagging_results record ID (None if not tagged yet)
    response_id: str
    respondent_id: str
    question: str  # Q8 or Q9
    question_type: str  # "praise" or "improvement"
    level: str  # Grammar/Middle/High/Generic
    response_text: str

    # Tags (from tagging_results, merged with overrides)
    tags: list[str]
    original_tags: list[str]
    has_override: bool

    # Quality indicators
    stability_score: float | None
    tag_votes: dict[str, int]
    keyword_mismatches: list[KeywordMismatch]

    # Status
    dismissed: bool
    dismissed_at: str | None


class TaggableResponsePage(BaseModel):
    """Paginated response for the tagging table."""
    items: list[TaggableResponse]
    page: int
    perPage: int
    totalItems: int
    totalPages: int


# Year validation pattern
YEAR_PATTERN = re.compile(r"^20\d{2}$")


def validate_year(year: str) -> None:
    """Validate year format."""
    if not YEAR_PATTERN.match(year):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid year format: {year}. Expected format: 20XX",
        )


@router.post("/import/{year}", response_model=ImportResult)
async def import_survey_data(
    year: str,
    file: UploadFile = File(...),
    replace_existing: bool = Form(default=False),
):
    """
    Import survey data from CSV file (FR-001, FR-002, FR-003).

    Transforms raw Survey Monkey CSV and persists to database.
    """
    validate_year(year)

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file",
        )

    try:
        from app.core.transform import transform_and_persist

        content = await file.read()
        result = await transform_and_persist(content, year, replace_existing)

        return ImportResult(**result)

    except ValueError as e:
        # Data already exists
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/load")
async def load_data(year: str, request: DataLoadRequest):
    """Load or reload data for a year (legacy in-memory approach)."""
    try:
        df = await pipeline_manager.load_data(
            year, force_reprocess=request.force_reprocess
        )
        return {
            "year": year,
            "status": "loaded",
            "row_count": len(df),
            "columns": df.columns,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/db-responses")
async def get_db_responses(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    school_level: str | None = Query(None, description="Filter by school level"),
    demographic: str | None = Query(
        None,
        description="Filter by demographic: year1, minority, support, tenure_3plus",
    ),
    sort: str = Query("-imported_at", description="Sort field with direction"),
):
    """
    Get paginated survey responses from database (FR-005).

    Returns survey responses with demographics and satisfaction scores.
    """
    validate_year(year)

    try:
        result = await pb_client.list_survey_responses(
            year=year,
            page=page,
            per_page=per_page,
            school_level=school_level,
            demographic=demographic,
            sort=sort,
        )
        return result
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/free-responses")
async def get_free_responses(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    question_type: str | None = Query(
        None, description="Filter by type: praise or improvement"
    ),
    level: str | None = Query(None, description="Filter by level: Grammar, Middle, High, Generic"),
    question: str | None = Query(None, description="Filter by question text (partial match)"),
):
    """
    Get paginated free-text responses from database (FR-006).
    """
    validate_year(year)

    try:
        result = await pb_client.list_free_responses(
            year=year,
            page=page,
            per_page=per_page,
            question_type=question_type,
            level=level,
            question=question,
        )
        return result
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/free-response-counts")
async def get_free_response_counts(year: str):
    """
    Get counts of Good Choice vs Better Serve respondents.

    Returns total counts, only-positive counts, and overlap.
    Groups by survey_response_id to count unique respondents.
    """
    validate_year(year)

    try:
        all_responses = await pb_client.get_full_list(
            "free_responses",
            filter_str=f'year = "{year}"',
        )

        # Group by survey_response_id to find which respondents gave which types
        respondent_types: dict[str, set[str]] = {}
        for r in all_responses:
            sid = r.get("survey_response_id") or r.get("response_id", "")
            qt = r.get("question_type", "")
            if sid not in respondent_types:
                respondent_types[sid] = set()
            respondent_types[sid].add(qt)

        total_gc = 0
        total_bs = 0
        only_gc = 0
        only_bs = 0
        both = 0

        for types in respondent_types.values():
            has_praise = "praise" in types
            has_improvement = "improvement" in types
            if has_praise:
                total_gc += 1
            if has_improvement:
                total_bs += 1
            if has_praise and has_improvement:
                both += 1
            elif has_praise:
                only_gc += 1
            elif has_improvement:
                only_bs += 1

        total_respondents = len(respondent_types)
        only_positive_pct = round(only_gc / total_respondents * 100, 1) if total_respondents > 0 else 0

        return {
            "total_good_choice": total_gc,
            "total_better_serve": total_bs,
            "only_good_choice": only_gc,
            "only_better_serve": only_bs,
            "both": both,
            "only_positive_pct": only_positive_pct,
        }
    except Exception as e:
        logger.error("Error computing free response counts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/free-responses-with-tags", response_model=TaggableResponsePage)
async def get_free_responses_with_tags(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    level: str | None = Query(None, description="Filter by level"),
    question_type: str | None = Query(None, description="Filter by type: praise or improvement"),
    has_keyword_mismatch: bool | None = Query(None, description="Filter to responses with missed keywords"),
    min_stability: float | None = Query(None, ge=0, le=1, description="Minimum stability score"),
    max_stability: float | None = Query(None, ge=0, le=1, description="Maximum stability score"),
    has_tag: str | None = Query(None, description="Filter to responses with specific tag"),
    missing_tag: str | None = Query(None, description="Filter to responses missing specific tag"),
    include_dismissed: bool = Query(False, description="Include dismissed responses"),
    sort: str = Query("response_id", description="Sort field"),
):
    """
    Get free responses with full tag data for the tagging table view.

    Returns paginated responses with their tagging data, stability scores,
    keyword mismatches, and override status.
    """
    validate_year(year)

    try:
        # Determine if we need full-fetch (stability sort or any post-filter)
        needs_full_fetch = sort in ("stability_asc", "stability_desc") or \
            has_keyword_mismatch is not None or \
            min_stability is not None or \
            max_stability is not None or \
            has_tag is not None or \
            missing_tag is not None or \
            not include_dismissed

        # Build filter for free_responses
        filters = [f'year = "{year}"']
        if level:
            filters.append(f'level = "{level}"')
        if question_type:
            filters.append(f'question_type = "{question_type}"')

        filter_str = " && ".join(filters)

        # Determine PocketBase sort (only for non-stability sorts)
        pb_sort = sort if sort not in ("stability_asc", "stability_desc") else "response_id"

        if needs_full_fetch:
            # Fetch ALL free_responses for accurate filtering + sorting
            all_free_responses = await pb_client.get_full_list(
                "free_responses",
                filter_str=filter_str,
                sort=pb_sort,
            )
        else:
            # PocketBase-paginated path (no post-filters needed)
            result = await pb_client.get_list(
                "free_responses",
                page=page,
                per_page=per_page,
                filter_str=filter_str,
                sort=pb_sort,
            )
            all_free_responses = result.get("items", [])

        # Enrich with tagging data
        items = []
        for fr in all_free_responses:
            response_id = fr["response_id"]

            # Get tagging result
            tagging = await pb_client.get_first_list_item(
                "tagging_results",
                filter_str=f'response_id = "{response_id}"',
            )

            # Get override if exists
            override = await pb_client.get_first_list_item(
                "tag_overrides",
                filter_str=f'response_id = "{response_id}"',
            )

            # Get effective tags (override takes precedence)
            effective_tags = []
            original_tags = []
            if tagging:
                original_tags = tagging.get("llm_tags") or []
                effective_tags = (override.get("modified_tags") if override else None) or original_tags
            elif override:
                effective_tags = override.get("modified_tags") or []

            # Apply post-filters that depend on tagging data
            stability = tagging.get("stability_score") if tagging else None
            keyword_mismatches = tagging.get("keyword_mismatches") or [] if tagging else []
            dismissed = tagging.get("dismissed", False) if tagging else False

            # Skip based on filters
            if min_stability is not None and (stability is None or stability < min_stability):
                continue
            if max_stability is not None and (stability is None or stability > max_stability):
                continue
            if has_keyword_mismatch is True and not keyword_mismatches:
                continue
            if has_keyword_mismatch is False and keyword_mismatches:
                continue
            if not include_dismissed and dismissed:
                continue
            if has_tag and has_tag not in effective_tags:
                continue
            if missing_tag and missing_tag in effective_tags:
                continue

            # Build response object
            has_override_flag = override is not None
            effective_stability = stability if tagging else None

            item = TaggableResponse(
                id=tagging["id"] if tagging else None,
                response_id=response_id,
                respondent_id=fr.get("survey_response_id") or fr.get("response_id", "").split("_")[0],
                question=fr["question"],
                question_type=fr["question_type"],
                level=fr["level"],
                response_text=fr["response_text"],
                tags=effective_tags,
                original_tags=original_tags,
                has_override=has_override_flag,
                stability_score=effective_stability,
                tag_votes=tagging.get("tag_votes") or {} if tagging else {},
                keyword_mismatches=[
                    KeywordMismatch(tag=km["tag"], keywords=km["keywords"])
                    for km in keyword_mismatches
                ] if keyword_mismatches else [],
                dismissed=dismissed,
                dismissed_at=tagging.get("dismissed_at") if tagging else None,
            )
            items.append(item)

        # Apply stability sort if requested
        if sort == "stability_asc":
            items.sort(key=lambda x: (x.stability_score is None, x.stability_score if x.stability_score is not None else 0.0))
        elif sort == "stability_desc":
            items.sort(key=lambda x: (x.stability_score is None, -(x.stability_score if x.stability_score is not None else 0.0)))

        if needs_full_fetch:
            # Manual pagination with accurate counts
            total_items = len(items)
            total_pages = max(1, (total_items + per_page - 1) // per_page)
            start = (page - 1) * per_page
            end = start + per_page
            paged_items = items[start:end]

            return TaggableResponsePage(
                items=paged_items,
                page=page,
                perPage=per_page,
                totalItems=total_items,
                totalPages=total_pages,
            )
        else:
            return TaggableResponsePage(
                items=items,
                page=result.get("page", page),
                perPage=result.get("perPage", per_page),
                totalItems=result.get("totalItems", 0),
                totalPages=result.get("totalPages", 0),
            )

    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/tagging-export")
async def export_tagging_data(year: str):
    """
    Export tagging results as CSV with stability scores and keyword flags.

    Each taxonomy tag gets its own boolean column (marked "x" if present).
    """
    validate_year(year)

    import csv
    import io

    try:
        # Fetch all free responses for the year
        all_free_responses = await pb_client.get_full_list(
            "free_responses",
            filter_str=f'year = "{year}"',
            sort="response_id",
        )

        if not all_free_responses:
            raise HTTPException(
                status_code=404,
                detail=f"No free responses found for year {year}",
            )

        taxonomy_tags = get_taxonomy_tags()

        # Bulk-fetch clustering results for this year
        all_clustering = await pb_client.get_full_list(
            "clustering_results",
            filter_str=f'year = "{year}"',
        )
        # Primary lookup by response_id; fallback by response_text
        # (handles ID mismatches after respondent ID fixes)
        cluster_by_response_id: dict[str, int | str] = {}
        cluster_by_text: dict[str, int | str] = {}
        for cr in all_clustering:
            cid = cr.get("cluster_id")
            if cid is not None:
                cluster_by_response_id[cr["response_id"]] = cid
                text = cr.get("response_text", "")
                if text:
                    cluster_by_text[text] = cid

        output = io.StringIO()
        headers = [
            "response_id",
            "respondent_id",
            "level",
            "question",
            "question_type",
            "response_text",
            *taxonomy_tags,
            "cluster_id",
            "stability_score",
            "keyword_mismatches",
            "has_override",
        ]
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        for fr in all_free_responses:
            response_id = fr["response_id"]

            # Get tagging result
            tagging = await pb_client.get_first_list_item(
                "tagging_results",
                filter_str=f'response_id = "{response_id}"',
            )

            # Get override if exists
            override = await pb_client.get_first_list_item(
                "tag_overrides",
                filter_str=f'response_id = "{response_id}"',
            )

            # Effective tags
            effective_tags = []
            if tagging:
                original_tags = tagging.get("llm_tags") or []
                effective_tags = (override.get("modified_tags") if override else None) or original_tags
            elif override:
                effective_tags = override.get("modified_tags") or []

            stability = tagging.get("stability_score") if tagging else None
            keyword_mismatches = tagging.get("keyword_mismatches") or [] if tagging else []
            has_override = override is not None

            # Format keyword mismatches as "tag:kw1,kw2; tag2:kw3"
            km_str = "; ".join(
                f"{km['tag']}:{','.join(km['keywords'])}"
                for km in keyword_mismatches
            ) if keyword_mismatches else ""

            # Lookup cluster_id: try response_id first, fall back to response_text
            cid = cluster_by_response_id.get(response_id)
            if cid is None:
                cid = cluster_by_text.get(fr["response_text"])

            row = {
                "response_id": response_id,
                "respondent_id": fr.get("survey_response_id") or response_id.split("_")[0],
                "level": fr["level"],
                "question": fr["question"],
                "question_type": fr["question_type"],
                "response_text": fr["response_text"],
                "cluster_id": cid if cid is not None else "",
                "stability_score": stability if stability is not None else "",
                "keyword_mismatches": km_str,
                "has_override": has_override,
            }

            # One column per tag: "x" if present, blank if not
            effective_tags_set = set(effective_tags)
            for tag in taxonomy_tags:
                row[tag] = "x" if tag in effective_tags_set else ""

            writer.writerow(row)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=tagging_{year}.csv"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/export")
async def export_data(year: str):
    """
    Export normalized survey data as CSV (FR-004).

    Streams CSV file from database records.
    """
    validate_year(year)

    import csv
    import io

    try:
        # Get all survey responses for the year
        responses = await pb_client.get_full_list(
            "survey_responses",
            filter_str=f'year = "{year}"',
        )

        if not responses:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for year {year}",
            )

        # Build CSV in memory
        output = io.StringIO()
        writer = None

        for resp in responses:
            if writer is None:
                # Use first record to determine headers
                headers = [
                    "year",
                    "respondent_id",
                    "school_level",
                    "submission_method",
                    "n_parents_represented",
                    "minority",
                    "support",
                    "tenure_years",
                    "year1_family",
                ]
                # Add satisfaction score columns
                if resp.get("satisfaction_scores"):
                    for q_key in sorted(resp["satisfaction_scores"].keys()):
                        for level in LEVELS:
                            headers.append(f"{q_key}_{level}")
                writer = csv.DictWriter(output, fieldnames=headers)
                writer.writeheader()

            # Flatten record
            demographics = resp.get("demographics", {}) or {}
            row = {
                "year": resp["year"],
                "respondent_id": resp["respondent_id"],
                "school_level": resp.get("school_level"),
                "submission_method": resp.get("submission_method"),
                "n_parents_represented": resp.get("n_parents_represented"),
                "minority": demographics.get("minority"),
                "support": demographics.get("support"),
                "tenure_years": demographics.get("tenure_years"),
                "year1_family": demographics.get("year1_family"),
            }

            # Add satisfaction scores
            scores = resp.get("satisfaction_scores", {}) or {}
            for q_key in sorted(scores.keys()):
                for level in LEVELS:
                    row[f"{q_key}_{level}"] = scores[q_key].get(level)

            writer.writerow(row)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=survey_data_{year}.csv"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/statistics")
async def get_statistics(year: str, weight_by_parents: bool = Query(True)):
    """Get computed statistics for a year."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    # Import and compute statistics
    from app.core.analysis import compute_statistics

    try:
        stats = compute_statistics(pipeline.data, pipeline.config, weight_by_parents=weight_by_parents)
        pipeline.statistics = stats
        return stats
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/responses")
async def get_responses(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    level: str | None = None,
    question: str | None = None,
    has_response: bool | None = None,
):
    """Get paginated survey responses."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    df = pipeline.data

    # Apply filters
    if level:
        df = df.filter(df["Level"].str.contains(level))

    if has_response is not None:
        # Filter for rows that have at least one free response
        free_response_cols = [
            c for c in df.columns if "good" in c.lower() or "serve" in c.lower()
        ]
        if free_response_cols:
            if has_response:
                df = df.filter(
                    df.select(free_response_cols).is_not_null().any_horizontal()
                )
            else:
                df = df.filter(
                    df.select(free_response_cols).is_null().all_horizontal()
                )

    # Calculate pagination
    total = len(df)
    total_pages = (total + per_page - 1) // per_page
    offset = (page - 1) * per_page

    # Get page of data
    page_df = df.slice(offset, per_page)

    return {
        "year": year,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "items": page_df.to_dicts(),
    }


@router.get("/{year}/summary")
async def get_summary(year: str):
    """Get summary information for a year's data."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    df = pipeline.data

    # Calculate summary stats
    summary = {
        "year": year,
        "total_responses": len(df),
        "columns": df.columns,
    }

    # Count by level if Level column exists
    if "Level" in df.columns:
        level_counts = df.group_by("Level").len().to_dicts()
        summary["by_level"] = {r["Level"]: r["len"] for r in level_counts}

    # Count free responses
    free_response_cols = [
        c
        for c in df.columns
        if "what makes" in c.lower() or "how" in c.lower() and "serve" in c.lower()
    ]
    if free_response_cols:
        non_empty = 0
        for col in free_response_cols:
            non_empty += df.filter(df[col].is_not_null()).height
        summary["free_response_count"] = non_empty

    return summary


class SyncRequest(BaseModel):
    force: bool = False
    run_async: bool = False  # If True, sync in background and return immediately


DEMOGRAPHIC_SEGMENTS = {
    "year_1": {
        "name": "Year 1 Families",
        "filter": lambda df: df.filter(
            (df["Years at GVCA"] == "1") |
            (df["Years at GVCA"] == "1 year") |
            (df["Years at GVCA"].str.starts_with("1"))
        ),
        "inverse_name": "Not Year 1",
    },
    "minority": {
        "name": "Minority",
        "filter": lambda df: df.filter(
            (df["Minority"].str.to_lowercase() == "yes")
        ),
        "inverse_name": "Not Minority",
    },
    "support": {
        "name": "Support (IEP/504/ALP)",
        "filter": lambda df: df.filter(
            (df["IEP, 504, ALP, or Read"].str.to_lowercase() == "yes")
        ),
        "inverse_name": "No Support",
    },
    "year_3_or_less": {
        "name": "Year 3 or Less",
        "filter": lambda df: df.filter(
            (df["Years at GVCA"] == "1") |
            (df["Years at GVCA"] == "2") |
            (df["Years at GVCA"] == "3") |
            (df["Years at GVCA"].str.starts_with("1")) |
            (df["Years at GVCA"].str.starts_with("2")) |
            (df["Years at GVCA"].str.starts_with("3"))
        ),
        "inverse_name": "Year 4+",
    },
}


async def _do_sync_to_pocketbase(year: str, force: bool = False) -> dict:
    """
    Internal function to sync survey data to PocketBase.

    Extracted to be callable from both sync and async contexts.
    """
    pipeline = await pipeline_manager.ensure_loaded(year)

    df = pipeline.data

    # If force is False, check if data already exists
    if not force:
        existing = await pb_client.get_list(
            "survey_responses",
            filter_str=f'year = "{year}"',
            per_page=1,
        )
        if existing.get("totalItems", 0) > 0:
            return {
                "year": year,
                "status": "already_synced",
                "existing_count": existing["totalItems"],
                "message": "Data already synced. Use force=true to re-sync.",
            }

    # Clear existing data for this year if force is True
    if force:
        deleted_survey = await pb_client.delete_by_filter(
            "survey_responses",
            f'year = "{year}"',
        )
        deleted_free = await pb_client.delete_by_filter(
            "free_responses",
            f'year = "{year}"',
        )
        logger.info(f"Deleted {deleted_survey} survey responses and {deleted_free} free responses for year {year}")

    synced = 0
    errors = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        respondent_id = row.get("Respondent ID", str(idx))

        try:
            # Determine level(s) for this respondent based on responses
            level_parts = []
            if row.get("Grade Selection"):
                grade_sel = row.get("Grade Selection", "")
                for level in LEVELS:
                    if level in grade_sel:
                        level_parts.append(level)
            level_str = ", ".join(level_parts) if level_parts else None

            # Parse years at GVCA
            years_at_gvca = None
            years_str = row.get("Years at GVCA")
            if years_str:
                try:
                    # Handle "X years" or just "X" formats
                    years_at_gvca = int(str(years_str).split()[0])
                except (ValueError, IndexError):
                    pass

            # Determine segment flags
            is_year_1 = years_at_gvca == 1 if years_at_gvca is not None else None

            support_val = row.get("IEP, 504, ALP, or Read")
            has_support = support_val is not None and support_val.lower() == "yes" if support_val else None

            minority_val = row.get("Minority")
            is_minority = minority_val is not None and minority_val.lower() == "yes" if minority_val else None

            # Get n_parents
            n_parents = row.get("N Parents Represented")
            if isinstance(n_parents, str):
                try:
                    n_parents = int(n_parents)
                except ValueError:
                    n_parents = 1
            elif n_parents is None:
                n_parents = 1

            # Build question_responses dict: {level: {question_short: answer}}
            question_responses: dict[str, dict[str, str | None]] = {}
            for level in LEVELS:
                level_responses: dict[str, str | None] = {}
                for q_idx, question in enumerate(QUESTIONS):
                    if question not in FREE_RESPONSE_QUESTIONS:
                        col_name = f"({level}) {question}"
                        if col_name in df.columns:
                            val = row.get(col_name)
                            if val is not None:
                                level_responses[f"Q{q_idx + 1}"] = val
                if level_responses:
                    question_responses[level] = level_responses

            # Build free_responses dict: {level: {question_short: text}}
            free_responses: dict[str, dict[str, str | None]] = {}
            for level in LEVELS:
                level_free: dict[str, str | None] = {}
                for q_idx, question in enumerate(QUESTIONS):
                    if question in FREE_RESPONSE_QUESTIONS:
                        col_name = f"({level}) {question}"
                        if col_name in df.columns:
                            val = row.get(col_name)
                            if val is not None and str(val).strip():
                                level_free[f"Q{q_idx + 1}"] = val
                if level_free:
                    free_responses[level] = level_free

            # Also check Generic responses
            generic_free: dict[str, str | None] = {}
            for q_idx, question in enumerate(QUESTIONS):
                if question in FREE_RESPONSE_QUESTIONS:
                    col_name = f"(Generic) {question}"
                    if col_name in df.columns:
                        val = row.get(col_name)
                        if val is not None and str(val).strip():
                            generic_free[f"Q{q_idx + 1}"] = val
            if generic_free:
                free_responses["Generic"] = generic_free

            # Build demographics object per schema
            demographics = {}
            if years_at_gvca is not None:
                demographics["years_at_gvca"] = years_at_gvca
            if is_year_1 is not None:
                demographics["is_year_1"] = is_year_1
            if has_support is not None:
                demographics["has_support"] = has_support
            if is_minority is not None:
                demographics["is_minority"] = is_minority

            record = {
                "year": year,
                "respondent_id": str(respondent_id),
                "school_level": level_str,
                "n_parents_represented": n_parents,
                "demographics": demographics if demographics else None,
                "satisfaction_scores": question_responses if question_responses else None,
                "imported_at": datetime.utcnow().isoformat() + "Z",
            }

            survey_record = await pb_client.create("survey_responses", record)

            # Create free_responses records for each free-text response
            for resp_level, level_responses in free_responses.items():
                for q_key, text in level_responses.items():
                    if text and str(text).strip():
                        # Determine question_type based on question key (Q8=praise, Q9=improvement)
                        question_type = "praise" if q_key == "Q8" else "improvement"
                        response_id = f"{respondent_id}_{resp_level}_{q_key}"

                        free_record = {
                            "year": year,
                            "response_id": response_id,
                            "survey_response_id": survey_record.get("id"),
                            "question": q_key,
                            "question_type": question_type,
                            "level": resp_level,
                            "response_text": str(text).strip(),
                        }
                        try:
                            await pb_client.create("free_responses", free_record)
                        except Exception as free_err:
                            logger.warning(
                                "Failed to create free_response %s: %s",
                                response_id, free_err,
                            )
                            errors.append({"response_id": response_id, "error": str(free_err)})
            synced += 1

        except Exception as e:
            import traceback
            logger.error(f"Error syncing respondent {respondent_id}: {e}")
            logger.error(f"Record data: {record}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            errors.append({"respondent_id": respondent_id, "error": str(e)})

    return {
        "year": year,
        "status": "completed",
        "synced": synced,
        "total": df.height,
        "errors": errors[:10] if errors else [],
        "error_count": len(errors),
    }


def _log_sync_error(task: asyncio.Task, year: str) -> None:
    """Callback to log errors from background sync tasks."""
    if task.cancelled():
        logger.warning("Background sync for year %s was cancelled", year)
    elif task.exception():
        logger.error(
            "Background sync for year %s failed: %s",
            year,
            task.exception(),
            exc_info=task.exception(),
        )


@router.post("/{year}/sync-to-pocketbase")
async def sync_to_pocketbase(year: str, request: SyncRequest):
    """
    Sync survey data to PocketBase for faster frontend queries.

    This endpoint transforms the raw survey data into a structured format
    and stores it in the survey_responses collection.

    If run_async=true, returns immediately and syncs in background.
    """
    validate_year(year)

    pipeline = await pipeline_manager.ensure_loaded(year)

    if request.run_async:
        # Run sync in background with error logging
        task = asyncio.create_task(_do_sync_to_pocketbase(year, request.force))
        task.add_done_callback(lambda t: _log_sync_error(t, year))
        return {
            "year": year,
            "status": "sync_started",
            "message": "Sync started in background. Data will be available shortly.",
        }

    # Synchronous execution
    try:
        result = await _do_sync_to_pocketbase(year, request.force)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{year}/data-status")
async def get_data_status(year: str):
    """
    Check if data exists in PocketBase for a year.

    Returns count of survey responses and free responses.
    """
    validate_year(year)

    try:
        survey_count = await pb_client.get_survey_response_count(year)
        free_count_result = await pb_client.get_list(
            "free_responses",
            filter_str=f'year = "{year}"',
            per_page=1,
        )
        free_response_count = free_count_result.get("totalItems", 0)
        return {
            "year": year,
            "has_data": survey_count > 0,
            "survey_response_count": survey_count,
            "free_response_count": free_response_count,
        }
    except Exception as e:
        logger.error(f"Error checking data status for year {year}: {e}")
        return {
            "year": year,
            "has_data": False,
            "survey_response_count": 0,
            "free_response_count": 0,
            "error": str(e),
        }


@router.get("/{year}/unified-responses")
async def get_unified_responses(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    # Filter parameters
    level: str | None = Query(None, description="Filter by school level"),
    is_minority: bool | None = Query(None, description="Filter by minority status"),
    has_support: bool | None = Query(None, description="Filter by support (IEP/504/ALP)"),
    is_year_1: bool | None = Query(None, description="Filter by Year 1 family status"),
    min_tenure: int | None = Query(None, description="Minimum years at GVCA"),
    max_tenure: int | None = Query(None, description="Maximum years at GVCA"),
    sort: str = Query("respondent_id", description="Sort field"),
):
    """
    Get unified survey responses with all data (survey + free responses) combined.

    Returns denormalized rows with demographics, satisfaction scores, and free responses.
    No tag columns - tags are handled in the Tagging page.
    """
    validate_year(year)

    try:
        # Build filter
        filters = [f'year = "{year}"']

        if level:
            filters.append(f'school_level ~ "{level}"')
        if is_minority is not None:
            filters.append(f'demographics.is_minority = {str(is_minority).lower()}')
        if has_support is not None:
            filters.append(f'demographics.has_support = {str(has_support).lower()}')
        if is_year_1 is not None:
            filters.append(f'demographics.is_year_1 = {str(is_year_1).lower()}')
        if min_tenure is not None:
            filters.append(f'demographics.years_at_gvca >= {min_tenure}')
        if max_tenure is not None:
            filters.append(f'demographics.years_at_gvca <= {max_tenure}')

        filter_str = " && ".join(filters)

        # Get survey responses
        result = await pb_client.get_list(
            "survey_responses",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort=sort,
        )

        # Transform into unified format
        items = []
        for resp in result.get("items", []):
            satisfaction_scores = resp.get("satisfaction_scores") or {}
            demographics = resp.get("demographics") or {}

            # Build satisfaction structure - Q1-Q7 for each level
            satisfaction = {}
            for q_idx in range(1, 8):
                q_key = f"Q{q_idx}"
                satisfaction[q_key] = {
                    "Grammar": satisfaction_scores.get("Grammar", {}).get(q_key),
                    "Middle": satisfaction_scores.get("Middle", {}).get(q_key),
                    "High": satisfaction_scores.get("High", {}).get(q_key),
                }

            unified = {
                "id": resp["id"],
                "respondent_id": resp["respondent_id"],
                "year": resp["year"],
                "school_level": resp.get("school_level"),
                # Demographics (from JSON object)
                "is_minority": demographics.get("is_minority"),
                "has_support": demographics.get("has_support"),
                "years_at_gvca": demographics.get("years_at_gvca"),
                "is_year_1": demographics.get("is_year_1"),
                "n_parents": resp.get("n_parents_represented", 1),
                # Satisfaction scores
                "satisfaction": satisfaction,
            }
            items.append(unified)

        return {
            "year": year,
            "page": result.get("page", page),
            "perPage": result.get("perPage", per_page),
            "totalItems": result.get("totalItems", 0),
            "totalPages": result.get("totalPages", 0),
            "items": items,
        }

    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/column-values/{column}")
async def get_column_values(year: str, column: str):
    """
    Get unique values for a column (for filter dropdowns).

    Supported columns: level, is_minority, has_support, is_year_1, years_at_gvca
    """
    validate_year(year)

    valid_columns = ["level", "is_minority", "has_support", "is_year_1", "years_at_gvca"]
    if column not in valid_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid column. Valid columns: {', '.join(valid_columns)}",
        )

    try:
        # Get all survey responses for the year (we need to dedupe in Python)
        all_responses = await pb_client.get_full_list(
            "survey_responses",
            filter_str=f'year = "{year}"',
        )

        # Extract unique values
        values = set()
        for resp in all_responses:
            val = resp.get(column)
            if val is not None:
                values.add(val)

        # Sort values
        sorted_values = sorted(values, key=lambda x: (x is None, x))

        return {
            "year": year,
            "column": column,
            "values": sorted_values,
        }

    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/segments")
async def get_segments(year: str):
    """Get available demographic segments and their sizes."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    df = pipeline.data
    total = df.height

    segments = []
    for key, config in DEMOGRAPHIC_SEGMENTS.items():
        try:
            filtered = config["filter"](df)
            count = filtered.height
            segments.append({
                "key": key,
                "name": config["name"],
                "inverse_name": config["inverse_name"],
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
                "inverse_count": total - count,
                "inverse_percentage": round((total - count) / total * 100, 1) if total > 0 else 0,
            })
        except Exception as e:
            logger.warning(f"Error computing segment {key}: {e}")

    return {
        "year": year,
        "total": total,
        "segments": segments,
    }


@router.get("/{year}/segments/{segment_key}/statistics")
async def get_segment_statistics(year: str, segment_key: str, inverse: bool = False):
    """Get statistics for a specific demographic segment."""
    from app.core.analysis import compute_statistics

    pipeline = await pipeline_manager.ensure_loaded(year)

    if segment_key not in DEMOGRAPHIC_SEGMENTS:
        raise HTTPException(
            status_code=404, detail=f"Unknown segment: {segment_key}"
        )

    df = pipeline.data
    config = DEMOGRAPHIC_SEGMENTS[segment_key]

    try:
        filtered = config["filter"](df)
        if inverse:
            # Get the inverse segment
            if "Respondent ID" not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="Data missing required 'Respondent ID' column",
                )
            filtered_ids = set(filtered["Respondent ID"].to_list())
            filtered = df.filter(~df["Respondent ID"].is_in(filtered_ids))

        stats = compute_statistics(filtered, pipeline.config)

        segment_name = config["inverse_name"] if inverse else config["name"]

        return {
            "year": year,
            "segment": segment_key,
            "segment_name": segment_name,
            "inverse": inverse,
            "statistics": stats,
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
