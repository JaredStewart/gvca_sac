"""Statistical analysis utilities using Polars."""

import logging
from typing import Any

import polars as pl

from app.core.survey_config import (
    FREE_RESPONSE_QUESTIONS,
    LEVELS,
    QUESTIONS,
    QUESTION_SCALES,
    SurveyConfig,
)

logger = logging.getLogger(__name__)


def compute_rank_value(response: str, scale: list[str]) -> int | None:
    """
    Compute rank value for a response.
    Higher rank = more positive (4 = best, 1 = worst).
    """
    if response is None or not isinstance(response, str):
        return None
    try:
        idx = scale.index(response)
        return 4 - idx
    except ValueError:
        return None


def compute_row_averages(row: dict[str, Any], config: SurveyConfig) -> dict[str, float | None]:
    """Compute average rank values for a single row."""
    overall_sum, overall_count = 0, 0
    level_sums = {level: 0 for level in LEVELS}
    level_counts = {level: 0 for level in LEVELS}

    for question in QUESTIONS:
        if question in FREE_RESPONSE_QUESTIONS:
            continue

        scale = QUESTION_SCALES.get(question, [])
        if not scale:
            continue

        for level in LEVELS:
            col = f"({level}) {question}"
            value = row.get(col)

            if value is None or value == "-":
                continue

            rank = compute_rank_value(value, scale)
            if rank is not None:
                overall_sum += rank
                overall_count += 1
                level_sums[level] += rank
                level_counts[level] += 1

    return {
        "Overall Average": overall_sum / overall_count if overall_count > 0 else None,
        "Grammar Average": level_sums["Grammar"] / level_counts["Grammar"] if level_counts["Grammar"] > 0 else None,
        "Middle Average": level_sums["Middle"] / level_counts["Middle"] if level_counts["Middle"] > 0 else None,
        "High Average": level_sums["High"] / level_counts["High"] if level_counts["High"] > 0 else None,
    }


def _family_weights(series: pl.Series) -> pl.Series:
    """Convert N Parents Represented to family weights: 2+ → 1.0, 1 → 0.75."""
    raw = series.cast(pl.Float64).fill_null(1.0)
    return (raw >= 2).cast(pl.Float64) * 0.25 + 0.75


def compute_averages(
    df: pl.DataFrame,
    weight_by_parents: bool = False,
    config: SurveyConfig | None = None
) -> tuple[pl.DataFrame, dict[str, float]]:
    """
    Compute per-respondent and weighted averages.

    Args:
        df: Survey DataFrame
        weight_by_parents: Whether to weight by parent count
        config: Survey configuration

    Returns:
        Tuple of (DataFrame with average columns, weighted totals dict)
    """
    if config is None:
        config = SurveyConfig()

    # Compute row averages
    averages = []
    for row in df.iter_rows(named=True):
        avg = compute_row_averages(row, config)
        averages.append(avg)

    # Add average columns to DataFrame
    avg_df = pl.DataFrame(averages)
    result_df = pl.concat([df, avg_df], how="horizontal")

    # Compute weighted totals
    # Weighting: coordinated (2 parents) = 1.0 family, individual (1 parent) = 0.75 family
    if weight_by_parents and "N Parents Represented" in result_df.columns:
        weights = _family_weights(result_df["N Parents Represented"])
    else:
        weights = pl.Series([1.0] * len(result_df))

    weighted_totals = {}
    for col in ["Overall Average", "Grammar Average", "Middle Average", "High Average"]:
        if col in result_df.columns:
            valid_mask = result_df[col].is_not_null()
            valid_values = result_df.filter(valid_mask)[col].cast(pl.Float64)
            valid_weights = weights.filter(valid_mask)

            if len(valid_values) > 0:
                weighted_sum = (valid_values * valid_weights).sum()
                weight_sum = valid_weights.sum()
                weighted_totals[col] = weighted_sum / weight_sum if weight_sum > 0 else None
            else:
                weighted_totals[col] = None

    return result_df, weighted_totals



def calculate_question_totals(
    df: pl.DataFrame,
    weight_by_parents: bool = False
) -> pl.DataFrame:
    """
    Calculate response totals by question, level, and segment.

    Args:
        df: Survey DataFrame
        weight_by_parents: Whether to weight by parent count

    Returns:
        pl.DataFrame with counts and percentages
    """
    # Define segment filters
    def get_segment_mask(df: pl.DataFrame, segment: str) -> pl.Series:
        if segment == "Year 1 Families":
            return pl.col("Years at GVCA").cast(pl.Int64, strict=False) == 1
        elif segment == "Not Year 1 Families":
            return pl.col("Years at GVCA").cast(pl.Int64, strict=False) > 1
        elif segment == "Year 3 or Less Families":
            return pl.col("Years at GVCA").cast(pl.Int64, strict=False) <= 3
        elif segment == "Year 4 or More Families":
            return pl.col("Years at GVCA").cast(pl.Int64, strict=False) > 3
        elif segment == "Minority":
            return pl.col("Minority") == "Yes"
        elif segment == "Not Minority":
            return pl.col("Minority") != "Yes"
        elif segment == "Support":
            return pl.col("IEP, 504, ALP, or Read") == "Yes"
        elif segment == "Not Support":
            return pl.col("IEP, 504, ALP, or Read") != "Yes"
        return pl.lit(True)

    segments = [
        "Year 1 Families",
        "Not Year 1 Families",
        "Year 3 or Less Families",
        "Year 4 or More Families",
        "Minority",
        "Not Minority",
        "Support",
        "Not Support",
    ]

    results = []

    for question in QUESTIONS:
        if question in FREE_RESPONSE_QUESTIONS:
            continue

        scale = QUESTION_SCALES.get(question, [])

        for response in scale:
            row_data = {"Question": question, "Response": response}

            # Calculate for all levels combined
            total_count = 0
            response_count = 0

            for level in LEVELS:
                col = f"({level}) {question}"
                if col not in df.columns:
                    continue

                level_df = df.filter(pl.col(col).is_not_null())
                if weight_by_parents and "N Parents Represented" in df.columns:
                    level_total = _family_weights(level_df["N Parents Represented"]).sum()
                    resp_df = level_df.filter(pl.col(col) == response)
                    response_total = _family_weights(resp_df["N Parents Represented"]).sum()
                else:
                    level_total = len(level_df)
                    response_total = len(level_df.filter(pl.col(col) == response))

                total_count += level_total or 0
                response_count += response_total or 0

            row_data["N_total"] = response_count
            row_data["%_total"] = (response_count / total_count * 100) if total_count > 0 else 0

            # Calculate for each level
            for level in LEVELS:
                col = f"({level}) {question}"
                if col not in df.columns:
                    row_data[f"N_{level}"] = 0
                    row_data[f"%_{level}"] = 0
                    continue

                level_df = df.filter(pl.col(col).is_not_null())

                if weight_by_parents and "N Parents Represented" in df.columns:
                    level_total = _family_weights(level_df["N Parents Represented"]).sum()
                    resp_df = level_df.filter(pl.col(col) == response)
                    level_response = _family_weights(resp_df["N Parents Represented"]).sum()
                else:
                    level_total = len(level_df)
                    level_response = len(level_df.filter(pl.col(col) == response))

                row_data[f"N_{level}"] = level_response or 0
                row_data[f"%_{level}"] = (
                    (level_response / level_total * 100) if level_total > 0 else 0
                )

            # Calculate for each segment
            for segment in segments:
                try:
                    segment_df = df.filter(get_segment_mask(df, segment))
                except Exception:
                    row_data[f"N_{segment}"] = 0
                    row_data[f"%_{segment}"] = 0
                    continue

                seg_total = 0
                seg_response = 0

                for level in LEVELS:
                    col = f"({level}) {question}"
                    if col not in segment_df.columns:
                        continue

                    level_df = segment_df.filter(pl.col(col).is_not_null())

                    if weight_by_parents and "N Parents Represented" in segment_df.columns:
                        seg_total += _family_weights(level_df["N Parents Represented"]).sum() or 0
                        resp_df = level_df.filter(pl.col(col) == response)
                        seg_response += _family_weights(resp_df["N Parents Represented"]).sum() or 0
                    else:
                        seg_total += len(level_df)
                        seg_response += len(level_df.filter(pl.col(col) == response))

                row_data[f"N_{segment}"] = seg_response
                row_data[f"%_{segment}"] = (seg_response / seg_total * 100) if seg_total > 0 else 0

            results.append(row_data)

    return pl.DataFrame(results)


def compute_statistics(df: pl.DataFrame, config: SurveyConfig | None = None) -> dict[str, Any]:
    """
    Compute all statistics for a survey dataset.

    Args:
        df: Survey DataFrame
        config: Survey configuration

    Returns:
        Dictionary with all computed statistics
    """
    if config is None:
        config = SurveyConfig()

    # Compute averages (weighted: coordinated=1.0, individual=0.75)
    df_with_avgs, weighted_totals = compute_averages(df, weight_by_parents=True, config=config)

    # Compute question totals (weighted: coordinated=1.0, individual=0.75)
    question_totals = calculate_question_totals(df, weight_by_parents=True)

    # Summary statistics
    total_responses = len(df)

    # Count by level
    level_counts = {}
    for level in LEVELS:
        # Count rows that have any non-null response for this level
        level_cols = [col for col in df.columns if col.startswith(f"({level})")]
        if level_cols:
            has_level = pl.any_horizontal([pl.col(c).is_not_null() for c in level_cols])
            level_counts[level] = df.filter(has_level).height

    return {
        "total_responses": total_responses,
        "weighted_averages": weighted_totals,
        "level_counts": level_counts,
        "question_totals": question_totals.to_dicts(),
    }
