"""Chart generation for school board presentations using matplotlib."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

from app.config import get_settings
from app.core.survey_config import LEVELS, QUESTIONS, FREE_RESPONSE_QUESTIONS, QUESTION_SCALES

logger = logging.getLogger(__name__)

# GVCA Color Scheme
GVCA_COLORS = {
    "positive": "#4CAF50",      # Green - Good Choice
    "negative": "#800020",      # Maroon - Better Serve
    "extremely": "#1a5f1a",     # Dark green - Extremely Satisfied
    "satisfied": "#4CAF50",     # Green - Satisfied
    "somewhat": "#FFC107",      # Amber - Somewhat
    "not": "#800020",           # Maroon - Not Satisfied
    "grammar": "#2196F3",       # Blue
    "middle": "#FF9800",        # Orange
    "high": "#9C27B0",          # Purple
}

SATISFACTION_COLORS = ["#1a5f1a", "#4CAF50", "#FFC107", "#800020"]
LEVEL_COLORS = {"Grammar": "#2196F3", "Middle": "#FF9800", "High": "#9C27B0"}

plt.style.use('seaborn-v0_8-whitegrid')


def setup_figure(figsize: tuple[int, int] = (10, 6), dpi: int = 150) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def save_chart(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    """Save a chart and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return str(filepath)


def compute_satisfaction_percentages(
    df: pl.DataFrame,
    question: str,
    level: str | None = None,
    scale: list[str] | None = None
) -> dict[str, float]:
    """Compute percentage for each satisfaction level."""
    if scale is None:
        scale = QUESTION_SCALES.get(question, ["Extremely Satisfied", "Satisfied", "Somewhat Satisfied", "Not Satisfied"])

    col_name = f"({level}) {question}" if level else question

    if col_name not in df.columns:
        return {s: 0.0 for s in scale}

    total = df.filter(pl.col(col_name).is_not_null()).height
    if total == 0:
        return {s: 0.0 for s in scale}

    result = {}
    for s in scale:
        count = df.filter(pl.col(col_name) == s).height
        result[s] = round(count / total * 100, 1)

    return result


def generate_composite_trend_chart(
    years_data: dict[str, pl.DataFrame],
    output_dir: Path,
) -> str:
    """Generate year-over-year composite satisfaction trend chart."""
    fig, ax = setup_figure((10, 6))

    years = sorted(years_data.keys())
    levels = ["Overall"] + LEVELS

    for level in levels:
        scores = []
        for year in years:
            df = years_data[year]
            # Compute weighted average for this level
            if level == "Overall":
                # Average across all levels and questions
                avg = compute_overall_average(df)
            else:
                avg = compute_level_average(df, level)
            scores.append(avg)

        color = LEVEL_COLORS.get(level, "#666666")
        ax.plot(years, scores, marker='o', linewidth=2, label=level, color=color)

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Score (1-4)')
    ax.set_title('Composite Satisfaction Scores Over Time')
    ax.set_ylim(1, 4)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return save_chart(fig, output_dir, 'composite_trend.png')


def compute_overall_average(df: pl.DataFrame) -> float:
    """Compute overall weighted average across all questions and levels."""
    total_score = 0
    total_count = 0

    for question in QUESTIONS:
        if question in FREE_RESPONSE_QUESTIONS:
            continue

        scale = QUESTION_SCALES.get(question, [])
        for level in LEVELS:
            col_name = f"({level}) {question}"
            if col_name not in df.columns:
                continue

            for idx, s in enumerate(scale):
                score = len(scale) - idx  # 4 for first, 1 for last
                count = df.filter(pl.col(col_name) == s).height
                total_score += score * count
                total_count += count

    return round(total_score / total_count, 2) if total_count > 0 else 0


def compute_level_average(df: pl.DataFrame, level: str) -> float:
    """Compute average for a specific level."""
    total_score = 0
    total_count = 0

    for question in QUESTIONS:
        if question in FREE_RESPONSE_QUESTIONS:
            continue

        scale = QUESTION_SCALES.get(question, [])
        col_name = f"({level}) {question}"
        if col_name not in df.columns:
            continue

        for idx, s in enumerate(scale):
            score = len(scale) - idx
            count = df.filter(pl.col(col_name) == s).height
            total_score += score * count
            total_count += count

    return round(total_score / total_count, 2) if total_count > 0 else 0


def generate_question_stacked_bar(
    df: pl.DataFrame,
    question: str,
    year: str,
    output_dir: Path,
    question_number: int,
) -> str:
    """Generate stacked bar chart for a single question by level."""
    fig, ax = setup_figure((10, 6))

    scale = QUESTION_SCALES.get(question, [])
    x = np.arange(len(LEVELS))
    width = 0.6
    bottom = np.zeros(len(LEVELS))

    for idx, s in enumerate(scale):
        values = []
        for level in LEVELS:
            pcts = compute_satisfaction_percentages(df, question, level, scale)
            values.append(pcts.get(s, 0))

        ax.bar(x, values, width, label=s, bottom=bottom, color=SATISFACTION_COLORS[idx])
        bottom += np.array(values)

    ax.set_xlabel('School Level')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Q{question_number}: {question[:60]}...' if len(question) > 60 else f'Q{question_number}: {question}')
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=8)

    filename = f'q{question_number}_by_level_{year}.png'
    return save_chart(fig, output_dir, filename)


def generate_question_yoy_chart(
    years_data: dict[str, pl.DataFrame],
    question: str,
    output_dir: Path,
    question_number: int,
) -> str:
    """Generate year-over-year stacked bar chart for a question."""
    fig, ax = setup_figure((12, 6))

    years = sorted(years_data.keys())
    scale = QUESTION_SCALES.get(question, [])

    # Group by year
    x = np.arange(len(years))
    width = 0.6
    bottom = np.zeros(len(years))

    for idx, s in enumerate(scale):
        values = []
        for year in years:
            df = years_data[year]
            # Average across all levels
            level_pcts = []
            for level in LEVELS:
                pcts = compute_satisfaction_percentages(df, question, level, scale)
                level_pcts.append(pcts.get(s, 0))
            values.append(np.mean(level_pcts) if level_pcts else 0)

        ax.bar(x, values, width, label=s, bottom=bottom, color=SATISFACTION_COLORS[idx])
        bottom += np.array(values)

    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Q{question_number}: Year-over-Year Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=8)

    filename = f'q{question_number}_yoy.png'
    return save_chart(fig, output_dir, filename)


def generate_tag_frequency_chart(
    good_choice_tags: dict[str, int],
    better_serve_tags: dict[str, int],
    year: str,
    output_dir: Path,
) -> str:
    """Generate dual bar chart for tag frequencies."""
    fig, ax = setup_figure((12, 8))

    # Combine and sort tags by total frequency
    all_tags = set(good_choice_tags.keys()) | set(better_serve_tags.keys())
    tag_totals = {
        tag: good_choice_tags.get(tag, 0) + better_serve_tags.get(tag, 0)
        for tag in all_tags
    }
    sorted_tags = sorted(tag_totals.keys(), key=lambda t: tag_totals[t], reverse=True)[:15]

    y = np.arange(len(sorted_tags))
    height = 0.35

    good_values = [good_choice_tags.get(tag, 0) for tag in sorted_tags]
    serve_values = [better_serve_tags.get(tag, 0) for tag in sorted_tags]

    ax.barh(y - height/2, good_values, height, label='Good Choice', color=GVCA_COLORS["positive"])
    ax.barh(y + height/2, serve_values, height, label='Better Serve', color=GVCA_COLORS["negative"])

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_tags)
    ax.set_xlabel('Count')
    ax.set_title(f'Tag Frequency by Question Type ({year})')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    filename = f'tag_frequency_{year}.png'
    return save_chart(fig, output_dir, filename)


def generate_category_yoy_chart(
    years_tag_data: dict[str, dict[str, int]],
    category: str,
    output_dir: Path,
) -> str:
    """Generate year-over-year chart for a tag category."""
    fig, ax = setup_figure((10, 6))

    years = sorted(years_tag_data.keys())
    counts = [years_tag_data[year].get(category, 0) for year in years]

    ax.bar(years, counts, color=GVCA_COLORS["positive"], edgecolor='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title(f'{category} Mentions Over Time')

    for i, (year, count) in enumerate(zip(years, counts)):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    filename = f'category_{category.lower().replace(" ", "_")}_yoy.png'
    return save_chart(fig, output_dir, filename)


def generate_level_composite_chart(
    years_data: dict[str, pl.DataFrame],
    output_dir: Path,
) -> str:
    """Generate composite scores by grade level across years."""
    fig, ax = setup_figure((10, 6))

    years = sorted(years_data.keys())
    x = np.arange(len(years))
    width = 0.25

    for i, level in enumerate(LEVELS):
        scores = []
        for year in years:
            df = years_data[year]
            avg = compute_level_average(df, level)
            scores.append(avg)

        offset = (i - 1) * width
        ax.bar(x + offset, scores, width, label=level, color=LEVEL_COLORS[level])

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Score (1-4)')
    ax.set_title('Composite Scores by Grade Level')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(1, 4)
    ax.legend(loc='lower right')

    return save_chart(fig, output_dir, 'level_composite_yoy.png')


def generate_satisfaction_summary_chart(
    df: pl.DataFrame,
    year: str,
    output_dir: Path,
) -> str:
    """Generate overall satisfaction summary pie chart."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    # Overall and by level
    titles = ["Overall", "Grammar", "Middle", "High"]

    for ax, title in zip(axes, titles):
        # Compute satisfaction distribution
        satisfied_pct = 0
        total = 0

        for question in QUESTIONS:
            if question in FREE_RESPONSE_QUESTIONS:
                continue

            scale = QUESTION_SCALES.get(question, [])
            if title == "Overall":
                for level in LEVELS:
                    pcts = compute_satisfaction_percentages(df, question, level, scale)
                    satisfied_pct += pcts.get(scale[0], 0) + pcts.get(scale[1], 0) if len(scale) >= 2 else 0
                    total += 1
            else:
                pcts = compute_satisfaction_percentages(df, question, title, scale)
                satisfied_pct += pcts.get(scale[0], 0) + pcts.get(scale[1], 0) if len(scale) >= 2 else 0
                total += 1

        avg_satisfied = satisfied_pct / total if total > 0 else 0
        unsatisfied = 100 - avg_satisfied

        colors = [GVCA_COLORS["positive"], GVCA_COLORS["somewhat"]]
        ax.pie([avg_satisfied, unsatisfied], colors=colors, startangle=90,
               autopct='%1.1f%%', pctdistance=0.6)
        ax.set_title(title)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=GVCA_COLORS["positive"], label='Satisfied/Very Satisfied'),
        mpatches.Patch(color=GVCA_COLORS["somewhat"], label='Other'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f'Satisfaction Summary ({year})', fontsize=14)
    fig.tight_layout()

    return save_chart(fig, output_dir, f'satisfaction_summary_{year}.png')


async def generate_all_charts(
    year: str,
    df: pl.DataFrame,
    tag_distribution: dict[str, Any] | None = None,
) -> list[str]:
    """Generate all charts for a single year."""
    settings = get_settings()
    output_dir = settings.artifacts_dir / year

    generated = []

    # Per-question charts
    for idx, question in enumerate(QUESTIONS, 1):
        if question in FREE_RESPONSE_QUESTIONS:
            continue
        try:
            path = generate_question_stacked_bar(df, question, year, output_dir, idx)
            generated.append(path)
        except Exception as e:
            logger.error(f"Error generating Q{idx} chart: {e}")

    # Satisfaction summary
    try:
        path = generate_satisfaction_summary_chart(df, year, output_dir)
        generated.append(path)
    except Exception as e:
        logger.error(f"Error generating satisfaction summary: {e}")

    # Tag frequency chart if tags available
    if tag_distribution:
        try:
            good_choice = tag_distribution.get("good_choice", {})
            better_serve = tag_distribution.get("better_serve", {})
            path = generate_tag_frequency_chart(good_choice, better_serve, year, output_dir)
            generated.append(path)
        except Exception as e:
            logger.error(f"Error generating tag frequency chart: {e}")

    return generated


async def generate_comparison_charts(
    years_data: dict[str, pl.DataFrame],
) -> list[str]:
    """Generate multi-year comparison charts."""
    settings = get_settings()
    output_dir = settings.artifacts_dir / "comparison"

    generated = []

    # Composite trend
    try:
        path = generate_composite_trend_chart(years_data, output_dir)
        generated.append(path)
    except Exception as e:
        logger.error(f"Error generating composite trend: {e}")

    # Level composite
    try:
        path = generate_level_composite_chart(years_data, output_dir)
        generated.append(path)
    except Exception as e:
        logger.error(f"Error generating level composite: {e}")

    # Per-question YoY
    for idx, question in enumerate(QUESTIONS, 1):
        if question in FREE_RESPONSE_QUESTIONS:
            continue
        try:
            path = generate_question_yoy_chart(years_data, question, output_dir, idx)
            generated.append(path)
        except Exception as e:
            logger.error(f"Error generating Q{idx} YoY chart: {e}")

    return generated


# ================ New Chart Generation Functions (FR-008, FR-009, FR-010) ================


def filter_responses_by_segment(
    responses: list[dict[str, Any]], segment: str
) -> list[dict[str, Any]]:
    """Filter survey responses by demographic segment."""
    if segment == "all":
        return responses

    filtered = []
    for r in responses:
        demographics = r.get("demographics") or {}
        school_level = r.get("school_level")

        match = False
        if segment == "year1":
            match = demographics.get("year1_family") is True
        elif segment == "minority":
            match = demographics.get("minority") is True
        elif segment == "support":
            match = demographics.get("support") is True
        elif segment == "grammar":
            match = school_level == "Grammar"
        elif segment == "middle":
            match = school_level == "Middle"
        elif segment == "high":
            match = school_level == "High"
        elif segment == "tenure_3plus":
            tenure = demographics.get("tenure_years")
            match = tenure is not None and tenure >= 3

        if match:
            filtered.append(r)

    return filtered


def compute_segment_satisfaction_average(
    responses: list[dict[str, Any]], questions: list[str] | None = None
) -> dict[str, float]:
    """
    Compute average satisfaction scores from PocketBase response records.

    Returns dict mapping question key to average score (1-4 scale).
    """
    # Map satisfaction text to numeric scores
    score_map = {
        "Extremely Satisfied": 4, "Satisfied": 3, "Somewhat Satisfied": 2, "Not Satisfied": 1,
        "Strongly Reflected": 4, "Reflected": 3, "Somewhat Reflected": 2, "Not Reflected": 1,
        "Extremely Effective": 4, "Effective": 3, "Somewhat Effective": 2, "Not Effective": 1,
        "Extremely Welcoming": 4, "Welcoming": 3, "Somewhat Welcoming": 2, "Not Welcoming": 1,
    }

    question_scores: dict[str, list[float]] = {}

    for r in responses:
        satisfaction = r.get("satisfaction_scores") or {}
        for q_key, levels_data in satisfaction.items():
            if questions and q_key not in questions:
                continue
            for level, response_text in levels_data.items():
                score = score_map.get(response_text)
                if score is not None:
                    key = f"{q_key}_{level}"
                    if key not in question_scores:
                        question_scores[key] = []
                    question_scores[key].append(score)

    # Compute averages
    return {k: sum(v) / len(v) if v else 0 for k, v in question_scores.items()}


async def generate_demographic_comparison_chart(
    responses: list[dict[str, Any]],
    segment_a: str,
    segment_b: str,
    questions: list[str] | None,
    year: str,
) -> dict[str, Any]:
    """
    Generate demographic comparison chart data and optional PNG (FR-008).

    Returns chart data for frontend rendering plus optional file_path.
    """
    # Filter responses by segment
    segment_a_data = filter_responses_by_segment(responses, segment_a)
    segment_b_data = filter_responses_by_segment(responses, segment_b)

    # Compute averages for each segment
    avg_a = compute_segment_satisfaction_average(segment_a_data, questions)
    avg_b = compute_segment_satisfaction_average(segment_b_data, questions)

    # Get unique question keys
    all_keys = sorted(set(avg_a.keys()) | set(avg_b.keys()))

    # Build chart data
    chart_data = {
        "segment_a": {
            "name": segment_a,
            "count": len(segment_a_data),
            "scores": {k: avg_a.get(k, 0) for k in all_keys},
        },
        "segment_b": {
            "name": segment_b,
            "count": len(segment_b_data),
            "scores": {k: avg_b.get(k, 0) for k in all_keys},
        },
        "questions": all_keys,
    }

    # Generate PNG
    settings = get_settings()
    output_dir = settings.artifacts_dir / year

    if all_keys:
        fig, ax = setup_figure((12, 6))

        x = np.arange(len(all_keys))
        width = 0.35

        bars_a = ax.bar(x - width/2, [avg_a.get(k, 0) for k in all_keys], width,
                        label=f'{segment_a} (n={len(segment_a_data)})',
                        color=GVCA_COLORS["positive"])
        bars_b = ax.bar(x + width/2, [avg_b.get(k, 0) for k in all_keys], width,
                        label=f'{segment_b} (n={len(segment_b_data)})',
                        color=GVCA_COLORS["negative"])

        ax.set_ylabel('Average Score (1-4)')
        ax.set_title(f'Demographic Comparison: {segment_a} vs {segment_b} ({year})')
        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 4.5)

        fig.tight_layout()
        file_path = save_chart(fig, output_dir, f'demographic_{segment_a}_vs_{segment_b}_{year}.png')
        chart_data["file_path"] = file_path

    return chart_data


async def generate_trend_comparison_chart(
    years_data: dict[str, list[dict[str, Any]]],
    school_level: str | None,
    questions: list[str] | None,
) -> dict[str, Any]:
    """
    Generate cross-year trend chart data and optional PNG (FR-009).

    Returns chart data for frontend rendering plus optional file_path.
    """
    years = sorted(years_data.keys())

    # Filter by school level if specified
    if school_level:
        years_data = {
            year: [r for r in responses if r.get("school_level") == school_level]
            for year, responses in years_data.items()
        }

    # Compute averages per year
    year_averages: dict[str, dict[str, float]] = {}
    for year, responses in years_data.items():
        year_averages[year] = compute_segment_satisfaction_average(responses, questions)

    # Get all question keys
    all_keys = sorted(
        set().union(*[set(avgs.keys()) for avgs in year_averages.values()])
    )

    # Build chart data
    chart_data = {
        "years": years,
        "school_level": school_level,
        "questions": all_keys,
        "data": {
            year: {k: year_averages[year].get(k, 0) for k in all_keys}
            for year in years
        },
        "counts": {year: len(years_data.get(year, [])) for year in years},
    }

    # Generate PNG - overall trend line
    settings = get_settings()
    output_dir = settings.artifacts_dir / "comparison"

    if all_keys:
        fig, ax = setup_figure((10, 6))

        # Compute overall average per year
        overall_avgs = []
        for year in years:
            avgs = year_averages[year]
            if avgs:
                overall_avgs.append(sum(avgs.values()) / len(avgs))
            else:
                overall_avgs.append(0)

        ax.plot(years, overall_avgs, marker='o', linewidth=2, markersize=8,
                color=GVCA_COLORS["positive"])

        ax.set_ylabel('Average Satisfaction Score (1-4)')
        ax.set_xlabel('Year')
        title = f'Satisfaction Trend Over Time'
        if school_level:
            title += f' ({school_level})'
        ax.set_title(title)
        ax.set_ylim(1, 4.5)

        fig.tight_layout()
        suffix = f'_{school_level.lower()}' if school_level else ''
        file_path = save_chart(fig, output_dir, f'trend_comparison{suffix}.png')
        chart_data["file_path"] = file_path

    return chart_data


async def generate_sentiment_chart(
    tagging_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate sentiment chart data (FR-010).

    Counts positive (praise) vs negative (improvement) occurrences per tag.
    """
    tag_counts: dict[str, dict[str, int]] = {}

    for result in tagging_results:
        tags = result.get("llm_tags", [])
        question_type = result.get("question_type", "improvement")

        for tag in tags:
            if tag not in tag_counts:
                tag_counts[tag] = {"positive": 0, "negative": 0}

            if question_type == "praise":
                tag_counts[tag]["positive"] += 1
            else:
                tag_counts[tag]["negative"] += 1

    # Convert to list format
    sentiment_data = [
        {
            "tag": tag,
            "positive_count": counts["positive"],
            "negative_count": counts["negative"],
        }
        for tag, counts in sorted(tag_counts.items(), key=lambda x: x[0])
    ]

    return sentiment_data
