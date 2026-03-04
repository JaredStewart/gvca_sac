"""Data transformation utilities using Polars."""

import csv
import logging
from pathlib import Path
from typing import Any

import polars as pl

from app.core.survey_config import (
    CONCLUDING_HEADERS,
    FREE_RESPONSE_QUESTIONS,
    INITIAL_HEADERS,
    LEVEL_SEQUENCES,
    LEVELS,
    QUESTIONS,
    SurveyConfig,
)

logger = logging.getLogger(__name__)

# Responses that are not meaningful free-text answers
_NON_ANSWERS = frozenset({
    "na", "n/a", "none", "nothing", "-", ".", "no", "no comment",
    "no comments", "n a", "nil", "nope", "not applicable",
})


def is_meaningful_response(text: Any) -> bool:
    """Return True if *text* is a non-trivial free-response answer.

    Rejects None, empty/whitespace, very short strings (< 3 chars),
    and common non-answers like "NA", "N/A", "None", "Nothing", "-", etc.
    """
    if text is None:
        return False
    s = str(text).strip()
    if len(s) < 3:
        return False
    return s.lower() not in _NON_ANSWERS


def build_input_headers() -> list[str]:
    """Build the expected input headers list."""
    headers = INITIAL_HEADERS.copy()

    for level_sequence in LEVEL_SEQUENCES:
        for question in QUESTIONS:
            for level in level_sequence:
                headers.append(f"({level}) {question}")
            if question in FREE_RESPONSE_QUESTIONS:
                headers.append(f"(Generic) {question}")

    headers.extend(CONCLUDING_HEADERS)
    return headers


def build_output_headers() -> tuple[dict[str, int], list[str]]:
    """Build output headers and index map."""
    index_map: dict[str, int] = {}
    output_headers: list[str] = []

    # Initial headers
    for header in INITIAL_HEADERS:
        index_map[header] = len(output_headers)
        output_headers.append(header)

    # Additional computed fields
    for field in ["N Parents Represented", "Empty Response"]:
        index_map[field] = len(output_headers)
        output_headers.append(field)

    # Concluding headers
    for header in CONCLUDING_HEADERS:
        index_map[header] = len(output_headers)
        output_headers.append(header)

    # Question columns by level
    for question in QUESTIONS:
        for level in LEVELS:
            header = f"({level}) {question}"
            index_map[header] = len(output_headers)
            output_headers.append(header)
        if question in FREE_RESPONSE_QUESTIONS:
            header = f"(Generic) {question}"
            index_map[header] = len(output_headers)
            output_headers.append(header)

    return index_map, output_headers


# Module-level constants
INPUT_HEADERS = build_input_headers()
INDEX_MAP, OUTPUT_HEADERS = build_output_headers()


def compute_parent_count(row: list[Any], index_map: dict[str, int]) -> int:
    """Compute number of parents represented."""
    submission_method = row[index_map["Submission Method"]]
    if (
        submission_method
        == "All parents and guardians will coordinate responses, and we will submit only one survey."
    ):
        return 2
    return 1


def compute_empty_response(row: list[Any], index_map: dict[str, int]) -> bool:
    """Check if response is empty."""
    for question in QUESTIONS:
        for level in LEVELS:
            header = f"({level}) {question}"
            if header in index_map and row[index_map[header]] != "-":
                return False
        if question in FREE_RESPONSE_QUESTIONS:
            header = f"(Generic) {question}"
            if header in index_map and row[index_map[header]] != "-":
                return False
    return True


def make_output_row(input_row: list[str]) -> list[Any]:
    """Transform a single input row to output format."""
    output_row: list[Any] = ["-"] * len(OUTPUT_HEADERS)

    for i, item in enumerate(input_row):
        if i < len(INITIAL_HEADERS):
            output_row[i] = input_row[i]
        elif item != "":
            if i < len(INPUT_HEADERS):
                input_header = INPUT_HEADERS[i]
                if input_header in INDEX_MAP:
                    output_index = INDEX_MAP[input_header]
                    output_row[output_index] = item

    # Compute additional fields
    output_row[INDEX_MAP["N Parents Represented"]] = compute_parent_count(
        output_row, INDEX_MAP
    )
    output_row[INDEX_MAP["Empty Response"]] = compute_empty_response(
        output_row, INDEX_MAP
    )

    return output_row


def transform_raw(input_path: str, output_path: str | None = None) -> list[list[Any]]:
    """
    Transform raw survey data to flattened format.

    Args:
        input_path: Path to raw CSV file
        output_path: Optional path to write output CSV

    Returns:
        List of flattened data rows
    """
    flattened_data = []

    # Try UTF-8 first (with BOM support), fall back to latin-1 for Excel exports
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(input_path, "r", encoding=encoding) as f:
                rows = list(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue

    for i, row in enumerate(rows):
        if i < 2:  # Skip header rows
            continue
        flattened_row = make_output_row(row)
        flattened_data.append(flattened_row)

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(OUTPUT_HEADERS)
            writer.writerows(flattened_data)

    return flattened_data


def load_flattened(filepath: str) -> pl.DataFrame:
    """
    Load processed survey data.

    Args:
        filepath: Path to processed CSV file

    Returns:
        pl.DataFrame: Survey data
    """
    df = pl.read_csv(filepath, infer_schema_length=0)  # Read all as strings initially

    # Detect and fix duplicate Respondent IDs (e.g., Excel scientific notation)
    if "Respondent ID" in df.columns:
        id_col = df["Respondent ID"]
        if id_col.n_unique() < len(id_col):
            n_dupes = len(id_col) - id_col.n_unique()
            logger.warning(
                "Found %d duplicate Respondent IDs (likely Excel scientific notation "
                "corruption). Assigning unique row-based IDs.", n_dupes
            )
            df = df.with_columns(
                pl.Series("Respondent ID", [f"R{i+1:04d}" for i in range(len(df))])
            )

    # Filter out empty responses
    if "Empty Response" in df.columns:
        df = df.filter(pl.col("Empty Response") != "True")

    # Replace "-" with null for all columns (they're all strings now)
    df = df.with_columns([
        pl.when(pl.col(col) == "-")
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in df.columns
    ])

    # Parse datetime columns
    if "Start" in df.columns:
        df = df.with_columns(
            pl.col("Start")
            .str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False)
            .alias("Start")
        )
    if "End" in df.columns:
        df = df.with_columns(
            pl.col("End")
            .str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False)
            .alias("End")
        )

    return df


def combine_years(years: list[str], data_dir: str = "processed") -> pl.DataFrame:
    """
    Load and combine processed data for multiple years.

    Args:
        years: List of year identifiers
        data_dir: Directory containing processed CSVs

    Returns:
        pl.DataFrame: Combined multi-year data
    """
    dfs = []
    for year in years:
        filepath = Path(data_dir) / f"{year}.csv"
        if filepath.exists():
            df = load_flattened(str(filepath))
            df = df.with_columns(pl.lit(year).alias("Year"))
            dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal")


def get_free_response_columns(df: pl.DataFrame) -> list[str]:
    """Get column names containing free response data."""
    cols = []
    for question in FREE_RESPONSE_QUESTIONS:
        for level in LEVELS:
            col = f"({level}) {question}"
            if col in df.columns:
                cols.append(col)
        generic_col = f"(Generic) {question}"
        if generic_col in df.columns:
            cols.append(generic_col)
    return cols


def extract_free_responses(
    df: pl.DataFrame,
    question: str | None = None
) -> pl.DataFrame:
    """
    Extract free response data for tagging/clustering.

    Args:
        df: Survey DataFrame
        question: Optional specific question to extract

    Returns:
        pl.DataFrame with response_id, question, level, response_text columns
    """
    rows = []

    for idx in range(df.height):
        row = df.row(idx, named=True)
        respondent_id = row.get("Respondent ID", idx)

        for q in FREE_RESPONSE_QUESTIONS:
            if question and q != question:
                continue

            q_key = f"Q{QUESTIONS.index(q) + 1}"

            for level in LEVELS:
                col = f"({level}) {q}"
                if col in row:
                    text = row[col]
                    if is_meaningful_response(text):
                        rows.append({
                            "response_id": f"{respondent_id}_{level}_{q_key}",
                            "respondent_id": respondent_id,
                            "question": q,
                            "level": level,
                            "response_text": str(text),
                        })

            # Generic responses
            generic_col = f"(Generic) {q}"
            if generic_col in row:
                text = row[generic_col]
                if is_meaningful_response(text):
                    rows.append({
                        "response_id": f"{respondent_id}_Generic_{q_key}",
                        "respondent_id": respondent_id,
                        "question": q,
                        "level": "Generic",
                        "response_text": str(text),
                    })

    return pl.DataFrame(rows)


def get_question_type(question: str) -> str:
    """Determine if a question is praise or improvement oriented."""
    question_lower = question.lower()
    if "good" in question_lower or "well" in question_lower:
        return "praise"
    elif "better serve" in question_lower or "improve" in question_lower:
        return "improvement"
    # Default based on known questions
    if "What makes" in question:
        return "praise"
    return "improvement"


def extract_survey_response_for_db(
    row: dict[str, Any],
    year: str,
    row_idx: int,
) -> dict[str, Any]:
    """
    Extract a single survey response record for database persistence.

    Args:
        row: Row data as dict
        year: Survey year
        row_idx: Index for fallback ID

    Returns:
        Dict ready for PocketBase insertion
    """
    respondent_id = str(row.get("Respondent ID", row_idx))

    # Determine primary school level
    grade_selection = row.get("Grade Selection", "")
    school_level = None
    if grade_selection:
        for level in LEVELS:
            if level in str(grade_selection):
                school_level = level
                break

    # Parse years at GVCA
    tenure_years = None
    years_str = row.get("Years at GVCA")
    if years_str:
        try:
            tenure_years = int(str(years_str).split()[0])
        except (ValueError, IndexError):
            pass

    # Build demographics object
    support_val = row.get("IEP, 504, ALP, or Read")
    minority_val = row.get("Minority")

    demographics = {
        "minority": minority_val is not None and str(minority_val).lower() == "yes",
        "support": support_val is not None and str(support_val).lower() == "yes",
        "tenure_years": tenure_years,
        "year1_family": tenure_years == 1 if tenure_years is not None else None,
    }

    # Compute n_parents_represented
    submission_method = row.get("Submission Method", "")
    n_parents = 2 if "coordinate" in str(submission_method).lower() else 1

    # Build satisfaction_scores: {question_short: {level: score}}
    satisfaction_scores: dict[str, dict[str, str]] = {}
    for q_idx, question in enumerate(QUESTIONS):
        if question not in FREE_RESPONSE_QUESTIONS:
            q_key = f"Q{q_idx + 1}"
            question_scores: dict[str, str] = {}
            for level in LEVELS:
                col_name = f"({level}) {question}"
                val = row.get(col_name)
                if val is not None and str(val).strip() and str(val) != "-":
                    question_scores[level] = str(val)
            if question_scores:
                satisfaction_scores[q_key] = question_scores

    return {
        "year": year,
        "respondent_id": respondent_id,
        "school_level": school_level,
        "submission_method": str(submission_method) if submission_method else None,
        "n_parents_represented": n_parents,
        "demographics": demographics,
        "satisfaction_scores": satisfaction_scores if satisfaction_scores else None,
    }


def extract_free_responses_for_db(
    row: dict[str, Any],
    year: str,
    survey_response_id: str,
    row_idx: int,
) -> list[dict[str, Any]]:
    """
    Extract free response records from a survey row for database persistence.

    Args:
        row: Row data as dict
        year: Survey year
        survey_response_id: PocketBase ID of the parent survey_response
        row_idx: Row index

    Returns:
        List of free response dicts
    """
    respondent_id = str(row.get("Respondent ID", row_idx))
    free_responses = []

    for question in FREE_RESPONSE_QUESTIONS:
        question_type = get_question_type(question)
        q_key = f"Q{QUESTIONS.index(question) + 1}"

        # Check each level
        for level in LEVELS:
            col_name = f"({level}) {question}"
            text = row.get(col_name)
            if is_meaningful_response(text):
                free_responses.append({
                    "year": year,
                    "response_id": f"{respondent_id}_{level}_{q_key}",
                    "survey_response_id": survey_response_id,
                    "question": question,
                    "question_type": question_type,
                    "level": level,
                    "response_text": str(text),
                })

        # Check Generic level
        generic_col = f"(Generic) {question}"
        text = row.get(generic_col)
        if is_meaningful_response(text):
            free_responses.append({
                "year": year,
                "response_id": f"{respondent_id}_Generic_{q_key}",
                "survey_response_id": survey_response_id,
                "question": question,
                "question_type": question_type,
                "level": "Generic",
                "response_text": str(text),
            })

    return free_responses


async def transform_and_persist(
    file_content: bytes,
    year: str,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """
    Transform raw Survey Monkey CSV and persist to PocketBase.

    Args:
        file_content: Raw CSV file bytes
        year: Survey year
        replace_existing: Whether to replace existing data

    Returns:
        Import result dict with counts
    """
    from app.services.pocketbase_client import pb_client

    # Check for existing data
    existing_count = await pb_client.get_survey_response_count(year)
    if existing_count > 0 and not replace_existing:
        raise ValueError(
            f"Data already exists for year {year}. Use replace_existing=true to overwrite."
        )

    # Parse CSV from bytes FIRST (before deleting existing data)
    import io
    csv_text = file_content.decode("utf-8")
    reader = csv.reader(io.StringIO(csv_text))

    # Skip header rows (Survey Monkey has 2 header rows)
    rows_data = []
    for i, row in enumerate(reader):
        if i < 2:
            continue
        flattened = make_output_row(row)
        rows_data.append(flattened)

    # Create DataFrame for easier processing
    df = pl.DataFrame(
        rows_data,
        schema={h: pl.Utf8 for h in OUTPUT_HEADERS},
        orient="row",
    )

    # Filter out empty responses
    if "Empty Response" in df.columns:
        df = df.filter(pl.col("Empty Response") != "True")

    # Delete existing data AFTER successful parse
    if existing_count > 0 and replace_existing:
        await pb_client.delete_survey_responses_by_year(year)
        await pb_client.delete_free_responses_by_year(year)

    total_responses = 0
    total_free_responses = 0

    # Process each row
    for idx in range(df.height):
        row = df.row(idx, named=True)

        # Create survey_response record
        survey_data = extract_survey_response_for_db(row, year, idx)
        try:
            created_response = await pb_client.create_survey_response(survey_data)
            total_responses += 1

            # Create free_response records
            free_responses = extract_free_responses_for_db(
                row, year, created_response["id"], idx
            )

            for fr in free_responses:
                await pb_client.create_free_response(fr)
                total_free_responses += 1

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    return {
        "year": year,
        "total_responses": total_responses,
        "free_responses_extracted": total_free_responses,
        "replaced_existing": replace_existing and existing_count > 0,
    }
