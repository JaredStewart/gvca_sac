"""LLM-based tagging for survey responses using Polars."""

import json
import logging
import re
from itertools import combinations
from typing import Any, Callable, Coroutine

import polars as pl
from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError
from pydantic import BaseModel

from app.config import get_settings
from app.core.survey_config import get_taxonomy, get_taxonomy_string, get_taxonomy_tags
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)


class TagResponse(BaseModel):
    """Schema for LLM tag response."""
    tags: list[str]


def find_keywords_in_text(text: str) -> dict[str, list[str]]:
    """
    Find taxonomy keywords present in text.

    Args:
        text: Response text to search

    Returns:
        Dict mapping tag names to list of found keywords
    """
    if not text or not isinstance(text, str):
        return {}

    text_lower = text.lower()
    found: dict[str, list[str]] = {}

    taxonomy_keywords = get_taxonomy()
    for tag, keywords in taxonomy_keywords.items():
        tag_found = []
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')
            if pattern.search(text_lower):
                tag_found.append(keyword)
        if tag_found:
            found[tag] = tag_found

    return found


def build_system_prompt() -> str:
    """Build the system prompt (simple role definition, no taxonomy)."""
    return (
        "You are an expert at analyzing school survey responses and categorizing "
        "them according to a predefined taxonomy.\n"
        "You carefully read each response and identify all relevant categories that apply.\n"
        "You are precise and only assign tags when the content clearly matches the category definition."
    )


def build_user_prompt(response_text: str, question_context: str | None = None) -> str:
    """Build the user prompt with question, response, taxonomy, and instructions."""
    parts = []

    if question_context:
        parts.append(f"Survey Question: {question_context}")

    parts.append(f"Response: {response_text}")

    parts.append(
        "\nAnalyze this survey response and assign relevant tags from the taxonomy.\n\n"
        "TAXONOMY:"
    )
    parts.append(get_taxonomy_string())
    parts.append(
        "\nINSTRUCTIONS:\n"
        "1. Read the response carefully\n"
        "2. Assign ALL tags that apply to the response content\n"
        "3. A response can have multiple tags\n"
        "4. Only assign tags that are clearly relevant to the response content\n"
        "5. Return the list of applicable tag names exactly as they appear in the taxonomy\n"
        "6. OPPOSITE-CONDITION MARKERS:\n"
        '   - "Concern": Use ONLY on Good Choice (Q8) responses that contain negative '
        "sentiment or concerns despite being in a positive-framed question\n"
        '   - "No improvement listed": Use ONLY on Better Serve (Q9) responses that do '
        "not actually identify an area for improvement (e.g., 'Nothing' or general praise)"
    )

    return "\n".join(parts)


async def call_openai_api(
    user_prompt: str,
    model: str | None = None,
    n_samples: int = 1,
) -> list[dict[str, Any]]:
    """
    Call OpenAI API for tagging using the official SDK with structured output.

    Uses ``client.beta.chat.completions.parse`` with the ``TagResponse``
    Pydantic model for validated structured output.  The ``n`` parameter
    sends a single request for multiple completions (cheaper and faster
    than looping).

    Args:
        user_prompt: User prompt (question + response text + taxonomy + instructions)
        model: Model to use (defaults to config ``default_llm_model``)
        n_samples: Number of completions

    Returns:
        List of parsed responses (dicts with ``tags``)
    """
    settings = get_settings()
    model = model or settings.default_llm_model

    if not settings.openai_api_key:
        logger.warning("No OpenAI API key configured")
        return []

    system_prompt = build_system_prompt()

    logger.debug(
        "OpenAI request: model=%s, n_samples=%d, prompt_length=%d",
        model, n_samples, len(user_prompt),
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            n=n_samples,
            response_format=TagResponse,
        )

        logger.debug(
            "OpenAI response: model=%s, choices=%d",
            response.model, len(response.choices),
        )

        results: list[dict[str, Any]] = []
        for i, choice in enumerate(response.choices):
            parsed = choice.message.parsed
            if parsed is not None:
                result = parsed.model_dump()
                logger.debug(
                    "Sample %d/%d tags: %s",
                    i + 1, n_samples, result.get("tags"),
                )
                results.append(result)
            else:
                # Fallback: try manual JSON parse of content
                content = choice.message.content
                logger.warning(
                    "Sample %d/%d: structured parse returned None, "
                    "raw content: %s",
                    i + 1, n_samples, content,
                )
                if content:
                    try:
                        results.append(json.loads(content))
                    except json.JSONDecodeError:
                        logger.error(
                            "Sample %d/%d: JSON parse also failed, "
                            "full content: %s",
                            i + 1, n_samples, content,
                        )

        if not results:
            logger.warning(
                "No valid responses from OpenAI: model=%s, "
                "n_samples_requested=%d, choices_received=%d",
                model, n_samples, len(response.choices),
            )

        return results

    except AuthenticationError as e:
        logger.error(
            "OpenAI authentication failed: status=%s, message=%s",
            e.status_code, e.message,
        )
        raise
    except RateLimitError as e:
        logger.error(
            "OpenAI rate limit exceeded: status=%s, message=%s",
            e.status_code, e.message,
        )
        raise
    except APIError as e:
        logger.error(
            "OpenAI API error: status=%s, message=%s, body=%s",
            e.status_code, e.message, e.body,
        )
        raise
    except Exception as e:
        logger.error("Unexpected error calling OpenAI: %s", e, exc_info=True)
        raise


def apply_voting(
    responses: list[dict[str, Any]],
    threshold: int
) -> tuple[list[str], dict[str, int]]:
    """
    Apply voting threshold to determine final tags.

    Args:
        responses: List of LLM responses with tags
        threshold: Minimum votes needed

    Returns:
        Tuple of (final_tags, vote_counts)
    """
    taxonomy_tags = get_taxonomy_tags()
    votes: dict[str, int] = {tag: 0 for tag in taxonomy_tags}

    for response in responses:
        tags = response.get("tags", [])
        for tag in tags:
            tag_normalized = tag.strip()
            if tag_normalized in votes:
                votes[tag_normalized] += 1

    final_tags = [tag for tag, count in votes.items() if count >= threshold]
    return final_tags, votes


async def tag_single_response(
    response_id: str,
    response_text: str,
    question: str | None,
    model: str,
    n_samples: int,
    threshold: int,
) -> dict[str, Any]:
    """Tag a single response."""
    prompt = build_user_prompt(response_text, question)

    # Call LLM
    llm_responses = await call_openai_api(prompt, model, n_samples)

    # Apply voting
    final_tags, votes = apply_voting(llm_responses, threshold)

    # Find keywords
    keywords_found = find_keywords_in_text(response_text)

    return {
        "response_id": response_id,
        "llm_tags": final_tags,
        "tag_votes": votes,
        "keywords_found": keywords_found,
        "model_used": model,
        "n_samples": n_samples,
        "threshold": threshold,
    }


# ================ Stability Scoring and Keyword Mismatch Functions ================


def compute_jaccard_stability(tag_lists: list[list[str]]) -> float:
    """
    Compute average pairwise Jaccard similarity (IoU) across N tag sets.

    For each pair (Si, Sj): Jaccard = |Si ∩ Sj| / |Si ∪ Sj| (both-empty → 1.0)
    Returns mean of all C(N,2) pairwise values.
    """
    n = len(tag_lists)
    if n < 2:
        return 1.0 if n == 1 else 0.0

    tag_sets = [set(tags) for tags in tag_lists]

    jaccard_sum = 0.0
    pair_count = 0

    for si, sj in combinations(tag_sets, 2):
        if not si and not sj:
            jaccard = 1.0
        else:
            intersection = len(si & sj)
            union = len(si | sj)
            jaccard = intersection / union
        jaccard_sum += jaccard
        pair_count += 1

    return jaccard_sum / pair_count if pair_count > 0 else 0.0


async def tag_response_with_stability(
    response_text: str,
    question: str | None = None,
    level: str | None = None,
    model: str | None = None,
    n_samples: int = 4,
) -> dict[str, Any]:
    """
    Tag a response with stability scoring (FR-014).

    Runs multiple samples and computes stability as the minimum agreement
    ratio across all assigned tags.

    Args:
        response_text: Text to tag
        question: Optional question context
        level: Optional school level
        model: Model to use
        n_samples: Number of samples for stability scoring

    Returns:
        Dict with llm_tags, tag_votes, stability_score, n_samples
    """
    prompt = build_user_prompt(response_text, question)

    # Get multiple samples
    llm_responses = await call_openai_api(prompt, model, n_samples)

    if not llm_responses:
        return {
            "llm_tags": [],
            "tag_votes": {},
            "stability_score": 0.0,
            "n_samples": 0,
        }

    # Count votes for each tag
    tag_votes: dict[str, int] = {}
    for response in llm_responses:
        tags = response.get("tags", [])
        for tag in tags:
            tag_normalized = tag.strip()
            tag_votes[tag_normalized] = tag_votes.get(tag_normalized, 0) + 1

    # Filter to valid taxonomy tags only
    valid_tags = set(get_taxonomy_tags())
    tag_votes = {tag: votes for tag, votes in tag_votes.items() if tag in valid_tags}

    actual_samples = len(llm_responses)

    # Tags that appear in majority of samples
    threshold = actual_samples / 2
    llm_tags = [
        tag for tag, votes in tag_votes.items()
        if votes > threshold
    ]

    # Compute stability score: average pairwise Jaccard (IoU)
    raw_tag_lists = [response.get("tags", []) for response in llm_responses]
    stability_score = compute_jaccard_stability(raw_tag_lists)

    return {
        "llm_tags": llm_tags,
        "tag_votes": tag_votes,
        "stability_score": round(stability_score, 3),
        "n_samples": actual_samples,
    }


def detect_keyword_mismatches(
    response_text: str,
    assigned_tags: list[str],
) -> list[dict[str, Any]]:
    """
    Detect keyword mismatches where keywords are present but tags not assigned (FR-015).

    Args:
        response_text: The response text
        assigned_tags: Tags that were assigned by the LLM

    Returns:
        List of {tag: str, keywords: list[str]} for unassigned tags with found keywords
    """
    # Find all keywords in text
    keywords_found = find_keywords_in_text(response_text)

    # Find mismatches: tags with keywords found but not assigned
    mismatches = []
    for tag, keywords in keywords_found.items():
        if tag not in assigned_tags and keywords:
            mismatches.append({
                "tag": tag,
                "keywords": keywords,
            })

    return mismatches


async def run_tagging_async(
    year: str,
    data: pl.DataFrame,
    model: str | None = None,
    n_samples: int = 4,
    threshold: int = 2,
    use_batch_api: bool = False,
    test_mode: bool = False,
    test_size: int = 20,
    progress_callback: Callable[[int, int], Coroutine[Any, Any, None]] | None = None,
) -> dict[str, Any]:
    """
    Run tagging on survey data asynchronously.

    Args:
        year: Survey year
        data: Survey DataFrame
        model: LLM model to use
        n_samples: Number of samples per response
        threshold: Voting threshold
        use_batch_api: Whether to use batch API
        test_mode: Whether to run on a small test subset
        test_size: Number of responses to tag in test mode
        progress_callback: Progress update callback

    Returns:
        Summary of tagging results
    """
    from app.core.transform import extract_free_responses, is_meaningful_response

    # Extract free responses
    responses_df = extract_free_responses(data)

    # Apply test mode slicing if enabled
    if test_mode:
        responses_df = responses_df.head(test_size)
        logger.info(f"Test mode enabled: processing {test_size} responses")

    total = responses_df.height

    if total == 0:
        return {"status": "no_responses", "tagged": 0}

    logger.info(f"Tagging {total} responses for year {year}")

    # Tag each response
    results = []

    for idx, row in enumerate(responses_df.iter_rows(named=True)):
        response_id = row["response_id"]
        response_text = row["response_text"]
        question = row.get("question")
        level = row.get("level")

        # Defense-in-depth: skip non-meaningful responses
        if not is_meaningful_response(response_text):
            if progress_callback:
                await progress_callback(idx + 1, total)
            continue

        try:
            result = await tag_single_response(
                response_id=response_id,
                response_text=response_text,
                question=question,
                model=model,
                n_samples=n_samples,
                threshold=threshold,
            )

            # Store in PocketBase
            record = {
                "year": year,
                "response_id": response_id,
                "question": question or "",
                "level": level or "",
                "response_text": response_text,
                "llm_tags": result["llm_tags"],
                "tag_votes": result["tag_votes"],
                "keywords_found": result["keywords_found"],
                "model_used": model,
                "n_samples": n_samples,
                "threshold": threshold,
            }

            await pb_client.upsert(
                "tagging_results",
                record,
                f'year = "{year}" && response_id = "{response_id}"',
            )

            results.append(result)

        except Exception as e:
            logger.error(f"Error tagging response {response_id}: {e}")

        if progress_callback:
            await progress_callback(idx + 1, total)

    return {
        "status": "completed",
        "year": year,
        "tagged": len(results),
        "total": total,
        "test_mode": test_mode,
    }
