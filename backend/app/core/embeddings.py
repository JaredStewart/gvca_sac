"""OpenAI embedding generation and Parquet file persistence."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Get or create the AsyncOpenAI client singleton."""
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying embedding batch due to {type(retry_state.outcome.exception()).__name__}, "
        f"attempt {retry_state.attempt_number}"
    ),
)
async def _embed_batch(
    texts: list[str], model: str, client: AsyncOpenAI
) -> list[list[float]]:
    """Embed a single batch with retry on transient errors."""
    response = await client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


async def generate_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    progress_callback=None,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts via OpenAI API.

    Sends texts in batched synchronous calls, chunked at batch_size.
    Retries on transient errors (rate limit, connection, server errors).
    Fails immediately on authentication errors.

    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model name.
        batch_size: Max texts per API call.
        progress_callback: Optional async callback(processed, total).

    Returns:
        np.ndarray of shape (N, embedding_dim) where N = len(non-empty texts).

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        AuthenticationError: If the API key is invalid.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in the .env file."
        )

    # Filter out empty/whitespace texts, tracking original indices
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text.strip())

    if not valid_texts:
        return np.array([], dtype=np.float32).reshape(0, 0)

    client = _get_client()
    all_embeddings: list[list[float]] = []
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(valid_texts), batch_size):
        batch = valid_texts[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        logger.info(
            f"Embedding batch {batch_num}/{total_batches}: {len(batch)} texts"
        )

        try:
            embeddings = await _embed_batch(batch, model, client)
            all_embeddings.extend(embeddings)
        except BadRequestError as e:
            if "max_tokens" in str(e).lower() or "too many" in str(e).lower():
                # Batch too large; split in half and retry
                logger.warning(
                    f"Batch {batch_num} too large, splitting in half"
                )
                mid = len(batch) // 2
                left = await _embed_batch(batch[:mid], model, client)
                right = await _embed_batch(batch[mid:], model, client)
                all_embeddings.extend(left)
                all_embeddings.extend(right)
            else:
                raise
        except AuthenticationError:
            logger.error("Invalid OpenAI API key")
            raise

        if progress_callback:
            await progress_callback(
                min(batch_idx + batch_size, len(valid_texts)),
                len(valid_texts),
            )

    return np.array(all_embeddings, dtype=np.float32)


def save_embeddings(
    df: pl.DataFrame,
    year: str,
    directory: str = "data/embeddings",
) -> Path:
    """
    Save embeddings DataFrame to a Parquet file.

    Expected columns: response_id, year, question, response_text,
    embedding, embed_model, created_at.

    Args:
        df: Polars DataFrame with embedding data.
        year: Survey year (used for filename).
        directory: Directory to write to.

    Returns:
        Path to the written Parquet file.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{year}.parquet"

    df.write_parquet(file_path)
    logger.info(f"Saved embeddings to {file_path} ({df.height} rows)")
    return file_path


def load_embeddings(
    year: str,
    directory: str = "data/embeddings",
) -> pl.DataFrame | None:
    """
    Load embeddings from a Parquet file.

    Args:
        year: Survey year.
        directory: Directory to read from.

    Returns:
        Polars DataFrame if file exists, None otherwise.
    """
    file_path = Path(directory) / f"{year}.parquet"

    if not file_path.exists():
        logger.info(f"No embedding file found at {file_path}")
        return None

    df = pl.read_parquet(file_path)
    logger.info(f"Loading embeddings from {file_path} ({df.height} rows)")
    return df


def embeddings_are_stale(
    stored: pl.DataFrame,
    current_model: str,
    current_response_ids: list[str] | None = None,
) -> bool:
    """
    Check if stored embeddings are stale.

    Stale when:
    - The embedding model changed.
    - The stored response_ids have zero overlap with current ones
      (e.g. response_id format changed from counter to natural key).

    Args:
        stored: DataFrame loaded from Parquet.
        current_model: The currently configured model name.
        current_response_ids: Optional list of current response IDs to
            check for format compatibility.

    Returns:
        True if embeddings should be regenerated from scratch.
    """
    if stored.is_empty():
        return True

    stored_model = stored["embed_model"][0]
    if stored_model != current_model:
        return True

    # If current response IDs provided, check overlap
    if current_response_ids is not None:
        stored_ids = set(stored["response_id"].to_list())
        overlap = sum(1 for rid in current_response_ids if rid in stored_ids)
        if overlap == 0 and len(current_response_ids) > 0:
            return True

    return False


def get_responses_needing_embedding(
    all_responses: pl.DataFrame,
    existing_embeddings: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Return only responses not yet embedded.

    Args:
        all_responses: All current responses (must have response_id column).
        existing_embeddings: Previously stored embeddings (or None).

    Returns:
        DataFrame of responses that need embedding.
    """
    if existing_embeddings is None:
        return all_responses

    existing_ids = set(existing_embeddings["response_id"].to_list())
    return all_responses.filter(~pl.col("response_id").is_in(existing_ids))


def merge_embeddings(
    existing: pl.DataFrame,
    new_embeddings: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge new embeddings into existing, deduplicating by response_id.

    Args:
        existing: Previously stored embeddings.
        new_embeddings: Newly generated embeddings.

    Returns:
        Combined DataFrame with duplicates removed (keeps latest).
    """
    combined = pl.concat([existing, new_embeddings])
    return combined.unique(subset=["response_id"], keep="last")


def build_embedding_dataframe(
    response_ids: list[str],
    years: list[str],
    questions: list[str],
    response_texts: list[str],
    embeddings: np.ndarray,
    model: str,
) -> pl.DataFrame:
    """
    Build a Polars DataFrame from raw embedding results.

    Args:
        response_ids: List of response IDs.
        years: List of year values.
        questions: List of question identifiers.
        response_texts: List of original response texts.
        embeddings: Numpy array of shape (N, dim).
        model: Embedding model name used.

    Returns:
        Polars DataFrame ready for Parquet storage.
    """
    now = datetime.now(timezone.utc).isoformat()

    return pl.DataFrame({
        "response_id": response_ids,
        "year": years,
        "question": questions,
        "response_text": response_texts,
        "embedding": embeddings.tolist(),
        "embed_model": [model] * len(response_ids),
        "created_at": [now] * len(response_ids),
    })
