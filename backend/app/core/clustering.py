"""Clustering and embedding utilities using Polars."""

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine

import numpy as np
import polars as pl

from app.core.embeddings import (
    build_embedding_dataframe,
    embeddings_are_stale,
    generate_embeddings,
    get_responses_needing_embedding,
    load_embeddings,
    merge_embeddings,
    save_embeddings,
)
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)


def split_into_fragments(text: str, max_sentences: int = 3) -> list[str]:
    """
    Split text into semantic fragments based on sentences.

    Args:
        text: Input text to fragment
        max_sentences: Maximum sentences per fragment

    Returns:
        List of text fragments
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    fragments = []
    current = []

    for sentence in sentences:
        current.append(sentence)
        if len(current) >= max_sentences:
            fragments.append(". ".join(current) + ".")
            current = []

    if current:
        fragments.append(". ".join(current) + ".")

    return fragments


def create_fragment_dataset(
    df: pl.DataFrame,
    response_col: str = "response_text",
    max_sentences: int = 3,
) -> pl.DataFrame:
    """
    Create a dataset of response fragments with metadata.

    Args:
        df: DataFrame with responses
        response_col: Column containing responses
        max_sentences: Max sentences per fragment

    Returns:
        pl.DataFrame: Fragment dataset
    """
    rows = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        text = row.get(response_col)
        if text is None or not str(text).strip():
            continue

        fragments = split_into_fragments(str(text), max_sentences)

        for frag_idx, fragment in enumerate(fragments):
            frag_row = {
                "parent_response_id": row.get("response_id", idx),
                "fragment_id": f"{idx}_{frag_idx}",
                "fragment_text": fragment,
                "fragment_index": frag_idx,
                "total_fragments": len(fragments),
                "original_response": text,
            }

            # Copy metadata
            for col in df.columns:
                if col != response_col:
                    frag_row[f"parent_{col}"] = row.get(col)

            rows.append(frag_row)

    return pl.DataFrame(rows)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    for_clustering: bool = False,
) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP with cosine metric.

    Args:
        embeddings: High-dimensional embeddings.
        n_components: Target dimensions (10 for clustering, 2 for viz).
        random_state: Random seed for reproducibility.
        for_clustering: If True, uses min_dist=0.0 for tight packing.
                        If False, uses min_dist=0.1 for visual clarity.

    Returns:
        np.ndarray: Reduced embeddings.
    """
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed")
        raise

    min_dist = 0.0 if for_clustering else 0.1

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 2,
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN.

    Args:
        embeddings: Embedding vectors (should be 10D UMAP output).
        min_cluster_size: Minimum cluster size.
        min_samples: HDBSCAN min_samples parameter.

    Returns:
        np.ndarray: Cluster labels (-1 for noise).
    """
    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed")
        raise

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def summarize_clusters(
    df: pl.DataFrame,
    cluster_col: str = "cluster_id",
    response_col: str = "response_text",
    tag_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate summary statistics for each cluster.

    Args:
        df: Clustered DataFrame
        cluster_col: Cluster label column
        response_col: Response text column
        tag_map: Mapping of response_id -> list of tag names (from tagging_results)

    Returns:
        List of cluster summaries
    """
    summaries = []
    tag_map = tag_map or {}

    unique_clusters = sorted([
        c
        for c in df[cluster_col].unique().to_list()
        if c is not None and c != -1
    ])

    for cluster_id in unique_clusters:
        cluster_data = df.filter(pl.col(cluster_col) == cluster_id)

        # Get centroid coordinates
        centroid_x = None
        centroid_y = None
        if "umap_x" in cluster_data.columns:
            centroid_x = cluster_data["umap_x"].mean()
        if "umap_y" in cluster_data.columns:
            centroid_y = cluster_data["umap_y"].mean()

        # Get sample responses
        sample_responses = []
        if response_col in cluster_data.columns:
            sample_responses = cluster_data[response_col].head(3).to_list()

        summary = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_data),
            "sample_responses": sample_responses,
            "centroid_x": float(centroid_x) if centroid_x is not None else None,
            "centroid_y": float(centroid_y) if centroid_y is not None else None,
        }

        # Tag distribution from tag_map
        tag_counts: dict[str, int] = {}
        cluster_response_ids = cluster_data["response_id"].to_list()
        for rid in cluster_response_ids:
            for tag in tag_map.get(rid, []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        cluster_size = len(cluster_data)
        tag_distribution = {
            tag: {
                "count": count,
                "percentage": round(count / cluster_size * 100, 2),
            }
            for tag, count in sorted(
                tag_counts.items(), key=lambda x: x[1], reverse=True
            )
        }

        summary["tag_distribution"] = tag_distribution
        summaries.append(summary)

    return summaries


async def run_embedding_async(
    year: str,
    data: pl.DataFrame,
    embed_model: str = "text-embedding-3-small",
    question: str | None = None,
    progress_callback: Callable[[int, int], Coroutine[Any, Any, None]]
    | None = None,
) -> dict[str, Any]:
    """
    Run only the embedding generation step (no UMAP/HDBSCAN).

    Extracts free-text responses, generates or loads cached embeddings,
    and saves them to Parquet. Useful for pre-generating embeddings
    before running the full clustering pipeline.

    Args:
        year: Survey year
        data: Survey DataFrame
        embed_model: OpenAI embedding model name
        question: Optional specific question to embed
        progress_callback: Progress update callback

    Returns:
        Summary with counts of generated/cached embeddings
    """
    from app.config import get_settings
    from app.core.transform import extract_free_responses

    settings = get_settings()
    embeddings_dir = str(settings.data_dir / "embeddings")

    # Extract responses
    responses_df = extract_free_responses(data, question=question)
    total = responses_df.height

    if total == 0:
        return {"status": "no_responses", "embedded": 0}

    logger.info(f"Embedding {total} responses for year {year}")

    if progress_callback:
        await progress_callback(0, total)

    texts = responses_df["response_text"].to_list()
    response_ids = responses_df["response_id"].to_list()
    questions = responses_df["question"].to_list()

    stored_df = load_embeddings(year, directory=embeddings_dir)
    cached_count = 0
    generated_count = 0

    if stored_df is not None and not embeddings_are_stale(stored_df, embed_model, current_response_ids=response_ids):
        stored_id_to_idx = {
            rid: i for i, rid in enumerate(stored_df["response_id"].to_list())
        }
        missing_ids = [rid for rid in response_ids if rid not in stored_id_to_idx]
        cached_count = len(response_ids) - len(missing_ids)

        if missing_ids:
            logger.info(
                f"{len(missing_ids)} new responses need embedding, "
                f"{cached_count} cached"
            )
            missing_texts = [
                texts[i]
                for i, rid in enumerate(response_ids)
                if rid in set(missing_ids)
            ]
            new_emb_array = await generate_embeddings(
                missing_texts, model=embed_model,
                progress_callback=progress_callback,
            )
            missing_response_data = responses_df.filter(
                pl.col("response_id").is_in(missing_ids)
            )
            new_emb_df = build_embedding_dataframe(
                response_ids=missing_response_data["response_id"].to_list(),
                years=[year] * len(missing_ids),
                questions=missing_response_data["question"].to_list(),
                response_texts=missing_response_data["response_text"].to_list(),
                embeddings=new_emb_array,
                model=embed_model,
            )
            stored_df = merge_embeddings(stored_df, new_emb_df)
            save_embeddings(stored_df, year, directory=embeddings_dir)
            generated_count = len(missing_ids)
        else:
            logger.info(f"All {total} embeddings cached for year {year}")
            if progress_callback:
                await progress_callback(total, total)
    else:
        if stored_df is not None:
            logger.info(
                f"Embeddings stale for year {year}, re-generating all embeddings"
            )

        logger.info(
            f"Generating embeddings for {len(texts)} responses via OpenAI API"
        )
        raw_embeddings = await generate_embeddings(
            texts, model=embed_model,
            progress_callback=progress_callback,
        )

        emb_df = build_embedding_dataframe(
            response_ids=response_ids,
            years=[year] * len(response_ids),
            questions=questions,
            response_texts=texts,
            embeddings=raw_embeddings,
            model=embed_model,
        )
        save_embeddings(emb_df, year, directory=embeddings_dir)
        generated_count = total

    if progress_callback:
        await progress_callback(total, total)

    logger.info(
        f"Embedding complete: {generated_count} generated, {cached_count} cached"
    )

    # Update pipeline status
    from app.services.pipeline_manager import pipeline_manager

    pipeline_manager.set_embeddings_complete(year)

    return {
        "status": "completed",
        "year": year,
        "total_responses": total,
        "generated": generated_count,
        "cached": cached_count,
        "embed_model": embed_model,
    }


async def run_clustering_async(
    year: str,
    data: pl.DataFrame,
    embed_model: str = "text-embedding-3-small",
    min_cluster_size: int = 5,
    question: str | None = None,
    progress_callback: Callable[[int, int], Coroutine[Any, Any, None]]
    | None = None,
) -> dict[str, Any]:
    """
    Run the full clustering pipeline asynchronously.

    Pipeline:
    1. Extract free-text responses
    2. Load or generate embeddings (OpenAI API)
    3. UMAP to 10D (for clustering, cosine metric, min_dist=0)
    4. HDBSCAN on 10D embeddings
    5. UMAP to 2D (for visualization, cosine metric, min_dist=0.1)
    6. Save embeddings to Parquet
    7. Upsert results to PocketBase

    Args:
        year: Survey year
        data: Survey DataFrame
        embed_model: OpenAI embedding model name
        min_cluster_size: Minimum cluster size for HDBSCAN
        question: Optional specific question to cluster
        progress_callback: Progress update callback

    Returns:
        Clustering results summary
    """
    from app.config import get_settings
    from app.core.transform import extract_free_responses

    settings = get_settings()
    embeddings_dir = str(settings.data_dir / "embeddings")

    # Step 0: Extract responses
    responses_df = extract_free_responses(data, question=question)
    total = responses_df.height

    if total == 0:
        return {"status": "no_responses", "clustered": 0}

    logger.info(f"Clustering {total} responses for year {year}")

    total_steps = 6  # load/generate, UMAP 10D, cluster, UMAP 2D, save, store

    if progress_callback:
        await progress_callback(0, total_steps)

    # Step 1: Load or generate embeddings
    texts = responses_df["response_text"].to_list()
    response_ids = responses_df["response_id"].to_list()
    questions = responses_df["question"].to_list()

    stored_df = load_embeddings(year, directory=embeddings_dir)

    if stored_df is not None and not embeddings_are_stale(stored_df, embed_model, current_response_ids=response_ids):
        logger.info(f"Loaded {stored_df.height} cached embeddings for year {year}")
        # Extract embedding matrix from stored data, matching response order
        stored_id_to_idx = {
            rid: i for i, rid in enumerate(stored_df["response_id"].to_list())
        }
        embeddings_list = stored_df["embedding"].to_list()

        # Check if all response IDs are in the stored data
        missing_ids = [rid for rid in response_ids if rid not in stored_id_to_idx]

        if missing_ids:
            logger.info(
                f"{len(missing_ids)} new responses need embedding, "
                f"{len(response_ids) - len(missing_ids)} cached"
            )
            # Generate only for missing responses
            missing_texts = [
                texts[i]
                for i, rid in enumerate(response_ids)
                if rid in set(missing_ids)
            ]
            new_emb_array = await generate_embeddings(
                missing_texts, model=embed_model
            )

            # Build new embedding records
            missing_response_data = responses_df.filter(
                pl.col("response_id").is_in(missing_ids)
            )
            new_emb_df = build_embedding_dataframe(
                response_ids=missing_response_data["response_id"].to_list(),
                years=[year] * len(missing_ids),
                questions=missing_response_data["question"].to_list(),
                response_texts=missing_response_data["response_text"].to_list(),
                embeddings=new_emb_array,
                model=embed_model,
            )
            stored_df = merge_embeddings(stored_df, new_emb_df)
            save_embeddings(stored_df, year, directory=embeddings_dir)
            embeddings_list = stored_df["embedding"].to_list()
            stored_id_to_idx = {
                rid: i
                for i, rid in enumerate(stored_df["response_id"].to_list())
            }

        # Reorder embeddings to match responses_df order
        ordered_embeddings = []
        for rid in response_ids:
            idx = stored_id_to_idx[rid]
            ordered_embeddings.append(embeddings_list[idx])
        raw_embeddings = np.array(ordered_embeddings, dtype=np.float32)
    else:
        if stored_df is not None:
            logger.info(
                f"Embeddings stale for year {year}, re-generating all embeddings"
            )

        logger.info(
            f"Generating embeddings for {len(texts)} responses via OpenAI API"
        )
        raw_embeddings = await generate_embeddings(texts, model=embed_model)

        # Build and save embedding DataFrame
        emb_df = build_embedding_dataframe(
            response_ids=response_ids,
            years=[year] * len(response_ids),
            questions=questions,
            response_texts=texts,
            embeddings=raw_embeddings,
            model=embed_model,
        )
        save_embeddings(emb_df, year, directory=embeddings_dir)

    if progress_callback:
        await progress_callback(1, total_steps)

    # Step 2: UMAP to 10D for clustering
    logger.info("Reducing dimensions to 10D for clustering...")
    embeddings_10d = await asyncio.to_thread(
        reduce_dimensions, raw_embeddings, n_components=10, for_clustering=True
    )

    if progress_callback:
        await progress_callback(2, total_steps)

    # Step 3: HDBSCAN on 10D
    logger.info("Clustering on 10D embeddings...")
    cluster_labels = await asyncio.to_thread(
        cluster_embeddings, embeddings_10d, min_cluster_size
    )

    if progress_callback:
        await progress_callback(3, total_steps)

    # Step 4: UMAP to 2D for visualization
    logger.info("Reducing dimensions to 2D for visualization...")
    coords_2d = await asyncio.to_thread(
        reduce_dimensions, raw_embeddings, n_components=2, for_clustering=False
    )

    if progress_callback:
        await progress_callback(4, total_steps)

    # Step 5: Add results to DataFrame
    responses_df = responses_df.with_columns([
        pl.Series("umap_x", coords_2d[:, 0]),
        pl.Series("umap_y", coords_2d[:, 1]),
        pl.Series("cluster_id", cluster_labels),
    ])

    # Delete old clustering results and summaries for this year before inserting
    logger.info("Clearing old clustering results for year %s...", year)
    await pb_client.delete_by_filter("clustering_results", f'year = "{year}"')
    await pb_client.delete_by_filter("cluster_summaries", f'year = "{year}"')

    # Store clustering results in PocketBase
    logger.info("Saving clustering results to PocketBase...")
    for row in responses_df.iter_rows(named=True):
        record = {
            "year": year,
            "response_id": row["response_id"],
            "question": row.get("question", ""),
            "response_text": row.get("response_text", ""),
            "level": row.get("level", ""),
            "umap_x": float(row["umap_x"]),
            "umap_y": float(row["umap_y"]),
            "cluster_id": int(row["cluster_id"]),
            "embed_model": embed_model,
        }

        await pb_client.create("clustering_results", record)

    if progress_callback:
        await progress_callback(5, total_steps)

    # Fetch tag data from PocketBase for tag_distribution
    tag_map: dict[str, list[str]] = {}
    try:
        tag_results = await pb_client.get_full_list(
            "tagging_results",
            filter_str=f'year = "{year}"',
        )
        for t in tag_results:
            rid = t.get("response_id")
            tags = t.get("llm_tags", [])
            if rid and tags:
                tag_map[rid] = tags
    except Exception as e:
        logger.warning(f"Could not fetch tagging results for tag_distribution: {e}")

    # Generate and store cluster summaries
    summaries = summarize_clusters(responses_df, tag_map=tag_map)

    # Use "_all_" as sentinel when no specific question filter is applied
    # (PocketBase requires non-empty string for required text fields)
    question_value = question if question else "_all_"

    for summary in summaries:
        # Ensure all values are Python native types for JSON serialization
        cluster_id = int(summary["cluster_id"])
        size = int(summary["size"])
        centroid_x = summary.get("centroid_x")
        centroid_y = summary.get("centroid_y")

        summary_record = {
            "year": year,
            "question": question_value,
            "cluster_id": cluster_id,
            "size": size,
            "sample_responses": summary["sample_responses"],
            "tag_distribution": summary.get("tag_distribution", {}),
            "centroid_x": float(centroid_x) if centroid_x is not None else None,
            "centroid_y": float(centroid_y) if centroid_y is not None else None,
        }

        await pb_client.create("cluster_summaries", summary_record)

    if progress_callback:
        await progress_callback(6, total_steps)

    # Count clusters (excluding noise = -1)
    n_clusters = len([c for c in set(cluster_labels) if c != -1])

    logger.info(
        f"Clustering complete: {n_clusters} clusters found from {total} responses"
    )

    # Update pipeline status
    from app.services.pipeline_manager import pipeline_manager

    pipeline_manager.set_embeddings_complete(year)
    pipeline_manager.set_clustering_complete(year)

    return {
        "status": "completed",
        "year": year,
        "total_responses": total,
        "n_clusters": n_clusters,
        "embed_model": embed_model,
    }


async def recluster_subset(
    response_ids: list[str],
    year: str,
    min_cluster_size: int = 3,
) -> dict[str, Any]:
    """
    Re-cluster a subset of responses using stored embeddings.

    No new API calls — reuses embeddings from Parquet file.
    Runs two-stage UMAP (10D + 2D) and HDBSCAN on the subset.

    Args:
        response_ids: List of response IDs to re-cluster.
        year: Survey year.
        min_cluster_size: HDBSCAN min_cluster_size for the subset.

    Returns:
        Dict with coordinates, clusters, total, cluster_count, noise_count.

    Raises:
        ValueError: If embeddings not found or too few response IDs.
    """
    from app.config import get_settings

    settings = get_settings()
    embeddings_dir = str(settings.data_dir / "embeddings")

    if len(response_ids) < 5:
        raise ValueError("At least 5 responses are required for re-clustering")

    # Load stored embeddings
    stored_df = load_embeddings(year, directory=embeddings_dir)
    if stored_df is None:
        raise FileNotFoundError(
            f"No embeddings found for year {year}. Run clustering first."
        )

    # Filter to requested response IDs
    subset_df = stored_df.filter(pl.col("response_id").is_in(response_ids))

    if subset_df.height < 5:
        raise ValueError(
            f"Only {subset_df.height} of {len(response_ids)} response IDs "
            f"found in stored embeddings"
        )

    # Extract embedding matrix
    embeddings_list = subset_df["embedding"].to_list()
    raw_embeddings = np.array(embeddings_list, dtype=np.float32)

    # Two-stage UMAP + HDBSCAN
    embeddings_10d = await asyncio.to_thread(
        reduce_dimensions, raw_embeddings, n_components=10, for_clustering=True
    )
    cluster_labels = await asyncio.to_thread(
        cluster_embeddings, embeddings_10d, min_cluster_size
    )
    coords_2d = await asyncio.to_thread(
        reduce_dimensions, raw_embeddings, n_components=2, for_clustering=False
    )

    # Build coordinate results
    subset_response_ids = subset_df["response_id"].to_list()
    subset_texts = subset_df["response_text"].to_list()
    subset_questions = subset_df["question"].to_list()

    coordinates = []
    for i, rid in enumerate(subset_response_ids):
        coordinates.append({
            "response_id": rid,
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "cluster_id": int(cluster_labels[i]),
            "response_text": subset_texts[i] if i < len(subset_texts) else None,
            "question": subset_questions[i] if i < len(subset_questions) else None,
        })

    # Build cluster summaries for the subset
    cluster_ids_set = set(int(c) for c in cluster_labels if c != -1)
    clusters = []
    for cid in sorted(cluster_ids_set):
        cluster_coords = [c for c in coordinates if c["cluster_id"] == cid]
        sample_texts = [
            c["response_text"]
            for c in cluster_coords[:3]
            if c.get("response_text")
        ]
        centroid_x = np.mean([c["x"] for c in cluster_coords])
        centroid_y = np.mean([c["y"] for c in cluster_coords])
        clusters.append({
            "cluster_id": cid,
            "size": len(cluster_coords),
            "sample_responses": sample_texts,
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "tag_distribution": {},
        })

    noise_count = sum(1 for c in cluster_labels if c == -1)

    return {
        "coordinates": coordinates,
        "clusters": clusters,
        "total": len(coordinates),
        "cluster_count": len(clusters),
        "noise_count": noise_count,
    }
