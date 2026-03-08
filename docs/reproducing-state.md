# Reproducing Pipeline State

How to transfer a fully-processed pipeline state from one environment to another.

## What the Pipeline Produces

After running the full pipeline (import -> tagging -> clustering -> charts), data lives in two places:

1. **PocketBase database** (`pocketbase/pb_data/`) — all collections with survey data, tags, clusters
2. **Files on disk** — raw CSVs, processed CSVs, embedding parquet files, chart artifacts

## Option A: Direct SQLite Copy (Fastest)

Copy these from the source environment:

```bash
# Required
pocketbase/pb_data/              # SQLite database (entire directory)
data/{year}.csv                  # Raw input CSVs (one per year)
data/embeddings/{year}.parquet   # Cached embedding vectors

# Optional (can regenerate for free)
artifacts/                       # Charts and CSV exports
# Note: data/processed/{year}.csv is NOT auto-created by the pipeline.
# It can be generated on-demand via load_flattened() / combine_years().
```

Start PocketBase and the app — everything works immediately.

## Option B: Structured Export

If a direct file copy isn't possible:

### Database Export Methods

- **PocketBase Admin UI**: Browse at `http://localhost:8090/_/` and export collections
- **API**: `GET /api/collections/{collection}/records` (paginated JSON)
- **SQLite dump**: `sqlite3 pocketbase/pb_data/data.db .dump > backup.sql`

### Collections to Export (in dependency order)

1. `survey_responses` — base survey data
2. `free_responses` — extracted free-text answers
3. `tagging_results` — LLM tags, stability scores, vote counts
4. `tag_overrides` — manual human corrections
5. `clustering_results` — UMAP coordinates, cluster assignments
6. `cluster_summaries` — per-cluster statistics
7. `cluster_metadata` — user-assigned cluster names
8. `batch_jobs` — job execution history

## What's Expensive to Recreate

| Stage | Cost | What to Preserve |
|-------|------|------------------|
| **Tagging** | 1 OpenAI API call with `n=4` samples per response | `tagging_results` collection |
| **Embeddings** | 1 API call per unique response | `data/embeddings/{year}.parquet` |
| **Manual reviews** | Human time | `tag_overrides`, `cluster_metadata` |
| **Clustering** | CPU only (seconds) | Can re-run, but results are stochastic |
| **Charts** | CPU only (seconds) | Can regenerate from database |

## Minimum Viable Transfer

If you only want to get charts and the frontend working:

1. `pocketbase/pb_data/` (the SQLite database)
2. `data/{year}.csv` (raw input files)
3. `data/embeddings/{year}.parquet` (avoids re-running embedding API)

With these three, start the app and all pages work: dashboard, tagging review, cluster explorer, visualizations, and comparisons.
