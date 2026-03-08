# Data Model & Database Schema

## Input CSV Format (Survey Monkey Export)

- **Filename pattern**: `data/{year}.csv` (years auto-discovered by scanning for `20XX.csv`)
- **Total columns**: 136
- **Header rows**: 2 (auto-skipped by transform pipeline)
- **Encoding**: UTF-8-sig or Latin-1

### Column Layout

| Columns | Content |
|---------|---------|
| 1-11 | Survey Monkey metadata: Respondent ID, Collector ID, Start/End timestamps, IP, Email, First/Last Name, Custom Data, Submission Method, Grade Selection |
| 12-133 | 9 survey questions x school levels (Grammar, Middle, High) |
| 134-136 | Demographics: Years at GVCA, IEP/504/ALP/Read, Minority |

### Survey Questions

| # | Question | Type | Scale |
|---|----------|------|-------|
| Q1 | Satisfaction with education provided | Likert | Extremely Satisfied -> Not Satisfied |
| Q2 | Satisfaction with intellectual growth | Likert | Extremely Satisfied -> Not Satisfied |
| Q3 | School culture reflects 7 core virtues | Likert | Strongly Reflected -> Not Reflected |
| Q4 | Satisfaction with moral character growth | Likert | Extremely Satisfied -> Not Satisfied |
| Q5 | Teacher communication effectiveness | Likert | Extremely Effective -> Not Effective |
| Q6 | Leadership communication effectiveness | Likert | Extremely Effective -> Not Effective |
| Q7 | How welcoming is the community | Likert | Extremely Welcoming -> Not Welcoming |
| Q8 | What makes GVCA a good choice? | Free text | — |
| Q9 | How can GVCA better serve you? | Free text | — |

Column naming convention: `({Level}) {Question text}` where Level is Grammar, Middle, High, or Generic (Q8/Q9 only).

### Data Handling Notes

- Missing values represented as `"-"` in CSV, converted to NULL on import
- Empty responses (all questions `"-"`) are filtered out
- Duplicate respondent IDs auto-fixed with `R0001`-style IDs
- Non-meaningful free text (NA, none, "-", no comment, etc.) filtered during processing

## PocketBase Collections

### survey_responses
Normalized survey data from CSV imports.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| respondent_id | text | required |
| school_level | text | Grammar/Middle/High |
| submission_method | text | |
| n_parents_represented | number | |
| demographics | json | Years at GVCA, IEP/504, Minority |
| satisfaction_scores | json | Q1-Q7 responses |
| imported_at | date | required |

### free_responses
Individual free-text responses extracted from survey data.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| response_id | text | required, unique |
| survey_response_id | text | links to survey_responses.id |
| question | text | required |
| question_type | text | "praise" (Q8) or "improvement" (Q9) |
| level | text | Grammar/Middle/High/Generic |
| response_text | text | required |

### tagging_results
LLM tagging output with stability scoring.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| response_id | text | links to free_responses.response_id |
| question | text | required |
| level | text | |
| response_text | text | required |
| llm_tags | json | list of assigned tags |
| tag_votes | json | dict of tag -> vote count |
| keywords_found | json | dict of tag -> keyword list |
| model_used | text | |
| n_samples | number | number of LLM samples (default 4) |
| threshold | number | voting threshold |
| stability_score | number | 0.0-1.0 Jaccard similarity |
| keyword_mismatches | json | list of {tag, keywords} for unmatched |
| review_status | text | pending/approved/hidden |
| batch_id | text | reference to batch_jobs |
| dismissed | bool | |
| dismissed_at | date | |

### tag_overrides
User modifications to automatically assigned tags.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| response_id | text | required |
| question | text | |
| original_tags | json | tags before override |
| modified_tags | json | tags after override |
| reason | text | |
| modified_at | date | |

### clustering_results
UMAP coordinates and cluster assignments.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| response_id | text | required |
| question | text | required |
| response_text | text | |
| level | text | |
| umap_x | number | 2D visualization X coordinate |
| umap_y | number | 2D visualization Y coordinate |
| cluster_id | number | HDBSCAN label (-1 = noise) |
| embed_model | text | embedding model used |

### cluster_summaries
Pre-computed metadata per cluster.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| question | text | "_all_" for global clustering |
| cluster_id | number | |
| size | number | response count |
| sample_responses | json | array of 3 sample texts |
| tag_distribution | json | tag -> {count, percentage} |
| centroid_x | number | mean UMAP X |
| centroid_y | number | mean UMAP Y |

### cluster_metadata
User-editable cluster names and descriptions.

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| cluster_id | number | |
| name | text | user-assigned |
| description | text | user-assigned |

Unique index on (year, cluster_id).

### batch_jobs
Tracks async batch processing jobs (OpenAI Batch API).

| Field | Type | Notes |
|-------|------|-------|
| job_type | text | tagging_batch/clustering/import |
| year | text | required |
| status | text | validating/in_progress/completed/failed/expired |
| progress | number | |
| total_items | number | |
| processed_items | number | |
| failed_items | number | |
| openai_batch_id | text | |
| input_file_id | text | |
| output_file_id | text | |
| error_file_id | text | |
| estimated_cost | number | |
| model_used | text | |
| error_message | text | |
| started_at | text | |
| completed_at | text | |
| metadata | json | |

### segment_summaries
Demographic segment statistics (placeholder).

| Field | Type | Notes |
|-------|------|-------|
| year | text | required |
| segment_name | text | required |
| tag_counts | json | |
| total_responses | number | |
| sample_responses | json | |

## Embedding Parquet Files

Stored at `data/embeddings/{year}.parquet`. One row per free-text response.

| Column | Type | Notes |
|--------|------|-------|
| response_id | string | links to free_responses |
| year | string | survey year |
| question | string | question identifier |
| response_text | string | original text |
| embedding | list[float] | 1536-dimension vector |
| embed_model | string | "text-embedding-3-small" |
| created_at | string | ISO timestamp |

These files act as a cache. The system checks for staleness (model change or zero response ID overlap) and only generates embeddings for new/missing responses, merging incrementally.

## Collection Relationships

```
survey_responses (1) --- M (free_responses)
                          |-- M (tagging_results)    [via response_id]
                          |-- M (clustering_results) [via response_id]
                          '-- M (tag_overrides)      [via response_id]

clustering_results --- cluster_summaries   [via cluster_id + year]
cluster_summaries  --- cluster_metadata    [via cluster_id + year]
tagging_results    --- batch_jobs          [via batch_id]
```

## File Artifacts

| Path | Content | Regenerable? |
|------|---------|--------------|
| `data/{year}.csv` | Raw Survey Monkey export | No (original input) |
| `data/processed/{year}.csv` | Normalized survey data | Yes (from raw CSV) |
| `data/embeddings/{year}.parquet` | Embedding vectors | Yes (costs API calls) |
| `artifacts/{year}_*.png` | Chart visualizations | Yes (free) |
| `artifacts/tagging_{year}.csv` | Tag export with columns | Yes (free) |
