# API Reference

Base URL: `http://localhost:8000/api`

## Pipeline Management

### GET /pipeline/years
List available survey years.

**Response:**
```json
{
  "years": ["2023", "2024", "2025"]
}
```

### POST /pipeline/init/{year}
Initialize and load data for a year.

**Response:**
```json
{
  "year": "2025",
  "status": "initialized",
  "row_count": 150
}
```

### GET /pipeline/status/{year}
Get pipeline state.

**Response:**
```json
{
  "year": "2025",
  "initialized": true,
  "loaded": true,
  "row_count": 150,
  "tagging_complete": false,
  "clustering_complete": false
}
```

## Data & Statistics

### GET /data/{year}/statistics
Get computed statistics.

**Response:**
```json
{
  "total_responses": 150,
  "weighted_averages": {
    "Overall Average": 3.45,
    "Grammar Average": 3.52,
    "Middle Average": 3.41,
    "High Average": 3.38
  },
  "level_counts": {
    "Grammar": 80,
    "Middle": 50,
    "High": 45
  }
}
```

### GET /data/{year}/responses
Get paginated raw responses.

**Query Parameters:**
- `page` (int, default: 1)
- `per_page` (int, default: 50, max: 200)
- `level` (string, optional)
- `has_response` (boolean, optional)

### GET /data/{year}/free-responses-with-tags
Get free responses with tagging data for the tagging interface.

**Query Parameters:**
- `page`, `per_page` (pagination)
- `level` (filter by school level)
- `question_type` (praise/improvement)
- `min_stability`, `max_stability` (stability score range)
- `has_keyword_mismatch` (boolean)
- `include_dismissed` (boolean)

## Tagging

### POST /tagging/{year}/start
Start tagging job.

**Request Body:**
```json
{
  "model": "gpt-5.2-nano",
  "n_samples": 4,
  "threshold": 2
}
```

### GET /tagging/{year}/results
Get paginated tagging results.

**Query Parameters:**
- `page`, `per_page`
- `level`, `tag`, `question`

### GET /tagging/{year}/distribution
Get tag distribution.

**Response:**
```json
{
  "year": "2025",
  "total_tags": 450,
  "unique_tags": 15,
  "distribution": [
    {"tag": "Teachers", "count": 85, "percentage": 18.89},
    {"tag": "Curriculum", "count": 72, "percentage": 16.00}
  ]
}
```

### PUT /tagging/{year}/responses/{response_id}/tags
Toggle a single tag (inline editing).

**Request Body:**
```json
{
  "tag": "Teachers",
  "value": true
}
```

### POST /tagging/{year}/batch
Start batch tagging via OpenAI Batch API.

**Request Body:**
```json
{
  "retag_existing": false
}
```

## Tags

### GET /tags/taxonomy
Get taxonomy from taxonomy.md.

**Response:**
```json
{
  "tags": [
    {"name": "Teachers", "keywords": ["teacher", "faculty", "staff"]},
    {"name": "Curriculum", "keywords": ["curriculum", "education", "learning"]}
  ]
}
```

### PUT /tags/{year}/responses/{id}
Override tags for a response.

## Clustering

### POST /clustering/{year}/start
Start clustering job.

### GET /clustering/{year}/coordinates
Get UMAP 2D coordinates for visualization.

**Query Parameters:**
- `question` (optional)
- `include_tags` (boolean)

### GET /clustering/{year}/summaries
Get cluster metadata.

## Jobs

### GET /jobs/{job_id}
Get job status.

**Response:**
```json
{
  "id": "uuid",
  "job_type": "tagging",
  "year": "2025",
  "status": "running",
  "progress": 45.5,
  "total_items": 200,
  "processed_items": 91
}
```

### POST /jobs/{job_id}/cancel
Cancel a running job.

## WebSocket

### /jobs/ws/{job_id}
Subscribe to real-time job updates.

**Message Format:**
```json
{
  "id": "uuid",
  "status": "running",
  "progress": 50.0,
  "processed_items": 100,
  "total_items": 200
}
```
