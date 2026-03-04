# Architecture

## System Overview

GVCA SAC is a full-stack web application for analyzing school satisfaction surveys. It uses LLM-based tagging, clustering, and statistical analysis to process parent feedback.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    FastAPI      │────▶│   PocketBase    │
│  React + Vite   │     │    Backend      │     │   (SQLite)      │
│  localhost:5173 │     │  localhost:8000 │     │  localhost:8090 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │
         │                      ▼
         │              ┌─────────────────┐
         └─────────────▶│   Vite Proxy    │
                        └─────────────────┘
```

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.13+)
- **Data Processing**: Polars
- **Validation**: Pydantic
- **AI/ML**: OpenAI API, sentence-transformers, UMAP, HDBSCAN

### Frontend
- **Framework**: React 18+ with TypeScript
- **State**: Zustand, React Query
- **UI**: shadcn/ui, Tailwind CSS
- **Charts**: Recharts, D3

### Database
- **PocketBase**: SQLite-backed with REST API

## Data Flow

### Survey Import
1. CSV uploaded via `/api/data/import/{year}`
2. Transform layer normalizes data
3. Survey responses stored in PocketBase
4. Free-text responses extracted to separate collection

### Tagging Pipeline
1. Free responses retrieved from PocketBase
2. Each response sent to OpenAI (4 samples for stability)
3. Tags aggregated with majority voting
4. Results stored with stability scores

### Clustering Pipeline
1. Responses embedded via sentence-transformers
2. UMAP reduces to 2D
3. HDBSCAN identifies clusters
4. Coordinates and summaries stored

## Directory Structure

```
gvca_sac/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI entry point
│       ├── config.py            # Settings
│       ├── constants.py         # Application constants
│       ├── routers/             # API endpoints
│       ├── services/            # Business logic services
│       └── core/                # Core algorithms
├── frontend/
│   └── src/
│       ├── api/                 # API client
│       ├── components/          # React components
│       ├── constants/           # Frontend constants
│       ├── pages/               # Page components
│       └── stores/              # Zustand stores
├── pocketbase/
│   └── pb_migrations/           # Database schema
└── docs/                        # This documentation
```

## Key Design Decisions

1. **Polars over Pandas**: Better performance, explicit typing, no index confusion
2. **PocketBase over PostgreSQL**: Zero-config SQLite, built-in REST API
3. **Vite Proxy**: Avoids CORS issues, cleaner URLs
4. **Background Jobs in-process**: Simple asyncio queue vs Redis/Celery complexity
5. **No Auth**: Single-user local development tool
