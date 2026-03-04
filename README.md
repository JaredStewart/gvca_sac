# GVCA SAC Survey Analysis

A full-stack web application for analyzing school satisfaction surveys using LLM-based tagging, clustering, and statistical analysis.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.13+ with uv for local development
- (Optional) Node.js 18+ for frontend development

### Run with Docker

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Access Points
- **Frontend**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **PocketBase Admin**: http://localhost:8090/_/

### Load Survey Data

1. Open the frontend at http://localhost:5173
2. Select a year from the dropdown
3. Click "Load Data" to initialize
4. Navigate to the Tagging page to start analyzing responses

## Features

- **Survey Import**: Upload and normalize Survey Monkey CSV exports
- **LLM Tagging**: Automated categorization with stability scoring
- **Clustering**: UMAP + HDBSCAN for discovering response patterns
- **Statistical Analysis**: Demographic segmentation and trend comparison
- **Export**: Generate CSV exports and visualization artifacts

## Development

```bash
# Backend development
cd backend
uv sync
uv run uvicorn app.main:app --reload

# Frontend development
cd frontend
npm install
npm run dev

# Run tests
cd backend && uv run pytest
cd frontend && npm test
```

## Documentation

- [Architecture](docs/architecture.md) - System design and data flow
- [API Reference](docs/api-reference.md) - REST API endpoints
- [Testing](docs/testing.md) - Test setup and patterns
- [Developer Guide](CLAUDE.md) - Detailed development reference

## Project Structure

```
gvca_sac/
├── backend/          # FastAPI + Polars backend
├── frontend/         # React + TypeScript frontend
├── pocketbase/       # Database migrations
├── data/             # Survey CSV files
├── docs/             # Documentation
└── specs/            # Feature specifications
```

## Environment Variables

Create a `.env` file (see `.env.example`):

```bash
OPENAI_API_KEY=sk-...  # Required for LLM tagging
```
