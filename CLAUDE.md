# Project Instructions

## Python Virtual Environment

When running Python/backend commands, always activate the venv first:
```bash
source /workspaces/gvca_sac/backend/.venv/bin/activate && <command>
```

## Package Management

Use `uv` to manage Python packages. Add dependencies to `backend/pyproject.toml` and install with:
```bash
cd /workspaces/gvca_sac/backend && uv pip install -e .
```
Do not use `pip install` directly.

## Deck Generation

See `docs/deck-generation.md` for the full workflow.

- Template: `data/templates/presentation.pptx` (gitignored local data file)
- Output: `artifacts/{year}/presentation_{year}.pptx`
- CLI: `gvca generate-deck <year> [--template PATH]`
- API: `POST /api/deck/generate`
