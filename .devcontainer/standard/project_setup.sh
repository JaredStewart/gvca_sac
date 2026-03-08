#!/bin/bash

### Script to install project dependencies (updateContentCommand)
### Runs after on_create_hook and whenever content changes

echo "Running Update Content Command Script: project_setup.sh"

set -e

export PATH="$HOME/.local/bin:$PATH"
WORKSPACE="/workspaces/gvca_sac"

# Install backend Python dependencies
echo "Installing backend dependencies..."
cd "${WORKSPACE}/backend"
uv sync

# Install root-level Python dependencies (notebooks/ML)
echo "Installing root-level dependencies..."
cd "${WORKSPACE}"
uv sync

# Install frontend Node dependencies
echo "Installing frontend dependencies..."
cd "${WORKSPACE}/frontend"
npm ci

# Verify installations
echo "Verifying installations..."
cd "${WORKSPACE}/backend"
uv run python -c "import fastapi; print(f'FastAPI {fastapi.__version__} OK')"
cd "${WORKSPACE}/frontend"
node -e "console.log('Node ' + process.version + ' OK')"
echo "PocketBase runs as Docker Compose sidecar"

echo "Update Content Command Script Complete"
