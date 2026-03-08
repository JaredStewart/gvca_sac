#!/bin/bash

### Script that runs every time the container starts (postStartCommand)
### Starts FastAPI backend and Vite frontend as background processes
### PocketBase runs as a Docker Compose sidecar service

echo "Running Post Start Command Script: post_start_hook.sh"

export PATH="$HOME/.local/bin:$PATH"
WORKSPACE="/workspaces/gvca_sac"

# Ensure directories exist
mkdir -p "${WORKSPACE}/data" "${WORKSPACE}/artifacts"

# Source .env if it exists
if [ -f "${WORKSPACE}/.env" ]; then
    set -a
    source "${WORKSPACE}/.env"
    set +a
fi

# Forward PocketBase sidecar port to localhost for Codespaces port forwarding
# Docker Compose sidecars are accessible by service name, not localhost
if ! command -v socat &>/dev/null; then
    echo "Installing socat..."
    sudo apt-get update && sudo apt-get install -y socat
fi
echo "Starting socat port forward for PocketBase..."
socat TCP4-LISTEN:8090,reuseaddr,fork TCP4:pocketbase:8090 &
echo "PocketBase port forward started (localhost:8090 → pocketbase:8090)"

# Wait for PocketBase sidecar (started by Docker Compose)
echo "Waiting for PocketBase..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8090/api/health > /dev/null 2>&1; then
        echo "PocketBase is healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNING: PocketBase health check timed out after 30s"
    fi
    sleep 1
done

# --- Start FastAPI Backend ---
echo "Starting FastAPI backend..."
cd "${WORKSPACE}/backend"
nohup uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > /tmp/backend.log 2>&1 &
echo "Backend started (PID: $!)"

# --- Start Vite Frontend ---
echo "Starting Vite frontend..."
cd "${WORKSPACE}/frontend"
nohup npm run dev -- --host 0.0.0.0 \
    > /tmp/frontend.log 2>&1 &
echo "Frontend started (PID: $!)"

# --- Summary ---
echo ""
echo "=== Services ==="
echo "PocketBase : http://localhost:8090  (Docker Compose sidecar)"
echo "Backend    : http://localhost:8000  Log: /tmp/backend.log"
echo "Frontend   : http://localhost:5173  Log: /tmp/frontend.log"
echo "================"

echo "Post Start Command Script Complete"
