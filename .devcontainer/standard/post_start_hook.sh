#!/bin/bash

### Script that runs every time the container starts (postStartCommand)
### Starts PocketBase, FastAPI backend, and Vite frontend as background processes

echo "Running Post Start Command Script: post_start_hook.sh"

export PATH="$HOME/.local/bin:$PATH"
WORKSPACE="/workspaces/gvca_sac"

# Ensure directories exist
mkdir -p "${WORKSPACE}/data" "${WORKSPACE}/artifacts" "${WORKSPACE}/pocketbase/pb_data"

# --- Start PocketBase ---
echo "Starting PocketBase..."
if command -v pocketbase >/dev/null 2>&1; then
    nohup pocketbase serve \
        --http=0.0.0.0:8090 \
        --dir="${WORKSPACE}/pocketbase/pb_data" \
        --migrationsDir="${WORKSPACE}/pocketbase/pb_migrations" \
        > /tmp/pocketbase.log 2>&1 &
    echo $! > /tmp/pocketbase.pid
    echo "PocketBase started (PID: $(cat /tmp/pocketbase.pid))"

    # Wait for PocketBase to be healthy
    echo "Waiting for PocketBase health check..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8090/api/health > /dev/null 2>&1; then
            echo "PocketBase is healthy"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "WARNING: PocketBase health check timed out after 30s"
            echo "Check logs: cat /tmp/pocketbase.log"
        fi
        sleep 1
    done
else
    echo "WARNING: pocketbase binary not found. Skipping PocketBase startup."
fi

# --- Start FastAPI Backend ---
echo "Starting FastAPI backend..."
cd "${WORKSPACE}/backend"
nohup uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > /tmp/backend.log 2>&1 &
echo $! > /tmp/backend.pid
echo "Backend started (PID: $(cat /tmp/backend.pid))"

# --- Start Vite Frontend ---
echo "Starting Vite frontend..."
cd "${WORKSPACE}/frontend"
nohup npm run dev \
    > /tmp/frontend.log 2>&1 &
echo $! > /tmp/frontend.pid
echo "Frontend started (PID: $(cat /tmp/frontend.pid))"

# --- Summary ---
echo ""
echo "=== Services Started ==="
echo "PocketBase : http://localhost:8090  (PID: $(cat /tmp/pocketbase.pid 2>/dev/null || echo 'N/A'))  Log: /tmp/pocketbase.log"
echo "Backend    : http://localhost:8000  (PID: $(cat /tmp/backend.pid 2>/dev/null || echo 'N/A'))  Log: /tmp/backend.log"
echo "Frontend   : http://localhost:5173  (PID: $(cat /tmp/frontend.pid 2>/dev/null || echo 'N/A'))  Log: /tmp/frontend.log"
echo "========================"

echo "Post Start Command Script Complete"
