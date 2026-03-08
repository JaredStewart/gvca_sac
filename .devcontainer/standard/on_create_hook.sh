#!/bin/bash

### Script to install base infrastructure on container creation (onCreateCommand)
### Runs once when the container is first created

echo "Running On Create Hook Script: on_create_hook.sh"

set -e

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Python 3.13 via uv
echo "Installing Python 3.13..."
uv python install 3.13

# Install build dependencies for native Python extensions (numba, hdbscan, etc.)
echo "Installing build-essential..."
sudo apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && sudo apt-get -y install --no-install-recommends build-essential socat \
    && sudo apt-get autoremove -y && sudo apt-get clean -y && sudo rm -rf /var/lib/apt/lists/*

# Create required directories
WORKSPACE="/workspaces/gvca_sac"
mkdir -p "${WORKSPACE}/data" "${WORKSPACE}/artifacts" "${WORKSPACE}/pocketbase/pb_data"

echo "On Create Hook Script Complete"
