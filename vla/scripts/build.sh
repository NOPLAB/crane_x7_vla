#!/bin/bash
# Build script for VLA training Docker images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$VLA_DIR"

# Parse arguments
TARGET="${1:-base}"

echo "Building VLA Docker image with target: $TARGET"

if [ "$TARGET" = "dev" ]; then
    docker compose build vla-dev
    echo "Development image built successfully!"
    echo "Run with: ./scripts/run.sh dev"
else
    docker compose build vla-train
    echo "Training image built successfully!"
    echo "Run with: ./scripts/run.sh"
fi
