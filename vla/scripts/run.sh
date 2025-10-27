#!/bin/bash
# Run script for VLA training container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$VLA_DIR"

# Check if .env exists, if not copy from template
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.template .env
    echo "Please edit .env file with your API keys and configuration"
fi

# Parse arguments
MODE="${1:-train}"

if [ "$MODE" = "dev" ]; then
    echo "Starting VLA development container..."
    docker compose --profile dev up -d vla-dev
    echo ""
    echo "Development container started!"
    echo "Access the container: docker exec -it crane_x7_vla_dev bash"
    echo "Start Jupyter: jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
    echo "Start TensorBoard: tensorboard --logdir=/workspace/logs --host=0.0.0.0"
else
    echo "Starting VLA training container..."
    docker compose run --rm vla-train bash
fi
