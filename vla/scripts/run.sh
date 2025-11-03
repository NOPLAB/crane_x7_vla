#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
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
    echo ""
fi

# Parse arguments
MODE="${1:-train}"
shift || true  # Remove first argument, ignore error if no arguments

# Display usage information
usage() {
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  train          - Start interactive training container (default)"
    echo "  dev            - Start development container with Jupyter and TensorBoard"
    echo "  openvla        - Run OpenVLA fine-tuning"
    echo "  openpi         - Run OpenPI fine-tuning"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run.sh train              # Interactive training mode"
    echo "  ./scripts/run.sh dev                # Development mode (detached)"
    echo "  ./scripts/run.sh openvla            # Run OpenVLA training"
    echo "  ./scripts/run.sh openpi             # Run OpenPI training"
    echo ""
}

case "$MODE" in
    train)
        echo "============================================"
        echo "Starting VLA training container"
        echo "============================================"
        echo ""
        docker compose --profile train run --rm vla-train bash "$@"
        ;;
    dev)
        echo "============================================"
        echo "Starting VLA development container"
        echo "============================================"
        echo ""
        docker compose --profile dev up -d vla-dev
        echo ""
        echo "✓ Development container started!"
        echo ""
        echo "Useful commands:"
        echo "  Access shell:        docker exec -it crane_x7_vla_dev bash"
        echo "  View logs:           docker compose logs -f vla-dev"
        echo "  Stop container:      docker compose --profile dev down"
        echo ""
        echo "Development tools:"
        echo "  Jupyter Lab:         http://localhost:8888"
        echo "  TensorBoard:         http://localhost:6006"
        echo ""
        echo "To start Jupyter Lab (inside container):"
        echo "  docker exec -it crane_x7_vla_dev jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
        echo ""
        echo "To start TensorBoard (inside container):"
        echo "  docker exec -it crane_x7_vla_dev tensorboard --logdir=/workspace/logs --host=0.0.0.0"
        ;;
    openvla)
        echo "============================================"
        echo "Starting OpenVLA fine-tuning"
        echo "============================================"
        echo ""
        docker compose --profile openvla up vla-finetune-openvla
        ;;
    openpi)
        echo "============================================"
        echo "Starting OpenPI fine-tuning"
        echo "============================================"
        echo ""
        docker compose --profile openpi up vla-finetune-openpi
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo ""
        usage
        exit 1
        ;;
esac
