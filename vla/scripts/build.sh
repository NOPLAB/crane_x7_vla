#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Build script for VLA training Docker images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$VLA_DIR"

# Check if .env exists, if not copy from template
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.template .env
    echo "Please edit .env file with your configuration before building"
    echo ""
fi

# Parse arguments
TARGET="${1:-base}"

case "$TARGET" in
    base|train)
        echo "============================================"
        echo "Building VLA training image (base target)"
        echo "============================================"
        docker compose build vla-train
        echo ""
        echo "✓ Training image built successfully!"
        echo ""
        echo "Usage:"
        echo "  Interactive mode: ./scripts/run.sh train"
        echo "  OpenVLA training: docker compose --profile openvla up"
        echo "  OpenPI training:  docker compose --profile openpi up"
        ;;
    dev)
        echo "============================================"
        echo "Building VLA development image (dev target)"
        echo "============================================"
        docker compose build vla-dev
        echo ""
        echo "✓ Development image built successfully!"
        echo ""
        echo "Usage:"
        echo "  Start dev container: ./scripts/run.sh dev"
        echo "  Access container:    docker exec -it crane_x7_vla_dev bash"
        echo "  Start Jupyter:       jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
        echo "  Start TensorBoard:   tensorboard --logdir=/workspace/logs --host=0.0.0.0"
        ;;
    all)
        echo "============================================"
        echo "Building all VLA images"
        echo "============================================"
        docker compose build vla-train vla-dev
        echo ""
        echo "✓ All images built successfully!"
        ;;
    *)
        echo "Error: Unknown target '$TARGET'"
        echo ""
        echo "Usage: $0 [target]"
        echo ""
        echo "Available targets:"
        echo "  base, train  - Build training image (default)"
        echo "  dev          - Build development image"
        echo "  all          - Build all images"
        exit 1
        ;;
esac
