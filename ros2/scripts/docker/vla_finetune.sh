#!/bin/bash
# Script to run VLA fine-tuning inside Docker container

set -e

echo "=========================================="
echo "CRANE-X7 OpenVLA Fine-tuning"
echo "=========================================="

# Change to workspace directory
cd /workspace

# Parse arguments
MODE=${1:-"train"}

case "$MODE" in
  "train")
    echo "Starting fine-tuning..."
    cd vla
    python3 finetune.py "$@"
    ;;

  "train-multi-gpu")
    echo "Starting multi-GPU fine-tuning..."
    NUM_GPUS=${2:-2}
    cd vla
    torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" finetune.py "${@:3}"
    ;;

  "test-dataset")
    echo "Testing dataset loading..."
    cd vla
    python3 crane_x7_dataset.py ../data/tfrecord_logs
    ;;

  "bash")
    echo "Starting interactive bash shell..."
    exec /bin/bash
    ;;

  *)
    echo "Usage: $0 {train|train-multi-gpu|test-dataset|bash} [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 train                              # Single GPU training"
    echo "  $0 train-multi-gpu 2                  # Multi-GPU training with 2 GPUs"
    echo "  $0 test-dataset                       # Test dataset loading"
    echo "  $0 bash                               # Interactive bash shell"
    exit 1
    ;;
esac
