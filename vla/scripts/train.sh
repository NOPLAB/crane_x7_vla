#!/bin/bash
# Training script wrapper for running inside Docker container

set -e

# Default parameters
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
DATASET_NAME="${DATASET_NAME:-crane_x7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

echo "Starting OpenVLA fine-tuning with CRANE-X7 dataset"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

python3 /workspace/finetune.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    "${@}"
