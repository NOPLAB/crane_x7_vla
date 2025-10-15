# VLA Fine-tuning with Docker

This guide explains how to use Docker for OpenVLA fine-tuning on CRANE-X7 data.

## Prerequisites

- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 12.4+ support
- At least 16GB GPU memory (recommended: 40GB+ for comfortable training)

## Quick Start

### 1. Build the VLA Docker Image

```bash
# Build the VLA fine-tuning image
docker compose build vla_finetune
```

This will create a separate Docker image with:
- CUDA 12.4
- PyTorch 2.2.0 with CUDA support
- OpenVLA dependencies
- Flash Attention 2 (if build succeeds)

### 2. Run Interactive Container

```bash
# Start interactive container
docker compose --profile vla run --rm vla_finetune

# Inside container, test dataset loading
python3 vla/crane_x7_dataset.py data/tfrecord_logs

# Run fine-tuning
cd vla
python3 finetune.py
```

### 3. Using the Helper Script

```bash
# Test dataset loading
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh test-dataset

# Single GPU training
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train

# Multi-GPU training (2 GPUs)
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train-multi-gpu 2
```

## Advanced Usage

### Custom Training Parameters

```bash
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && python3 finetune.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --lora_rank 64 \
    --use_wandb \
    --wandb_project my-crane-x7
"
```

### Multi-GPU Training

```bash
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py \
    --batch_size 8 \
    --learning_rate 5e-4
"
```

### Mount Additional Directories

Edit `docker-compose.yml` to add more volume mounts:

```yaml
volumes:
  - type: bind
    source: "./my_custom_data"
    target: "/workspace/custom_data"
```

## Directory Structure in Container

```
/workspace/
├── vla/                    # Fine-tuning scripts (mounted)
│   ├── finetune.py
│   ├── crane_x7_dataset.py
│   ├── finetune_config.py
│   └── README.md
├── data/                   # Training data (mounted)
│   └── tfrecord_logs/
│       ├── episode_0000_*/
│       ├── episode_0001_*/
│       └── ...
└── outputs/                # Checkpoints and logs (mounted)
    └── crane_x7_finetune/
```

## GPU Configuration

The docker-compose.yml is configured to use all available GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all          # Use all GPUs
          capabilities: [gpu]
```

To use specific GPUs, modify `count`:
- `count: 1` - Use 1 GPU
- `count: 2` - Use 2 GPUs
- `count: all` - Use all available GPUs

Or use `CUDA_VISIBLE_DEVICES` environment variable:

```bash
docker compose --profile vla run --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vla_finetune
```

## Shared Memory Configuration

The container is configured with 16GB shared memory (`shm_size: '16gb'`) for PyTorch DataLoader multiprocessing. If you encounter shared memory errors, increase this value in `docker-compose.yml`:

```yaml
shm_size: '32gb'  # Increase if needed
```

## Monitoring Training

### Using Weights & Biases

```bash
# Login to W&B (one-time setup)
docker compose --profile vla run --rm vla_finetune bash -c "
  pip3 install wandb && wandb login
"

# Run training with W&B logging
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && python3 finetune.py \
    --use_wandb \
    --wandb_project crane-x7-openvla \
    --wandb_entity YOUR_USERNAME
"
```

### Viewing Logs

Training logs are printed to stdout. To save logs to a file:

```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train 2>&1 | tee training.log
```

## Troubleshooting

### Out of Memory

1. Reduce batch size in `finetune_config.py` or via CLI:
   ```bash
   --batch_size 4
   ```

2. Enable gradient checkpointing:
   ```python
   gradient_checkpointing: bool = True
   ```

3. Reduce LoRA rank:
   ```bash
   --lora_rank 16
   ```

### Flash Attention Build Failed

Flash Attention 2 is optional. The container will continue without it if the build fails. Training will be slightly slower but still functional.

To manually install inside the container:

```bash
docker compose --profile vla run --rm vla_finetune bash
# Inside container:
pip3 install packaging ninja
pip3 install flash-attn==2.5.5 --no-build-isolation
```

### CUDA Version Mismatch

The Dockerfile uses CUDA 12.4. If you have a different CUDA version on your host, modify the base image in Dockerfile:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS vla  # Change to your CUDA version
```

Then rebuild:
```bash
docker compose build vla_finetune --no-cache
```

### Permission Issues

If you encounter permission errors with mounted volumes:

```bash
# Run container as current user
docker compose --profile vla run --rm --user $(id -u):$(id -g) vla_finetune
```

## Development Workflow

1. **Data Collection**: Use ROS2 containers to collect demonstration data
   ```bash
   docker compose --profile real up  # or --profile sim
   ```

2. **Fine-tuning**: Switch to VLA container for training
   ```bash
   docker compose --profile vla run --rm vla_finetune
   ```

3. **Inference**: Load fine-tuned model for robot control
   ```bash
   # (Implementation depends on your deployment setup)
   ```

## Resource Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with 16GB VRAM (e.g., V100, RTX 4090)
- RAM: 32GB system RAM
- Storage: 50GB free space

### Recommended Requirements
- GPU: NVIDIA A100 40GB/80GB
- RAM: 64GB+ system RAM
- Storage: 100GB+ free space (for datasets and checkpoints)

### Multi-GPU Scaling
- 1x A100 (40GB): Batch size ~8-12 with LoRA
- 2x A100 (40GB): Batch size ~16-24 with LoRA
- 4x A100 (40GB): Batch size ~32-48 with LoRA

## Next Steps

After fine-tuning completes:

1. Checkpoints are saved in `outputs/crane_x7_finetune/checkpoint-*/`
2. Load the fine-tuned model for inference (see `vla/README.md`)
3. Integrate with your robot control stack
