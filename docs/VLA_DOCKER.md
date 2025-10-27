# VLA Training Docker Environment

This directory contains Docker configurations for training OpenVLA models with CRANE-X7 datasets.

## Prerequisites

- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Initial Setup

Copy the environment template and configure your API keys:

```bash
cd vla
cp .env.template .env
# Edit .env with your favorite editor
nano .env  # or vim .env
```

Required configurations:
- `WANDB_API_KEY`: Your Weights & Biases API key (optional, for experiment tracking)
- `HF_TOKEN`: Your HuggingFace token (required for downloading pretrained models)
- `DATA_DIR`: Path to your TFRecord datasets (default: `../data/tfrecord_logs`)

### 2. Build Docker Image

Build the training image:

```bash
./scripts/build.sh
```

Or build the development image (includes Jupyter, debugging tools):

```bash
./scripts/build.sh dev
```

### 3. Run Training

#### Option A: Interactive Training Session

Start an interactive container:

```bash
./scripts/run.sh
```

Inside the container, run training:

```bash
# Using the wrapper script
./scripts/train.sh

# Or run Python directly with custom parameters
python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 2e-5
```

#### Option B: Direct Training (One Command)

```bash
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints
```

### 4. Development Mode

For interactive development with Jupyter and TensorBoard:

```bash
# Start development container
./scripts/run.sh dev

# Access the container
docker exec -it crane_x7_vla_dev bash

# Start Jupyter Lab (inside container)
jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Or start TensorBoard (inside container)
tensorboard --logdir=/workspace/logs --host=0.0.0.0
```

Access points:
- Jupyter Lab: http://localhost:8888
- TensorBoard: http://localhost:6006

## Directory Structure

```
vla/
├── Dockerfile              # Multi-stage build (base + dev)
├── docker-compose.yml      # Container orchestration
├── .env.template           # Environment variable template
├── .env                    # Your local config (git-ignored)
├── scripts/
│   ├── build.sh           # Build Docker images
│   ├── run.sh             # Run containers
│   └── train.sh           # Training wrapper script
├── data/                  # Mounted dataset directory
├── checkpoints/           # Saved model checkpoints
└── logs/                  # Training logs
```

## Environment Variables

All environment variables can be set in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases API key | (empty) |
| `HF_TOKEN` | HuggingFace API token | (empty) |
| `DATA_DIR` | Dataset directory path | `../data/tfrecord_logs` |
| `CHECKPOINT_DIR` | Model checkpoint directory | `./checkpoints` |
| `LOG_DIR` | Training logs directory | `./logs` |
| `HF_CACHE` | HuggingFace cache directory | `~/.cache/huggingface` |
| `JUPYTER_PORT` | Jupyter port (dev mode) | `8888` |
| `TENSORBOARD_PORT` | TensorBoard port (dev mode) | `6006` |

## Training Parameters

Configure training via command-line arguments or environment variables:

```bash
# Via environment variables
export MODEL_NAME="openvla/openvla-7b"
export DATASET_NAME="crane_x7"
export BATCH_SIZE=8
export NUM_EPOCHS=10
export LEARNING_RATE=2e-5
./scripts/train.sh

# Via command-line arguments
python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --use_lora \
    --gradient_checkpointing
```

## Advanced Usage

### Multi-GPU Training

```bash
# Automatically uses all available GPUs
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --multi_gpu
```

### Custom Dataset Path

```bash
# Modify DATA_DIR in .env or use custom mount
docker compose run --rm \
    -v /path/to/my/data:/workspace/data \
    vla-train python3 finetune.py --data_dir /workspace/data
```

### Resume from Checkpoint

```bash
python3 finetune.py \
    --resume_from_checkpoint /workspace/checkpoints/checkpoint-1000
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config (/etc/docker/daemon.json)
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### Out of Memory

Reduce batch size or enable gradient checkpointing:

```bash
python3 finetune.py --batch_size 4 --gradient_checkpointing
```

### HuggingFace Authentication Failed

Ensure `HF_TOKEN` is set in `.env` and you have access to the model:

```bash
# Login manually inside container
huggingface-cli login
```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run with current user (add to docker-compose.yml)
user: "${UID}:${GID}"
```

## Monitoring Training

### Weights & Biases

Training automatically logs to W&B if `WANDB_API_KEY` is set:

```bash
# View your runs at https://wandb.ai/<your-username>/<project-name>
```

### TensorBoard

```bash
# Start TensorBoard in container
tensorboard --logdir=/workspace/logs --host=0.0.0.0

# Access at http://localhost:6006
```

### GPU Utilization

```bash
# Inside container
watch -n 1 nvidia-smi

# Or from host
docker exec crane_x7_vla_train watch -n 1 nvidia-smi
```

## Clean Up

```bash
# Stop all containers
docker compose down

# Remove volumes
docker compose down -v

# Remove images
docker rmi crane_x7_vla:latest crane_x7_vla:dev
```

## References

- [OpenVLA Documentation](https://github.com/openvla/openvla)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
