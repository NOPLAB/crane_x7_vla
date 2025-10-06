# CRANE-X7 OpenVLA Fine-tuning

This directory contains scripts for fine-tuning OpenVLA on CRANE-X7 robot demonstration data.

## Overview

The fine-tuning pipeline consists of:

1. **crane_x7_dataset.py**: PyTorch Dataset loader for CRANE-X7 TFRecord data
2. **finetune_config.py**: Configuration dataclass for fine-tuning parameters
3. **finetune.py**: Main training script with LoRA support

## Requirements

### Option 1: Using Docker (Recommended)

Use the pre-configured Docker environment with all dependencies:

```bash
# Build the VLA fine-tuning image
docker compose build vla_finetune

# Run interactive container
docker compose --profile vla run --rm vla_finetune

# Or use the helper script
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train
```

See [docker_usage.md](docker_usage.md) for detailed Docker instructions.

### Option 2: Local Installation

Install the required dependencies:

```bash
# Install all requirements
pip install -r requirements.txt

# Optional: Flash Attention 2 for faster training
pip install flash-attn==2.5.5 --no-build-isolation
```

## Data Format

The fine-tuning scripts expect data in the following structure:

```
data/tfrecord_logs/
├── episode_0000_TIMESTAMP/
│   └── episode_data.tfrecord
├── episode_0001_TIMESTAMP/
│   └── episode_data.tfrecord
└── ...
```

Each TFRecord file contains:
- `observation/state`: Joint positions (7-DOF float array)
- `observation/image`: RGB image (JPEG encoded bytes)
- `observation/timestamp`: Timestamp (float)
- `action`: Next state / target joint positions (7-DOF float array)

## Quick Start

### 1. Test Dataset Loading

Verify that your dataset can be loaded correctly:

```bash
cd vla
python crane_x7_dataset.py ../data/tfrecord_logs
```

### 2. Run Fine-tuning

#### Single GPU

```bash
cd vla
python finetune.py
```

#### Multi-GPU (PyTorch DDP)

```bash
cd vla
torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py
```

#### With Custom Parameters

```bash
python finetune.py \
  --data_root ../data/tfrecord_logs \
  --output_dir ../outputs/my_finetune \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_epochs 10 \
  --lora_rank 32
```

#### With Weights & Biases Logging

```bash
python finetune.py \
  --use_wandb \
  --wandb_project crane-x7-openvla \
  --wandb_entity your-username
```

## Configuration

Key configuration parameters (see `finetune_config.py` for full details):

### Model Configuration
- `vla_path`: HuggingFace model path (default: `"openvla/openvla-7b"`)
- `use_flash_attention`: Use Flash Attention 2 for faster training (default: `True`)

### Data Configuration
- `data_root`: Path to TFRecord data (default: `"data/tfrecord_logs"`)
- `instruction`: Task instruction for conditioning (default: `"Pick and place the object"`)
- `use_image`: Whether to use camera images (default: `True`)
- `image_size`: Target image size (default: `(224, 224)`)

### Training Configuration
- `batch_size`: Batch size per GPU (default: `8`)
- `num_epochs`: Number of training epochs (default: `10`)
- `learning_rate`: Learning rate (default: `5e-4`)
- `grad_accumulation_steps`: Gradient accumulation (default: `1`)

### LoRA Configuration
- `use_lora`: Use LoRA for parameter-efficient fine-tuning (default: `True`)
- `lora_rank`: LoRA rank (default: `32`)
- `lora_alpha`: LoRA alpha scaling factor (default: `64`)
- `lora_dropout`: LoRA dropout (default: `0.1`)

### Checkpointing
- `output_dir`: Output directory (default: `"outputs/crane_x7_finetune"`)
- `save_steps`: Save checkpoint interval (default: `500`)
- `save_total_limit`: Max checkpoints to keep (default: `3`)

## Memory Requirements

### LoRA Fine-tuning (Recommended)

With LoRA (rank=32), the memory requirements are:

- **Single GPU (A100 40GB)**: Batch size 8 - 12
- **Single GPU (A100 80GB)**: Batch size 16 - 24
- **Multi-GPU**: Scale batch size accordingly

### Full Fine-tuning

Full fine-tuning requires significantly more memory. For the 7B parameter model:

- **Single GPU (A100 80GB)**: Batch size 2 - 4 (with gradient checkpointing)
- **Multi-GPU (8x A100)**: Recommended for stable training

## Output

The training script saves:

1. **Checkpoints**: Saved to `output_dir/checkpoint-{step}/`
   - LoRA adapter weights (if using LoRA)
   - Full model weights (if not using LoRA)
   - Processor configuration

2. **Logs**: Training metrics logged to:
   - Console output
   - Weights & Biases (if enabled)

## Loading Fine-tuned Model

After fine-tuning, load your model for inference:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "outputs/crane_x7_finetune/checkpoint-5000"
)

# Load processor
processor = AutoProcessor.from_pretrained(
    "outputs/crane_x7_finetune/checkpoint-5000",
    trust_remote_code=True
)

# Inference
model.eval()
model.to("cuda")

# ... use model for inference ...
```

## Advanced Usage

### Custom Dataset Instruction

Modify the task instruction in `finetune_config.py`:

```python
instruction: str = "Grasp the red block and place it in the bin"
```

### Training Without Images

If your dataset doesn't have images (state-only):

```python
use_image: bool = False
```

### Gradient Checkpointing

To save memory at the cost of training speed:

```python
gradient_checkpointing: bool = True
```

### Resume from Checkpoint

```python
resume_from_checkpoint: Optional[str] = "outputs/crane_x7_finetune/checkpoint-5000"
```

## Troubleshooting

### Out of Memory

1. Reduce `batch_size`
2. Increase `grad_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing`
4. Reduce `lora_rank`

### Slow Training

1. Enable `use_flash_attention` (requires flash-attn installation)
2. Increase `num_workers` for data loading
3. Use multiple GPUs with PyTorch DDP

### Dataset Loading Issues

1. Verify TFRecord files are valid: `python crane_x7_dataset.py <data_root>`
2. Check TensorFlow is installed: `pip install tensorflow`
3. Ensure data structure matches expected format

## Reference

For more information on OpenVLA:
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA Models](https://huggingface.co/openvla)

For CRANE-X7 robot:
- [CRANE-X7 ROS 2](https://github.com/rt-net/crane_x7_ros)
- [CRANE-X7 Description](https://github.com/rt-net/crane_x7_description)
