# CRANE-X7 VLA Training Framework

A unified framework for training Vision-Language-Action (VLA) models on CRANE-X7 robot data. Supports multiple VLA backends with automatic data format conversion.

## Features

- **Multiple VLA Backends**: OpenVLA and OpenPI support
- **Unified Configuration**: Single configuration format for all backends
- **Automatic Data Conversion**: TFRecord → LeRobot format conversion on-the-fly
- **Multi-Camera Support**: Handle multiple camera views with synchronization
- **Flexible Training**: LoRA, gradient checkpointing, mixed precision, DDP

## Architecture

```
crane_x7_vla/
├── backends/          # VLA backend implementations
│   ├── base.py       # Abstract backend interface
│   ├── openvla.py    # OpenVLA backend
│   └── openpi.py     # OpenPI backend
├── config/           # Configuration system
│   ├── base.py       # Unified configuration
│   ├── openvla_config.py
│   └── openpi_config.py
├── data/             # Data loading and conversion
│   ├── adapters.py   # TFRecord data adapter
│   ├── converters.py # Format converters
│   └── camera_manager.py # Multi-camera management
├── transforms/       # Data transformations
│   ├── action_transforms.py  # Action padding, chunking, normalization
│   ├── image_transforms.py   # Image processing
│   └── state_transforms.py   # State normalization
└── training/         # Training interface
    ├── trainer.py    # Unified trainer
    └── cli.py        # Command-line interface
```

## Installation

```bash
# Install package in development mode
cd vla/src
pip install -e crane_x7_vla
```

## Quick Start

### 1. Generate Default Configuration

```bash
# For OpenVLA
python -m crane_x7_vla.training.cli config \
  --backend openvla \
  --output openvla_config.yaml \
  --data-root ./data/tfrecord_logs

# For OpenPI
python -m crane_x7_vla.training.cli config \
  --backend openpi \
  --output openpi_config.yaml \
  --data-root ./data/tfrecord_logs
```

### 2. Train a Model

```bash
# Train with OpenVLA
python -m crane_x7_vla.training.cli train \
  --config openvla_config.yaml

# Train with OpenPI
python -m crane_x7_vla.training.cli train \
  --config openpi_config.yaml

# Or train with command-line arguments
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --data-root ./data/tfrecord_logs \
  --experiment-name my_experiment \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --num-epochs 100
```

### 3. Evaluate a Model

```bash
python -m crane_x7_vla.training.cli evaluate \
  --config openvla_config.yaml \
  --checkpoint ./outputs/checkpoint-1000 \
  --test-data ./data/test
```

## Configuration

### YAML Configuration Example

```yaml
backend: openvla  # or 'openpi'
output_dir: ./outputs
experiment_name: crane_x7_vla
seed: 42

data:
  data_root: ./data/tfrecord_logs
  format: tfrecord
  train_split: 0.9
  val_split: 0.1
  shuffle: true
  num_workers: 4
  cameras:
    - name: primary
      topic: /camera/color/image_raw
      width: 640
      height: 480
      enabled: true

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 5e-4
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  mixed_precision: bf16
  gradient_checkpointing: false
  save_interval: 1000
  eval_interval: 500
  log_interval: 10

# Backend-specific configuration
backend_config:
  # OpenVLA specific
  model_id: openvla/openvla-7b
  use_lora: true
  lora_rank: 32
  lora_alpha: 16

  # OpenPI specific
  # model_type: pi0_fast
  # action_horizon: 50
  # action_dim: 32
```

## Python API Usage

```python
from crane_x7_vla.config.openvla_config import OpenVLAConfig, OpenVLASpecificConfig
from crane_x7_vla.config.base import DataConfig, TrainingConfig, CameraConfig
from crane_x7_vla.training.trainer import VLATrainer

# Create configuration
data_config = DataConfig(
    data_root="./data/tfrecord_logs",
    cameras=[
        CameraConfig(name="primary", topic="/camera/color/image_raw")
    ]
)

training_config = TrainingConfig(
    batch_size=16,
    num_epochs=100,
    learning_rate=5e-4
)

config = OpenVLAConfig(
    backend="openvla",
    data=data_config,
    training=training_config,
    experiment_name="my_experiment",
    openvla=OpenVLASpecificConfig(
        model_id="openvla/openvla-7b",
        use_lora=True,
        lora_rank=32
    )
)

# Create trainer and train
trainer = VLATrainer(config)
results = trainer.train()
```

## Backend Comparison

| Feature               | OpenVLA             | OpenPI                   |
| --------------------- | ------------------- | ------------------------ |
| **Action Prediction** | Single-step         | Action chunks (50 steps) |
| **Action Dimension**  | 8 (native)          | 32 (padded from 8)       |
| **Data Format**       | TFRecord (direct)   | LeRobot (converted)      |
| **Cameras**           | Single camera       | Multi-camera (3 views)   |
| **Base Model**        | Prismatic VLM       | PaliGemma + Gemma        |
| **Training**          | LoRA via PEFT       | LoRA or full fine-tuning |
| **Current Status**    | ✅ Fully implemented | ⚠️ Partially implemented  |

## Data Format

### Input (CRANE-X7 TFRecord)

```python
{
    'observation/state': [8] float32,      # Joint positions
    'observation/image': bytes,            # JPEG-encoded RGB
    'observation/depth': bytes,            # Depth (optional)
    'action': [8] float32,                 # Target actions
    'prompt': string,                      # Language instruction
}
```

### OpenVLA Format (used directly)

Same as input TFRecord format.

### OpenPI Format (converted automatically)

```python
{
    'observation/state': [32] float32,     # Padded state
    'observation/image': {                 # Multi-camera images
        'base_0_rgb': [224, 224, 3] uint8,
        'left_wrist_0_rgb': [224, 224, 3] uint8,
        'right_wrist_0_rgb': [224, 224, 3] uint8,
    },
    'actions': [50, 32] float32,           # Action chunks
    'prompt': string,
}
```

## Multi-Camera Support

To use multiple cameras, configure them in your YAML:

```yaml
data:
  cameras:
    - name: primary
      topic: /camera/color/image_raw
      width: 640
      height: 480
      enabled: true
    - name: wrist_left
      topic: /camera_wrist_left/color/image_raw
      width: 640
      height: 480
      enabled: true
    - name: wrist_right
      topic: /camera_wrist_right/color/image_raw
      width: 640
      height: 480
      enabled: false  # Optional camera
```

## Advanced Features

### Action Transformations

- **Padding**: Automatically pad 8-dim actions to 32-dim for OpenPI
- **Chunking**: Convert single actions to 50-step action horizons
- **Normalization**: Quantile or z-score normalization

### Image Processing

- **Resizing**: Automatic resize to model's expected size
- **Normalization**: ImageNet-style normalization
- **Multi-camera**: Handle multiple camera views with padding for missing cameras

### Training Options

- **LoRA**: Parameter-efficient fine-tuning
- **Mixed Precision**: BF16/FP16 for faster training
- **Gradient Checkpointing**: Reduce memory usage
- **DDP**: Multi-GPU distributed training

## Troubleshooting

### OpenPI Backend Not Working

OpenPI integration is partially implemented. For full OpenPI support:

1. The data conversion pipeline is complete
2. Training loop integration is pending
3. Use OpenPI's native training script for now, or contribute to complete the integration

### CUDA Out of Memory

- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Reduce `lora_rank`
- Use `gradient_accumulation_steps` > 1

### Data Loading Slow

- Increase `num_workers`
- Enable dataset caching (for LeRobot conversion)
- Use SSD for data storage

## Contributing

Contributions are welcome! Priority areas:

1. Complete OpenPI training loop integration
2. Add evaluation metrics
3. Implement inference optimization
4. Add more VLA backends

## License

MIT License - Copyright (c) 2025 nop

## References

- OpenVLA: https://openvla.github.io/
- OpenPI: https://github.com/Physical-Intelligence/openpi
- CRANE-X7: https://github.com/rt-net/crane_x7
