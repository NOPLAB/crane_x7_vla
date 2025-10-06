#!/usr/bin/env python3
"""
Fine-tuning configuration for CRANE-X7 OpenVLA.

This module defines the configuration parameters for fine-tuning OpenVLA
on CRANE-X7 robot demonstration data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CraneX7FinetuneConfig:
    """Configuration for CRANE-X7 OpenVLA fine-tuning."""

    # === Model Configuration ===
    vla_path: str = "openvla/openvla-7b"  # HuggingFace model path
    use_flash_attention: bool = True       # Use Flash Attention 2 (faster, requires flash-attn)

    # === Data Configuration ===
    data_root: Path = Path("data/tfrecord_logs")  # Root directory with episode folders
    instruction: str = "Pick and place the object"  # Task instruction for conditioning
    use_image: bool = True                 # Whether to use camera images
    image_size: tuple = (224, 224)         # Target image size (H, W)

    # === Training Configuration ===
    batch_size: int = 8                    # Training batch size per GPU
    num_epochs: int = 10                   # Number of training epochs
    max_steps: Optional[int] = None        # Max training steps (overrides epochs if set)
    learning_rate: float = 5e-4            # Learning rate
    weight_decay: float = 0.01             # Weight decay for AdamW
    warmup_steps: int = 100                # Learning rate warmup steps
    grad_accumulation_steps: int = 1       # Gradient accumulation steps
    max_grad_norm: float = 1.0             # Gradient clipping norm

    # === LoRA Configuration ===
    use_lora: bool = True                  # Use LoRA for parameter-efficient fine-tuning
    lora_rank: int = 32                    # LoRA rank
    lora_alpha: int = 64                   # LoRA alpha (scaling factor)
    lora_dropout: float = 0.1              # LoRA dropout
    # Target all linear layers in the model for LoRA
    lora_target_modules: list = None       # Will be set to all linear layers if None

    # === Data Loading ===
    num_workers: int = 4                   # Number of data loading workers
    shuffle: bool = True                   # Shuffle training data
    pin_memory: bool = True                # Pin memory for faster GPU transfer

    # === Checkpoint & Logging ===
    output_dir: Path = Path("outputs/crane_x7_finetune")  # Output directory
    save_steps: int = 500                  # Save checkpoint every N steps
    eval_steps: int = 500                  # Evaluate every N steps
    logging_steps: int = 10                # Log metrics every N steps
    save_total_limit: int = 3              # Maximum number of checkpoints to keep

    # === Weights & Biases ===
    use_wandb: bool = False                # Use Weights & Biases for logging
    wandb_project: str = "crane-x7-openvla"  # W&B project name
    wandb_entity: Optional[str] = None     # W&B entity (username/org)
    wandb_run_name: Optional[str] = None   # W&B run name

    # === Device Configuration ===
    device: str = "cuda"                   # Device to use (cuda/cpu)
    mixed_precision: str = "bf16"          # Mixed precision training (bf16/fp16/no)
    seed: int = 42                         # Random seed

    # === Advanced ===
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    gradient_checkpointing: bool = False   # Use gradient checkpointing to save memory

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate data root exists
        if not self.data_root.exists():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")

        # Set default LoRA target modules (all linear layers)
        if self.use_lora and self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Validate image size
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2:
            raise ValueError("image_size must be a tuple/list of (height, width)")

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


def get_default_config() -> CraneX7FinetuneConfig:
    """Get default fine-tuning configuration."""
    return CraneX7FinetuneConfig()


def get_lora_config(config: CraneX7FinetuneConfig):
    """
    Create PEFT LoRA configuration.

    Args:
        config: Fine-tuning configuration

    Returns:
        LoraConfig for PEFT
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


if __name__ == '__main__':
    # Print default configuration
    config = get_default_config()
    print("Default CRANE-X7 Fine-tuning Configuration:")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key:30s}: {value}")
