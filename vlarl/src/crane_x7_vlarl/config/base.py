# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Base configuration for VLA-RL training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crane_x7_vlarl.config.ppo_config import PPOConfig
from crane_x7_vlarl.config.rollout_config import RolloutConfig


@dataclass
class VLARLConfig:
    """Unified configuration for VLA-RL training."""

    # Basic settings
    experiment_name: str = "crane_x7_vlarl"
    """Name of the experiment."""

    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    """Directory for saving checkpoints and logs."""

    seed: int = 42
    """Random seed for reproducibility."""

    # Model settings
    pretrained_checkpoint: str = "openvla/openvla-7b"
    """Path or HuggingFace model ID for pretrained VLA model."""

    sft_checkpoint: str | None = None
    """Path to SFT checkpoint from vla/ training (optional)."""

    use_lora: bool = True
    """Whether to use LoRA for fine-tuning."""

    lora_rank: int = 32
    """LoRA rank."""

    lora_alpha: int = 16
    """LoRA alpha."""

    lora_dropout: float = 0.05
    """LoRA dropout."""

    # PPO configuration
    ppo: PPOConfig = field(default_factory=PPOConfig)
    """PPO algorithm configuration."""

    # Rollout configuration
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    """Environment rollout configuration."""

    # Training settings
    num_updates: int = 1000
    """Total number of PPO updates."""

    total_timesteps: int = 100000
    """Total timesteps for training."""

    save_interval: int = 100
    """Interval for saving checkpoints (in updates)."""

    eval_interval: int = 50
    """Interval for evaluation (in updates)."""

    num_eval_episodes: int = 10
    """Number of episodes for evaluation."""

    eval_episodes: int = 10
    """Number of episodes for evaluation (alias)."""

    language_instruction: str = "pick up the object and place it"
    """Language instruction for the task."""

    # Logging
    use_wandb: bool = True
    """Whether to use Weights & Biases for logging."""

    wandb_project: str = "crane_x7_vlarl"
    """W&B project name."""

    wandb_entity: str | None = None
    """W&B entity (team/username)."""

    log_interval: int = 10
    """Interval for logging metrics (in updates)."""

    # Hardware
    device: str = "cuda"
    """Device for training (cuda, cpu)."""

    num_gpus: int = 1
    """Number of GPUs for training."""

    mixed_precision: str = "bf16"
    """Mixed precision mode (bf16, fp16, no)."""

    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing."""

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, (PPOConfig, RolloutConfig)):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VLARLConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        if "ppo" in data and isinstance(data["ppo"], dict):
            data["ppo"] = PPOConfig(**data["ppo"])
        if "rollout" in data and isinstance(data["rollout"], dict):
            data["rollout"] = RolloutConfig(**data["rollout"])

        return cls(**data)
