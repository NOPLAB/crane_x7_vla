"""
Wrapper around OpenVLA finetune.py to integrate with crane_x7_vla architecture.

This module provides adapter classes that bridge the crane_x7_vla backend interface
with the existing OpenVLA fine-tuning code.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# Add OpenVLA scripts to path
openvla_scripts_path = Path(__file__).parent.parent.parent / "openvla" / "vla-scripts"
if str(openvla_scripts_path) not in sys.path:
    sys.path.insert(0, str(openvla_scripts_path))


@dataclass
class CraneX7FinetuneConfig:
    """
    Configuration for CRANE-X7 VLA fine-tuning.

    This is an adapter class that maps crane_x7_vla configuration to OpenVLA's FinetuneConfig.
    """
    # fmt: off
    # Model settings
    vla_path: str = "openvla/openvla-7b"

    # Data settings
    data_root: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "crane_x7"
    instruction: str = "manipulate objects"
    image_size: tuple = (224, 224)
    use_image: bool = True

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    num_epochs: int = 100
    max_steps: int = 200_000
    warmup_steps: int = 0
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Training settings
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["all-linear"])
    use_quantization: bool = False

    # Flash Attention
    use_flash_attention: bool = False

    # Output settings
    output_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    logging_steps: int = 10
    save_steps: int = 5000
    save_latest_checkpoint_only: bool = True

    # Data loading
    num_workers: int = 0
    shuffle: bool = True

    # Tracking
    wandb_project: str = "crane-x7-vla"
    wandb_entity: Optional[str] = None
    run_id_note: Optional[str] = None
    # fmt: on


class CraneX7Trainer:
    """
    Wrapper around OpenVLA fine-tuning code for CRANE-X7.

    This class adapts the OpenVLA finetune() function to work with the
    crane_x7_vla backend architecture.
    """

    def __init__(self, config: CraneX7FinetuneConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: CraneX7FinetuneConfig instance
        """
        self.config = config
        self.global_step = 0
        self.epoch = 0

    def train(self) -> None:
        """
        Execute the training loop.

        This wraps the OpenVLA finetune() function with our configuration.
        """
        # Import the OpenVLA finetune module
        try:
            import sys
            import os

            # The finetune module should be in the path already from __init__
            # but let's ensure it
            openvla_scripts = Path(__file__).parent.parent.parent / "openvla" / "vla-scripts"
            if str(openvla_scripts) not in sys.path:
                sys.path.insert(0, str(openvla_scripts))

            # Import finetune module
            import finetune as finetune_module

        except ImportError as e:
            raise ImportError(
                f"Failed to import OpenVLA finetune module. "
                f"Make sure openvla/vla-scripts exists and has finetune.py. "
                f"Error: {e}"
            )

        # Convert CraneX7FinetuneConfig to OpenVLA FinetuneConfig
        openvla_config = self._convert_to_openvla_config()

        print(f"Starting OpenVLA fine-tuning")
        print(f"Dataset: {openvla_config.dataset_name}")
        print(f"Data root: {openvla_config.data_root_dir}")
        print(f"Output dir: {openvla_config.run_root_dir}")
        print(f"Batch size: {openvla_config.batch_size}")
        print(f"Learning rate: {openvla_config.learning_rate}")
        print(f"LoRA: {openvla_config.use_lora} (rank={openvla_config.lora_rank})")

        # Call the finetune function directly with config object
        # The finetune function is decorated with @draccus.wrap() but can accept
        # a config object directly
        try:
            finetune_module.finetune(openvla_config)
        except TypeError:
            # If direct call doesn't work, the function expects CLI parsing
            # In this case, we need to call it differently
            print("Warning: Direct config passing failed, using CLI argument emulation")

            # Build command-line style arguments
            args = [
                f"--vla_path={openvla_config.vla_path}",
                f"--data_root_dir={openvla_config.data_root_dir}",
                f"--dataset_name={openvla_config.dataset_name}",
                f"--run_root_dir={openvla_config.run_root_dir}",
                f"--batch_size={openvla_config.batch_size}",
                f"--learning_rate={openvla_config.learning_rate}",
                f"--max_steps={openvla_config.max_steps}",
                f"--save_steps={openvla_config.save_steps}",
            ]

            if openvla_config.use_lora:
                args.extend([
                    "--use_lora",
                    f"--lora_rank={openvla_config.lora_rank}",
                    f"--lora_dropout={openvla_config.lora_dropout}",
                ])

            # Temporarily replace sys.argv
            old_argv = sys.argv
            sys.argv = ["finetune.py"] + args

            try:
                finetune_module.finetune()
            finally:
                sys.argv = old_argv

        # Update tracking variables
        self.global_step = openvla_config.max_steps
        self.epoch = -1  # OpenVLA uses steps, not epochs

    def _convert_to_openvla_config(self):
        """
        Convert CraneX7FinetuneConfig to OpenVLA FinetuneConfig.

        Returns:
            FinetuneConfig instance for OpenVLA
        """
        from finetune import FinetuneConfig

        return FinetuneConfig(
            vla_path=self.config.vla_path,
            data_root_dir=self.config.data_root,
            dataset_name=self.config.dataset_name,
            run_root_dir=self.config.output_dir,
            adapter_tmp_dir=self.config.adapter_tmp_dir,
            batch_size=self.config.batch_size,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            learning_rate=self.config.learning_rate,
            grad_accumulation_steps=self.config.grad_accumulation_steps,
            image_aug=self.config.image_aug,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            save_latest_checkpoint_only=self.config.save_latest_checkpoint_only,
            use_lora=self.config.use_lora,
            lora_rank=self.config.lora_rank,
            lora_dropout=self.config.lora_dropout,
            use_quantization=self.config.use_quantization,
            wandb_project=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            run_id_note=self.config.run_id_note,
        )
