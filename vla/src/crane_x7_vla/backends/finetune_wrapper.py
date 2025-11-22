# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

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

            # Disable distributed training for single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # The finetune module should be in the path already from __init__
            # but let's ensure it
            openvla_scripts = Path(__file__).parent.parent.parent / "openvla" / "vla-scripts"
            if str(openvla_scripts) not in sys.path:
                sys.path.insert(0, str(openvla_scripts))

            # Import finetune module
            import finetune as finetune_module

            # Register CRANE-X7 dataset configuration with OpenVLA
            self._register_crane_x7_dataset()

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

        # Build command-line style arguments for finetune.py
        # The finetune function is decorated with @draccus.wrap() which parses sys.argv
        args = [
            f"--vla_path={openvla_config.vla_path}",
            f"--data_root_dir={openvla_config.data_root_dir}",
            f"--dataset_name={openvla_config.dataset_name}",
            f"--run_root_dir={openvla_config.run_root_dir}",
            f"--batch_size={openvla_config.batch_size}",
            f"--learning_rate={openvla_config.learning_rate}",
            f"--max_steps={openvla_config.max_steps}",
            f"--save_steps={openvla_config.save_steps}",
            f"--grad_accumulation_steps={openvla_config.grad_accumulation_steps}",
            f"--shuffle_buffer_size={openvla_config.shuffle_buffer_size}",
        ]

        # Add boolean flags with values
        args.append(f"--image_aug={str(openvla_config.image_aug).lower()}")
        args.append(f"--save_latest_checkpoint_only={str(openvla_config.save_latest_checkpoint_only).lower()}")
        args.append(f"--use_lora={str(openvla_config.use_lora).lower()}")
        args.append(f"--use_quantization={str(openvla_config.use_quantization).lower()}")

        # Add LoRA parameters
        if openvla_config.use_lora:
            args.extend([
                f"--lora_rank={openvla_config.lora_rank}",
                f"--lora_dropout={openvla_config.lora_dropout}",
            ])

        # Add wandb parameters if provided
        if openvla_config.wandb_project:
            args.append(f"--wandb_project={openvla_config.wandb_project}")
        if openvla_config.wandb_entity:
            args.append(f"--wandb_entity={openvla_config.wandb_entity}")
        if openvla_config.run_id_note:
            args.append(f"--run_id_note={openvla_config.run_id_note}")

        # Temporarily replace sys.argv to pass arguments to draccus-wrapped function
        old_argv = sys.argv
        sys.argv = ["finetune.py"] + args

        try:
            # Call the finetune function (draccus will parse sys.argv)
            finetune_module.finetune()
        finally:
            # Restore original sys.argv
            sys.argv = old_argv

        # Update tracking variables
        self.global_step = openvla_config.max_steps
        self.epoch = -1  # OpenVLA uses steps, not epochs

    def _register_crane_x7_dataset(self):
        """
        Register CRANE-X7 dataset configuration with OpenVLA.

        This dynamically adds crane_x7 dataset config to OXE_DATASET_CONFIGS
        and OXE_STANDARDIZATION_TRANSFORMS without modifying the OpenVLA source.
        """
        try:
            # Import OpenVLA dataset configuration modules
            from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding, ActionEncoding
            from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
            import tensorflow as tf
            import tensorflow_datasets as tfds

            # Register TFDS builder by importing and registering it
            # This makes the builder discoverable by tfds.builder()
            try:
                import sys
                # Add the data directory to path so we can import the builder
                tfds_builders_dir = Path(__file__).parent.parent / "data" / "tfds_builders"
                if str(tfds_builders_dir) not in sys.path:
                    sys.path.insert(0, str(tfds_builders_dir))

                # Import the builder class
                from crane_x7 import Crane_x7

                # Register the builder with TFDS internal registry
                # This allows tfds.builder('crane_x7') to find it
                if hasattr(tfds.core, 'registered'):
                    # Method 1: Add to builder modules (TFDS < 4.9)
                    if hasattr(tfds.core.registered, '_BUILDER_MODULES'):
                        tfds.core.registered._BUILDER_MODULES['crane_x7'] = 'crane_x7'

                    # Method 2: Add to registry (TFDS >= 4.9)
                    if hasattr(tfds.core.registered, '_DATASET_REGISTRY'):
                        tfds.core.registered._DATASET_REGISTRY['crane_x7'] = Crane_x7

                    # Method 3: Direct registration
                    if hasattr(tfds.core.registered, '_RegisteredBuilder'):
                        tfds.core.registered._RegisteredBuilder._all_builders['crane_x7'] = Crane_x7

                print(f"✓ Registered crane_x7 TFDS builder from {tfds_builders_dir}")

            except Exception as builder_error:
                print(f"Warning: Failed to register TFDS builder: {builder_error}")
                print(f"  Attempting fallback method...")

                # Fallback: Copy builder to data directory
                import shutil
                builder_src = Path(__file__).parent.parent / "data" / "tfds_builders" / "crane_x7.py"
                builder_dest_dir = self.config.data_root / "crane_x7"
                builder_dest_dir.mkdir(parents=True, exist_ok=True)
                builder_dest = builder_dest_dir / "crane_x7.py"
                shutil.copy2(builder_src, builder_dest)

                # Create __init__.py
                init_file = builder_dest_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text(
                        'from .crane_x7 import Crane_x7\n__all__ = ["Crane_x7"]\n'
                    )
                print(f"  ✓ Copied builder to {builder_dest_dir}")

            # Prepare the dataset by calling download_and_prepare()
            # This is required before TFDS can load the dataset
            try:
                print(f"Preparing CRANE-X7 dataset from {self.config.data_root}...")
                builder_instance = Crane_x7(data_dir=str(self.config.data_root))
                builder_instance.download_and_prepare()
                print("✓ Dataset prepared successfully")
            except Exception as prepare_error:
                print(f"Warning: Dataset preparation encountered an issue: {prepare_error}")
                print("  This may be expected if the dataset is already prepared.")

            # Define CRANE-X7 dataset configuration
            crane_x7_config = {
                "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
                "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
                "state_obs_keys": ["state"],
                "state_encoding": StateEncoding.JOINT,
                "action_encoding": ActionEncoding.JOINT_POS,
            }

            # Define CRANE-X7 standardization transform
            def crane_x7_dataset_transform(trajectory):
                """
                Transform CRANE-X7 dataset to match OpenVLA format.

                CRANE-X7 format:
                - observation/state: [8] joint positions (7 arm + 1 gripper)
                - observation/image: JPEG encoded RGB image
                - observation/depth: optional depth image
                - action: [8] next joint positions
                - prompt: language instruction
                """
                # Ensure all required keys exist
                if "observation" not in trajectory:
                    trajectory["observation"] = {}

                # Handle language instruction
                if "prompt" in trajectory:
                    trajectory["language_instruction"] = trajectory["prompt"]
                elif "task" in trajectory:
                    trajectory["language_instruction"] = trajectory["task"]
                else:
                    # Default instruction
                    trajectory["language_instruction"] = tf.constant("manipulate objects", dtype=tf.string)

                # Ensure state is in observation dict
                if "state" not in trajectory["observation"] and "observation/state" in trajectory:
                    trajectory["observation"]["state"] = trajectory["observation/state"]

                # Ensure image is in observation dict
                if "image" not in trajectory["observation"] and "observation/image" in trajectory:
                    trajectory["observation"]["image"] = trajectory["observation/image"]

                # Handle depth if present
                if "depth" not in trajectory["observation"] and "observation/depth" in trajectory:
                    trajectory["observation"]["depth"] = trajectory["observation/depth"]

                return trajectory

            # Register with OpenVLA (monkey patch)
            if "crane_x7" not in OXE_DATASET_CONFIGS:
                OXE_DATASET_CONFIGS["crane_x7"] = crane_x7_config
                print("✓ Registered crane_x7 dataset configuration")

            if "crane_x7" not in OXE_STANDARDIZATION_TRANSFORMS:
                OXE_STANDARDIZATION_TRANSFORMS["crane_x7"] = crane_x7_dataset_transform
                print("✓ Registered crane_x7 standardization transform")

        except Exception as e:
            print(f"Warning: Failed to register crane_x7 dataset: {e}")
            print("This may cause issues during training.")

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
