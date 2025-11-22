# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenVLA backend implementation.

Wraps the existing OpenVLA fine-tuning code with the unified VLA backend interface.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from PIL import Image

# Add parent directory to import existing OpenVLA code
vla_src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(vla_src_path))

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openvla_config import OpenVLAConfig
from crane_x7_vla.backends.finetune_wrapper import CraneX7Trainer, CraneX7FinetuneConfig
from transformers import AutoModelForVision2Seq, AutoProcessor


class OpenVLABackend(VLABackend):
    """
    OpenVLA backend implementation.

    Wraps the existing OpenVLA fine-tuning pipeline with the unified interface.
    """

    def __init__(self, config: OpenVLAConfig):
        """
        Initialize OpenVLA backend.

        Args:
            config: OpenVLA configuration
        """
        super().__init__(config)
        self.vla_config = config
        self.trainer = None
        self._action_dim = config.action_dim
        self._action_horizon = config.action_horizon
        self._image_size = config.openvla.image_size

    def _create_finetune_config(self) -> CraneX7FinetuneConfig:
        """
        Convert UnifiedVLAConfig to CraneX7FinetuneConfig.

        Returns:
            CraneX7FinetuneConfig instance
        """
        ft_config = CraneX7FinetuneConfig()

        # Data settings
        ft_config.data_root = self.vla_config.data.data_root
        ft_config.instruction = "manipulate objects"  # Default instruction
        ft_config.image_size = self.vla_config.openvla.image_size
        ft_config.use_image = True

        # Training settings
        ft_config.batch_size = self.vla_config.training.batch_size
        ft_config.learning_rate = self.vla_config.training.learning_rate
        ft_config.weight_decay = self.vla_config.training.weight_decay
        ft_config.num_epochs = self.vla_config.training.num_epochs
        ft_config.warmup_steps = self.vla_config.training.warmup_steps
        ft_config.grad_accumulation_steps = self.vla_config.training.gradient_accumulation_steps
        ft_config.max_grad_norm = self.vla_config.training.max_grad_norm
        ft_config.mixed_precision = self.vla_config.training.mixed_precision
        ft_config.gradient_checkpointing = self.vla_config.training.gradient_checkpointing

        # Model settings
        ft_config.vla_path = self.vla_config.openvla.model_id
        ft_config.use_lora = self.vla_config.openvla.use_lora
        ft_config.lora_rank = self.vla_config.openvla.lora_rank
        ft_config.lora_alpha = self.vla_config.openvla.lora_alpha
        ft_config.lora_dropout = self.vla_config.openvla.lora_dropout
        ft_config.lora_target_modules = self.vla_config.openvla.lora_target_modules
        ft_config.use_flash_attention = self.vla_config.openvla.use_flash_attention

        # Logging settings
        ft_config.output_dir = self.vla_config.output_dir
        ft_config.logging_steps = self.vla_config.training.log_interval
        ft_config.save_steps = self.vla_config.training.save_interval

        # Data loading
        ft_config.num_workers = self.vla_config.data.num_workers
        ft_config.shuffle = self.vla_config.data.shuffle

        return ft_config

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            Dictionary containing training metrics and results
        """
        # Create fine-tune config from unified config
        ft_config = self._create_finetune_config()

        # Create trainer
        self.trainer = CraneX7Trainer(ft_config)

        # Run training
        self.trainer.train()

        # Return training results
        results = {
            'final_step': self.trainer.global_step,
            'final_epoch': self.trainer.epoch,
            'output_dir': str(ft_config.output_dir),
        }

        return results

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load model if checkpoint provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first or train a model.")

        # TODO: Implement evaluation logic
        # This would involve:
        # 1. Loading test dataset
        # 2. Running inference on test set
        # 3. Computing metrics (e.g., action prediction error)

        raise NotImplementedError("Evaluation not yet implemented for OpenVLA backend")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Dictionary containing:
                - 'state': Robot state [8]
                - 'image': RGB image [H, W, 3]
            language_instruction: Task instruction

        Returns:
            Predicted action as numpy array [8]
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # Prepare instruction
        if language_instruction is None:
            language_instruction = "manipulate objects"

        prompt = f"In: What action should the robot take to {language_instruction}?\nOut:"

        # Convert image to PIL
        image = observation['image']
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Process inputs
        inputs = self.processor([prompt], [image])

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self._action_dim)

        # Decode action
        # OpenVLA outputs tokenized actions that need to be decoded
        # This is a simplified version - actual decoding depends on how actions are tokenized
        action = outputs[0, -self._action_dim:].cpu().numpy()

        return action

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        if self.vla_config.openvla.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(path)
        else:
            # Save full model
            self.model.save_pretrained(path)

        # Save processor
        if self.processor is not None:
            self.processor.save_pretrained(path)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")

        print(f"Loading model from {path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True
        )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.vla_config.training.mixed_precision == "bf16" else torch.float32,
            "low_cpu_mem_usage": True,
        }

        if self.vla_config.openvla.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            **model_kwargs
        )

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully")

    @property
    def action_dim(self) -> int:
        """Get the action dimension of the model."""
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        """Get the action horizon (OpenVLA predicts single-step actions)."""
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        """Get the expected image size for the model."""
        return self._image_size
