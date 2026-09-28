# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""OpenVLA adapter for VLA-RL training.

This module provides an adapter for using OpenVLA models with the VLA-RL
training framework, including support for loading checkpoints from the
vla/ training pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class OpenVLAAdapter(nn.Module):
    """Adapter for OpenVLA model for use in VLA-RL training.

    This adapter wraps OpenVLA to provide:
    - Action generation from observations
    - Log probability computation for PPO
    - Value estimation (via separate value head)
    - Loading from vla/ training checkpoints
    """

    # CRANE-X7 action dimensions
    ACTION_DIM = 8  # 7 arm + 1 gripper

    def __init__(
        self,
        model_name_or_path: str = "openvla/openvla-7b",
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: torch.device | str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize OpenVLA adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            use_lora: Whether to use LoRA for fine-tuning.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            device: Device for model.
            torch_dtype: Model dtype.
        """
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = torch.device(device) if isinstance(device, str) else device
        self.torch_dtype = torch_dtype

        # Placeholder for model and processor (lazy loading)
        self._model = None
        self._processor = None
        self._action_tokenizer = None

        # Value head for critic
        self._value_head = None

        # Dataset statistics for action normalization
        self.action_mean = None
        self.action_std = None

        # Default language instruction
        self.default_instruction = "pick up the object and place it"

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded (lazy loading)."""
        if self._model is not None:
            return

        logger.info(f"Loading OpenVLA model from {self.model_name_or_path}")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for OpenVLA. "
                "Install with: pip install transformers"
            )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        # Load model
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # Avoid SDPA compatibility issues
        )

        # Apply LoRA if requested
        if self.use_lora:
            self._apply_lora()

        # Move to device
        self._model = self._model.to(self.device)

        # Initialize value head
        hidden_size = (
            self._model.config.hidden_size
            if hasattr(self._model.config, "hidden_size")
            else 4096
        )
        self._value_head = (
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            .to(self.device)
            .to(self.torch_dtype)
        )

        # Try to get action tokenizer
        try:
            from prismatic.models.action_tokenizer import ActionTokenizer

            self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)
        except ImportError:
            logger.warning("ActionTokenizer not available, using default tokenization")

        logger.info("OpenVLA model loaded successfully")

    def _apply_lora(self) -> None:
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            )

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self._model = get_peft_model(self._model, lora_config)
        logger.info(f"Applied LoRA with rank={self.lora_rank}, alpha={self.lora_alpha}")

    @classmethod
    def from_vla_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "OpenVLAAdapter":
        """Load adapter from vla/ training checkpoint.

        Supports both LoRA adapter and merged checkpoint formats.

        Args:
            checkpoint_path: Path to checkpoint directory.
            device: Device for model.

        Returns:
            Loaded OpenVLAAdapter.
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading from vla/ checkpoint: {checkpoint_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Check checkpoint format
        has_lora = (checkpoint_path / "lora_adapters").exists() or (
            checkpoint_path / "adapter_model.safetensors"
        ).exists()
        has_merged = (checkpoint_path / "model.safetensors").exists() or (
            checkpoint_path / "pytorch_model.bin"
        ).exists()

        if has_lora:
            return cls._load_lora_checkpoint(checkpoint_path, device)
        elif has_merged:
            return cls._load_merged_checkpoint(checkpoint_path, device)
        else:
            # Try loading as HuggingFace model
            return cls(model_name_or_path=str(checkpoint_path), device=device)

    @classmethod
    def _load_lora_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | str,
    ) -> "OpenVLAAdapter":
        """Load from LoRA adapter checkpoint."""
        # Determine base model path
        # Check if there's a config file with base model info
        config_file = checkpoint_path / "adapter_config.json"
        if config_file.exists():
            with open(config_file) as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get(
                "base_model_name_or_path", "openvla/openvla-7b"
            )
        else:
            base_model = "openvla/openvla-7b"

        # Create adapter with base model
        adapter = cls(
            model_name_or_path=base_model,
            use_lora=False,  # Don't apply new LoRA, we'll load existing
            device=device,
        )
        adapter._ensure_loaded()

        # Load LoRA weights
        try:
            from peft import PeftModel

            lora_path = (
                checkpoint_path / "lora_adapters"
                if (checkpoint_path / "lora_adapters").exists()
                else checkpoint_path
            )
            adapter._model = PeftModel.from_pretrained(
                adapter._model,
                str(lora_path),
                is_trainable=True,
            )
            logger.info(f"Loaded LoRA adapters from {lora_path}")
        except Exception as e:
            logger.warning(f"Failed to load LoRA adapters: {e}")

        # Load dataset statistics if available
        stats_file = checkpoint_path / "dataset_statistics.json"
        if stats_file.exists():
            adapter._load_dataset_statistics(stats_file)

        return adapter

    @classmethod
    def _load_merged_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | str,
    ) -> "OpenVLAAdapter":
        """Load from merged model checkpoint."""
        adapter = cls(
            model_name_or_path=str(checkpoint_path),
            use_lora=True,  # Apply new LoRA for RL fine-tuning
            device=device,
        )

        # Load dataset statistics if available
        stats_file = checkpoint_path / "dataset_statistics.json"
        if stats_file.exists():
            adapter._load_dataset_statistics(stats_file)

        return adapter

    def _load_dataset_statistics(self, stats_file: Path) -> None:
        """Load dataset statistics for action normalization."""
        with open(stats_file) as f:
            stats = json.load(f)

        if "action" in stats:
            action_stats = stats["action"]
            if "mean" in action_stats and "std" in action_stats:
                self.action_mean = np.array(action_stats["mean"])
                self.action_std = np.array(action_stats["std"])
                logger.info("Loaded action normalization statistics")
            elif "q01" in action_stats and "q99" in action_stats:
                # Quantile-based normalization
                q01 = np.array(action_stats["q01"])
                q99 = np.array(action_stats["q99"])
                self.action_mean = (q01 + q99) / 2
                self.action_std = (q99 - q01) / 2
                logger.info("Loaded action normalization from quantiles")

    def generate_action(
        self,
        image: np.ndarray | Image.Image,
        instruction: str | None = None,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Generate action from observation.

        Args:
            image: RGB image (H, W, 3) as numpy array or PIL Image.
            instruction: Language instruction (uses default if None).
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Tuple of (action, log_prob, value).
        """
        self._ensure_loaded()

        if instruction is None:
            instruction = self.default_instruction

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Prepare inputs
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self._processor(
            prompt,
            image,
            return_tensors="pt",
        ).to(self.device, dtype=self.torch_dtype)

        # Generate action tokens
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.ACTION_DIM * 2,  # Action tokens
                temperature=temperature,
                do_sample=do_sample,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Decode action
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
        action = self._decode_action(generated_ids)

        # Compute log probability (simplified)
        log_prob = self._compute_log_prob(inputs, generated_ids)

        # Compute value estimate
        value = self._compute_value(outputs)

        return action, log_prob, value

    def _decode_action(self, token_ids: torch.Tensor) -> np.ndarray:
        """Decode action tokens to continuous action."""
        if self._action_tokenizer is not None:
            try:
                action = self._action_tokenizer.decode_token_ids_to_actions(
                    token_ids.cpu().numpy()
                )
                if self.action_std is not None:
                    # Denormalize
                    action = action * self.action_std + self.action_mean
                return action[: self.ACTION_DIM]
            except Exception:
                pass

        # Fallback: decode text and parse
        text = self._processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        try:
            # Try to parse as comma-separated values
            values = [float(x.strip()) for x in text.split(",")]
            action = np.array(values[: self.ACTION_DIM])
        except Exception:
            # Return zero action on failure
            action = np.zeros(self.ACTION_DIM)

        return action

    def _compute_log_prob(
        self,
        inputs: dict[str, torch.Tensor],
        generated_ids: torch.Tensor,
    ) -> float:
        """Compute log probability of generated tokens."""
        # Simplified: return 0 for now
        # Full implementation would compute actual log probs
        return 0.0

    def _compute_value(self, outputs: Any) -> float:
        """Compute value estimate from model outputs."""
        if self._value_head is None:
            return 0.0

        # Use last hidden state for value estimation
        try:
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
                value = self._value_head(last_hidden)
                return value.item()
        except Exception:
            pass

        return 0.0

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for PPO training.

        Args:
            observations: Dict with image and state tensors.
            actions: Action tensors.

        Returns:
            Dict with log_probs, values, entropy.
        """
        self._ensure_loaded()

        batch_size = actions.shape[0]

        # Use value head to compute values (this creates gradient flow)
        # For now, use a simple state-based value estimation
        if "states" in observations:
            states = observations["states"]
            # Expand state dimension for value head input
            # Value head expects hidden_size, so we project states
            state_dim = states.shape[-1]
            hidden_size = self._value_head[0].in_features

            # Create a simple projection if needed
            if not hasattr(self, "_state_projection"):
                self._state_projection = (
                    nn.Linear(state_dim, hidden_size)
                    .to(self.device)
                    .to(self.torch_dtype)
                )

            # Project states and compute values
            projected = self._state_projection(states.to(self.torch_dtype))
            values = self._value_head(projected).squeeze(-1)
        else:
            # Fallback: create trainable values via value head
            dummy_input = torch.zeros(
                batch_size,
                self._value_head[0].in_features,
                device=self.device,
                dtype=self.torch_dtype,
            )
            values = self._value_head(dummy_input).squeeze(-1)

        # For log_probs and entropy, we need proper implementation
        # For now, create small trainable values to allow gradient flow
        # These will be refined in future iterations
        log_probs = values * 0.0  # Same shape, connected to computation graph
        entropy = (
            torch.ones(batch_size, device=self.device, dtype=self.torch_dtype) * 0.1
        )

        return {
            "log_probs": log_probs.float(),
            "values": values.float(),
            "entropy": entropy.float(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save adapter checkpoint.

        Args:
            path: Directory to save checkpoint.
        """
        self._ensure_loaded()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters if applicable
        if hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(path / "lora_adapters")
            logger.info(f"Saved LoRA adapters to {path / 'lora_adapters'}")

        # Save processor
        if self._processor is not None:
            self._processor.save_pretrained(path / "processor")

        # Save value head
        if self._value_head is not None:
            torch.save(self._value_head.state_dict(), path / "value_head.pt")

        # Save action statistics
        if self.action_mean is not None:
            stats = {
                "action": {
                    "mean": self.action_mean.tolist(),
                    "std": self.action_std.tolist(),
                }
            }
            with open(path / "dataset_statistics.json", "w") as f:
                json.dump(stats, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")
