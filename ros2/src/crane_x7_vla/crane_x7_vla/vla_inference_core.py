#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Core VLA inference logic shared between ROS 2 node and rosbridge client.

This module provides a ROS-independent implementation of VLA model loading
and inference, allowing reuse across different transport mechanisms.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

# Add VLA directory to path
VLA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "vla"
sys.path.insert(0, str(VLA_PATH))


class VLAInferenceCore:
    """Core VLA inference logic without ROS dependencies.

    This class handles model loading and inference for OpenVLA models,
    providing a clean interface that can be used by both ROS 2 nodes
    and rosbridge clients.

    Attributes:
        model_path: Path to the VLA model (local or HuggingFace Hub ID)
        device: PyTorch device for inference ('cuda' or 'cpu')
        unnorm_key: Key for action normalization statistics
        model: Loaded VLA model
        processor: Model processor for image/text preprocessing
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        unnorm_key: str = 'crane_x7',
        model_base_name: str = 'openvla',
        use_flash_attention: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize VLA inference core.

        Args:
            model_path: Path to model (local path or HuggingFace Hub ID)
            device: Device for inference ('cuda' or 'cpu')
            unnorm_key: Key for action normalization statistics
            model_base_name: Base model name for prompt formatting
            use_flash_attention: Whether to use Flash Attention 2
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.device_name = device
        self.unnorm_key = unnorm_key
        self.model_base_name = model_base_name
        self.use_flash_attention = use_flash_attention

        self.logger = logger or logging.getLogger(__name__)

        self.model = None
        self.processor = None
        self.device = None

    def setup_device(self) -> torch.device:
        """Setup PyTorch device.

        Returns:
            Configured PyTorch device
        """
        if torch.cuda.is_available() and self.device_name == 'cuda':
            self.device = torch.device('cuda')
            self.logger.info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.logger.info('Using CPU device')
        return self.device

    def is_huggingface_hub_id(self, path: str) -> bool:
        """Check if path looks like a HuggingFace Hub model ID.

        Args:
            path: Path string to check

        Returns:
            True if path looks like a HF Hub ID (e.g., 'username/model-name')
        """
        if not path:
            return False
        if path.startswith('/') or path.startswith('./') or path.startswith('..'):
            return False
        if '\\' in path:  # Windows path
            return False
        parts = path.split('/')
        return len(parts) == 2 and all(p for p in parts)

    def load_model(self) -> bool:
        """Load VLA model and processor.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path:
            self.logger.error(
                'Model path not specified! '
                'Set VLA_MODEL_PATH environment variable. '
                'Example: VLA_MODEL_PATH=/workspace/vla/outputs/<your_model_dir> '
                'or VLA_MODEL_PATH=your-username/crane_x7_openvla (HuggingFace Hub)'
            )
            return False

        # Setup device if not already done
        if self.device is None:
            self.setup_device()

        # Check if this is a HuggingFace Hub ID or local path
        is_hf_hub = self.is_huggingface_hub_id(self.model_path)

        if is_hf_hub:
            self.logger.info(f'Loading model from HuggingFace Hub: {self.model_path}')
            model_path_str = self.model_path
            model_path = None
        else:
            model_path = Path(self.model_path)
            model_path_str = str(model_path)
            if not model_path.exists():
                self.logger.error(f'Model path does not exist: {model_path}')
                outputs_dir = Path('/workspace/vla/outputs')
                if outputs_dir.exists():
                    available = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
                    if available:
                        self.logger.info(f'Available models in {outputs_dir}: {available}')
                return False

        self.logger.info(f'Loading VLA model from {model_path_str}...')

        try:
            # Register custom OpenVLA classes for HuggingFace Auto classes
            self._register_openvla_classes()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path_str,
                trust_remote_code=True
            )
            self.logger.info('Processor loaded successfully')

            # Check if this is a LoRA checkpoint
            is_lora_checkpoint, lora_adapter_path = self._check_lora_checkpoint(
                is_hf_hub, model_path, model_path_str
            )

            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager",
            }

            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.logger.info('Using Flash Attention 2')

            if is_lora_checkpoint and lora_adapter_path:
                self.model = self._load_lora_model(lora_adapter_path, model_kwargs)
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path_str,
                    **model_kwargs
                )

            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load normalization statistics
            self._load_norm_stats(is_hf_hub, model_path, model_path_str)

            # Verify unnorm_key
            self._verify_unnorm_key()

            # Debug: log model configuration
            self.logger.info(
                f'Model config: vocab_size={self.model.config.text_config.vocab_size}, '
                f'pad_to_multiple_of={self.model.config.pad_to_multiple_of}, '
                f'n_action_bins={self.model.config.n_action_bins}'
            )

            self.logger.info('VLA model loaded successfully')
            return True

        except Exception as e:
            self.logger.error(f'Failed to load model: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _register_openvla_classes(self) -> None:
        """Register custom OpenVLA classes for HuggingFace Auto classes."""
        try:
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as HFOpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            from transformers import AutoImageProcessor

            AutoConfig.register("openvla", HFOpenVLAConfig)
            AutoImageProcessor.register(HFOpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(HFOpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(HFOpenVLAConfig, OpenVLAForActionPrediction)
            self.logger.info('Registered OpenVLA custom classes')
        except ImportError:
            self.logger.warning(
                'Could not import prismatic classes - relying on trust_remote_code'
            )

    def _check_lora_checkpoint(
        self,
        is_hf_hub: bool,
        model_path: Optional[Path],
        model_path_str: str
    ) -> tuple[bool, Optional[Path]]:
        """Check if model is a LoRA checkpoint.

        Returns:
            Tuple of (is_lora_checkpoint, lora_adapter_path)
        """
        is_lora_checkpoint = False
        lora_adapter_path = None

        if is_hf_hub:
            try:
                from huggingface_hub import HfFileSystem, snapshot_download
                fs = HfFileSystem()
                lora_config_path = f"{model_path_str}/lora_adapters/adapter_config.json"
                if fs.exists(lora_config_path):
                    is_lora_checkpoint = True
                    local_dir = snapshot_download(
                        model_path_str,
                        allow_patterns=["lora_adapters/*"],
                        local_dir="/tmp/vla_model"
                    )
                    lora_adapter_path = Path(local_dir) / "lora_adapters"
                    self.logger.info(f'Downloaded LoRA adapter to {lora_adapter_path}')
            except Exception as e:
                self.logger.debug(f'LoRA check for HF Hub failed: {e}')
        else:
            lora_adapter_path = model_path / "lora_adapters"
            is_lora_checkpoint = lora_adapter_path.exists()

        return is_lora_checkpoint, lora_adapter_path

    def _load_lora_model(self, lora_adapter_path: Path, model_kwargs: dict):
        """Load base model and apply LoRA adapter.

        Args:
            lora_adapter_path: Path to LoRA adapter
            model_kwargs: Model loading kwargs

        Returns:
            Merged model
        """
        self.logger.info('Detected LoRA checkpoint, loading base model + adapter...')

        # Read adapter config to get base model path
        adapter_config_path = lora_adapter_path / "adapter_config.json"
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", "openvla/openvla-7b")
        self.logger.info(f'Base model: {base_model_path}')

        # Load base model
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        self.logger.info('Base model loaded')

        # Load and apply LoRA adapter using PEFT
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_adapter_path),
            is_trainable=False,
        )
        self.logger.info('LoRA adapter applied')

        # Merge LoRA weights for faster inference
        model = model.merge_and_unload()
        self.logger.info('LoRA weights merged')

        return model

    def _load_norm_stats(
        self,
        is_hf_hub: bool,
        model_path: Optional[Path],
        model_path_str: str
    ) -> None:
        """Load normalization statistics from checkpoint."""
        checkpoint_stats = None

        if is_hf_hub:
            try:
                from huggingface_hub import hf_hub_download
                stats_file = hf_hub_download(
                    model_path_str,
                    filename="dataset_statistics.json"
                )
                with open(stats_file, 'r') as f:
                    checkpoint_stats = json.load(f)
            except Exception as e:
                self.logger.debug(f'Could not download dataset_statistics.json: {e}')
        else:
            norm_stats_path = model_path / "dataset_statistics.json"
            if norm_stats_path.exists():
                with open(norm_stats_path, 'r') as f:
                    checkpoint_stats = json.load(f)

        if checkpoint_stats:
            if not hasattr(self.model, 'norm_stats') or self.model.norm_stats is None:
                self.model.norm_stats = {}
            self.model.norm_stats.update(checkpoint_stats)
            self.logger.info(
                f'Loaded checkpoint statistics: {list(checkpoint_stats.keys())}'
            )
        elif hasattr(self.model, 'norm_stats') and self.model.norm_stats:
            self.logger.info(
                f'Using model norm_stats: {list(self.model.norm_stats.keys())}'
            )
        else:
            self.logger.warning(
                'No dataset_statistics.json found - using default normalization'
            )

    def _verify_unnorm_key(self) -> None:
        """Verify unnorm_key is available and select fallback if needed."""
        if hasattr(self.model, 'norm_stats') and self.model.norm_stats:
            if self.unnorm_key not in self.model.norm_stats:
                available_keys = list(self.model.norm_stats.keys())
                self.logger.warning(
                    f'unnorm_key "{self.unnorm_key}" not in norm_stats. '
                    f'Available keys: {available_keys}'
                )
                if 'bridge_orig' in available_keys:
                    self.unnorm_key = 'bridge_orig'
                elif available_keys:
                    self.unnorm_key = available_keys[0]
                self.logger.info(f'Using fallback unnorm_key: {self.unnorm_key}')

            action_dim = len(self.model.norm_stats[self.unnorm_key]["action"]["q01"])
            self.logger.info(
                f'Using unnorm_key: {self.unnorm_key} (action_dim={action_dim})'
            )

    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[np.ndarray]:
        """Run VLA inference and return predicted action.

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            log_callback: Optional callback for logging debug info

        Returns:
            Action array (8-dim for CRANE-X7) or None if inference fails
        """
        if self.model is None or self.processor is None:
            self.logger.error('Model not loaded')
            return None

        try:
            # Prepare image
            pil_image = PILImage.fromarray(image)
            pil_image = pil_image.convert("RGB")

            # Debug logging
            if log_callback:
                image_hash = hash(image.tobytes())
                log_callback(
                    f'Inference input: image_size={pil_image.size}, '
                    f'image_hash={image_hash % 10000:04d}, '
                    f'mean_pixel={image.mean():.2f}'
                )

            # Build prompt based on model version
            prompt = self._build_prompt(instruction)

            # Process inputs
            inputs = self.processor(prompt, pil_image)

            # Move tensors to device
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(self.device)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

            # Run inference
            with torch.no_grad():
                action = self._run_autoregressive_inference(
                    input_ids, attention_mask, pixel_values, log_callback
                )

            return action

        except Exception as e:
            self.logger.error(f'Inference failed: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _build_prompt(self, instruction: str) -> str:
        """Build prompt string for the model.

        Args:
            instruction: Task instruction

        Returns:
            Formatted prompt string
        """
        if "openvla-v01" in self.model_base_name or "v01" in self.model_base_name:
            # OpenVLA v0.1 format (VicunaV15ChatPromptBuilder)
            system_prompt = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            prompt = (
                f"{system_prompt} USER: What action should the robot take to "
                f"{instruction.lower()}? ASSISTANT:"
            )
        else:
            # OpenVLA format (PurePromptBuilder)
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        return prompt

    def _run_autoregressive_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> np.ndarray:
        """Run autoregressive inference to generate action tokens.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            pixel_values: Processed image tensor
            log_callback: Optional logging callback

        Returns:
            Predicted action array
        """
        action_dim = self.model.get_action_dim(self.unnorm_key)

        # Add special empty token if needed
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
                )

        # Autoregressive generation
        generated_tokens = []
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        past_key_values = None

        for step in range(action_dim):
            if step == 0:
                forward_output = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    pixel_values=pixel_values,
                )
            else:
                forward_output = self.model(
                    input_ids=current_input_ids[:, -1:],
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                )

            logits = forward_output.logits
            past_key_values = forward_output.past_key_values

            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())

            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1
            )

        predicted_token_ids = np.array(generated_tokens)

        if log_callback:
            log_callback(f'Generated token IDs: {predicted_token_ids}')

        # Decode tokens to actions
        action = self._decode_action_tokens(predicted_token_ids, log_callback)

        return action

    def _decode_action_tokens(
        self,
        predicted_token_ids: np.ndarray,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> np.ndarray:
        """Decode action tokens to continuous action values.

        Args:
            predicted_token_ids: Array of generated token IDs
            log_callback: Optional logging callback

        Returns:
            Decoded action array
        """
        vocab_size = self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of

        if log_callback:
            bin_indices = vocab_size - predicted_token_ids - 1
            log_callback(f'Bin indices: {bin_indices} (vocab_size={vocab_size})')

        discretized_actions = vocab_size - predicted_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1)
        normalized_actions = self.model.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.model.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        # Ensure 1D array
        action = np.asarray(action).flatten()

        return action

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None and self.processor is not None
