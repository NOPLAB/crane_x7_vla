# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
MiniVLA backend implementation.

This module implements MiniVLA fine-tuning using:
- Qwen 2.5 0.5B LLM backbone (~7x smaller than OpenVLA)
- DINO-SigLIP vision encoder (same as OpenVLA)
- VQ Action Chunking for multi-step action prediction
- Multi-image support (history and wrist camera)
"""

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import wandb

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.minivla_config import MiniVLAConfig
from crane_x7_vla.action_tokenizer.vq import ResidualVQ
from crane_x7_vla.action_tokenizer.vq_tokenizer import VQActionTokenizer, BinActionTokenizer
from crane_x7_vla.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class MiniVLAFinetuneConfig:
    """
    Configuration for MiniVLA fine-tuning.

    Similar to OpenVLA's FinetuneConfig but with MiniVLA-specific settings.
    """

    # Model settings
    llm_path: str = "Qwen/Qwen2.5-0.5B"
    vision_backbone: str = "dinosiglip-vit-so-224px"

    # Directory Paths
    data_root_dir: Path = Path("datasets")
    dataset_name: str = "crane_x7"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # VQ settings
    use_vq: bool = True
    vq_path: Optional[Path] = None
    action_horizon: int = 8
    vq_n_embed: int = 256
    vq_n_latent: int = 512
    vq_n_groups: int = 7

    # Multi-image settings
    use_multi_image: bool = True
    image_history: int = 2
    use_wrist_camera: bool = True

    # Fine-tuning Parameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    # Overfitting Detection
    overfit_split_ratio: float = 0.1
    overfit_check_interval: int = 500
    overfit_check_steps: int = 50

    # Memory Optimization
    gradient_checkpointing: bool = False

    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    use_quantization: bool = False

    # Tokenization
    use_extra_tokens: bool = True
    n_extra_tokens: int = 256

    # Tracking
    wandb_project: str = "minivla"
    wandb_entity: Optional[str] = None
    run_id_note: Optional[str] = None

    # Checkpoint
    skip_merge_on_save: bool = True


class MiniVLAModel(torch.nn.Module):
    """
    MiniVLA model architecture.

    Combines:
    - Vision encoder (DINO-SigLIP ViT)
    - LLM backbone (Qwen 2.5 0.5B)
    - Multi-image processing
    - VQ action tokenization
    """

    def __init__(
        self,
        llm_model_id: str = "Qwen/Qwen2.5-0.5B",
        vision_backbone: str = "dinosiglip-vit-so-224px",
        image_size: Tuple[int, int] = (224, 224),
        num_images: int = 1,
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize MiniVLA model.

        Args:
            llm_model_id: HuggingFace model ID for LLM
            vision_backbone: Vision backbone type
            image_size: Input image size (H, W)
            num_images: Number of images per observation
            use_flash_attention: Use Flash Attention 2
            torch_dtype: Model dtype
        """
        super().__init__()

        self.llm_model_id = llm_model_id
        self.vision_backbone = vision_backbone
        self.image_size = image_size
        self.num_images = num_images

        # Load LLM
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_id,
            trust_remote_code=True,
        )

        # Vision encoder (simplified - using timm DINO-SigLIP)
        self._init_vision_encoder(vision_backbone, torch_dtype)

        # Projection layer: vision features -> LLM embedding space
        self.vision_proj = torch.nn.Linear(
            self.vision_hidden_size,
            self.llm.config.hidden_size,
            dtype=torch_dtype,
        )

        # Patch embed info for computing action logit positions
        self.num_patches_per_image = (image_size[0] // 14) * (image_size[1] // 14)

    def _init_vision_encoder(self, backbone: str, dtype: torch.dtype):
        """Initialize vision encoder."""
        try:
            import timm
            # Use DINO-SigLIP fused backbone similar to Prismatic
            if "dinosiglip" in backbone.lower():
                # Fused DINO + SigLIP backbone
                self.vision_encoder_dino = timm.create_model(
                    "vit_large_patch14_dinov2.lvd142m",
                    pretrained=True,
                    num_classes=0,
                )
                self.vision_encoder_siglip = timm.create_model(
                    "vit_so400m_patch14_siglip_224",
                    pretrained=True,
                    num_classes=0,
                )
                self.vision_hidden_size = 1024 + 1152  # DINO + SigLIP
                self.use_fused_vision = True
            else:
                # Single backbone
                self.vision_encoder = timm.create_model(
                    "vit_large_patch14_dinov2.lvd142m",
                    pretrained=True,
                    num_classes=0,
                )
                self.vision_hidden_size = 1024
                self.use_fused_vision = False
        except ImportError:
            logger.warning("timm not available, using placeholder vision encoder")
            self.vision_encoder = torch.nn.Identity()
            self.vision_hidden_size = 768
            self.use_fused_vision = False

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature tokens.

        Args:
            images: Tensor of shape (B, N, C, H, W) where N is num_images

        Returns:
            Image features of shape (B, N*num_patches, hidden_size)
        """
        B, N, C, H, W = images.shape

        # Flatten batch and image dims
        images_flat = images.view(B * N, C, H, W)

        if self.use_fused_vision:
            # Fused DINO + SigLIP
            features_dino = self.vision_encoder_dino.forward_features(images_flat)
            features_siglip = self.vision_encoder_siglip.forward_features(images_flat)
            features = torch.cat([features_dino, features_siglip], dim=-1)
        else:
            features = self.vision_encoder.forward_features(images_flat)

        # Project to LLM space
        features = self.vision_proj(features)

        # Reshape to (B, N*num_patches, hidden_size)
        num_patches = features.shape[1]
        features = features.view(B, N * num_patches, -1)

        return features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            pixel_values: Images (B, N, C, H, W) or (B, C, H, W)
            labels: Target token IDs for training (B, seq_len)

        Returns:
            Model output with loss if labels provided
        """
        # Add image dimension if needed
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        # Encode images
        image_features = self.encode_images(pixel_values)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate image and text embeddings
        # [image_tokens, text_tokens]
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

        # Extend attention mask for image tokens
        image_attn = torch.ones(
            image_features.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([image_attn, attention_mask], dim=1)

        # Extend labels if provided
        if labels is not None:
            image_labels = torch.full(
                image_features.shape[:2],
                fill_value=-100,  # Ignore index
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([image_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 7,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate action tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Images
            max_new_tokens: Number of action tokens to generate

        Returns:
            Generated token IDs
        """
        # Add image dimension if needed
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        # Encode images
        image_features = self.encode_images(pixel_values)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

        # Extend attention mask
        image_attn = torch.ones(
            image_features.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([image_attn, attention_mask], dim=1)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        return outputs


class MiniVLABackend(VLABackend):
    """
    MiniVLA backend implementation.

    Wraps the MiniVLA model with the unified VLA backend interface.
    Features:
    - ~1B parameters (7x smaller than OpenVLA)
    - VQ Action Chunking for efficient multi-step prediction
    - Multi-image input (history + wrist camera)
    - ~12.5Hz inference (2.5x faster than OpenVLA)
    """

    def __init__(self, config: MiniVLAConfig):
        """
        Initialize MiniVLA backend.

        Args:
            config: MiniVLA configuration
        """
        super().__init__(config)
        self.minivla_config = config
        self.trainer = None

        # Set properties from config
        self._action_dim = config.action_dim
        self._action_horizon = config.minivla.action_horizon
        self._image_size = config.minivla.image_size

        # VQ tokenizer (initialized later)
        self.vq_tokenizer: Optional[VQActionTokenizer] = None
        self.bin_tokenizer: Optional[BinActionTokenizer] = None

        # Initialize tokenizer based on VQ setting
        if config.minivla.vq.enabled and config.minivla.vq.vq_path:
            self.vq_tokenizer = VQActionTokenizer(
                vq_path=config.minivla.vq.vq_path,
                use_extra_tokens=config.minivla.use_extra_tokens,
            )
        else:
            self.bin_tokenizer = BinActionTokenizer(
                action_dim=self._action_dim,
                n_bins=config.minivla.action_tokenization_bins,
                action_range=config.minivla.action_range,
                use_extra_tokens=config.minivla.use_extra_tokens,
            )

    def _create_finetune_config(self) -> MiniVLAFinetuneConfig:
        """Convert UnifiedVLAConfig to MiniVLAFinetuneConfig."""
        cfg = self.minivla_config

        return MiniVLAFinetuneConfig(
            # Model settings
            llm_path=cfg.minivla.llm_model_id,
            vision_backbone=cfg.minivla.vision_backbone,
            # Directory Paths
            data_root_dir=cfg.data.data_root,
            dataset_name=getattr(cfg.data, 'dataset_name', 'crane_x7'),
            run_root_dir=cfg.output_dir,
            # VQ settings
            use_vq=cfg.minivla.vq.enabled,
            vq_path=Path(cfg.minivla.vq.vq_path) if cfg.minivla.vq.vq_path else None,
            action_horizon=cfg.minivla.vq.action_horizon,
            vq_n_embed=cfg.minivla.vq.n_embed,
            vq_n_latent=cfg.minivla.vq.n_latent,
            vq_n_groups=cfg.minivla.vq.n_groups,
            # Multi-image settings
            use_multi_image=cfg.minivla.multi_image.enabled,
            image_history=cfg.minivla.multi_image.image_history,
            use_wrist_camera=cfg.minivla.multi_image.use_wrist_camera,
            # Fine-tuning Parameters
            batch_size=cfg.training.batch_size,
            max_steps=cfg.training.max_steps,
            save_steps=cfg.training.save_interval,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            grad_accumulation_steps=cfg.training.gradient_accumulation_steps,
            image_aug=cfg.minivla.image_aug,
            shuffle_buffer_size=getattr(cfg.data, 'shuffle_buffer_size', 100_000),
            # Overfitting Detection
            overfit_split_ratio=cfg.overfitting.overfit_split_ratio,
            overfit_check_interval=cfg.overfitting.overfit_check_interval,
            overfit_check_steps=cfg.overfitting.overfit_check_steps,
            # Memory Optimization
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            # LoRA Arguments
            use_lora=cfg.minivla.use_lora,
            lora_rank=cfg.minivla.lora_rank,
            lora_alpha=cfg.minivla.lora_alpha,
            lora_dropout=cfg.minivla.lora_dropout,
            use_quantization=cfg.minivla.use_quantization,
            # Tokenization
            use_extra_tokens=cfg.minivla.use_extra_tokens,
            n_extra_tokens=cfg.minivla.n_extra_tokens,
            # Tracking
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            run_id_note=cfg.experiment_name,
            # Checkpoint
            skip_merge_on_save=cfg.minivla.skip_merge_on_save,
        )

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            Dictionary containing training metrics and results
        """
        logger.info("Starting MiniVLA training...")
        cfg = self._create_finetune_config()

        # Validate GPU
        assert torch.cuda.is_available(), "MiniVLA training requires GPU!"
        distributed_state = PartialState()
        torch.cuda.set_device(device_id := distributed_state.local_process_index)
        torch.cuda.empty_cache()

        # Create experiment ID
        exp_id = (
            f"minivla+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_vq:
            exp_id += f"+vq-h{cfg.action_horizon}"
        if cfg.use_lora:
            exp_id += f"+lora-r{cfg.lora_rank}"
        if cfg.run_id_note:
            exp_id += f"--{cfg.run_id_note}"

        # Create directories
        run_dir = cfg.run_root_dir / exp_id
        os.makedirs(run_dir, exist_ok=True)

        # Quantization config
        quantization_config = None
        if cfg.use_quantization:
            assert cfg.use_lora, "Quantization only supported with LoRA!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        # Calculate number of images
        num_images = 1
        if cfg.use_multi_image:
            num_images = cfg.image_history
            if cfg.use_wrist_camera:
                num_images += 1

        # Create model
        model = MiniVLAModel(
            llm_model_id=cfg.llm_path,
            vision_backbone=cfg.vision_backbone,
            image_size=self._image_size,
            num_images=num_images,
            use_flash_attention=self.minivla_config.minivla.use_flash_attention,
            torch_dtype=torch.bfloat16,
        )

        # Device placement
        if cfg.use_quantization:
            model = prepare_model_for_kbit_training(model)
        else:
            model = model.to(device_id)

        # Gradient checkpointing
        if cfg.gradient_checkpointing:
            model.llm.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # LoRA
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=self.minivla_config.minivla.lora_target_modules,
                init_lora_weights="gaussian",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # DDP
        is_distributed = distributed_state.num_processes > 1
        if is_distributed:
            model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
            if cfg.gradient_checkpointing:
                model._set_static_graph()
            dist.barrier()

        unwrapped_model = model.module if is_distributed else model

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Create action tokenizer
        if cfg.use_vq and cfg.vq_path:
            action_tokenizer = VQActionTokenizer(vq_path=cfg.vq_path)
        else:
            action_tokenizer = BinActionTokenizer(
                action_dim=self._action_dim,
                n_bins=self.minivla_config.minivla.action_tokenization_bins,
            )

        # TODO: Create dataset and dataloader
        # This would use CraneX7Dataset or a MiniVLA-specific variant
        # For now, log that training setup is complete
        logger.info(f"MiniVLA model initialized: {exp_id}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # Initialize W&B
        if distributed_state.is_main_process and wandb.run is None:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=f"ft+{exp_id}",
            )

        # Return setup info (actual training loop to be implemented with dataset)
        return {
            'exp_id': exp_id,
            'run_dir': str(run_dir),
            'model_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in trainable_params),
            'status': 'initialized',
        }

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        if self.model is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # TODO: Implement evaluation loop
        raise NotImplementedError("Evaluation not yet implemented for MiniVLA backend")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Dictionary containing:
                - 'state': Robot state [8]
                - 'image': RGB image [H, W, 3] or list of images
                - 'wrist_image': Wrist camera image (optional)
            language_instruction: Task instruction

        Returns:
            Predicted action(s) as numpy array
            - Shape [8] if VQ disabled (single action)
            - Shape [H, 8] if VQ enabled (action chunk)
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # Prepare instruction
        if language_instruction is None:
            language_instruction = "manipulate objects"

        prompt = f"In: What action should the robot take to {language_instruction}?\nOut:"

        # Prepare images
        images = []
        primary_image = observation.get('image')
        if primary_image is not None:
            if isinstance(primary_image, list):
                images.extend(primary_image)
            else:
                images.append(primary_image)

        wrist_image = observation.get('wrist_image')
        if wrist_image is not None and self.minivla_config.minivla.multi_image.use_wrist_camera:
            images.append(wrist_image)

        # Convert images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        # Process inputs
        inputs = self.processor(prompt, pil_images)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate action tokens
        n_tokens = self.vq_tokenizer.n_groups if self.vq_tokenizer else self._action_dim
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=n_tokens)

        # Decode actions
        action_tokens = outputs[0, -n_tokens:].cpu().numpy()

        if self.vq_tokenizer:
            # VQ decoding -> action chunk [H, A]
            actions = self.vq_tokenizer.decode(action_tokens)
        else:
            # Bin decoding -> single action [A]
            actions = self.bin_tokenizer.decode(action_tokens)

        return actions

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        if self.minivla_config.minivla.use_lora:
            self.model.save_pretrained(path / "lora_adapters")
        else:
            self.model.save_pretrained(path)

        # Save processor
        if self.processor is not None:
            self.processor.save_pretrained(path)

        # Save VQ model if used
        if self.vq_tokenizer is not None and self.vq_tokenizer.vq is not None:
            self.vq_tokenizer.vq.save(path / "vq_model.pt")

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")

        logger.info(f"Loading MiniVLA model from {path}")

        # Determine model configuration
        cfg = self.minivla_config.minivla
        num_images = cfg.total_image_count

        # Create model
        self.model = MiniVLAModel(
            llm_model_id=cfg.llm_model_id,
            vision_backbone=cfg.vision_backbone,
            image_size=cfg.image_size,
            num_images=num_images,
            use_flash_attention=cfg.use_flash_attention,
            torch_dtype=torch.bfloat16 if self.minivla_config.training.mixed_precision == "bf16" else torch.float32,
        )

        # Load LoRA weights if applicable
        lora_path = path / "lora_adapters"
        if lora_path.exists() and cfg.use_lora:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info("Loaded LoRA adapters")

        # Load VQ model if exists
        vq_path = path / "vq_model.pt"
        if vq_path.exists():
            self.vq_tokenizer = VQActionTokenizer(
                vq_path=vq_path,
                use_extra_tokens=cfg.use_extra_tokens,
            )
            logger.info("Loaded VQ model")

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        # Set processor (tokenizer)
        self.processor = self.model.tokenizer

        logger.info("MiniVLA model loaded successfully")

    @property
    def action_dim(self) -> int:
        """Get the action dimension of the model."""
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        """Get the action horizon (VQ chunking returns multiple actions)."""
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        """Get the expected image size for the model."""
        return self._image_size
