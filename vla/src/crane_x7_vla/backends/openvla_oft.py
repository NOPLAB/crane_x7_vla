# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenVLA-OFT (Optimized Fine-Tuning) backend implementation.

This module implements OpenVLA-OFT which provides:
- L1 Regression Action Head (continuous actions instead of tokens)
- Action Chunking (predict multiple future actions)
- FiLM (Feature-wise Linear Modulation for language conditioning)
- Proprioceptive Input support
- Multi-image Input support
- 26x faster inference through parallel decoding

Reference: https://arxiv.org/abs/2502.19645
"""

import gc
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as HFOpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openvla_oft_config import OpenVLAOFTConfig
from crane_x7_vla.data.openvla_oft_dataset import (
    CraneX7OFTDataset,
    OpenVLAOFTBatchTransform,
    PaddedCollatorForOFT,
)
from crane_x7_vla.models.openvla_oft_components import (
    L1RegressionActionHead,
    ProprioProjector,
    FiLMedVisionBackbone,
)
from crane_x7_vla.utils.logging import get_logger

logger = get_logger(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class OpenVLAOFTFinetuneConfig:
    """
    Configuration for OpenVLA-OFT fine-tuning.

    This config matches the OpenVLA-OFT paper's training setup.
    """

    # fmt: off
    vla_path: str = "openvla/openvla-7b"
    """Path to base OpenVLA model (on HuggingFace Hub)"""

    # Action settings
    action_dim: int = 8
    """Action dimension (CRANE-X7: 8)"""

    action_horizon: int = 8
    """Action chunk horizon (number of future actions)"""

    # OFT components
    use_film: bool = True
    """Whether to use FiLM for language-vision modulation"""

    use_proprio: bool = True
    """Whether to use proprioceptive input"""

    use_multi_image: bool = False
    """Whether to use multiple camera images"""

    num_images: int = 1
    """Number of images in input"""

    proprio_dim: int = 8
    """Proprioceptive state dimension"""

    # Action head settings
    action_head_hidden_dim: int = 4096
    """Hidden dimension for action head MLP"""

    action_head_num_blocks: int = 2
    """Number of residual blocks in action head"""

    # Directory paths
    data_root_dir: Path = Path("datasets")
    dataset_name: str = "crane_x7"
    run_root_dir: Path = Path("runs")

    # Fine-tuning parameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    # LR schedule
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000

    # Overfitting detection
    overfit_split_ratio: float = 0.1
    overfit_check_interval: int = 500
    overfit_check_steps: int = 50

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Memory optimization
    gradient_checkpointing: bool = False
    use_quantization: bool = False

    # W&B tracking
    wandb_project: str = "openvla-oft"
    wandb_entity: Optional[str] = None
    run_id_note: Optional[str] = None

    # Checkpoint saving
    skip_merge_on_save: bool = True
    # fmt: on


def count_parameters(module: nn.Module, name: str) -> int:
    """Count and print trainable parameters in a module."""
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info(f"# trainable params in {name}: {num_params:,}")
    return num_params


def wrap_ddp(
    module: nn.Module, device_id: int, find_unused: bool = False
) -> nn.Module:
    """Wrap module with DistributedDataParallel."""
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


class OpenVLAOFTTrainer:
    """
    Trainer for OpenVLA-OFT fine-tuning.

    Implements the training loop with:
    - L1 regression loss on continuous action chunks
    - Optional FiLM conditioning
    - Optional proprioceptive input
    - Multi-GPU support via DDP
    """

    def __init__(self, cfg: OpenVLAOFTFinetuneConfig):
        """Initialize trainer with configuration."""
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    def train(self) -> None:
        """Execute OpenVLA-OFT fine-tuning loop."""
        logger.info(f"Fine-tuning OpenVLA-OFT on `{self.cfg.dataset_name}`")
        logger.info(f"  Action Horizon: {self.cfg.action_horizon}")
        logger.info(f"  Use FiLM: {self.cfg.use_film}")
        logger.info(f"  Use Proprio: {self.cfg.use_proprio}")

        # Initialize distributed state
        assert torch.cuda.is_available(), "Fine-tuning requires GPU!"
        distributed_state = PartialState()
        torch.cuda.set_device(device_id := distributed_state.local_process_index)
        torch.cuda.empty_cache()

        # Configure experiment ID
        exp_id = (
            f"openvla-oft+{self.cfg.dataset_name}"
            f"+h{self.cfg.action_horizon}"
            f"+b{self.cfg.batch_size * self.cfg.grad_accumulation_steps}"
            f"+lr-{self.cfg.learning_rate}"
        )
        if self.cfg.use_film:
            exp_id += "+film"
        if self.cfg.use_proprio:
            exp_id += "+proprio"
        if self.cfg.use_lora:
            exp_id += f"+lora-r{self.cfg.lora_rank}"
        if self.cfg.run_id_note:
            exp_id += f"--{self.cfg.run_id_note}"
        if self.cfg.image_aug:
            exp_id += "--image_aug"

        # Build directories
        run_dir = self.cfg.run_root_dir / exp_id
        os.makedirs(run_dir, exist_ok=True)

        # Quantization config
        quantization_config = None
        if self.cfg.use_quantization:
            assert self.cfg.use_lora, "Quantization only supported with LoRA!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        # Register OpenVLA model
        AutoConfig.register("openvla", HFOpenVLAConfig)
        AutoImageProcessor.register(HFOpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(HFOpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(HFOpenVLAConfig, OpenVLAForActionPrediction)

        # Load processor and base model
        processor = AutoProcessor.from_pretrained(
            self.cfg.vla_path, trust_remote_code=True
        )
        vla = AutoModelForVision2Seq.from_pretrained(
            self.cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # Device placement
        if self.cfg.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        else:
            vla = vla.to(device_id)

        # Enable gradient checkpointing
        if self.cfg.gradient_checkpointing:
            vla.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Set number of images
        if self.cfg.use_multi_image:
            vla.vision_backbone.set_num_images_in_input(self.cfg.num_images)

        # Get LLM hidden dimension
        llm_dim = vla.llm_dim  # Usually 4096 for Llama-2 7B

        # Apply LoRA to LLM layers
        if self.cfg.use_lora:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            lora_config = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=lora_target_modules,
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

        # Apply FiLM wrapper if enabled
        if self.cfg.use_film:
            logger.info("Wrapping vision backbone with FiLM...")
            count_parameters(vla.vision_backbone, "vision_backbone (before FiLM)")

            # Wrap vision backbone with FiLM
            # For LoRA models, access through .model
            if hasattr(vla, 'model'):
                vla.model.vision_backbone = FiLMedVisionBackbone(
                    vision_backbone=vla.model.vision_backbone,
                    llm_dim=llm_dim,
                )
                vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)
            else:
                vla.vision_backbone = FiLMedVisionBackbone(
                    vision_backbone=vla.vision_backbone,
                    llm_dim=llm_dim,
                )
                vla.vision_backbone = vla.vision_backbone.to(device_id)

            count_parameters(vla.vision_backbone, "vision_backbone (after FiLM)")

        # Initialize L1 Regression Action Head
        action_head = L1RegressionActionHead(
            llm_hidden_dim=llm_dim,
            action_dim=self.cfg.action_dim,
            action_horizon=self.cfg.action_horizon,
            num_blocks=self.cfg.action_head_num_blocks,
        ).to(torch.bfloat16).to(device_id)
        count_parameters(action_head, "action_head")

        # Initialize Proprio Projector if enabled
        proprio_projector = None
        if self.cfg.use_proprio:
            proprio_projector = ProprioProjector(
                proprio_dim=self.cfg.proprio_dim,
                llm_dim=llm_dim,
            ).to(torch.bfloat16).to(device_id)
            count_parameters(proprio_projector, "proprio_projector")

        # Distributed training setup
        is_distributed = distributed_state.num_processes > 1
        if is_distributed:
            vla = wrap_ddp(vla, device_id, find_unused=True)
            action_head = wrap_ddp(action_head, device_id)
            if proprio_projector:
                proprio_projector = wrap_ddp(proprio_projector, device_id)

            if self.cfg.gradient_checkpointing:
                vla._set_static_graph()

            dist.barrier()

        # Get unwrapped model
        unwrapped_vla = vla.module if is_distributed else vla
        unwrapped_action_head = action_head.module if is_distributed else action_head
        unwrapped_proprio_projector = (
            proprio_projector.module if is_distributed and proprio_projector else proprio_projector
        )

        # Calculate number of patches
        num_patches = (
            unwrapped_vla.vision_backbone.get_num_patches()
            * (self.cfg.num_images if self.cfg.use_multi_image else 1)
        )
        if self.cfg.use_proprio:
            num_patches += 1  # Proprio embedding
        logger.info(f"Number of vision patches: {num_patches}")

        # Collect trainable parameters
        trainable_params = [p for p in vla.parameters() if p.requires_grad]
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
        if proprio_projector:
            trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]

        total_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Total trainable params: {total_params:,}")

        # Optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # LR scheduler
        original_lr = optimizer.param_groups[0]["lr"]
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.num_steps_before_decay],
            gamma=0.1,
        )

        # Create batch transform
        batch_transform = OpenVLAOFTBatchTransform(
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in self.cfg.vla_path else VicunaV15ChatPromptBuilder,
            action_horizon=self.cfg.action_horizon,
            action_dim=self.cfg.action_dim,
            include_proprio=self.cfg.use_proprio,
            include_wrist_image=self.cfg.use_multi_image and self.cfg.num_images > 1,
        )

        # Create dataset
        train_dataset = CraneX7OFTDataset(
            data_root_dir=self.cfg.data_root_dir,
            data_mix=self.cfg.dataset_name,
            batch_transform=batch_transform,
            resize_resolution=tuple(unwrapped_vla.config.image_sizes),
            action_horizon=self.cfg.action_horizon,
            shuffle_buffer_size=self.cfg.shuffle_buffer_size,
            image_aug=self.cfg.image_aug,
            overfit_split_ratio=self.cfg.overfit_split_ratio,
            split="train",
            rank=distributed_state.process_index,
            world_size=distributed_state.num_processes,
        )

        # Overfit detection dataset
        overfit_dataset = None
        if self.cfg.overfit_split_ratio > 0:
            overfit_dataset = CraneX7OFTDataset(
                data_root_dir=self.cfg.data_root_dir,
                data_mix=self.cfg.dataset_name,
                batch_transform=batch_transform,
                resize_resolution=tuple(unwrapped_vla.config.image_sizes),
                action_horizon=self.cfg.action_horizon,
                shuffle_buffer_size=1000,
                image_aug=False,
                overfit_split_ratio=self.cfg.overfit_split_ratio,
                split="overfit",
                train=False,
                rank=distributed_state.process_index,
                world_size=distributed_state.num_processes,
            )

        # Save dataset statistics
        if distributed_state.is_main_process:
            save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

        # Create dataloader
        collator = PaddedCollatorForOFT(
            max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collator,
            num_workers=0,
        )

        overfit_dataloader = None
        if overfit_dataset:
            overfit_dataloader = DataLoader(
                overfit_dataset,
                batch_size=self.cfg.batch_size,
                collate_fn=collator,
                num_workers=0,
            )

        # Initialize W&B
        if distributed_state.is_main_process:
            if wandb.run is None:
                wandb.init(
                    entity=self.cfg.wandb_entity,
                    project=self.cfg.wandb_project,
                    name=f"oft+{exp_id}",
                )

        # Training metrics
        recent_losses = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_curr_action_l1 = deque(maxlen=self.cfg.grad_accumulation_steps)

        # Training loop
        logger.info("Starting training...")
        with tqdm.tqdm(total=self.cfg.max_steps, leave=False) as progress:
            vla.train()
            action_head.train()
            if proprio_projector:
                proprio_projector.train()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                # Forward pass
                loss, metrics = self._forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    batch=batch,
                    device_id=device_id,
                    num_patches=num_patches,
                    use_film=self.cfg.use_film,
                    use_proprio=self.cfg.use_proprio,
                )

                # Normalize loss for gradient accumulation
                normalized_loss = loss / self.cfg.grad_accumulation_steps
                normalized_loss.backward()

                # Store metrics
                recent_losses.append(loss.item())
                recent_l1_losses.append(metrics["l1_loss"])
                recent_curr_action_l1.append(metrics["curr_action_l1"])

                gradient_step_idx = batch_idx // self.cfg.grad_accumulation_steps

                # Compute smoothed metrics
                smoothed_loss = sum(recent_losses) / len(recent_losses)
                smoothed_l1 = sum(recent_l1_losses) / len(recent_l1_losses)
                smoothed_curr_l1 = sum(recent_curr_action_l1) / len(recent_curr_action_l1)

                # Log to W&B
                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    wandb.log(
                        {
                            "train/loss": smoothed_loss,
                            "train/l1_loss": smoothed_l1,
                            "train/curr_action_l1": smoothed_curr_l1,
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=gradient_step_idx,
                    )

                # LR warmup
                if self.cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / self.cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                # Optimizer step
                if (batch_idx + 1) % self.cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

                    # Overfitting check
                    if (
                        overfit_dataloader is not None
                        and gradient_step_idx > 0
                        and gradient_step_idx % self.cfg.overfit_check_interval == 0
                    ):
                        overfit_metrics = self._run_overfit_check(
                            vla=vla,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            overfit_dataloader=overfit_dataloader,
                            device_id=device_id,
                            num_patches=num_patches,
                            distributed_state=distributed_state,
                        )

                        if distributed_state.is_main_process:
                            wandb.log(
                                {
                                    "eval/loss": overfit_metrics["loss"],
                                    "eval/l1_loss": overfit_metrics["l1_loss"],
                                    "eval/curr_action_l1": overfit_metrics["curr_action_l1"],
                                },
                                step=gradient_step_idx,
                            )

                        vla.train()
                        action_head.train()
                        if proprio_projector:
                            proprio_projector.train()

                # Save checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % self.cfg.save_steps == 0:
                    self._save_checkpoint(
                        run_dir=run_dir,
                        step=gradient_step_idx,
                        vla=vla,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        processor=processor,
                        dataset_statistics=train_dataset.dataset_statistics,
                        distributed_state=distributed_state,
                        is_distributed=is_distributed,
                    )

                # Stop at max steps
                if gradient_step_idx >= self.cfg.max_steps:
                    logger.info(f"Reached max steps {self.cfg.max_steps}")
                    break

        self.global_step = gradient_step_idx

    def _forward_pass(
        self,
        vla: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        device_id: int,
        num_patches: int,
        use_film: bool,
        use_proprio: bool,
    ) -> tuple:
        """Run forward pass and compute L1 loss."""
        # Get ground truth actions
        gt_actions = batch["actions"].to(device_id).to(torch.bfloat16)

        # Prepare proprio embedding if enabled
        proprio_embedding = None
        if use_proprio and proprio_projector is not None and "proprio" in batch:
            proprio = batch["proprio"].to(device_id).to(torch.bfloat16)
            proprio_embedding = proprio_projector(proprio)

        # VLA forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
            )

        # Get last hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        # Extract hidden states for action tokens
        # We need to identify which positions correspond to actions
        # In OFT, we use the hidden states after the vision patches
        batch_size = batch["input_ids"].shape[0]
        action_horizon = self.cfg.action_horizon
        action_dim = self.cfg.action_dim

        # Get text hidden states (after vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:]

        # For OFT, we take the first action_horizon * action_dim tokens after the prompt
        # as the action representation
        num_action_tokens = action_horizon * action_dim
        if text_hidden_states.shape[1] >= num_action_tokens:
            actions_hidden_states = text_hidden_states[:, :num_action_tokens]
        else:
            # Pad if not enough tokens
            pad_size = num_action_tokens - text_hidden_states.shape[1]
            actions_hidden_states = torch.cat([
                text_hidden_states,
                text_hidden_states[:, -1:].expand(-1, pad_size, -1),
            ], dim=1)

        actions_hidden_states = actions_hidden_states.to(torch.bfloat16)

        # Predict actions through action head
        # Get the module if wrapped in DDP
        if hasattr(action_head, 'module'):
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
        else:
            predicted_actions = action_head.predict_action(actions_hidden_states)

        # Compute L1 loss
        loss = torch.nn.L1Loss()(predicted_actions, gt_actions)

        # Compute detailed metrics
        with torch.no_grad():
            # Current action (first timestep) L1
            curr_action_l1 = torch.nn.L1Loss()(
                predicted_actions[:, 0], gt_actions[:, 0]
            ).item()

            # Full chunk L1
            l1_loss = loss.item()

        metrics = {
            "l1_loss": l1_loss,
            "curr_action_l1": curr_action_l1,
        }

        return loss, metrics

    def _run_overfit_check(
        self,
        vla: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module],
        overfit_dataloader: DataLoader,
        device_id: int,
        num_patches: int,
        distributed_state,
    ) -> Dict[str, float]:
        """Run overfitting check on held-out steps."""
        vla.eval()
        action_head.eval()
        if proprio_projector:
            proprio_projector.eval()

        losses = []
        l1_losses = []
        curr_action_l1s = []

        with torch.no_grad():
            for i, batch in enumerate(overfit_dataloader):
                if i >= self.cfg.overfit_check_steps:
                    break

                loss, metrics = self._forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    batch=batch,
                    device_id=device_id,
                    num_patches=num_patches,
                    use_film=self.cfg.use_film,
                    use_proprio=self.cfg.use_proprio,
                )

                losses.append(loss.item())
                l1_losses.append(metrics["l1_loss"])
                curr_action_l1s.append(metrics["curr_action_l1"])

        return {
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "l1_loss": sum(l1_losses) / len(l1_losses) if l1_losses else 0.0,
            "curr_action_l1": sum(curr_action_l1s) / len(curr_action_l1s) if curr_action_l1s else 0.0,
        }

    def _save_checkpoint(
        self,
        run_dir: Path,
        step: int,
        vla: nn.Module,
        action_head: nn.Module,
        proprio_projector: Optional[nn.Module],
        processor,
        dataset_statistics: Dict,
        distributed_state,
        is_distributed: bool,
    ) -> None:
        """Save training checkpoint."""
        if distributed_state.is_main_process:
            logger.info(f"Saving checkpoint at step {step}")

            checkpoint_dir = run_dir / f"checkpoint-{step}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Get unwrapped modules
            unwrapped_vla = vla.module if is_distributed else vla
            unwrapped_action_head = action_head.module if is_distributed else action_head

            # Save processor
            processor.save_pretrained(checkpoint_dir)

            # Save dataset statistics
            save_dataset_statistics(dataset_statistics, checkpoint_dir)

            # Save LoRA adapters
            if self.cfg.use_lora:
                lora_dir = checkpoint_dir / "lora_adapters"
                os.makedirs(lora_dir, exist_ok=True)
                unwrapped_vla.save_pretrained(lora_dir)
                logger.info(f"Saved LoRA adapters at {lora_dir}")

            # Save action head
            action_head_path = checkpoint_dir / "action_head.pt"
            torch.save(unwrapped_action_head.state_dict(), action_head_path)

            # Save proprio projector if used
            if proprio_projector is not None:
                unwrapped_proprio = (
                    proprio_projector.module if is_distributed else proprio_projector
                )
                proprio_path = checkpoint_dir / "proprio_projector.pt"
                torch.save(unwrapped_proprio.state_dict(), proprio_path)

            # Save FiLM weights if used
            if self.cfg.use_film:
                vision_backbone = unwrapped_vla.vision_backbone
                if hasattr(unwrapped_vla, 'model'):
                    vision_backbone = unwrapped_vla.model.vision_backbone
                film_path = checkpoint_dir / "vision_backbone_film.pt"
                torch.save(vision_backbone.state_dict(), film_path)

            # Save config
            config_dict = {
                "action_dim": self.cfg.action_dim,
                "action_horizon": self.cfg.action_horizon,
                "use_film": self.cfg.use_film,
                "use_proprio": self.cfg.use_proprio,
                "proprio_dim": self.cfg.proprio_dim,
                "vla_path": self.cfg.vla_path,
            }
            import json
            with open(checkpoint_dir / "oft_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Checkpoint saved at {checkpoint_dir}")

        if is_distributed:
            dist.barrier()


class OpenVLAOFTBackend(VLABackend):
    """
    OpenVLA-OFT backend implementation.

    Provides VLABackend interface for OpenVLA-OFT training and inference.
    """

    def __init__(self, config: OpenVLAOFTConfig):
        """Initialize backend with configuration."""
        super().__init__(config)
        self.vla_config = config
        self.trainer = None
        self._action_dim = config.openvla_oft.action_dim
        self._action_horizon = config.openvla_oft.action_horizon
        self._image_size = config.openvla_oft.image_size

    def _create_finetune_config(self) -> OpenVLAOFTFinetuneConfig:
        """Convert OpenVLAOFTConfig to OpenVLAOFTFinetuneConfig."""
        return OpenVLAOFTFinetuneConfig(
            vla_path=self.vla_config.openvla_oft.model_id,
            action_dim=self.vla_config.openvla_oft.action_dim,
            action_horizon=self.vla_config.openvla_oft.action_horizon,
            use_film=self.vla_config.openvla_oft.film.enabled,
            use_proprio=self.vla_config.openvla_oft.proprio.enabled,
            use_multi_image=self.vla_config.openvla_oft.multi_image.enabled,
            num_images=self.vla_config.openvla_oft.multi_image.num_images,
            proprio_dim=self.vla_config.openvla_oft.proprio.proprio_dim,
            data_root_dir=self.vla_config.data.data_root,
            dataset_name=getattr(self.vla_config.data, 'dataset_name', 'crane_x7'),
            run_root_dir=self.vla_config.output_dir,
            batch_size=self.vla_config.training.batch_size,
            max_steps=self.vla_config.training.max_steps,
            save_steps=self.vla_config.training.save_interval,
            learning_rate=self.vla_config.training.learning_rate,
            weight_decay=self.vla_config.training.weight_decay,
            grad_accumulation_steps=self.vla_config.training.gradient_accumulation_steps,
            image_aug=self.vla_config.openvla_oft.image_aug,
            gradient_checkpointing=self.vla_config.training.gradient_checkpointing,
            overfit_split_ratio=self.vla_config.overfitting.overfit_split_ratio,
            overfit_check_interval=self.vla_config.overfitting.overfit_check_interval,
            overfit_check_steps=self.vla_config.overfitting.overfit_check_steps,
            use_lora=self.vla_config.openvla_oft.use_lora,
            lora_rank=self.vla_config.openvla_oft.lora_rank,
            lora_alpha=self.vla_config.openvla_oft.lora_alpha,
            lora_dropout=self.vla_config.openvla_oft.lora_dropout,
            use_quantization=self.vla_config.openvla_oft.use_quantization,
            wandb_project=self.vla_config.wandb_project,
            wandb_entity=self.vla_config.wandb_entity,
            run_id_note=self.vla_config.experiment_name,
            skip_merge_on_save=self.vla_config.openvla_oft.skip_merge_on_save,
        )

    def train(self) -> Dict[str, Any]:
        """Execute training loop."""
        cfg = self._create_finetune_config()
        self.trainer = OpenVLAOFTTrainer(cfg)
        self.trainer.train()

        return {
            'final_step': self.trainer.global_step,
            'final_epoch': self.trainer.epoch,
            'run_root_dir': str(cfg.run_root_dir),
        }

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        raise NotImplementedError("Evaluation not yet implemented for OpenVLA-OFT")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform inference to predict action chunk.

        Args:
            observation: Dict with 'state' and 'image' keys
            language_instruction: Task instruction

        Returns:
            Action chunk: (action_horizon, action_dim) or (action_dim,) for single action
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # TODO: Implement full inference pipeline
        raise NotImplementedError("Inference not yet implemented for OpenVLA-OFT")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        raise NotImplementedError("Use trainer's checkpoint saving")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        # TODO: Implement checkpoint loading for inference
        raise NotImplementedError("Checkpoint loading not yet implemented")

    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        """Get action horizon (number of predicted actions)."""
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        """Get expected image size."""
        return self._image_size
