# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pi0/Pi0.5 Backend for CRANE-X7.

This module implements the VLABackend interface for Pi0 and Pi0.5 models.
Uses PaliGemma + Expert Gemma architecture with flow matching.
"""

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from crane_x7_vla.backends.pi0.config import Pi0Config
from crane_x7_vla.backends.pi0.dataset import CraneX7Pi0Dataset, collate_pi0_batch
from crane_x7_vla.backends.pi0.model import Pi0Model, Pi0ModelConfig
from crane_x7_vla.core.base import VLABackend
from crane_x7_vla.core.transforms.action_transforms import ActionNormalizer, ActionPadder


logger = logging.getLogger(__name__)


@dataclass
class Pi0TrainerConfig:
    """Configuration for Pi0 training."""

    # Model
    model_type: str = "pi0"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pretrained_checkpoint: str | None = None
    use_pretrained: bool = True
    paligemma_pretrained_id: str = "google/paligemma-3b-pt-224"
    openpi_checkpoint: str | None = None  # e.g., "pi0_base", "pi05_base"

    # Data
    data_root_dir: str = ""
    output_dir: str = ""
    dataset_name: str = "crane_x7"

    # Training
    batch_size: int = 8
    max_steps: int = 100_000
    save_steps: int = 5000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1

    # Action chunk
    action_horizon: int = 50
    action_dim: int = 32
    crane_x7_action_dim: int = 8

    # Token
    max_token_len: int = 48

    # Flow matching
    num_denoise_steps: int = 10

    # Precision
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True

    # Normalization
    normalize_actions: bool = True
    normalization_mode: str = "quantile"

    # Image
    image_size: tuple[int, int] = (224, 224)
    image_aug: bool = True

    # Cameras
    camera_names: list[str] | None = None

    # Freeze settings
    freeze_vlm: bool = True
    freeze_action_expert: bool = False

    # LoRA settings (OpenPI準拠: スケーリング=1.0)
    use_lora: bool = False
    # VLM (PaliGemma) LoRA settings
    vlm_lora_rank: int = 16
    vlm_lora_alpha: int = 16  # scaling = 1.0
    # Expert (Gemma-300M) LoRA settings
    expert_lora_rank: int = 32
    expert_lora_alpha: int = 32  # scaling = 1.0
    lora_dropout: float = 0.05
    lora_rslora: bool = False  # Rank-Stabilized LoRA
    lora_target_modules: list[str] | None = None
    lora_apply_to_vlm: bool = False
    lora_apply_to_expert: bool = True
    lora_skip_merge_on_save: bool = True

    # EMA settings
    ema_decay: float | None = None  # None to disable, OpenPI default: 0.99

    # Overfitting detection
    overfit_split_ratio: float = 0.1
    overfit_check_interval: int = 500
    overfit_check_steps: int = 50

    # Logging
    wandb_project: str = "crane-x7-pi0"
    wandb_entity: str | None = None
    log_interval: int = 10

    # Prompt
    default_prompt: str = "manipulate objects"


class Pi0Trainer:
    """Trainer for Pi0/Pi0.5 models."""

    def __init__(self, cfg: Pi0TrainerConfig):
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    def train(self) -> dict[str, Any]:
        """Execute training loop."""
        cfg = self.cfg

        logger.info("=" * 60)
        logger.info("  CRANE-X7 Pi0/Pi0.5 Training")
        logger.info(f"  Model Type: {cfg.model_type}")
        logger.info(f"  PaliGemma: {cfg.paligemma_variant}")
        logger.info(f"  Action Expert: {cfg.action_expert_variant}")
        logger.info(f"  Batch Size: {cfg.batch_size}")
        logger.info("=" * 60)

        # GPU setup
        assert torch.cuda.is_available(), "Training requires GPU"
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

        is_main_process = distributed_state.is_main_process
        is_distributed = distributed_state.num_processes > 1

        # Precision
        dtype = torch.bfloat16 if cfg.precision == "bfloat16" else torch.float32

        # Create model
        logger.info("Creating Pi0 model...")
        model_config = Pi0ModelConfig(
            pi05=cfg.model_type == "pi0.5",
            paligemma_variant=cfg.paligemma_variant,
            action_expert_variant=cfg.action_expert_variant,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            max_token_len=cfg.max_token_len,
            dtype=cfg.precision,
            use_pretrained=cfg.use_pretrained,
            paligemma_pretrained_id=cfg.paligemma_pretrained_id,
            openpi_checkpoint=cfg.openpi_checkpoint,
        )
        model = Pi0Model(model_config)
        model = model.to(device_id)

        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply LoRA if enabled
        self.use_lora = cfg.use_lora
        if cfg.use_lora:
            # Default target modules for Gemma architecture
            target_modules = cfg.lora_target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

            # Apply LoRA to Action Expert (gemma_expert.model)
            # Note: gemma_expert.model.embed_tokens is set to None in gemma_pytorch.py,
            # which causes get_input_embeddings() to return None and breaks PEFT's
            # prepare_model_for_gradient_checkpointing. We temporarily set a dummy embedding
            # to work around this, then restore None after PEFT wrapping.
            if cfg.lora_apply_to_expert:
                expert_lora_config = LoraConfig(
                    r=cfg.expert_lora_rank,
                    lora_alpha=cfg.expert_lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=target_modules,
                    init_lora_weights="gaussian",
                    bias="none",
                    use_rslora=cfg.lora_rslora,
                )
                gemma_model = model.paligemma_with_expert.gemma_expert.model
                # Temporarily set dummy embed_tokens to satisfy PEFT's gradient checkpointing setup
                hidden_size = gemma_model.config.hidden_size
                model_device = next(gemma_model.parameters()).device
                gemma_model.embed_tokens = torch.nn.Embedding(1, hidden_size, device=model_device)
                # Apply LoRA
                model.paligemma_with_expert.gemma_expert.model = get_peft_model(gemma_model, expert_lora_config)
                # Restore embed_tokens to None (access through base_model.model for PeftModel)
                model.paligemma_with_expert.gemma_expert.model.base_model.model.embed_tokens = None
                model.paligemma_with_expert.gemma_expert.model.print_trainable_parameters()
                logger.info(
                    f"Applied LoRA to Action Expert (rank={cfg.expert_lora_rank}, alpha={cfg.expert_lora_alpha})"
                )

            # Apply LoRA to VLM (paligemma language model) if specified
            if cfg.lora_apply_to_vlm:
                vlm_lora_config = LoraConfig(
                    r=cfg.vlm_lora_rank,
                    lora_alpha=cfg.vlm_lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=target_modules,
                    init_lora_weights="gaussian",
                    bias="none",
                    use_rslora=cfg.lora_rslora,
                )
                vlm_model = model.paligemma_with_expert.paligemma.language_model
                model.paligemma_with_expert.paligemma.language_model = get_peft_model(vlm_model, vlm_lora_config)
                # Check if PeftModel was returned (has print_trainable_parameters method)
                if hasattr(model.paligemma_with_expert.paligemma.language_model, "print_trainable_parameters"):
                    model.paligemma_with_expert.paligemma.language_model.print_trainable_parameters()
                logger.info(f"Applied LoRA to PaliGemma VLM (rank={cfg.vlm_lora_rank}, alpha={cfg.vlm_lora_alpha})")
        else:
            # Freeze layers if specified (only when not using LoRA)
            if cfg.freeze_vlm:
                for _name, param in model.paligemma_with_expert.paligemma.named_parameters():
                    param.requires_grad = False
                logger.info("Froze PaliGemma VLM weights")

            if cfg.freeze_action_expert:
                for _name, param in model.paligemma_with_expert.gemma_expert.named_parameters():
                    param.requires_grad = False
                logger.info("Froze Action Expert weights")

        # EMA initialization
        self.ema_model = None
        self.ema_decay = cfg.ema_decay
        if cfg.ema_decay is not None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            logger.info(f"Initialized EMA model with decay={cfg.ema_decay}")

        # DDP
        if is_distributed:
            model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
            dist.barrier()

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        # Optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.max_steps,
        )

        # Dataset
        camera_names = cfg.camera_names or ["base_0_rgb"]
        train_dataset = CraneX7Pi0Dataset(
            data_root_dir=Path(cfg.data_root_dir),
            action_horizon=cfg.action_horizon,
            source_action_dim=cfg.crane_x7_action_dim,
            target_action_dim=cfg.action_dim,
            max_token_len=cfg.max_token_len,
            resize_resolution=cfg.image_size,
            train=True,
            image_aug=cfg.image_aug,
            normalize_actions=cfg.normalize_actions,
            normalization_mode=cfg.normalization_mode,
            default_prompt=cfg.default_prompt,
            camera_names=camera_names,
            discrete_state_input=cfg.model_type == "pi0.5",
            overfit_split_ratio=cfg.overfit_split_ratio,
            split="train",
            rank=device_id if is_distributed else 0,
            world_size=distributed_state.num_processes,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_pi0_batch,
            num_workers=0,
            pin_memory=True,
        )

        # Overfit dataset
        overfit_dataloader = None
        if cfg.overfit_split_ratio > 0:
            overfit_dataset = CraneX7Pi0Dataset(
                data_root_dir=Path(cfg.data_root_dir),
                action_horizon=cfg.action_horizon,
                source_action_dim=cfg.crane_x7_action_dim,
                target_action_dim=cfg.action_dim,
                max_token_len=cfg.max_token_len,
                resize_resolution=cfg.image_size,
                train=False,
                image_aug=False,
                normalize_actions=cfg.normalize_actions,
                normalization_mode=cfg.normalization_mode,
                default_prompt=cfg.default_prompt,
                camera_names=camera_names,
                discrete_state_input=cfg.model_type == "pi0.5",
                overfit_split_ratio=cfg.overfit_split_ratio,
                split="overfit",
                rank=device_id if is_distributed else 0,
                world_size=distributed_state.num_processes,
            )
            overfit_dataloader = DataLoader(
                overfit_dataset,
                batch_size=cfg.batch_size,
                collate_fn=collate_pi0_batch,
                num_workers=0,
            )

        # Output directory
        exp_id = f"{cfg.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if cfg.output_dir:
            run_dir = Path(cfg.output_dir) / f"crane_x7_{cfg.model_type}_{exp_id}"
        else:
            run_dir = Path(cfg.data_root_dir).parent / "outputs" / f"crane_x7_{cfg.model_type}_{exp_id}"
        if is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        use_wandb = False
        if is_main_process:
            try:
                import wandb

                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    name=f"{cfg.model_type}-{exp_id}",
                    config=vars(cfg),
                )
                use_wandb = True
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")

        # Training loop
        model.train()
        data_iter = iter(train_dataloader)
        loss_accumulator = 0.0
        grad_step = 0

        logger.info(f"Starting training for {cfg.max_steps} steps...")

        for step in range(cfg.max_steps * cfg.grad_accumulation_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
                self.epoch += 1

            # Move to device
            images = [batch["images"][cam].to(device_id) for cam in camera_names]
            img_masks = [batch["image_masks"][cam].to(device_id) for cam in camera_names]
            lang_tokens = batch["lang_tokens"].to(device_id)
            lang_masks = batch["lang_masks"].to(device_id)
            state = batch["state"].to(device_id)
            actions = batch["actions"].to(device_id)

            # Forward pass
            with torch.amp.autocast("cuda", dtype=dtype):
                model_ref = model.module if is_distributed else model
                loss = model_ref.forward(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=state,
                    actions=actions,
                )
                loss = loss.mean()

            # Backward
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            loss_accumulator += loss.item()

            # Gradient step
            if (step + 1) % cfg.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                if self.ema_model is not None:
                    model_ref = model.module if is_distributed else model
                    with torch.no_grad():
                        for ema_param, param in zip(self.ema_model.parameters(), model_ref.parameters()):
                            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

                grad_step += 1
                self.global_step = grad_step

                # Logging
                if grad_step % cfg.log_interval == 0 and is_main_process:
                    avg_loss = loss_accumulator / (cfg.log_interval * cfg.grad_accumulation_steps)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(f"Step {grad_step}/{cfg.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                    if use_wandb:
                        wandb.log(
                            {"train/loss": avg_loss, "train/lr": lr, "train/epoch": self.epoch},
                            step=grad_step,
                        )

                    loss_accumulator = 0.0

                # Overfitting check
                if overfit_dataloader is not None and grad_step % cfg.overfit_check_interval == 0:
                    overfit_loss = self._run_overfit_check(
                        model, overfit_dataloader, device_id, dtype, is_distributed, camera_names
                    )
                    if is_main_process:
                        logger.info(f"  Overfit Loss: {overfit_loss:.4f}")
                        if use_wandb:
                            wandb.log({"eval/overfit_loss": overfit_loss}, step=grad_step)
                    model.train()

                # Save checkpoint
                if grad_step % cfg.save_steps == 0 and is_main_process:
                    self._save_checkpoint(
                        model, optimizer, scheduler, run_dir / f"checkpoint-{grad_step}", is_distributed
                    )

        # Final save
        if is_main_process:
            self._save_checkpoint(model, optimizer, scheduler, run_dir / "checkpoint-final", is_distributed)
            if use_wandb:
                wandb.finish()

        return {
            "final_step": self.global_step,
            "final_epoch": self.epoch,
            "run_dir": str(run_dir),
        }

    def _run_overfit_check(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device_id: int,
        dtype: torch.dtype,
        is_distributed: bool,
        camera_names: list[str],
    ) -> float:
        """Run overfitting check on held-out steps."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.cfg.overfit_check_steps:
                    break

                images = [batch["images"][cam].to(device_id) for cam in camera_names]
                img_masks = [batch["image_masks"][cam].to(device_id) for cam in camera_names]
                lang_tokens = batch["lang_tokens"].to(device_id)
                lang_masks = batch["lang_masks"].to(device_id)
                state = batch["state"].to(device_id)
                actions = batch["actions"].to(device_id)

                with torch.amp.autocast("cuda", dtype=dtype):
                    model_ref = model.module if is_distributed else model
                    loss = model_ref.forward(
                        images=images,
                        img_masks=img_masks,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=state,
                        actions=actions,
                    )
                    loss = loss.mean()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: AdamW,
        scheduler: Any,
        path: Path,
        is_distributed: bool,
    ) -> None:
        """Save training checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if is_distributed else model

        # Save full checkpoint
        checkpoint_data = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": vars(self.cfg),
            "use_lora": self.use_lora,
        }

        # Save EMA model if enabled
        if self.ema_model is not None:
            checkpoint_data["ema_model_state_dict"] = self.ema_model.state_dict()
            logger.info("Saved EMA model state")

        torch.save(checkpoint_data, path / "checkpoint.pt")

        # Save LoRA adapters separately if using LoRA
        if self.use_lora:
            lora_save_dir = path / "lora_adapters"
            lora_save_dir.mkdir(parents=True, exist_ok=True)

            # Save action expert LoRA adapters
            # Note: LoRA is applied to gemma_expert.model (GemmaModel), not gemma_expert (GemmaForCausalLM)
            if self.cfg.lora_apply_to_expert:
                expert_lora_dir = lora_save_dir / "gemma_expert"
                expert_lora_dir.mkdir(parents=True, exist_ok=True)
                model_to_save.paligemma_with_expert.gemma_expert.model.save_pretrained(expert_lora_dir)
                logger.info(f"Saved Action Expert LoRA adapters to {expert_lora_dir}")

            # Save VLM LoRA adapters if applied
            if self.cfg.lora_apply_to_vlm:
                vlm_lora_dir = lora_save_dir / "paligemma_lm"
                vlm_lora_dir.mkdir(parents=True, exist_ok=True)
                model_to_save.paligemma_with_expert.paligemma.language_model.save_pretrained(vlm_lora_dir)
                logger.info(f"Saved PaliGemma LM LoRA adapters to {vlm_lora_dir}")

        with (path / "config.json").open("w") as f:
            json.dump(vars(self.cfg), f, indent=2, default=str)

        # Copy dataset_statistics.json for inference
        data_root = Path(self.cfg.data_root_dir)
        stats_file = data_root / "dataset_statistics.json"
        if stats_file.exists():
            shutil.copy(stats_file, path / "dataset_statistics.json")
            logger.info("Copied dataset_statistics.json to checkpoint")

        logger.info(f"Saved checkpoint to {path}")


class Pi0Backend(VLABackend):
    """
    Pi0/Pi0.5 backend for CRANE-X7.

    Implements VLABackend interface for training and inference
    with Pi0 and Pi0.5 models.
    """

    def __init__(self, config: Pi0Config):
        super().__init__(config)
        self._action_dim = 8  # CRANE-X7 native
        self._action_horizon = config.pi0.action_horizon
        self._image_size = config.pi0.image_size
        self._target_action_dim = config.pi0.action_dim
        self.model: Pi0Model | None = None

        # Transform utilities
        self.action_padder = ActionPadder(self._action_dim, self._target_action_dim)
        self.action_normalizer = ActionNormalizer(mode=config.pi0.normalization_mode)

    def _create_trainer_config(self) -> Pi0TrainerConfig:
        """Create trainer config from unified config."""
        cfg = self.config
        pi0_cfg = cfg.pi0

        # Use Pi0-specific LoRA config (OpenPI準拠)
        use_lora = pi0_cfg.use_lora
        lora_dropout = pi0_cfg.lora_dropout
        lora_target_modules = None  # Use default Gemma modules
        lora_skip_merge = True

        return Pi0TrainerConfig(
            model_type=pi0_cfg.model_type,
            paligemma_variant=pi0_cfg.paligemma_variant,
            action_expert_variant=pi0_cfg.action_expert_variant,
            pretrained_checkpoint=pi0_cfg.pretrained_checkpoint,
            use_pretrained=pi0_cfg.use_pretrained,
            paligemma_pretrained_id=pi0_cfg.paligemma_pretrained_id,
            openpi_checkpoint=pi0_cfg.openpi_checkpoint,
            data_root_dir=str(cfg.data.data_root),
            output_dir=str(cfg.output_dir) if cfg.output_dir else "",
            dataset_name=cfg.experiment_name,
            batch_size=cfg.training.batch_size,
            max_steps=cfg.training.max_steps,
            save_steps=cfg.training.save_interval,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            grad_accumulation_steps=cfg.training.gradient_accumulation_steps,
            action_horizon=pi0_cfg.action_horizon,
            action_dim=pi0_cfg.action_dim,
            crane_x7_action_dim=8,
            max_token_len=pi0_cfg.max_token_len,
            num_denoise_steps=pi0_cfg.num_denoise_steps,
            precision=pi0_cfg.precision,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            normalize_actions=pi0_cfg.normalize_actions,
            normalization_mode=pi0_cfg.normalization_mode,
            image_size=pi0_cfg.image_size,
            image_aug=True,
            camera_names=pi0_cfg.camera_names,
            freeze_vlm=pi0_cfg.freeze_vlm,
            freeze_action_expert=pi0_cfg.freeze_action_expert,
            # LoRA settings (OpenPI準拠: VLMとExpert別設定)
            use_lora=use_lora,
            vlm_lora_rank=pi0_cfg.vlm_lora_rank,
            vlm_lora_alpha=pi0_cfg.vlm_lora_alpha,
            expert_lora_rank=pi0_cfg.expert_lora_rank,
            expert_lora_alpha=pi0_cfg.expert_lora_alpha,
            lora_dropout=lora_dropout,
            lora_rslora=pi0_cfg.lora_rslora,
            lora_target_modules=lora_target_modules,
            lora_apply_to_vlm=pi0_cfg.lora_apply_to_vlm,
            lora_apply_to_expert=pi0_cfg.lora_apply_to_expert,
            lora_skip_merge_on_save=lora_skip_merge,
            # EMA settings
            ema_decay=pi0_cfg.ema_decay,
            # Other settings
            overfit_split_ratio=cfg.overfitting.overfit_split_ratio,
            overfit_check_interval=cfg.overfitting.overfit_check_interval,
            overfit_check_steps=cfg.overfitting.overfit_check_steps,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            log_interval=cfg.training.log_interval,
            default_prompt=pi0_cfg.default_prompt,
        )

    def train(self) -> dict[str, Any]:
        """Execute training."""
        cfg = self._create_trainer_config()
        trainer = Pi0Trainer(cfg)
        return trainer.train()

    def evaluate(
        self, checkpoint_path: str | Path | None = None, test_data_path: str | Path | None = None
    ) -> dict[str, float]:
        """Evaluate model on test data."""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint first.")

        data_path = Path(test_data_path) if test_data_path else self.config.data.data_root
        camera_names = self.config.pi0.camera_names or ["base_0_rgb"]

        test_dataset = CraneX7Pi0Dataset(
            data_root_dir=data_path,
            action_horizon=self._action_horizon,
            source_action_dim=self._action_dim,
            target_action_dim=self._target_action_dim,
            max_token_len=self.config.pi0.max_token_len,
            resize_resolution=self._image_size,
            train=False,
            image_aug=False,
            normalize_actions=self.config.pi0.normalize_actions,
            normalization_mode=self.config.pi0.normalization_mode,
            default_prompt=self.config.pi0.default_prompt,
            camera_names=camera_names,
            split="test",
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            collate_fn=collate_pi0_batch,
            num_workers=0,
        )

        device = next(self.model.parameters()).device
        dtype = torch.bfloat16 if self.config.pi0.precision == "bfloat16" else torch.float32

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        logger.info("Starting evaluation...")

        with torch.no_grad():
            for batch in test_dataloader:
                images = [batch["images"][cam].to(device) for cam in camera_names]
                img_masks = [batch["image_masks"][cam].to(device) for cam in camera_names]
                lang_tokens = batch["lang_tokens"].to(device)
                lang_masks = batch["lang_masks"].to(device)
                state = batch["state"].to(device)
                actions = batch["actions"].to(device)

                with torch.amp.autocast("cuda", dtype=dtype):
                    loss = self.model.forward(
                        images=images,
                        img_masks=img_masks,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=state,
                        actions=actions,
                    )
                    loss = loss.mean()

                    pred_actions = self.model.sample_actions(
                        images=images,
                        img_masks=img_masks,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=state,
                        num_steps=self.config.pi0.num_denoise_steps,
                    )
                    mse = F.mse_loss(pred_actions, actions)

                total_loss += loss.item()
                total_mse += mse.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        num_samples = num_batches * self.config.training.batch_size

        logger.info(f"Evaluation complete: loss={avg_loss:.4f}, mse={avg_mse:.4f}, samples={num_samples}")

        return {
            "eval/loss": avg_loss,
            "eval/action_mse": avg_mse,
            "eval/num_samples": float(num_samples),
        }

    def infer(self, observation: dict[str, np.ndarray], language_instruction: str | None = None) -> np.ndarray:
        """Perform inference on a single observation."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint first.")

        device = next(self.model.parameters()).device

        # Prepare state
        state = observation["state"]
        if state.shape[-1] == self._action_dim:
            state = self.action_padder.pad(state)

        # Prepare image
        image = observation["image"]
        if image.dtype == np.uint8:
            image = (image.astype(np.float32) / 127.5) - 1.0
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)  # HWC -> CHW

        # Tokenize prompt
        prompt = language_instruction or self.config.pi0.default_prompt
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", trust_remote_code=True)
        encoding = tokenizer(
            prompt,
            max_length=self.config.pi0.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert to tensors
        images = [torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)]
        img_masks = [torch.tensor([True]).to(device)]
        lang_tokens = encoding["input_ids"].to(device)
        lang_masks = encoding["attention_mask"].bool().to(device)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Sample actions
        self.model.eval()
        with torch.no_grad():
            action_chunk = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state_tensor,
                num_steps=self.config.pi0.num_denoise_steps,
            )

        # Get first action and truncate
        action = action_chunk[0, 0, : self._action_dim].cpu().numpy()

        # Denormalize
        if self.action_normalizer.stats:
            action = self.action_normalizer.denormalize(action)

        return action

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({"model_state_dict": self.model.state_dict()}, path / "model.pt")

        with (path / "config.json").open("w") as f:
            json.dump(vars(self.config.pi0), f, indent=2, default=str)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        path = Path(path)

        checkpoint_path = path / "checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = path / "model.pt"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model (use_pretrained=False since we're loading from checkpoint)
        model_config = Pi0ModelConfig(
            pi05=self.config.pi0.pi05,
            paligemma_variant=self.config.pi0.paligemma_variant,
            action_expert_variant=self.config.pi0.action_expert_variant,
            action_dim=self._target_action_dim,
            action_horizon=self._action_horizon,
            max_token_len=self.config.pi0.max_token_len,
            dtype=self.config.pi0.precision,
            use_pretrained=False,  # Skip pretrained loading when restoring from checkpoint
        )
        self.model = Pi0Model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        logger.info(f"Loaded checkpoint from {path}")

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        return self._image_size
