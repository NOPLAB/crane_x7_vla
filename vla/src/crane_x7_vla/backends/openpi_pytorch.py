# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenPI PyTorch Backend for CRANE-X7.

This module implements the OpenPI backend using PyTorch and HuggingFace Pi0 model.
It uses flow matching for action chunk prediction, following the OpenVLA training pattern.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from crane_x7_vla.utils.logging import get_logger

logger = get_logger(__name__)
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import PartialState
from transformers import (
    AutoModel,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openpi_pytorch_config import (
    OpenPIPytorchConfig,
    OpenPIPytorchSpecificConfig,
)
from crane_x7_vla.data.openpi_pytorch_dataset import (
    CraneX7ActionChunkDataset,
    collate_action_chunk_batch,
)
from crane_x7_vla.transforms.action_transforms import ActionNormalizer, ActionPadder


@dataclass
class CraneX7Pi0FinetuneConfig:
    """Configuration for Pi0 finetuning on CRANE-X7 data."""

    # Model
    model_name: str = "lerobot/pi0_base"
    pretrained_checkpoint: Optional[str] = None

    # Data
    data_root_dir: str = ""
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

    # Flow matching
    num_denoise_steps: int = 10

    # Precision
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False

    # Normalization
    normalize_actions: bool = True
    normalization_mode: str = "quantile"

    # Image
    image_size: Tuple[int, int] = (224, 224)
    image_aug: bool = True

    # Overfitting detection
    overfit_split_ratio: float = 0.1
    overfit_check_interval: int = 500
    overfit_check_steps: int = 50

    # Logging
    wandb_project: str = "crane-x7-pi0"
    wandb_entity: Optional[str] = None
    log_interval: int = 10

    # Prompt
    default_prompt: str = "manipulate objects"


class FlowMatchingModule(torch.nn.Module):
    """
    Flow Matching wrapper for Pi0 model.

    Implements flow matching training and inference for action prediction.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        action_dim: int = 32,
        action_horizon: int = 50,
    ):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Action head for flow matching
        # This projects the model output to velocity field
        hidden_dim = getattr(backbone.config, "hidden_size", 2048)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, action_horizon * action_dim),
        )

        # Time embedding
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, hidden_dim),
        )

    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for flow matching.

        Args:
            observation: Dict with 'state' and 'image' tensors
            noisy_actions: Noisy action chunk (B, horizon, action_dim)
            timestep: Flow matching timestep (B,)

        Returns:
            Predicted velocity field (B, horizon, action_dim)
        """
        B = noisy_actions.shape[0]
        device = noisy_actions.device

        # Time embedding
        t_embed = self.time_embed(timestep.unsqueeze(-1))  # (B, hidden_dim)

        # Flatten noisy actions for conditioning
        noisy_flat = noisy_actions.reshape(B, -1)  # (B, horizon * action_dim)

        # Get backbone features
        # Note: This is a simplified version. Real Pi0 uses more complex encoding.
        image = observation["image"]["base_0_rgb"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW

        # For now, use a simple encoding
        # In production, this should use the full Pi0 architecture
        with torch.amp.autocast("cuda", enabled=False):
            # Simple pooling of image features
            features = self.backbone.get_image_features(image.float())
            if features.dim() == 3:
                features = features.mean(dim=1)  # (B, hidden_dim)

        # Add time embedding
        features = features + t_embed

        # Predict velocity
        velocity = self.action_head(features)
        velocity = velocity.reshape(B, self.action_horizon, self.action_dim)

        return velocity

    def compute_loss(
        self,
        observation: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            observation: Dict with 'state' and 'image' tensors
            actions: Target action chunk (B, horizon, action_dim)

        Returns:
            Flow matching loss
        """
        B = actions.shape[0]
        device = actions.device

        # Sample random noise
        noise = torch.randn_like(actions)

        # Sample random timesteps
        t = torch.rand(B, device=device)

        # Interpolate between noise and actions
        # x_t = (1 - t) * noise + t * actions
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * noise + t_expand * actions

        # Predict velocity
        v_pred = self.forward(observation, x_t, t)

        # Target velocity: v = actions - noise
        v_target = actions - noise

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def sample_actions(
        self,
        observation: Dict[str, torch.Tensor],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Sample actions using flow matching ODE integration.

        Args:
            observation: Dict with 'state' and 'image' tensors
            num_steps: Number of integration steps

        Returns:
            Sampled action chunk (B, horizon, action_dim)
        """
        B = observation["image"]["base_0_rgb"].shape[0]
        device = observation["image"]["base_0_rgb"].device

        # Start from noise
        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        # Euler integration
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((B,), step / num_steps, device=device)
            v = self.forward(observation, x, t)
            x = x + v * dt

        return x


class CraneX7Pi0Trainer:
    """
    Trainer for Pi0 on CRANE-X7 data.

    Follows OpenVLA training pattern with flow matching loss.
    """

    def __init__(self, cfg: CraneX7Pi0FinetuneConfig):
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    def train(self) -> Dict[str, Any]:
        """Execute training loop."""
        cfg = self.cfg
        logger.info("=" * 60)
        logger.info("  CRANE-X7 Pi0 PyTorch Training")
        logger.info(f"  Model: {cfg.model_name}")
        logger.info(f"  Action Horizon: {cfg.action_horizon}")
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

        # Load model
        logger.info(f"Loading model: {cfg.model_name}")
        try:
            backbone = AutoModel.from_pretrained(
                cfg.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load {cfg.model_name}: {e}")
            logger.info("Using fallback vision model...")
            from transformers import CLIPModel
            backbone = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=dtype,
            )

        # Create flow matching module
        model = FlowMatchingModule(
            backbone=backbone,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
        )
        model = model.to(device_id)

        if cfg.gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()

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
        train_dataset = CraneX7ActionChunkDataset(
            data_root_dir=Path(cfg.data_root_dir),
            action_horizon=cfg.action_horizon,
            source_action_dim=cfg.crane_x7_action_dim,
            target_action_dim=cfg.action_dim,
            resize_resolution=cfg.image_size,
            train=True,
            image_aug=cfg.image_aug,
            normalize_actions=cfg.normalize_actions,
            normalization_mode=cfg.normalization_mode,
            default_prompt=cfg.default_prompt,
            overfit_split_ratio=cfg.overfit_split_ratio,
            split="train",
            rank=device_id if is_distributed else 0,
            world_size=distributed_state.num_processes,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_action_chunk_batch,
            num_workers=4,
            pin_memory=True,
        )

        # Overfit dataset (for validation)
        if cfg.overfit_split_ratio > 0:
            overfit_dataset = CraneX7ActionChunkDataset(
                data_root_dir=Path(cfg.data_root_dir),
                action_horizon=cfg.action_horizon,
                source_action_dim=cfg.crane_x7_action_dim,
                target_action_dim=cfg.action_dim,
                resize_resolution=cfg.image_size,
                train=False,
                image_aug=False,
                normalize_actions=cfg.normalize_actions,
                normalization_mode=cfg.normalization_mode,
                default_prompt=cfg.default_prompt,
                overfit_split_ratio=cfg.overfit_split_ratio,
                split="overfit",
                rank=device_id if is_distributed else 0,
                world_size=distributed_state.num_processes,
            )
            overfit_dataloader = DataLoader(
                overfit_dataset,
                batch_size=cfg.batch_size,
                collate_fn=collate_action_chunk_batch,
                num_workers=2,
            )
        else:
            overfit_dataloader = None

        # Output directory
        exp_id = f"{cfg.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(cfg.data_root_dir).parent / "outputs" / f"crane_x7_pi0_{exp_id}"
        if is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        if is_main_process:
            try:
                import wandb
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    name=f"pi0-{exp_id}",
                    config=vars(cfg),
                )
                use_wandb = True
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
                use_wandb = False
        else:
            use_wandb = False

        # Training loop
        model.train()
        data_iter = iter(train_dataloader)
        loss_accumulator = 0.0
        grad_step = 0

        logger.info(f"Starting training for {cfg.max_steps} steps...")

        for step in range(cfg.max_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
                self.epoch += 1

            # Move to device
            observation = {
                "state": batch["observation"]["state"].to(device_id),
                "image": {
                    "base_0_rgb": batch["observation"]["image"]["base_0_rgb"].to(device_id),
                },
            }
            actions = batch["actions"].to(device_id)

            # Forward pass
            with torch.amp.autocast("cuda", dtype=dtype):
                if is_distributed:
                    loss = model.module.compute_loss(observation, actions)
                else:
                    loss = model.compute_loss(observation, actions)

            # Backward
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            loss_accumulator += loss.item()

            # Gradient step
            if (step + 1) % cfg.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                grad_step += 1
                self.global_step = grad_step

                # Logging
                if grad_step % cfg.log_interval == 0 and is_main_process:
                    avg_loss = loss_accumulator / cfg.log_interval
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {grad_step}/{cfg.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": self.epoch,
                        }, step=grad_step)

                    loss_accumulator = 0.0

                # Overfitting check
                if (
                    overfit_dataloader is not None
                    and grad_step % cfg.overfit_check_interval == 0
                ):
                    overfit_loss = self._run_overfit_check(
                        model, overfit_dataloader, device_id, dtype, is_distributed
                    )
                    if is_main_process:
                        logger.info(f"  Overfit Loss: {overfit_loss:.4f}")
                        if use_wandb:
                            wandb.log({
                                "eval/overfit_loss": overfit_loss,
                            }, step=grad_step)
                    model.train()

                # Save checkpoint
                if grad_step % cfg.save_steps == 0 and is_main_process:
                    self._save_checkpoint(
                        model, optimizer, scheduler,
                        run_dir / f"checkpoint-{grad_step}",
                        is_distributed,
                    )

        # Final save
        if is_main_process:
            self._save_checkpoint(
                model, optimizer, scheduler,
                run_dir / "checkpoint-final",
                is_distributed,
            )
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
    ) -> float:
        """Run overfitting check on held-out steps."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.cfg.overfit_check_steps:
                    break

                observation = {
                    "state": batch["observation"]["state"].to(device_id),
                    "image": {
                        "base_0_rgb": batch["observation"]["image"]["base_0_rgb"].to(device_id),
                    },
                }
                actions = batch["actions"].to(device_id)

                with torch.amp.autocast("cuda", dtype=dtype):
                    if is_distributed:
                        loss = model.module.compute_loss(observation, actions)
                    else:
                        loss = model.compute_loss(observation, actions)

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

        # Get underlying model for DDP
        model_to_save = model.module if is_distributed else model

        # Save model state
        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": vars(self.cfg),
        }, path / "checkpoint.pt")

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.cfg), f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {path}")


class OpenPIPytorchBackend(VLABackend):
    """
    PyTorch-based OpenPI backend for CRANE-X7.

    Uses HuggingFace Pi0 model with flow matching for action prediction.
    """

    def __init__(self, config: OpenPIPytorchConfig):
        super().__init__(config)
        self._action_dim = 8  # CRANE-X7 native
        self._action_horizon = config.openpi_pytorch.action_horizon
        self._image_size = config.openpi_pytorch.image_size
        self._target_action_dim = config.openpi_pytorch.action_dim

        # Transform utilities
        self.action_padder = ActionPadder(self._action_dim, self._target_action_dim)
        self.action_normalizer = ActionNormalizer(
            mode=config.openpi_pytorch.normalization_mode
        )

    def _create_finetune_config(self) -> CraneX7Pi0FinetuneConfig:
        """Create finetune config from unified config."""
        cfg = self.config
        pi_cfg = cfg.openpi_pytorch

        return CraneX7Pi0FinetuneConfig(
            model_name=pi_cfg.model_name,
            pretrained_checkpoint=pi_cfg.pretrained_checkpoint,
            data_root_dir=str(cfg.data.data_root),
            dataset_name=cfg.experiment_name,
            batch_size=cfg.training.batch_size,
            max_steps=cfg.training.max_steps,
            save_steps=cfg.training.save_interval,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            grad_accumulation_steps=cfg.training.gradient_accumulation_steps,
            action_horizon=pi_cfg.action_horizon,
            action_dim=pi_cfg.action_dim,
            crane_x7_action_dim=8,
            num_denoise_steps=pi_cfg.num_denoise_steps,
            precision=pi_cfg.precision,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            normalize_actions=pi_cfg.normalize_actions,
            normalization_mode=pi_cfg.normalization_mode,
            image_size=pi_cfg.image_size,
            image_aug=True,
            overfit_split_ratio=cfg.overfitting.overfit_split_ratio,
            overfit_check_interval=cfg.overfitting.overfit_check_interval,
            overfit_check_steps=cfg.overfitting.overfit_check_steps,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            log_interval=cfg.training.log_interval,
            default_prompt=pi_cfg.default_prompt,
        )

    def train(self) -> Dict[str, Any]:
        """Execute training."""
        cfg = self._create_finetune_config()
        trainer = CraneX7Pi0Trainer(cfg)
        return trainer.train()

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """Evaluate model (not yet implemented)."""
        raise NotImplementedError("Evaluation not yet implemented for OpenPI PyTorch backend")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Dict with 'state' (8,) and 'image' (H, W, 3)
            language_instruction: Task instruction

        Returns:
            Predicted action (8,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint first.")

        device = next(self.model.parameters()).device

        # Prepare observation
        state = observation["state"]
        if state.shape[-1] == self._action_dim:
            state = self.action_padder.pad(state)

        image = observation["image"]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to tensors
        obs_tensor = {
            "state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device),
            "image": {
                "base_0_rgb": torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device),
            },
        }

        # Sample actions
        self.model.eval()
        with torch.no_grad():
            action_chunk = self.model.sample_actions(
                obs_tensor,
                num_steps=self.config.openpi_pytorch.num_denoise_steps,
            )

        # Get first action and truncate to CRANE-X7 dimension
        action = action_chunk[0, 0, :self._action_dim].cpu().numpy()

        # Denormalize if needed
        if self.action_normalizer.stats:
            action = self.action_normalizer.denormalize(action)

        return action

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
        }, path / "model.pt")

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.config.openpi_pytorch), f, indent=2, default=str)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)

        # Load model
        checkpoint_path = path / "checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = path / "model.pt"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model
        dtype = torch.bfloat16 if self.config.openpi_pytorch.precision == "bfloat16" else torch.float32

        try:
            backbone = AutoModel.from_pretrained(
                self.config.openpi_pytorch.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception:
            from transformers import CLIPModel
            backbone = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=dtype,
            )

        self.model = FlowMatchingModule(
            backbone=backbone,
            action_dim=self._target_action_dim,
            action_horizon=self._action_horizon,
        )

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
