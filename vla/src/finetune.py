#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Fine-tune OpenVLA on CRANE-X7 robot demonstration data.

This script fine-tunes the OpenVLA model on CRANE-X7 TFRecord data using
LoRA (Low-Rank Adaptation) for parameter-efficient training.

Usage:
    # Single GPU
    python finetune.py

    # Multi-GPU with PyTorch DDP
    torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py

    # With custom configuration
    python finetune.py --batch_size 16 --learning_rate 1e-4 --use_wandb
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from crane_x7_dataset import CraneX7Dataset
from finetune_config import CraneX7FinetuneConfig, get_lora_config


class CraneX7Trainer:
    """Trainer for fine-tuning OpenVLA on CRANE-X7 data."""

    def __init__(self, config: CraneX7FinetuneConfig):
        """Initialize trainer."""
        self.config = config
        self.setup_distributed()
        self.setup_logging()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()

        self.global_step = 0
        self.epoch = 0

    def setup_distributed(self):
        """Setup distributed training if available."""
        if torch.cuda.is_available() and 'RANK' in os.environ:
            # Distributed training
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.is_distributed = True
            self.is_main_process = (self.rank == 0)
        else:
            # Single GPU or CPU
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
            self.is_distributed = False
            self.is_main_process = True

        if self.is_main_process:
            print(f"Training on {self.world_size} GPU(s)")
            print(f"Device: {self.device}")

    def setup_logging(self):
        """Setup Weights & Biases logging."""
        self.use_wandb = self.config.use_wandb and self.is_main_process

        if self.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
            print("Weights & Biases logging enabled")

    def setup_model(self):
        """Load and setup OpenVLA model with LoRA."""
        if self.is_main_process:
            print(f"Loading model: {self.config.vla_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.vla_path,
            trust_remote_code=True
        )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
            "low_cpu_mem_usage": True,
        }

        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.vla_path,
            **model_kwargs
        )

        # Apply LoRA
        if self.config.use_lora:
            from peft import get_peft_model

            lora_config = get_lora_config(self.config)
            self.model = get_peft_model(self.model, lora_config)

            if self.is_main_process:
                print("LoRA configuration:")
                print(f"  Rank: {self.config.lora_rank}")
                print(f"  Alpha: {self.config.lora_alpha}")
                print(f"  Target modules: {self.config.lora_target_modules}")
                self.model.print_trainable_parameters()

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Move to device
        self.model = self.model.to(self.device)

        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

    def setup_data(self):
        """Setup dataset and dataloader."""
        if self.is_main_process:
            print(f"Loading data from: {self.config.data_root}")

        # Create dataset
        self.dataset = CraneX7Dataset(
            data_root=str(self.config.data_root),
            instruction=self.config.instruction,
            image_size=self.config.image_size,
            use_image=self.config.use_image,
        )

        # Create sampler for distributed training
        if self.is_distributed:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.config.shuffle,
            )
        else:
            sampler = None

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and self.config.shuffle),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        if self.is_main_process:
            print(f"Dataset size: {len(self.dataset)} steps")
            print(f"Batch size per GPU: {self.config.batch_size}")
            print(f"Gradient accumulation steps: {self.config.grad_accumulation_steps}")
            print(f"Effective batch size: {self.config.batch_size * self.world_size * self.config.grad_accumulation_steps}")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        if self.config.use_lora:
            # Only optimize LoRA parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = self.model.parameters()

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate total steps
        steps_per_epoch = len(self.dataloader) // self.config.grad_accumulation_steps
        if self.config.max_steps is not None:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs

        # Create learning rate scheduler with warmup
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return max(0.0, 1.0 - (step - self.config.warmup_steps) / max(1, self.total_steps - self.config.warmup_steps))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        if self.is_main_process:
            print(f"Total training steps: {self.total_steps}")
            print(f"Warmup steps: {self.config.warmup_steps}")

    def train_step(self, batch):
        """Execute one training step."""
        # Prepare inputs
        images = batch['image'].to(self.device) if 'image' in batch else None
        states = batch['state'].to(self.device)
        actions = batch['action'].to(self.device)
        instructions = batch['instruction']

        # Create text prompts
        prompts = [f"In: What action should the robot take to {inst}?\nOut:" for inst in instructions]

        # Process inputs through processor
        if images is not None:
            # Convert tensor images back to PIL for processor
            from PIL import Image
            import numpy as np

            pil_images = []
            for img_tensor in images:
                img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_array))

            inputs = self.processor(prompts, pil_images)
        else:
            # Text-only mode (though OpenVLA expects images)
            inputs = self.processor(prompts, None)

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs, labels=actions)
        loss = outputs.loss

        # Backward pass
        loss = loss / self.config.grad_accumulation_steps
        loss.backward()

        return loss.item() * self.config.grad_accumulation_steps

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        if self.is_distributed:
            self.dataloader.sampler.set_epoch(self.epoch)

        total_loss = 0.0
        progress_bar = None

        if self.is_main_process:
            progress_bar = tqdm(total=len(self.dataloader), desc=f"Epoch {self.epoch}")

        for step, batch in enumerate(self.dataloader):
            # Training step
            loss = self.train_step(batch)
            total_loss += loss

            # Gradient accumulation
            if (step + 1) % self.config.grad_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.is_main_process and self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps / self.config.grad_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": self.epoch,
                            "train/global_step": self.global_step,
                        })

                    total_loss = 0.0

                # Save checkpoint
                if self.is_main_process and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Check if max steps reached
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    def save_checkpoint(self):
        """Save model checkpoint."""
        output_dir = self.config.output_dir / f"checkpoint-{self.global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.config.use_lora:
            # Save LoRA adapters
            model = self.model.module if isinstance(self.model, DDP) else self.model
            model.save_pretrained(output_dir)
        else:
            # Save full model
            model = self.model.module if isinstance(self.model, DDP) else self.model
            model.save_pretrained(output_dir)

        # Save processor
        self.processor.save_pretrained(output_dir)

        print(f"Checkpoint saved to {output_dir}")

        # Cleanup old checkpoints
        if self.config.save_total_limit > 0:
            checkpoints = sorted(self.config.output_dir.glob("checkpoint-*"))
            if len(checkpoints) > self.config.save_total_limit:
                for ckpt in checkpoints[:-self.config.save_total_limit]:
                    import shutil
                    shutil.rmtree(ckpt)
                    print(f"Removed old checkpoint: {ckpt}")

    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print("\n" + "=" * 60)
            print("Starting Training")
            print("=" * 60)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

        # Save final checkpoint
        if self.is_main_process:
            print("\nTraining complete!")
            self.save_checkpoint()

            if self.use_wandb:
                import wandb
                wandb.finish()

        # Cleanup distributed
        if self.is_distributed:
            dist.destroy_process_group()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA on CRANE-X7 data")

    # Override config parameters
    parser.add_argument("--data_root", type=str, help="Data root directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load default config
    config = CraneX7FinetuneConfig()

    # Override with command line arguments
    if args.data_root:
        config.data_root = Path(args.data_root)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.lora_rank:
        config.lora_rank = args.lora_rank
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity

    # Create trainer and start training
    trainer = CraneX7Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
