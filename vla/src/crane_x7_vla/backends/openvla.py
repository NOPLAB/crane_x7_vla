# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenVLA backend implementation.

This module implements OpenVLA-style fine-tuning using the CRANE-X7 dataset,
directly based on the OpenVLA finetune.py implementation.
"""

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as HFOpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Import CRANE-X7 specific modules
from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openvla_config import OpenVLAConfig
from crane_x7_vla.data.crane_x7_dataset import CraneX7Dataset, CraneX7BatchTransform

# Add parent directory to import existing OpenVLA code
vla_src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(vla_src_path))

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class CraneX7FinetuneConfig:
    """
    Configuration for CRANE-X7 OpenVLA fine-tuning.

    This configuration exactly matches the OpenVLA finetune.py FinetuneConfig.
    """
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "crane_x7"                                  # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    weight_decay: float = 0.01                                      # Weight decay for AdamW optimizer
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # Overfitting Detection Parameters
    overfit_split_ratio: float = 0.1                                # Ratio of steps for overfitting detection (0.0 to disable)
    overfit_check_interval: int = 500                               # Check overfitting every N gradient steps
    overfit_check_steps: int = 50                                   # Number of steps per overfitting check

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # Checkpoint Saving Parameters
    skip_merge_on_save: bool = True                                 # Skip LoRA merge during checkpoint saving
                                                                    #   => Avoids NCCL timeout on multi-GPU setups
                                                                    #   => Merge can be done post-training
    # fmt: on


class CraneX7Trainer:
    """
    OpenVLA fine-tuning trainer for CRANE-X7.

    This trainer exactly matches the OpenVLA finetune.py implementation.
    """

    def __init__(self, cfg: CraneX7FinetuneConfig):
        """
        Initialize trainer with configuration.

        Args:
            cfg: CraneX7FinetuneConfig instance
        """
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    def train(self) -> None:
        """
        Execute the OpenVLA fine-tuning loop.

        This method exactly matches the finetune() function in openvla/vla-scripts/finetune.py.
        """
        print(f"Fine-tuning OpenVLA Model `{self.cfg.vla_path}` on `{self.cfg.dataset_name}`")

        # [Validate] Ensure GPU Available & Set Device / Distributed Context
        assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
        distributed_state = PartialState()
        torch.cuda.set_device(device_id := distributed_state.local_process_index)
        torch.cuda.empty_cache()

        # Configure Unique Experiment ID & Log Directory
        exp_id = (
            f"{self.cfg.vla_path.split('/')[-1]}+{self.cfg.dataset_name}"
            f"+b{self.cfg.batch_size * self.cfg.grad_accumulation_steps}"
            f"+lr-{self.cfg.learning_rate}"
        )
        if self.cfg.use_lora:
            exp_id += f"+lora-r{self.cfg.lora_rank}+dropout-{self.cfg.lora_dropout}"
        if self.cfg.use_quantization:
            exp_id += "+q-4bit"
        if self.cfg.run_id_note is not None:
            exp_id += f"--{self.cfg.run_id_note}"
        if self.cfg.image_aug:
            exp_id += "--image_aug"

        # Start =>> Build Directories
        run_dir, adapter_dir = self.cfg.run_root_dir / exp_id, self.cfg.adapter_tmp_dir / exp_id
        os.makedirs(run_dir, exist_ok=True)

        # Quantization Config =>> only if LoRA fine-tuning
        quantization_config = None
        if self.cfg.use_quantization:
            assert self.cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )

        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", HFOpenVLAConfig)
        AutoImageProcessor.register(HFOpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(HFOpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(HFOpenVLAConfig, OpenVLAForActionPrediction)

        # Load OpenVLA Processor and Model using HF AutoClasses
        processor = AutoProcessor.from_pretrained(self.cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            self.cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid SDPA compatibility issues
        )

        # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
        if self.cfg.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        else:
            vla = vla.to(device_id)

        # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
        if self.cfg.use_lora:
            lora_config = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=min(self.cfg.lora_rank, 16),
                lora_dropout=self.cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

        # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training (only if distributed)
        is_distributed = distributed_state.num_processes > 1
        if is_distributed:
            vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

        # Get unwrapped model for accessing config/internals
        unwrapped_vla = vla.module if is_distributed else vla

        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        # Create Action Tokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)

        # Load Fine-tuning Dataset =>> using CRANE-X7 dataset with RLDS-compatible interface
        batch_transform = CraneX7BatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in self.cfg.vla_path else VicunaV15ChatPromptBuilder,
        )

        # Create training dataset (with optional step-level split for overfitting detection)
        vla_dataset = CraneX7Dataset(
            self.cfg.data_root_dir,
            self.cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(unwrapped_vla.config.image_sizes),
            shuffle_buffer_size=self.cfg.shuffle_buffer_size,
            image_aug=self.cfg.image_aug,
            overfit_split_ratio=self.cfg.overfit_split_ratio,
            split="train",
        )

        # Create overfitting detection dataset if overfit_split_ratio > 0
        # This dataset uses step-level splitting (not episode-level) to properly detect overfitting
        overfit_dataset = None
        overfit_dataloader = None
        if self.cfg.overfit_split_ratio > 0:
            overfit_dataset = CraneX7Dataset(
                self.cfg.data_root_dir,
                self.cfg.dataset_name,
                batch_transform,
                resize_resolution=tuple(unwrapped_vla.config.image_sizes),
                shuffle_buffer_size=1000,  # Small buffer for overfitting check
                image_aug=False,  # No augmentation for overfitting check
                overfit_split_ratio=self.cfg.overfit_split_ratio,
                split="overfit",
                train=False,  # Don't repeat overfitting check data
            )

        # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
        if distributed_state.is_main_process:
            save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

        # Create Collator and DataLoader
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        )

        # Create overfitting check dataloader if enabled
        if overfit_dataset is not None:
            overfit_dataloader = DataLoader(
                overfit_dataset,
                batch_size=self.cfg.batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
            )

        # Initialize Logging =>> W&B
        if distributed_state.is_main_process:
            wandb.init(entity=self.cfg.wandb_entity, project=self.cfg.wandb_project, name=f"ft+{exp_id}")

        # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
        recent_losses = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_action_accuracies = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=self.cfg.grad_accumulation_steps)

        # Train!
        with tqdm.tqdm(total=self.cfg.max_steps, leave=False) as progress:
            vla.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / self.cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, unwrapped_vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // self.cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                # Push Metrics to W&B (every 10 gradient steps)
                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                        },
                        step=gradient_step_idx,
                    )

                # Optimizer Step
                if (batch_idx + 1) % self.cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                    # Run overfitting check at specified interval
                    if (
                        overfit_dataloader is not None
                        and gradient_step_idx > 0
                        and gradient_step_idx % self.cfg.overfit_check_interval == 0
                    ):
                        overfit_metrics = self._run_overfit_check(
                            vla=vla,
                            overfit_dataloader=overfit_dataloader,
                            device_id=device_id,
                            unwrapped_vla=unwrapped_vla,
                            action_tokenizer=action_tokenizer,
                            distributed_state=distributed_state,
                        )

                        # Log overfitting metrics to W&B
                        if distributed_state.is_main_process:
                            wandb.log(
                                {
                                    "overfit_loss": overfit_metrics["overfit_loss"],
                                    "overfit_action_accuracy": overfit_metrics["overfit_action_accuracy"],
                                    "overfit_l1_loss": overfit_metrics["overfit_l1_loss"],
                                },
                                step=gradient_step_idx,
                            )
                            print(
                                f"[Step {gradient_step_idx}] Overfit Loss: {overfit_metrics['overfit_loss']:.4f}, "
                                f"Overfit Accuracy: {overfit_metrics['overfit_action_accuracy']:.4f}, "
                                f"Overfit L1: {overfit_metrics['overfit_l1_loss']:.4f}"
                            )

                        # Return model to training mode
                        vla.train()

                # Save Model Checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % self.cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # Create checkpoint directory with step number
                        checkpoint_dir = run_dir / f"checkpoint-{gradient_step_idx}"
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Save Processor
                        processor.save_pretrained(checkpoint_dir)

                        # Save dataset statistics
                        save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                        if self.cfg.use_lora:
                            # Save adapter weights to checkpoint directory
                            lora_save_dir = checkpoint_dir / "lora_adapters"
                            os.makedirs(lora_save_dir, exist_ok=True)
                            unwrapped_vla.save_pretrained(lora_save_dir)
                            print(f"Saved LoRA adapters at: {lora_save_dir}")

                            if not self.cfg.skip_merge_on_save:
                                # Merge LoRA weights into model backbone for faster inference
                                # NOTE: This is slow and can cause NCCL timeout on multi-GPU setups
                                #       Consider setting skip_merge_on_save=True and merging post-training
                                import gc

                                print("Merging LoRA weights (this may take several minutes)...")

                                # Load base model on CPU and merge with adapters
                                base_vla = AutoModelForVision2Seq.from_pretrained(
                                    self.cfg.vla_path,
                                    torch_dtype=torch.bfloat16,
                                    low_cpu_mem_usage=True,
                                    trust_remote_code=True,
                                    attn_implementation="eager",
                                    device_map="cpu",
                                )
                                merged_vla = PeftModel.from_pretrained(base_vla, lora_save_dir, device_map="cpu")
                                merged_vla = merged_vla.merge_and_unload()

                                # Save merged model to checkpoint directory
                                merged_vla.save_pretrained(checkpoint_dir)
                                print(f"Saved merged model at: {checkpoint_dir}")

                                # Clean up
                                del base_vla, merged_vla
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                print(f"Skipping LoRA merge (skip_merge_on_save=True). Run merge_lora.py post-training.")
                        else:
                            # Full model (no LoRA) - save directly
                            unwrapped_vla.save_pretrained(checkpoint_dir)

                        print(f"Checkpoint saved at: {checkpoint_dir}")

                    # Synchronize all processes after checkpoint saving
                    # This barrier is lightweight since only file I/O is performed (no heavy merge)
                    if is_distributed:
                        dist.barrier()

                # Stop training when max_steps is reached
                if gradient_step_idx == self.cfg.max_steps:
                    print(f"Max step {self.cfg.max_steps} reached! Stopping training...")
                    break

        # Update tracking variables
        self.global_step = gradient_step_idx

    def _run_overfit_check(
        self,
        vla,
        overfit_dataloader,
        device_id: int,
        unwrapped_vla,
        action_tokenizer: ActionTokenizer,
        distributed_state,
    ) -> Dict[str, float]:
        """
        Run overfitting check loop and compute metrics.

        This uses held-out steps from the same episodes (not separate episodes)
        to properly detect overfitting/memorization.

        Args:
            vla: The VLA model
            overfit_dataloader: Overfitting check data loader
            device_id: GPU device ID
            unwrapped_vla: Unwrapped model for accessing internals
            action_tokenizer: Action tokenizer instance
            distributed_state: Distributed training state

        Returns:
            Dictionary containing overfitting check metrics
        """
        vla.eval()

        overfit_losses = []
        overfit_action_accuracies = []
        overfit_l1_losses = []

        with torch.no_grad():
            for overfit_step, batch in enumerate(overfit_dataloader):
                if overfit_step >= self.cfg.overfit_check_steps:
                    break

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Compute Accuracy and L1 Loss
                action_logits = output.logits[
                    :, unwrapped_vla.vision_backbone.featurizer.patch_embed.num_patches : -1
                ]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                overfit_losses.append(loss.item())
                overfit_action_accuracies.append(action_accuracy.item())
                overfit_l1_losses.append(action_l1_loss.item())

        # Compute average metrics
        metrics = {
            "overfit_loss": sum(overfit_losses) / len(overfit_losses) if overfit_losses else 0.0,
            "overfit_action_accuracy": sum(overfit_action_accuracies) / len(overfit_action_accuracies) if overfit_action_accuracies else 0.0,
            "overfit_l1_loss": sum(overfit_l1_losses) / len(overfit_l1_losses) if overfit_l1_losses else 0.0,
        }

        return metrics


class OpenVLABackend(VLABackend):
    """
    OpenVLA backend implementation.

    Wraps the OpenVLA fine-tuning pipeline with the unified VLA backend interface.
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
            CraneX7FinetuneConfig instance matching OpenVLA finetune.py FinetuneConfig
        """
        ft_config = CraneX7FinetuneConfig(
            vla_path=self.vla_config.openvla.model_id,
            # Directory Paths
            data_root_dir=self.vla_config.data.data_root,
            dataset_name=self.vla_config.data.dataset_name if hasattr(self.vla_config.data, 'dataset_name') else "crane_x7",
            run_root_dir=self.vla_config.output_dir,
            # Fine-tuning Parameters
            batch_size=self.vla_config.training.batch_size,
            max_steps=self.vla_config.training.max_steps,
            save_steps=self.vla_config.training.save_interval,
            learning_rate=self.vla_config.training.learning_rate,
            weight_decay=self.vla_config.training.weight_decay,
            grad_accumulation_steps=self.vla_config.training.gradient_accumulation_steps,
            image_aug=self.vla_config.openvla.image_aug if hasattr(self.vla_config.openvla, 'image_aug') else True,
            shuffle_buffer_size=self.vla_config.data.shuffle_buffer_size if hasattr(self.vla_config.data, 'shuffle_buffer_size') else 100_000,
            # Overfitting Detection Parameters
            overfit_split_ratio=self.vla_config.overfitting.overfit_split_ratio if hasattr(self.vla_config, 'overfitting') and hasattr(self.vla_config.overfitting, 'overfit_split_ratio') else 0.1,
            overfit_check_interval=self.vla_config.overfitting.overfit_check_interval if hasattr(self.vla_config, 'overfitting') and hasattr(self.vla_config.overfitting, 'overfit_check_interval') else 500,
            overfit_check_steps=self.vla_config.overfitting.overfit_check_steps if hasattr(self.vla_config, 'overfitting') and hasattr(self.vla_config.overfitting, 'overfit_check_steps') else 50,
            # LoRA Arguments
            use_lora=self.vla_config.openvla.use_lora,
            lora_rank=self.vla_config.openvla.lora_rank,
            lora_dropout=self.vla_config.openvla.lora_dropout,
            use_quantization=self.vla_config.openvla.use_quantization,
            # Tracking Parameters
            wandb_project=self.vla_config.wandb_project if hasattr(self.vla_config, 'wandb_project') else "openvla",
            wandb_entity=self.vla_config.wandb_entity if hasattr(self.vla_config, 'wandb_entity') else "stanford-voltron",
            run_id_note=self.vla_config.experiment_name if hasattr(self.vla_config, 'experiment_name') else None,
            # Checkpoint Saving Parameters
            skip_merge_on_save=self.vla_config.openvla.skip_merge_on_save if hasattr(self.vla_config.openvla, 'skip_merge_on_save') else True,
        )

        return ft_config

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            Dictionary containing training metrics and results
        """
        # Create fine-tune config from unified config
        cfg = self._create_finetune_config()

        # Create trainer
        self.trainer = CraneX7Trainer(cfg)

        # Run training
        self.trainer.train()

        # Return training results
        results = {
            'final_step': self.trainer.global_step,
            'final_epoch': self.trainer.epoch,
            'run_root_dir': str(cfg.run_root_dir),
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
            "attn_implementation": "eager",  # Avoid SDPA compatibility issues with OpenVLA
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
