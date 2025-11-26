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
from crane_x7_vla.data.crane_x7_dataset import (
    CraneX7Dataset,
    CraneX7DatasetConfig,
    CraneX7BatchTransform,
)

# Add parent directory to import existing OpenVLA code
vla_src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(vla_src_path))

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class CraneX7FinetuneConfig:
    """
    Configuration for CRANE-X7 OpenVLA fine-tuning.

    This configuration is compatible with the OpenVLA finetune.py implementation
    but uses the CRANE-X7 dataset loader directly.
    """
    # fmt: off
    # Model settings
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root: Path = Path("data/tfrecord_logs")                    # Path to CRANE-X7 TFRecord dataset directory
    output_dir: Path = Path("outputs")                              # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    shuffle_buffer_size: int = 1000                                 # Dataloader shuffle buffer size
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    wandb_project: str = "crane-x7-vla"                             # Name of W&B project to log to
    wandb_entity: Optional[str] = None                              # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging

    # CRANE-X7 Dataset Parameters
    action_dim: int = 8                                             # CRANE-X7 action dimension (7 joints + 1 gripper)
    normalize_actions: bool = True                                  # Whether to normalize actions
    default_instruction: str = "manipulate the object"              # Default language instruction
    # fmt: on


class CraneX7Trainer:
    """
    OpenVLA fine-tuning trainer for CRANE-X7.

    This trainer implements the OpenVLA fine-tuning loop using CRANE-X7 dataset,
    following the architecture from openvla/vla-scripts/finetune.py.
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
        Execute the OpenVLA fine-tuning loop.

        This method implements the complete training pipeline:
        1. Setup distributed training environment
        2. Load OpenVLA model and processor
        3. Setup LoRA if enabled
        4. Load CRANE-X7 dataset
        5. Run training loop with W&B logging
        6. Save checkpoints
        """
        print(f"Fine-tuning OpenVLA Model `{self.config.vla_path}` on CRANE-X7 dataset")

        # [Validate] Ensure GPU Available & Set Device / Distributed Context
        assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
        distributed_state = PartialState()
        torch.cuda.set_device(device_id := distributed_state.local_process_index)
        torch.cuda.empty_cache()

        # Configure Unique Experiment ID & Log Directory
        exp_id = (
            f"{self.config.vla_path.split('/')[-1]}+crane_x7"
            f"+b{self.config.batch_size * self.config.grad_accumulation_steps}"
            f"+lr-{self.config.learning_rate}"
        )
        if self.config.use_lora:
            exp_id += f"+lora-r{self.config.lora_rank}+dropout-{self.config.lora_dropout}"
        if self.config.use_quantization:
            exp_id += "+q-4bit"
        if self.config.run_id_note is not None:
            exp_id += f"--{self.config.run_id_note}"

        # Start =>> Build Directories
        run_dir = self.config.output_dir / exp_id
        adapter_dir = self.config.adapter_tmp_dir / exp_id
        os.makedirs(run_dir, exist_ok=True)

        # Quantization Config =>> only if LoRA fine-tuning
        quantization_config = None
        if self.config.use_quantization:
            assert self.config.use_lora, "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )

        # Register OpenVLA model to HF Auto Classes
        AutoConfig.register("openvla", HFOpenVLAConfig)
        AutoImageProcessor.register(HFOpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(HFOpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(HFOpenVLAConfig, OpenVLAForActionPrediction)

        # Load OpenVLA Processor and Model using HF AutoClasses
        print(f"Loading OpenVLA model from {self.config.vla_path}...")
        processor = AutoProcessor.from_pretrained(self.config.vla_path, trust_remote_code=True)

        # Load model with appropriate settings for quantization
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization_config,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # Avoid SDPA compatibility issues with OpenVLA
        }
        if self.config.use_quantization:
            # For quantized models, use device_map="auto" to let BitsAndBytes handle placement
            model_kwargs["device_map"] = "auto"

        vla = AutoModelForVision2Seq.from_pretrained(
            self.config.vla_path,
            **model_kwargs,
        )

        # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
        if self.config.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        else:
            vla = vla.to(device_id)

        # [LoRA] Wrap Model w/ PEFT `LoraConfig`
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=min(self.config.lora_rank, 16),
                lora_dropout=self.config.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

        # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training (only if distributed training is enabled)
        if dist.is_initialized():
            vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
            use_ddp = True
        else:
            use_ddp = False
            print("Running in single-GPU mode (no DDP)")

        # Helper to access underlying model (DDP wraps model in .module)
        def get_unwrapped_model(model):
            return model.module if use_ddp else model

        # Create Optimizer
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.config.learning_rate)

        # Create Action Tokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)

        # Load CRANE-X7 Dataset
        print(f"Loading CRANE-X7 dataset from {self.config.data_root}...")

        # Create dataset configuration
        dataset_config = CraneX7DatasetConfig(
            data_root=self.config.data_root,
            action_dim=self.config.action_dim,
            normalize_actions=self.config.normalize_actions,
            default_instruction=self.config.default_instruction,
            image_size=tuple(get_unwrapped_model(vla).config.image_sizes),
        )

        # Create batch transform
        batch_transform = CraneX7BatchTransform(
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder if "v01" not in self.config.vla_path else VicunaV15ChatPromptBuilder,
            config=dataset_config,
            predict_stop_token=True,
        )

        # Create dataset
        vla_dataset = CraneX7Dataset(
            config=dataset_config,
            batch_transform=batch_transform,
            train=True,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
        )

        # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
        if distributed_state.is_main_process:
            # Create dataset statistics in expected format (nested under dataset name)
            # The save_dataset_statistics function expects: {dataset_name: {action: {...}}}
            dataset_statistics = {
                "crane_x7": {
                    "action": {
                        "mean": batch_transform.normalization_stats.get("action", {}).get("mean", [0.0] * self.config.action_dim),
                        "std": batch_transform.normalization_stats.get("action", {}).get("std", [1.0] * self.config.action_dim),
                        "min": batch_transform.normalization_stats.get("action", {}).get("min", [-1.0] * self.config.action_dim),
                        "max": batch_transform.normalization_stats.get("action", {}).get("max", [1.0] * self.config.action_dim),
                        "q01": batch_transform.normalization_stats.get("action", {}).get("q01", [-1.0] * self.config.action_dim),
                        "q99": batch_transform.normalization_stats.get("action", {}).get("q99", [1.0] * self.config.action_dim),
                    }
                }
            }
            save_dataset_statistics(dataset_statistics, run_dir)
            print(f"Saved dataset statistics to {run_dir}")

        # Create Collator and DataLoader
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.config.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important =>> Set to 0 if using TensorFlow dataset loader
        )

        # Initialize Logging =>> W&B
        if distributed_state.is_main_process:
            wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=f"ft+{exp_id}",
                config={
                    "vla_path": self.config.vla_path,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "max_steps": self.config.max_steps,
                    "lora_rank": self.config.lora_rank if self.config.use_lora else None,
                    "action_dim": self.config.action_dim,
                }
            )

        # Deque to store recent train metrics
        recent_losses = deque(maxlen=self.config.grad_accumulation_steps)
        recent_action_accuracies = deque(maxlen=self.config.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=self.config.grad_accumulation_steps)

        # Train!
        print(f"Starting training for {self.config.max_steps} steps...")
        with tqdm.tqdm(total=self.config.max_steps, leave=False) as progress:
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
                normalized_loss = loss / self.config.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, get_unwrapped_model(vla).vision_backbone.featurizer.patch_embed.num_patches : -1]
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
                gradient_step_idx = batch_idx // self.config.grad_accumulation_steps

                # Compute smoothened train metrics
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
                if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                # Save Model Checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % self.config.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model
                        save_dir = adapter_dir if self.config.use_lora else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        get_unwrapped_model(vla).save_pretrained(save_dir)

                    # Wait for processor and adapter weights to be saved by main process
                    if use_ddp:
                        dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    if self.config.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            self.config.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
                            attn_implementation="eager"  # Avoid SDPA compatibility issues with OpenVLA
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if self.config.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(run_dir)
                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save dataset statistics to new directory
                                save_dataset_statistics(dataset_statistics, checkpoint_dir)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                    # Block on Main Process Checkpointing
                    if use_ddp:
                        dist.barrier()

                # Stop training when max_steps is reached
                if gradient_step_idx >= self.config.max_steps:
                    print(f"Max step {self.config.max_steps} reached! Stopping training...")
                    break

        # Update tracking variables
        self.global_step = gradient_step_idx
        print(f"Training completed! Final step: {self.global_step}")

        # Close W&B
        if distributed_state.is_main_process:
            wandb.finish()


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
            CraneX7FinetuneConfig instance
        """
        # Model settings
        ft_config = CraneX7FinetuneConfig(
            vla_path=self.vla_config.openvla.model_id,
            # Data settings
            data_root=self.vla_config.data.data_root,
            output_dir=self.vla_config.output_dir,
            # Training settings
            batch_size=self.vla_config.training.batch_size,
            learning_rate=self.vla_config.training.learning_rate,
            max_steps=self.vla_config.training.max_steps,
            save_steps=self.vla_config.training.save_interval,
            grad_accumulation_steps=self.vla_config.training.gradient_accumulation_steps,
            shuffle_buffer_size=self.vla_config.data.shuffle_buffer_size,
            # LoRA settings
            use_lora=self.vla_config.openvla.use_lora,
            lora_rank=self.vla_config.openvla.lora_rank,
            lora_dropout=self.vla_config.openvla.lora_dropout,
            use_quantization=self.vla_config.openvla.use_quantization,
            # Tracking settings
            wandb_project=self.vla_config.wandb_project,
            wandb_entity=self.vla_config.wandb_entity,
            run_id_note=self.vla_config.experiment_name,
            # CRANE-X7 specific
            action_dim=self._action_dim,
            normalize_actions=True,
            default_instruction="manipulate the object",
        )

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
