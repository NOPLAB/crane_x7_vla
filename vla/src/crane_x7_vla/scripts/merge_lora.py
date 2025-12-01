#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Merge LoRA adapters into base OpenVLA model.

This script merges LoRA adapter weights saved during training into the base
OpenVLA model to create a standalone model for inference.

Usage:
    python -m crane_x7_vla.scripts.merge_lora \
        --adapter_path /path/to/lora_adapters \
        --output_path /path/to/merged_model \
        --base_model openvla/openvla-7b

Example:
    python -m crane_x7_vla.scripts.merge_lora \
        --adapter_path outputs/crane_x7_openvla/lora_adapters \
        --output_path outputs/crane_x7_openvla_merged
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Register OpenVLA model to HF Auto Classes
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_classes():
    """Register OpenVLA classes with HuggingFace Auto classes."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def merge_lora_adapters(
    adapter_path: str,
    output_path: str,
    base_model: str = "openvla/openvla-7b",
    copy_processor: bool = True,
    copy_statistics: bool = True,
) -> None:
    """
    Merge LoRA adapters into base model.

    Args:
        adapter_path: Path to directory containing LoRA adapter weights
        output_path: Path to save merged model
        base_model: HuggingFace model ID or path to base model
        copy_processor: Whether to copy processor files from adapter directory
        copy_statistics: Whether to copy dataset statistics from adapter directory
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    # Validate adapter path
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.exists():
        raise ValueError(f"No adapter_config.json found in {adapter_path}")

    print(f"=" * 60)
    print(f"LoRA Merge Script")
    print(f"=" * 60)
    print(f"Adapter path: {adapter_path}")
    print(f"Base model: {base_model}")
    print(f"Output path: {output_path}")
    print(f"=" * 60)

    # Register OpenVLA classes
    print("Registering OpenVLA classes...")
    register_openvla_classes()

    # Load base model on CPU to avoid GPU memory issues
    print(f"Loading base model from {base_model}...")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="cpu",
    )

    # Load LoRA adapters
    print(f"Loading LoRA adapters from {adapter_path}...")
    merged_vla = PeftModel.from_pretrained(
        base_vla,
        adapter_path,
        device_map="cpu",
    )

    # Merge and unload adapters
    print("Merging LoRA weights into base model...")
    merged_vla = merged_vla.merge_and_unload()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_vla.save_pretrained(output_path)

    # Copy processor files if they exist in parent directory
    parent_dir = adapter_path.parent
    processor_files = [
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    if copy_processor:
        print("Copying processor files...")
        for filename in processor_files:
            src_file = parent_dir / filename
            if src_file.exists():
                shutil.copy(src_file, output_path / filename)
                print(f"  Copied: {filename}")

    # Copy dataset statistics if they exist
    if copy_statistics:
        stats_file = parent_dir / "dataset_statistics.json"
        if stats_file.exists():
            shutil.copy(stats_file, output_path / "dataset_statistics.json")
            print("  Copied: dataset_statistics.json")

    # Clean up to free memory
    del base_vla, merged_vla
    gc.collect()
    torch.cuda.empty_cache()

    print(f"=" * 60)
    print(f"Merge completed successfully!")
    print(f"Merged model saved to: {output_path}")
    print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base OpenVLA model"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to directory containing LoRA adapter weights",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openvla/openvla-7b",
        help="HuggingFace model ID or path to base model (default: openvla/openvla-7b)",
    )
    parser.add_argument(
        "--no_copy_processor",
        action="store_true",
        help="Do not copy processor files from adapter directory",
    )
    parser.add_argument(
        "--no_copy_statistics",
        action="store_true",
        help="Do not copy dataset statistics from adapter directory",
    )

    args = parser.parse_args()

    merge_lora_adapters(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        base_model=args.base_model,
        copy_processor=not args.no_copy_processor,
        copy_statistics=not args.no_copy_statistics,
    )


if __name__ == "__main__":
    main()
