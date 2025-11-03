#!/usr/bin/env python3
"""
Example: Train OpenVLA on CRANE-X7 data.

This script demonstrates how to train OpenVLA using the Python API.
"""

import logging
from pathlib import Path

from crane_x7_vla.config.openvla_config import OpenVLAConfig, OpenVLASpecificConfig
from crane_x7_vla.config.base import DataConfig, TrainingConfig, CameraConfig
from crane_x7_vla.training.trainer import VLATrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def main():
    """Main training script."""

    # Configure camera(s)
    cameras = [
        CameraConfig(
            name="primary",
            topic="/camera/color/image_raw",
            width=640,
            height=480,
            enabled=True
        )
    ]

    # Configure data
    data_config = DataConfig(
        data_root=Path("./data/tfrecord_logs"),
        train_split=0.9,
        val_split=0.1,
        shuffle=True,
        num_workers=4,
        cameras=cameras
    )

    # Configure training
    training_config = TrainingConfig(
        batch_size=16,
        num_epochs=100,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        mixed_precision="bf16",
        save_interval=1000,
        log_interval=10
    )

    # Configure OpenVLA-specific settings
    openvla_specific = OpenVLASpecificConfig(
        model_id="openvla/openvla-7b",
        use_lora=True,
        lora_rank=32,
        lora_alpha=16,
        lora_dropout=0.05,
        use_flash_attention=False
    )

    # Create full config
    config = OpenVLAConfig(
        backend="openvla",
        data=data_config,
        training=training_config,
        output_dir=Path("./outputs/openvla_example"),
        experiment_name="openvla_crane_x7",
        seed=42,
        openvla=openvla_specific
    )

    # Save config
    config_path = config.output_dir / "config.yaml"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_path)
    logging.info(f"Configuration saved to {config_path}")

    # Create trainer
    trainer = VLATrainer(config)

    # Start training
    logging.info("Starting training...")
    results = trainer.train()

    logging.info(f"Training completed!")
    logging.info(f"Results: {results}")


if __name__ == "__main__":
    main()
