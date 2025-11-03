#!/usr/bin/env python3
"""
Example: Train VLA from configuration file.

This script demonstrates how to train using a YAML configuration file.
"""

import logging
import argparse
from pathlib import Path

from crane_x7_vla.training.trainer import VLATrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train VLA from config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logging.info(f"Loading configuration from {config_path}")

    # Create trainer from config file
    trainer = VLATrainer.from_config_file(config_path)

    # Start training
    logging.info("Starting training...")
    results = trainer.train()

    logging.info(f"Training completed!")
    logging.info(f"Results: {results}")


if __name__ == "__main__":
    main()
