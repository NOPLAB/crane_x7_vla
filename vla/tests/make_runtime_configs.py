"""Create one-step VLA configs for integration checks with synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

from crane_x7_vla.training.cli import create_default_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for name in ("openvla", "openvla-oft", "minivla", "pi0", "pi0.5"):
        config = create_default_config(name, args.data_root, args.output_root / name, f"smoke_{name}")
        config.training.batch_size = 1
        config.training.max_steps = 1
        config.training.gradient_accumulation_steps = 1
        config.training.save_interval = 100
        config.training.eval_interval = 100
        config.training.log_interval = 1
        config.overfitting.overfit_split_ratio = 0.0
        config.data.num_workers = 0
        if name == "openvla":
            config.openvla.use_quantization = True
            config.openvla.image_aug = False
        elif name == "openvla-oft":
            config.openvla_oft.use_quantization = True
            config.openvla_oft.image_aug = False
            config.openvla_oft.multi_image.enabled = False
        elif name.startswith("pi0"):
            config.pi0.paligemma_variant = "dummy"
            config.pi0.action_expert_variant = "dummy"
            config.pi0.use_pretrained = False
            config.pi0.action_horizon = 4
            config.pi0.camera_names = ["base_0_rgb"]
            config.pi0.num_cameras = 1
            config.pi0.normalize_actions = False
        output = args.output_root / f"{name}.yaml"
        config.to_yaml(output)
        print(f"{name}: {output}")


if __name__ == "__main__":
    main()
