"""Run one CRANE-X7 action prediction from a Pi0/Pi0.5 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crane_x7_vla.backends.pi0 import Pi0Backend
from crane_x7_vla.backends.pi0.config import Pi0Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    config = Pi0Config.from_yaml(args.config)
    backend = Pi0Backend(config)
    backend.load_checkpoint(args.checkpoint)
    image = np.zeros((*config.pi0.image_size, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    action = backend.infer({"state": state, "image": image}, "move the red cube")
    assert action.shape == (8,)
    assert np.isfinite(action).all()
    print(f"{config.backend} action_shape={action.shape} finite=True")


if __name__ == "__main__":
    main()
