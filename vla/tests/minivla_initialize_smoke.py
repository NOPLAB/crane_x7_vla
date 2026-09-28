"""Exercise the MiniVLA model setup path without claiming a training step."""

from __future__ import annotations

import argparse
from pathlib import Path

from crane_x7_vla.backends.minivla.backend import MiniVLABackend
from crane_x7_vla.backends.minivla.config import MiniVLAConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = MiniVLABackend(MiniVLAConfig.from_yaml(args.config)).initialize()
    print(result)
