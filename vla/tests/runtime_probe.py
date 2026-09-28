"""Check CUDA, CRANE-X7 TFRecord input, and every VLA backend import."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from crane_x7_vla.backends import get_backend
from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files
from crane_x7_vla.training.cli import create_default_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        value = torch.ones(2, device="cuda") * 2
        assert value.tolist() == [2.0, 2.0]

    files = find_tfrecord_files(args.data_root)
    print(f"tfrecord_files={len(files)}")
    if not files:
        raise RuntimeError("No TFRecord files found")
    example = next(iter(TFRecordReader(files)))
    assert len(example["action"]) == 8
    assert len(example["observation/proprio"]) == 8
    print(f"first_example_keys={sorted(example)}")

    for name in ("openvla", "openvla-oft", "minivla", "pi0", "pi0.5"):
        config = create_default_config(name, args.data_root, args.data_root / "outputs", "runtime_probe")
        backend = get_backend(name)(config)
        print(f"backend={name} class={type(backend).__name__}")


if __name__ == "__main__":
    main()
