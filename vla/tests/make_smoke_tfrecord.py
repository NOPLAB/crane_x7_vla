"""Generate small synthetic CRANE-X7 episodes for runtime checks.

The generated data only exercises I/O and model wiring. It does not represent
real robot behavior or measure policy quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tfrecord.writer import TFRecordWriter


def make_episode(output: Path, steps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = TFRecordWriter(str(output))
    try:
        for step in range(steps):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            x = 20 + (step * 2) % 170
            cv2.rectangle(image, (x, 92), (x + 28, 120), (0, 0, 255), -1)
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            phase = step / max(steps - 1, 1)
            action = np.array([phase, -phase, 0.1, 0.2, -0.1, 0.3, 0.0, phase], dtype=np.float32)
            writer.write(
                {
                    "observation/proprio": (action.tolist(), "float"),
                    "observation/image_primary": (encoded.tobytes(), "byte"),
                    "observation/image_wrist": (encoded.tobytes(), "byte"),
                    "observation/timestep": (step, "int"),
                    "action": (action.tolist(), "float"),
                    "task/language_instruction": (b"move the red cube", "byte"),
                    "dataset_name": (b"crane_x7", "byte"),
                }
            )
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    for index in range(2):
        make_episode(args.output_dir / f"episode_{index:04d}" / "episode_data.tfrecord", args.steps)
    print(f"Wrote 2 synthetic episodes with {args.steps} steps each to {args.output_dir}")
