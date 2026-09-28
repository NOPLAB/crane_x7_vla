# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.crane_x7 import CraneX7  # noqa: E402

CraneX7.mjcf_path = str((PROJECT_ROOT / "robot" / "crane_x7.xml").resolve())

env = gym.make(
    "PickPlace-CRANE-X7",
    render_mode="rgb_array",
    sim_backend="cpu",
    render_backend="cpu",
    robot_uids="CRANE-X7",
)
env.reset()
img = env.render()
if isinstance(img, (list, tuple)):
    img = img[0]
elif hasattr(img, "shape") and len(img.shape) == 4:
    img = img[0]

plt.imshow(img)
plt.axis("off")
backend = plt.get_backend().lower()
plt.show()

env.close()
