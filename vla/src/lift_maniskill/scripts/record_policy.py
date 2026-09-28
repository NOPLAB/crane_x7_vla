# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Record a rollout of the PickPlace-CRANE-X7 policy (Dreamer or random fallback).

Example
-------
TMPDIR=$PWD python record_policy.py --output policy_rollout.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import dreamer_pickplace as dp  # noqa: E402

# Allow torch.load on checkpoints created when dreamer_pickplace.py was run as __main__
os.sys.modules["__main__"] = dp


def load_agent(checkpoint: Path) -> Optional[dp.Agent]:
    if not checkpoint.exists():
        return None
    obj = torch.load(checkpoint, map_location="cpu")
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Module):
        obj.to("cpu")
        obj.eval()
    return obj


def to_frame_array(frame) -> np.ndarray:
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    if hasattr(frame, "squeeze"):
        frame = frame.squeeze(0)
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame)


def record_episode(
    agent: Optional[dp.Agent], output: Path, steps: int, fps: int
) -> None:
    base_env = gym.make(
        "PickPlace-CRANE-X7",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
        robot_uids="CRANE-X7",
        obs_mode="rgb",
    )
    env = dp.HandCameraWrapper(base_env, image_size=64)
    obs, info = env.reset()

    if agent is not None:
        agent.reset()

    frames = []
    frames.append(to_frame_array(base_env.render()))

    for _ in range(steps):
        if agent is None:
            action = base_env.action_space.sample()
        else:
            action, _ = agent(obs, eval=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(to_frame_array(base_env.render()))
        if terminated or truncated:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record PickPlace-CRANE-X7 policy rollout."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("dreamer_agent.pth"))
    parser.add_argument(
        "--output", type=Path, default=Path("policy_videos/policy_rollout.mp4")
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    tmp_dir = os.environ.get("TMPDIR")
    if not tmp_dir:
        os.environ["TMPDIR"] = str(PROJECT_ROOT)

    agent = load_agent(args.checkpoint)
    if agent is None:
        print(
            f"[INFO] No valid checkpoint at {args.checkpoint}. Falling back to random actions."
        )
    else:
        print(f"[INFO] Loaded policy from {args.checkpoint}.")

    record_episode(agent, args.output, steps=args.steps, fps=args.fps)
    print(f"[INFO] Saved rollout video to {args.output.resolve()}")


if __name__ == "__main__":
    main()
