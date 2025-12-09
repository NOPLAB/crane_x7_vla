# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ManiSkill simulator adapter implementing lift interface."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from lift.interface import Simulator
from lift.types import Observation, SimulatorConfig, StepResult
from lift.factory import register_simulator
from robot.crane_x7 import CraneX7Config


@register_simulator("maniskill")
class ManiSkillSimulator(Simulator):
    """ManiSkill simulator implementation."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self._env: Optional[gym.Env] = None
        self._current_obs = None
        self._init_environment()

    def _init_environment(self) -> None:
        """Initialize ManiSkill environment."""
        from maniskill.agent import CraneX7  # noqa: F401
        from maniskill.environments import PickPlace  # noqa: F401

        self._env = gym.make(
            self.config.env_id,
            render_mode=self.config.render_mode,
            sim_backend=self.config.backend,
            robot_uids="CRANE-X7",
            obs_mode="rgb",
            control_mode=self.config.control_mode,
        )

        obs, info = self._env.reset()
        self._current_obs = obs
        self._is_running = True

    @property
    def arm_joint_names(self) -> list[str]:
        return CraneX7Config.ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return CraneX7Config.GRIPPER_JOINT_NAMES

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            obs, info = self._env.reset(seed=seed)
        else:
            obs, info = self._env.reset()
        self._current_obs = obs
        return self._convert_observation(obs), info

    def step(self, action: np.ndarray) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._current_obs = obs
        return StepResult(
            observation=self._convert_observation(obs),
            reward=float(reward) if np.isscalar(reward) else float(reward[0]),
            terminated=bool(terminated)
            if np.isscalar(terminated)
            else bool(terminated[0]),
            truncated=bool(truncated) if np.isscalar(truncated) else bool(truncated[0]),
            info=info,
        )

    def get_observation(self) -> Observation:
        return self._convert_observation(self._current_obs)

    def get_qpos(self) -> np.ndarray:
        qpos = self._env.agent.robot.get_qpos()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        if qpos.ndim == 2:
            qpos = qpos[0]
        return qpos

    def get_qvel(self) -> np.ndarray:
        qvel = self._env.agent.robot.get_qvel()
        if isinstance(qvel, torch.Tensor):
            qvel = qvel.cpu().numpy()
        if qvel.ndim == 2:
            qvel = qvel[0]
        return qvel

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._is_running = False

    def _convert_observation(self, obs: dict) -> Observation:
        """Convert ManiSkill observation to lift Observation."""
        rgb_image = None
        depth_image = None

        if "sensor_data" in obs:
            hand_cam = obs["sensor_data"].get("hand_camera", {})
            rgb = hand_cam.get("rgb")
            if rgb is not None:
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)
                rgb_image = rgb

            depth = hand_cam.get("depth")
            if depth is not None:
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                if depth.ndim == 4:
                    depth = depth[0]
                depth_image = depth

        qpos = self.get_qpos()
        qvel = self.get_qvel()

        return Observation(
            rgb_image=rgb_image,
            depth_image=depth_image,
            qpos=qpos,
            qvel=qvel,
            extra={"raw_obs": obs},
        )
