# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Pick and place environment for CRANE-X7."""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from lift_maniskill.agent import CraneX7


@register_env("PickPlace-CRANE-X7", max_episode_steps=200)
class PickPlace(BaseEnv):
    """Pick and place task environment for CRANE-X7."""

    SUPPORTED_ROBOTS = ["CRANE-X7"]
    agent: Union[CraneX7]

    goal_radius = 0.1
    cube_half_size = 0.02
    cube_spawn_half_size = 0.1
    cube_spawn_center = np.array([0.15, 0.02])
    cube_spawn_jitter = np.array([0.01, 0.01])
    lift_height_offset = 0.12
    grasp_distance_threshold = 0.05

    def __init__(self, *args, robot_uids="robot", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.lift_success_height = self.cube_half_size + self.lift_height_offset
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self) -> SimConfig:
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    def _load_agent(self, options: dict) -> None:
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict) -> None:
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)]

    def _setup_sensors(self, options: dict):
        return super()._setup_sensors(options)

    def _load_lighting(self, options: dict):
        return super()._load_lighting(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.robot_uids == CraneX7.uid:
                rest_qpos = self.agent.keyframes["rest"].qpos
                if rest_qpos.ndim == 1:
                    rest_qpos = np.tile(rest_qpos, (batch, 1))
                else:
                    rest_qpos = np.array(rest_qpos, copy=True)
                qpos = rest_qpos.copy()
                if self.robot_init_qpos_noise > 0:
                    if self._enhanced_determinism:
                        noise = self._batched_episode_rng[env_idx].normal(
                            0, self.robot_init_qpos_noise, qpos.shape[1]
                        )
                    else:
                        noise = self._episode_rng.normal(
                            0, self.robot_init_qpos_noise, qpos.shape
                        )
                    qpos = qpos + noise
                self.agent.reset(qpos)
            xyz = torch.zeros((batch, 3), device=self.device)
            xy = torch.as_tensor(
                self.cube_spawn_center, dtype=torch.float32, device=self.device
            ).repeat(batch, 1)
            if np.any(self.cube_spawn_jitter):
                jitter = (
                    torch.rand((batch, 2), device=self.device) * 2 - 1
                ) * torch.as_tensor(
                    self.cube_spawn_jitter, dtype=torch.float32, device=self.device
                )
                xy = xy + jitter
            xyz[:, :2] = xy
            xyz[:, 2] = self.cube_half_size
            quat = torch.zeros((batch, 4), device=self.device)
            quat[:, 3] = 1.0
            self.obj.set_pose(Pose.create_from_pq(xyz, quat))

    def evaluate(self):
        metrics = self._compute_task_metrics()
        return {
            "success": metrics["success"],
            "fail": torch.zeros_like(metrics["success"]),
            "height_reached": metrics["height_reached"],
            "is_close": metrics["is_close"],
            "gripper_to_cube_dist": metrics["distance"],
        }

    def _get_obs_extra(self, info: dict):
        metrics = self._compute_task_metrics()
        return {
            "height_reached": metrics["height_reached"],
            "gripper_to_cube_dist": metrics["distance"],
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        metrics = self._compute_task_metrics()
        distance = metrics["distance"]
        reaching_reward = 1 - torch.tanh(5 * distance)

        cube_height = metrics["cube_height"]
        lift_progress = torch.clamp(
            (cube_height - self.cube_half_size)
            / max(self.lift_success_height - self.cube_half_size, 1e-6),
            min=0.0,
            max=1.0,
        )
        lift_reward = lift_progress

        grasp_bonus = torch.exp(-10 * distance)

        reward = reaching_reward + lift_reward + grasp_bonus
        success = metrics["success"]
        reward[success] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    def get_state_dict(self):
        return super().get_state_dict()

    def set_state_dict(self, state):
        super().set_state_dict(state)

    def _compute_task_metrics(self) -> Dict[str, torch.Tensor]:
        cube_pos = self.obj.pose.p
        gripper_link = self.agent.robot.links_map["crane_x7_gripper_base_link"]
        gripper_pos = gripper_link.pose.p
        distance = torch.linalg.norm(cube_pos - gripper_pos, axis=1)
        height = cube_pos[:, 2]
        height_reached = height >= self.lift_success_height
        is_close = distance <= self.grasp_distance_threshold
        success = height_reached & is_close
        return {
            "distance": distance,
            "cube_height": height,
            "height_reached": height_reached,
            "is_close": is_close,
            "success": success,
        }
