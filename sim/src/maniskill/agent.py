# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""CRANE-X7 ManiSkill agent definition."""

import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

from robot.crane_x7 import CraneX7Config, get_mjcf_path


@register_agent()
class CraneX7(BaseAgent):
    """CRANE-X7 robot agent for ManiSkill."""

    uid = "CRANE-X7"
    mjcf_path = get_mjcf_path()

    keyframes = dict(
        rest=Keyframe(
            qpos=CraneX7Config.REST_QPOS,
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = CraneX7Config.ARM_JOINT_NAMES
    gripper_joint_names = CraneX7Config.GRIPPER_JOINT_NAMES

    arm_stiffness = CraneX7Config.ARM_STIFFNESS
    arm_damping = CraneX7Config.ARM_DAMPING
    arm_force_limit = CraneX7Config.ARM_FORCE_LIMIT

    gripper_stiffness = CraneX7Config.GRIPPER_STIFFNESS
    gripper_damping = CraneX7Config.GRIPPER_DAMPING
    gripper_force_limit = CraneX7Config.GRIPPER_FORCE_LIMIT

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                "crane_x7_gripper_finger_b_joint": {
                    "joint": "crane_x7_gripper_finger_a_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                }
            },
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
        )
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):
        p = [0.0, 0.0445, 0.034]
        q = [np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25)]

        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=p, q=q),
                width=640,
                height=480,
                fov=np.deg2rad(69),
                near=0.01,
                far=10.0,
                mount=self.robot.links_map["crane_x7_gripper_base_link"],
            )
        ]
