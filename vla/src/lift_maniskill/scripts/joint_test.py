# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import MJCFLoader

loader = MJCFLoader()
loader.set_scene(ManiSkillScene())
robot = loader.load("robot.xml")
print(robot.active_joints_map.keys())
