# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

import time

import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot.xml")

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
