#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop


SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORKSPACE_DIR=$SCRIPT_DIR/../..
ROS2_WORKSPACE=$WORKSPACE_DIR/ros2

source $ROS2_WORKSPACE/install/setup.bash

echo "ROS 2 workspace ready!"

# Start bash shell
exec /bin/bash
