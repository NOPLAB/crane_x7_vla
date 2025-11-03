#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

set -e  # Exit on error

# Resolve script directory using realpath for absolute path resolution
SCRIPT_DIR=$(dirname $(realpath $0))
# Script is at /workspace/ros2/scripts/docker, go up 3 levels to /workspace
WORKSPACE_DIR=$(realpath $SCRIPT_DIR/../../..)
ROS2_WORKSPACE=$WORKSPACE_DIR/ros2

echo "=== ROS 2 Workspace Setup ==="
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "ROS2_WORKSPACE: $ROS2_WORKSPACE"

if [ ! -d "$ROS2_WORKSPACE" ]; then
    echo "ERROR: ROS2_WORKSPACE directory not found: $ROS2_WORKSPACE"
    exit 1
fi

if [ ! -f "$ROS2_WORKSPACE/install/setup.bash" ]; then
    echo "ERROR: setup.bash not found: $ROS2_WORKSPACE/install/setup.bash"
    echo "Please build the workspace first."
    exit 1
fi

source $ROS2_WORKSPACE/install/setup.bash

echo "ROS 2 workspace ready!"
echo "============================="
echo ""

# Start bash shell
exec /bin/bash
