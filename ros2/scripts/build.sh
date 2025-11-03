#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop


SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$SCRIPT_DIR/../..
DOCKER_DEV_IMAGE_NAME=ros-dev

# Build dev
docker build -t $DOCKER_DEV_IMAGE_NAME --target dev -f $PROJECT_ROOT/ros2/Dockerfile $PROJECT_ROOT
