#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DOCKER_FILE_DIR=$SCRIPT_DIR
DOCKER_DEV_IMAGE_NAME=ros-dev

# Build dev
docker build -t $DOCKER_DEV_IMAGE_NAME --target dev $DOCKER_FILE_DIR

