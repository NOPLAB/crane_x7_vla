#!/bin/bash

DOCKER_IMAGE_NAME=ros-dev
DOCKER_CONTAINER_NAME=ros-dev
DISPLAY=host.docker.internal:0.0

docker run \
    -e DISPLAY=$DISPLAY	\
    --name=$DOCKER_CONTAINER_NAME \
    --rm \
    -it \
    $DOCKER_IMAGE_NAME

