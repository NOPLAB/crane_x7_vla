#!/bin/bash

DOCKER_IMAGE_NAME=ros-dev
DOCKER_CONTAINER_NAME=ros-dev

DOCKER_OPTION=
DOCKER_WSL_OPTION="-v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg"

if [[ "$(uname -r)" == *-microsoft-standard-WSL2 ]]; then
    # WSL
    DOCKER_OPTION=$DOCKER_WSL_OPTION
else
    # Other
    DISPLAY=host.docker.internal:0.0
fi

docker run \
    -e DISPLAY=$DISPLAY	\
    --name=$DOCKER_CONTAINER_NAME \
    --rm \
    -it \
    $DOCKER_OPTION \
    $DOCKER_IMAGE_NAME

