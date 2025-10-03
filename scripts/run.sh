#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
DOCKER_IMAGE_NAME=ros-dev
DOCKER_CONTAINER_NAME=ros-dev

DOCKER_OPTION=" -v $SCRIPT_DIR/..:/workspace"
DOCKER_WSL_OPTION="-v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg"
DOCKER_LINUX_OPTION="-v /tmp/.X11-unix:/tmp/.X11-unix"

if [[ "$(uname -r)" == *-microsoft-standard-WSL2 ]]; then
    # WSL
    echo "Run on WSL"
    DOCKER_OPTION="$DOCKER_OPTION $DOCKER_WSL_OPTION"
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Run on Linux"
    xhost +
    DOCKER_OPTION="$DOCKER_OPTION $DOCKER_LINUX_OPTION"
else
    echo "Cloudn't detect your system"
    return -1;
fi

docker run \
    -e DISPLAY=$DISPLAY	\
    --name=$DOCKER_CONTAINER_NAME \
    --rm \
    -it \
    $DOCKER_OPTION \
    $DOCKER_IMAGE_NAME \
    /bin/bash
