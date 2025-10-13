#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORKSPACE_DIR=$SCRIPT_DIR/../..
ROS2_WORKSPACE=$WORKSPACE_DIR/ros2

cd $ROS2_WORKSPACE

# Wait for build to complete
while [ ! -f "$ROS2_WORKSPACE/install/setup.bash" ]; do
    echo "Waiting for build to complete..."
    sleep 2
done

source $ROS2_WORKSPACE/install/setup.bash

echo "ROS 2 workspace ready!"

exec "$@"
