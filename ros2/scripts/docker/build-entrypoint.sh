#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORKSPACE_DIR=$SCRIPT_DIR/../..
ROS2_WORKSPACE=$WORKSPACE_DIR/ros2

cd $ROS2_WORKSPACE

rm -rf $ROS2_WORKSPACE/build $ROS2_WORKSPACE/install $ROS2_WORKSPACE/log

colcon build --symlink-install

source $ROS2_WORKSPACE/install/setup.bash

echo "Build completed successfully!"
