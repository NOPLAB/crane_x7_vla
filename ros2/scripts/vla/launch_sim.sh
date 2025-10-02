#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
ROS2_WORKSPACE=$SCRIPT_DIR/../..

cd $ROS2_WORKSPACE

colcon build --symlink-install

source $ROS2_WORKSPACE/install/setup.bash

ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py

