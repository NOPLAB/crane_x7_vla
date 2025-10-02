#!/bin/bash

USB_DEVICE=$1

SCRIPT_DIR=$(cd $(dirname $0); pwd)
ROS2_WORKSPACE=$SCRIPT_DIR/../..

if [ ! -e USB_DEVICE ]; then
    echo "Not found USB device: $USB_DEVICE"
fi

cd $ROS2_WORKSPACE

colcon build --symlink-install

source $ROS2_WORKSPACE/install/setup.bash

ros2 launch crane_x7_examples demo.launch.py port_name:=$USB_DEVICE

