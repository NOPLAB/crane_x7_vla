#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORKSPACE_DIR=$SCRIPT_DIR/../..
ROS2_WORKSPACE=$WORKSPACE_DIR/ros2

cd $ROS2_WORKSPACE

source $ROS2_WORKSPACE/install/setup.bash

echo "ROS 2 workspace ready!"

# Check user groups
echo "=== User Groups Check ==="
echo "Current user: $(whoami) (UID: $(id -u), GID: $(id -g))"
echo "Groups: $(groups)"
echo ""

# Check video devices
echo "=== Video Device Check ==="
if command -v v4l2-ctl &> /dev/null; then
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "No /dev/video* devices found"
    echo ""

    # Check if user can access video devices
    if [ -e /dev/video0 ]; then
        if [ -r /dev/video0 ] && [ -w /dev/video0 ]; then
            echo "✓ User has read/write access to /dev/video0"
        else
            echo "✗ WARNING: User does NOT have access to /dev/video0"
            echo "  Device permissions: $(ls -l /dev/video0)"
            echo "  Required group: $(stat -c '%G (%g)' /dev/video0)"
        fi
    fi
    echo ""
fi

# Check RealSense camera connection
if command -v rs-enumerate-devices &> /dev/null; then
    echo "=== RealSense Device Check ==="
    RS_OUTPUT=$(rs-enumerate-devices -s 2>&1)
    if echo "$RS_OUTPUT" | grep -q "Intel RealSense"; then
        echo "✓ RealSense camera detected by librealsense!"
        echo "$RS_OUTPUT" | head -n 20
    else
        echo "ℹ INFO: rs-enumerate-devices did not detect RealSense camera."
        echo "  This is normal if:"
        echo "  - ROS 2 realsense2_camera node is working (check 'ros2 topic list')"
        echo "  - Camera topics are publishing (e.g., /camera/color/image_raw)"
        echo ""
        echo "  If ROS 2 topics are NOT working, troubleshoot:"
        echo "  1. Check USB connection: lsusb | grep Intel"
        echo "  2. Verify privileged mode is enabled in docker-compose.yml"
        echo "  3. Check user has access to video devices (see above)"
        echo ""
        echo "  rs-enumerate-devices output:"
        echo "$RS_OUTPUT" | head -n 10
    fi
else
    echo "=== RealSense Device Check ==="
    echo "ℹ INFO: rs-enumerate-devices not available."
    echo "  RealSense detection via librealsense2-utils skipped."
    echo "  ROS 2 realsense2_camera node should still work if camera is connected."
fi
echo "=========================="
echo ""

exec "$@"
