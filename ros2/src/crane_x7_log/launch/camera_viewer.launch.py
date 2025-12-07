#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for RealSense camera viewer.
Displays camera image streams using rviz2.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch camera viewer for RealSense D435."""

    # Launch arguments
    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Image topic to display'
    )

    # rviz2 camera viewer
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('crane_x7_log'),
        'config', 'camera_viewer.rviz'
    ])
    rviz_viewer = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription([
        declare_image_topic,
        rviz_viewer,
    ])
