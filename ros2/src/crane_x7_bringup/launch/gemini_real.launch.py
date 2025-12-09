#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Gemini API統合（実機）のbringup launchファイル。

crane_x7_gemini/gemini_with_robot.launch.pyをラップする。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7のUSBポート名
  - use_d435 (default: true): RealSense D435カメラを使用
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gemini API integration with real robot."""
    # Declare launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Use RealSense D435 camera'
    )

    # Include gemini_with_robot.launch.py
    gemini_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_gemini'),
                'launch',
                'gemini_with_robot.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name'),
            'use_d435': LaunchConfiguration('use_d435'),
        }.items()
    )

    return LaunchDescription([
        declare_port_name,
        declare_use_d435,
        gemini_launch,
    ])
