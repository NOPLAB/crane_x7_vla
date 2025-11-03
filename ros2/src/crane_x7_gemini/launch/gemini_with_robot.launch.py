#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for complete Gemini + CRANE-X7 system.

This launch file starts:
- CRANE-X7 robot control with MoveIt
- RealSense camera
- Gemini trajectory planner
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for complete Gemini + robot system."""

    # Declare launch arguments
    port_name_arg = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port for CRANE-X7'
    )

    use_d435_arg = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Use RealSense D435 camera'
    )

    api_key_arg = DeclareLaunchArgument(
        'api_key',
        default_value=os.environ.get('GEMINI_API_KEY', ''),
        description='Google Gemini API key'
    )

    execute_trajectory_arg = DeclareLaunchArgument(
        'execute_trajectory',
        default_value='true',
        description='Whether to execute planned trajectories'
    )

    # Include CRANE-X7 demo launch (MoveIt + hardware control)
    crane_x7_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_examples'),
                'launch',
                'demo.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name'),
            'use_d435': LaunchConfiguration('use_d435'),
        }.items()
    )

    # Include trajectory planner
    trajectory_planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_gemini'),
                'launch',
                'trajectory_planner.launch.py'
            ])
        ]),
        launch_arguments={
            'api_key': LaunchConfiguration('api_key'),
            'execute_trajectory': LaunchConfiguration('execute_trajectory'),
        }.items()
    )

    return LaunchDescription([
        port_name_arg,
        use_d435_arg,
        api_key_arg,
        execute_trajectory_arg,
        crane_x7_demo,
        trajectory_planner,
    ])
