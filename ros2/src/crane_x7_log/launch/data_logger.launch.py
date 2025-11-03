#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch file for CRANE-X7 data logger."""

    # Get config file path
    config_file = os.path.join(
        get_package_share_directory('crane_x7_log'),
        'config',
        'logger_config.yaml'
    )

    # Declare launch arguments (these override config file values)
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to YAML config file'
    )

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='',
        description='Directory to save logged data (overrides config)'
    )

    declare_use_camera = DeclareLaunchArgument(
        'use_camera',
        default_value='',
        description='Enable camera data logging (overrides config)'
    )

    # Data Logger Node
    data_logger_node = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        parameters=[LaunchConfiguration('config_file')]
    )

    return LaunchDescription([
        declare_config_file,
        declare_output_dir,
        declare_use_camera,
        data_logger_node,
    ])
