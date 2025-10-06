#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file that starts CRANE-X7 real robot control with data logger.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch real robot control with data logger."""

    # Get package directories
    crane_x7_examples_dir = get_package_share_directory('crane_x7_examples')
    crane_x7_log_dir = get_package_share_directory('crane_x7_log')

    # Paths
    demo_launch = os.path.join(crane_x7_examples_dir, 'launch', 'demo.launch.py')
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
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

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/workspace/data/tfrecord_logs',
        description='Directory to save logged data'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=logger_config,
        description='Path to logger config file'
    )

    # Include real robot demo launch
    robot_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(demo_launch),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name'),
            'use_d435': LaunchConfiguration('use_d435'),
        }.items()
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
        declare_port_name,
        declare_use_d435,
        declare_output_dir,
        declare_config_file,
        robot_demo,
        data_logger_node,
    ])
