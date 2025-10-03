#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file that starts CRANE-X7 Gazebo simulation with TFRecord data logger.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch Gazebo simulation with TFRecord logger."""

    # Get package directories
    crane_x7_gazebo_dir = get_package_share_directory('crane_x7_gazebo')
    crane_x7_log_dir = get_package_share_directory('crane_x7_log')

    # Paths
    gazebo_launch = os.path.join(crane_x7_gazebo_dir, 'launch', 'crane_x7_with_table.launch.py')
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
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

    # Include Gazebo simulation launch
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch)
    )

    # TFRecord Logger Node
    tfrecord_logger_node = Node(
        package='crane_x7_log',
        executable='tfrecord_logger',
        name='tfrecord_logger',
        output='screen',
        parameters=[LaunchConfiguration('config_file')]
    )

    return LaunchDescription([
        declare_output_dir,
        declare_config_file,
        gazebo_sim,
        tfrecord_logger_node,
    ])
