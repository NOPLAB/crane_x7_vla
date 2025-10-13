#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file for CRANE-X7 teleoperation Follower mode with data logger.
This launches both the Follower hardware node and the data logger for VLA training.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7'
    )

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/workspace/data/tfrecord_logs',
        description='Directory to save logged data'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_log'),
            'config', 'logger_config.yaml'
        ]),
        description='Path to logger config file'
    )

    # Include teleop_follower.launch.py
    teleop_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_teleop'),
                'launch', 'teleop_follower.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name')
        }.items()
    )

    # Data logger node
    data_logger = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'output_dir': LaunchConfiguration('output_dir')}
        ]
    )

    return LaunchDescription([
        declare_port_name,
        declare_output_dir,
        declare_config_file,
        teleop_follower,
        data_logger
    ])
