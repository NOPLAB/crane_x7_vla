#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file for CRANE-X7 teleoperation Leader mode with data logger.
This launches the Leader hardware node (for manual teaching) and the data logger for VLA training.

The Leader robot can be manually moved (torque OFF) and publishes joint angles.
The data logger subscribes to the Leader's joint state topic for demonstration recording.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch teleop leader with data logger."""

    # Get package directories
    crane_x7_log_dir = get_package_share_directory('crane_x7_log')

    # Paths
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7 Leader robot'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='false',
        description='Use RealSense D435 camera for visual observations'
    )

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/workspace/data/tfrecord_logs',
        description='Directory to save logged demonstration data'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=logger_config,
        description='Path to data logger config file'
    )

    # Include teleop_leader.launch.py from crane_x7_teleop package
    teleop_leader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_teleop'),
                'launch', 'teleop_leader.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name')
        }.items()
    )

    # RealSense D435 camera node (optional)
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('use_d435')),
        launch_arguments={
            'camera_namespace': '',
            'device_type': 'd435',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
        }.items()
    )

    # Data logger node - subscribes to /joint_states published by leader
    data_logger = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'output_dir': LaunchConfiguration('output_dir'),
                # Override joint_states topic to use leader's topic
                # Leader publishes to both /joint_states and /teleop/leader/state
                'joint_states_topic': '/joint_states'
            }
        ]
    )

    return LaunchDescription([
        declare_port_name,
        declare_use_d435,
        declare_output_dir,
        declare_config_file,
        teleop_leader,
        realsense_node,
        data_logger
    ])
