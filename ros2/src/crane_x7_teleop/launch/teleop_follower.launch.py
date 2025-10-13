#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file for CRANE-X7 teleoperation Follower mode.
The Follower robot receives and follows Leader's joint angles (torque ON).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    # Config files
    crane_x7_control_dir = get_package_share_directory('crane_x7_control')
    config_file = PathJoinSubstitution([
        FindPackageShare('crane_x7_teleop'),
        'config', 'teleop_config.yaml'
    ])

    # teleop_hardware_node (Follower mode)
    teleop_hardware = Node(
        package='crane_x7_teleop',
        executable='teleop_hardware_node',
        name='teleop_hardware_node',
        output='screen',
        parameters=[
            config_file,
            {
                'mode': 'follower',
                'port_name': LaunchConfiguration('port_name'),
                'config_file': os.path.join(crane_x7_control_dir, 'config', 'manipulator_config.yaml'),
                'links_file': os.path.join(crane_x7_control_dir, 'config', 'manipulator_links.csv')
            }
        ]
    )

    return LaunchDescription([
        declare_port_name,
        teleop_hardware
    ])
