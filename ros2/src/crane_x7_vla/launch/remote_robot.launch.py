#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch CRANE-X7 robot for remote VLA control (robot_controller only, no VLA inference).

This launch file starts the CRANE-X7 hardware control and robot_controller node
without the VLA inference node. The inference node should be running on a remote
GPU server (Runpod/Vast.ai) and communicating via Tailscale VPN.

Usage:
  ros2 launch crane_x7_vla remote_robot.launch.py

The remote inference server should publish to /vla/predicted_action topic,
which robot_controller subscribes to and executes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch robot demo + robot_controller (no VLA inference node)."""
    # Declare launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port for CRANE-X7'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Use RealSense D435 camera'
    )

    declare_use_viewer = DeclareLaunchArgument(
        'use_viewer',
        default_value='false',
        description='Display RViz camera viewer'
    )

    declare_auto_execute = DeclareLaunchArgument(
        'auto_execute',
        default_value='true',
        description='Automatically execute received VLA actions'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_vla'),
            'config',
            'vla_config.yaml'
        ]),
        description='Path to VLA config file'
    )

    # Include CRANE-X7 demo launch (MoveIt2 + hardware control + camera)
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

    # Robot controller node (receives actions from remote VLA inference)
    robot_controller_node = Node(
        package='crane_x7_vla',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'auto_execute': LaunchConfiguration('auto_execute'),
            }
        ],
    )

    # Camera viewer (optional RViz)
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('crane_x7_log'),
        'config',
        'camera_viewer.rviz'
    ])
    camera_viewer = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_viewer'))
    )

    return LaunchDescription([
        declare_port_name,
        declare_use_d435,
        declare_use_viewer,
        declare_auto_execute,
        declare_config_file,
        crane_x7_demo,
        robot_controller_node,
        camera_viewer,
    ])
