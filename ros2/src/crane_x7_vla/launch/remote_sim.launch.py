#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch CRANE-X7 Gazebo simulation for remote VLA control (no VLA inference node).

This launch file starts the Gazebo simulation and robot_controller node
without the VLA inference node. The inference node should be running on a remote
GPU server (Runpod/Vast.ai) and communicating via Tailscale VPN.

Usage:
  ros2 launch crane_x7_vla remote_sim.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gazebo simulation + robot_controller (no VLA inference node)."""
    # Declare launch arguments
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

    # Include CRANE-X7 Gazebo simulation with D435 camera
    crane_x7_sim_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_sim_gazebo'),
                'launch',
                'pick_and_place.launch.py'
            ])
        ]),
        launch_arguments={
            'use_d435': 'true',
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
        declare_use_viewer,
        declare_auto_execute,
        declare_config_file,
        crane_x7_sim_gazebo,
        robot_controller_node,
        camera_viewer,
    ])
