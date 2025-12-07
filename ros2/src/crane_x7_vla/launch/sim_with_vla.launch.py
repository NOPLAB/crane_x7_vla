#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch CRANE-X7 Gazebo simulation with VLA control."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description."""
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to VLA model (e.g., /workspace/vla/outputs/<model_dir>/checkpoint-1500)'
    )

    declare_task_instruction = DeclareLaunchArgument(
        'task_instruction',
        default_value='pick up the object',
        description='Task instruction for the robot'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run inference on (cuda or cpu)'
    )

    # Include CRANE-X7 Gazebo launch with D435 camera
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

    # Include VLA control launch
    vla_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_vla'),
                'launch',
                'vla_control.launch.py'
            ])
        ]),
        launch_arguments={
            'model_path': LaunchConfiguration('model_path'),
            'task_instruction': LaunchConfiguration('task_instruction'),
            'device': LaunchConfiguration('device'),
        }.items()
    )

    return LaunchDescription([
        declare_model_path,
        declare_task_instruction,
        declare_device,
        crane_x7_sim_gazebo,
        vla_control,
    ])
