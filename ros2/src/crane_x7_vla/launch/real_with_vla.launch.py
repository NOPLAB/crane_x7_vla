#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch CRANE-X7 real robot with VLA control."""

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

    # Include CRANE-X7 demo launch (MoveIt2 + hardware control)
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
        declare_port_name,
        declare_use_d435,
        declare_task_instruction,
        declare_device,
        crane_x7_demo,
        vla_control,
    ])
