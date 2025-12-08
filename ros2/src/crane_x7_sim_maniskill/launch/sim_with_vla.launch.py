#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch ManiSkill simulation with VLA inference."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description."""

    pkg_dir = get_package_share_directory('crane_x7_sim_maniskill')
    config_file = os.path.join(pkg_dir, 'config', 'maniskill_config.yaml')

    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to VLA model'
    )

    declare_task_instruction = DeclareLaunchArgument(
        'task_instruction',
        default_value='pick up the object',
        description='Task instruction for VLA'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for VLA inference'
    )

    declare_sim_backend = DeclareLaunchArgument(
        'sim_backend',
        default_value='gpu',
        description='Simulation backend (cpu or gpu)'
    )

    maniskill_sim_node = Node(
        package='crane_x7_sim_maniskill',
        executable='maniskill_sim_node',
        name='maniskill_sim_node',
        output='screen',
        parameters=[
            config_file,
            {
                'sim_backend': LaunchConfiguration('sim_backend'),
            }
        ]
    )

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
        declare_sim_backend,
        maniskill_sim_node,
        vla_control,
    ])
