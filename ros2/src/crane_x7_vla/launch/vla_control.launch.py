#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch VLA inference and robot control nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Setup launch configuration."""
    # Get launch arguments
    task_instruction = LaunchConfiguration('task_instruction')
    config_file = LaunchConfiguration('config_file')
    use_flash_attention = LaunchConfiguration('use_flash_attention')
    device = LaunchConfiguration('device')
    auto_execute = LaunchConfiguration('auto_execute')

    # VLA inference node
    vla_inference_node = Node(
        package='crane_x7_vla',
        executable='vla_inference_node',
        name='vla_inference_node',
        output='screen',
        parameters=[
            config_file,
            {
                'task_instruction': task_instruction,
                'use_flash_attention': use_flash_attention,
                'device': device,
            }
        ],
    )

    # Robot controller node
    robot_controller_node = Node(
        package='crane_x7_vla',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'auto_execute': auto_execute,
            }
        ],
    )

    return [vla_inference_node, robot_controller_node]


def generate_launch_description():
    """Generate launch description."""
    # Declare launch arguments
    declare_task_instruction = DeclareLaunchArgument(
        'task_instruction',
        default_value='pick up the object',
        description='Task instruction for the robot'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_vla'),
            'config',
            'vla_config.yaml'
        ]),
        description='Path to configuration file'
    )

    declare_use_flash_attention = DeclareLaunchArgument(
        'use_flash_attention',
        default_value='false',
        description='Enable Flash Attention 2'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run inference on (cuda or cpu)'
    )

    declare_auto_execute = DeclareLaunchArgument(
        'auto_execute',
        default_value='true',
        description='Automatically execute VLA-predicted actions'
    )

    return LaunchDescription([
        declare_task_instruction,
        declare_config_file,
        declare_use_flash_attention,
        declare_device,
        declare_auto_execute,
        OpaqueFunction(function=launch_setup)
    ])
