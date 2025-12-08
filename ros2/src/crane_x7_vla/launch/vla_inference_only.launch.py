#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch VLA inference node only (for remote GPU deployment).

This launch file starts only the VLA inference node without the robot controller.
Used for remote GPU servers (Runpod/Vast.ai) that communicate with the local robot
via Tailscale VPN.

The robot_controller node should be running on the local robot machine
using remote_robot.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Setup launch configuration."""
    model_path = LaunchConfiguration('model_path')
    task_instruction = LaunchConfiguration('task_instruction')
    config_file = LaunchConfiguration('config_file')
    use_flash_attention = LaunchConfiguration('use_flash_attention')
    device = LaunchConfiguration('device')

    # VLA inference node only (no robot_controller)
    vla_inference_node = Node(
        package='crane_x7_vla',
        executable='vla_inference_node',
        name='vla_inference_node',
        output='screen',
        parameters=[
            config_file,
            {
                'model_path': model_path,
                'task_instruction': task_instruction,
                'use_flash_attention': use_flash_attention,
                'device': device,
            }
        ],
    )

    return [vla_inference_node]


def generate_launch_description():
    """Generate launch description for VLA inference only."""
    # Declare launch arguments with environment variable defaults
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value=EnvironmentVariable('VLA_MODEL_PATH', default_value=''),
        description='Path to VLA model checkpoint (required)'
    )

    declare_task_instruction = DeclareLaunchArgument(
        'task_instruction',
        default_value=EnvironmentVariable(
            'VLA_TASK_INSTRUCTION',
            default_value='pick up the object'
        ),
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
        default_value=EnvironmentVariable('VLA_DEVICE', default_value='cuda'),
        description='Device to run inference on (cuda or cpu)'
    )

    return LaunchDescription([
        declare_model_path,
        declare_task_instruction,
        declare_config_file,
        declare_use_flash_attention,
        declare_device,
        OpaqueFunction(function=launch_setup)
    ])
