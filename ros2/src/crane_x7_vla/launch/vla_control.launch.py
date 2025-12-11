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
    model_path = LaunchConfiguration('model_path')
    task_instruction = LaunchConfiguration('task_instruction')
    config_file = LaunchConfiguration('config_file')
    use_flash_attention = LaunchConfiguration('use_flash_attention')
    device = LaunchConfiguration('device')
    auto_execute = LaunchConfiguration('auto_execute')
    image_topic = LaunchConfiguration('image_topic')

    # VLA inference node
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
                'image_topic': image_topic,
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

    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='RGB image topic for VLA inference'
    )

    return LaunchDescription([
        declare_model_path,
        declare_task_instruction,
        declare_config_file,
        declare_use_flash_attention,
        declare_device,
        declare_auto_execute,
        declare_image_topic,
        OpaqueFunction(function=launch_setup)
    ])
