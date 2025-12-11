#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 VLA推論（実機）のbringup launchファイル。

実機制御 + VLA制御ノードを統合して起動する。

引数:
  - model_path: VLAモデルのパス
  - port_name (default: /dev/ttyUSB0): CRANE-X7のUSBポート名
  - use_d435 (default: true): RealSense D435カメラを使用
  - task_instruction (default: 'pick up the object'): タスク指示
  - device (default: cuda): 推論デバイス (cuda or cpu)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch VLA inference with real robot."""
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to VLA model'
    )

    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7'
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

    # Include crane_x7_examples demo (robot control + optional D435 camera)
    demo_launch = IncludeLaunchDescription(
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

    # Include VLA control nodes from crane_x7_vla
    vla_control_launch = IncludeLaunchDescription(
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
        demo_launch,
        vla_control_launch,
    ])
