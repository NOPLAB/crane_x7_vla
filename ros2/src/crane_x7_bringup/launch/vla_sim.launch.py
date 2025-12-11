#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 VLA推論（シミュレーション）のbringup launchファイル。

Gazeboシミュレーション + VLA制御ノードを統合して起動する。

引数:
  - model_path: VLAモデルのパス
  - task_instruction (default: 'pick up the object'): タスク指示
  - device (default: cuda): 推論デバイス (cuda or cpu)
  - use_d435 (default: true): D435カメラを有効化
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch VLA inference with Gazebo simulation."""
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to VLA model'
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

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Enable RealSense D435 camera in simulation'
    )

    # Include Gazebo simulation from crane_x7_sim_gazebo
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_sim_gazebo'),
                'launch',
                'pick_and_place.launch.py'
            ])
        ]),
        launch_arguments={
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
        declare_task_instruction,
        declare_device,
        declare_use_d435,
        gazebo_launch,
        vla_control_launch,
    ])
