#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
ManiSkillシミュレーションのbringup launchファイル。

crane_x7_sim_maniskill/sim_only.launch.pyをラップする。

引数:
  - sim_backend (default: gpu): シミュレーションバックエンド
  - sim_rate (default: 30.0): シミュレーションレート
  - env_id (default: PickPlace-CRANE-X7): 環境ID
  - auto_reset (default: true): エピソード終了時の自動リセット
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch ManiSkill simulation."""
    declare_sim_backend = DeclareLaunchArgument(
        'sim_backend',
        default_value='gpu',
        description='Simulation backend (cpu or gpu)'
    )

    declare_sim_rate = DeclareLaunchArgument(
        'sim_rate',
        default_value='30.0',
        description='Simulation rate in Hz'
    )

    declare_env_id = DeclareLaunchArgument(
        'env_id',
        default_value='PickPlace-CRANE-X7',
        description='ManiSkill environment ID'
    )

    declare_auto_reset = DeclareLaunchArgument(
        'auto_reset',
        default_value='true',
        description='Auto reset on episode end'
    )

    maniskill_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_sim_maniskill'),
                'launch',
                'sim_only.launch.py'
            ])
        ]),
        launch_arguments={
            'sim_backend': LaunchConfiguration('sim_backend'),
            'sim_rate': LaunchConfiguration('sim_rate'),
            'env_id': LaunchConfiguration('env_id'),
            'auto_reset': LaunchConfiguration('auto_reset'),
        }.items()
    )

    return LaunchDescription([
        declare_sim_backend,
        declare_sim_rate,
        declare_env_id,
        declare_auto_reset,
        maniskill_sim,
    ])
