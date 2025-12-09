#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Liftシミュレーションのbringup launchファイル。

crane_x7_lift/sim.launch.pyをラップする。

引数:
  - simulator (default: maniskill): シミュレータバックエンド
  - backend (default: gpu): シミュレーションバックエンド
  - sim_rate (default: 30.0): シミュレーションレート
  - env_id (default: PickPlace-CRANE-X7): 環境ID
  - auto_reset (default: true): エピソード終了時の自動リセット
  - render_mode (default: rgb_array): レンダリングモード
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch lift simulation."""
    declare_simulator = DeclareLaunchArgument(
        'simulator',
        default_value='maniskill',
        description='Simulator backend (maniskill, genesis, isaacsim)'
    )

    declare_backend = DeclareLaunchArgument(
        'backend',
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
        description='Environment ID'
    )

    declare_auto_reset = DeclareLaunchArgument(
        'auto_reset',
        default_value='true',
        description='Auto reset on episode end'
    )

    declare_render_mode = DeclareLaunchArgument(
        'render_mode',
        default_value='rgb_array',
        description='Render mode (rgb_array, human, none)'
    )

    lift_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_lift'),
                'launch',
                'sim.launch.py'
            ])
        ]),
        launch_arguments={
            'simulator': LaunchConfiguration('simulator'),
            'backend': LaunchConfiguration('backend'),
            'sim_rate': LaunchConfiguration('sim_rate'),
            'env_id': LaunchConfiguration('env_id'),
            'auto_reset': LaunchConfiguration('auto_reset'),
            'render_mode': LaunchConfiguration('render_mode'),
        }.items()
    )

    return LaunchDescription([
        declare_simulator,
        declare_backend,
        declare_sim_rate,
        declare_env_id,
        declare_auto_reset,
        declare_render_mode,
        lift_sim,
    ])
