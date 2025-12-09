#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Gazeboシミュレーションのbringup launchファイル。

crane_x7_log/sim.launch.pyをラップして、Gazeboシミュレーションを起動する。

引数:
  - use_logger (default: true): データロガーを有効化
  - use_viewer (default: false): カメラビューア(rviz2)を表示
  - output_dir: ログデータの保存先
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gazebo simulation."""
    # Declare launch arguments
    declare_use_logger = DeclareLaunchArgument(
        'use_logger',
        default_value='true',
        description='Enable data logger'
    )

    declare_use_viewer = DeclareLaunchArgument(
        'use_viewer',
        default_value='false',
        description='Display camera viewer (rviz2)'
    )

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/workspace/data/tfrecord_logs',
        description='Directory to save logged data'
    )

    # Include crane_x7_log sim.launch.py
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_log'),
                'launch',
                'sim.launch.py'
            ])
        ]),
        launch_arguments={
            'use_logger': LaunchConfiguration('use_logger'),
            'use_viewer': LaunchConfiguration('use_viewer'),
            'output_dir': LaunchConfiguration('output_dir'),
        }.items()
    )

    return LaunchDescription([
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        sim_launch,
    ])
