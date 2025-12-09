#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Gazeboシミュレーションのbringup launchファイル。

crane_x7_sim_gazebo + データロガー + カメラビューアを統合して起動する。

引数:
  - use_d435 (default: true): D435カメラを有効化
  - use_logger (default: true): データロガーを有効化
  - use_viewer (default: false): カメラビューア(rviz2)を表示
  - output_dir: ログデータの保存先
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gazebo simulation with optional data logging."""
    # Declare launch arguments
    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Enable RealSense D435 camera in simulation'
    )

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

    # Include data logger from crane_x7_log (conditional)
    data_logger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_log'),
                'launch',
                'data_logger.launch.py'
            ])
        ]),
        launch_arguments={
            'output_dir': LaunchConfiguration('output_dir'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_logger'))
    )

    # Include camera viewer from crane_x7_log (conditional)
    camera_viewer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_log'),
                'launch',
                'camera_viewer.launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('use_viewer'))
    )

    return LaunchDescription([
        declare_use_d435,
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        gazebo_launch,
        data_logger_launch,
        camera_viewer_launch,
    ])
