#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7実機制御のbringup launchファイル。

crane_x7_log/real.launch.pyをラップして、実機制御を起動する。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7のUSBポート名
  - use_d435 (default: false): RealSense D435カメラを使用
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
    """Launch real robot control."""
    # Declare launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='false',
        description='Use RealSense D435 camera'
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

    # Include crane_x7_log real.launch.py
    real_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_log'),
                'launch',
                'real.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name'),
            'use_d435': LaunchConfiguration('use_d435'),
            'use_logger': LaunchConfiguration('use_logger'),
            'use_viewer': LaunchConfiguration('use_viewer'),
            'output_dir': LaunchConfiguration('output_dir'),
        }.items()
    )

    return LaunchDescription([
        declare_port_name,
        declare_use_d435,
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        real_launch,
    ])
