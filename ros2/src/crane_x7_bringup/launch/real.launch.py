#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7実機制御のbringup launchファイル。

crane_x7_control + MoveIt2 + データロガー + カメラビューアを統合して起動する。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7のUSBポート名
  - use_d435 (default: false): RealSense D435カメラを使用
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
    """Launch real robot control with optional data logging."""
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

    # Include robot control from crane_x7_control
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_control'),
                'launch',
                'crane_x7_control.launch.py'
            ])
        ])
    )

    # Include MoveIt2 from crane_x7_moveit_config
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_moveit_config'),
                'launch',
                'run_move_group.launch.py'
            ])
        ])
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
        declare_port_name,
        declare_use_d435,
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        control_launch,
        moveit_launch,
        data_logger_launch,
        camera_viewer_launch,
    ])
