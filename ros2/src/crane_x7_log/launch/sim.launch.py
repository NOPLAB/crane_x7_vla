#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Gazeboシミュレーションの統合launchファイル。

引数:
  - use_logger (default: true): データロガーを有効化
  - use_viewer (default: false): カメラビューア(rviz2)を表示
  - output_dir: ログデータの保存先
  - config_file: ロガー設定ファイルのパス
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gazebo simulation with optional logger and viewer."""

    # Get package directories
    crane_x7_sim_gazebo_dir = get_package_share_directory('crane_x7_sim_gazebo')
    crane_x7_log_dir = get_package_share_directory('crane_x7_log')

    # Paths
    gazebo_launch = os.path.join(crane_x7_sim_gazebo_dir, 'launch', 'pick_and_place.launch.py')
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
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

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=logger_config,
        description='Path to logger config file'
    )

    # Include Gazebo simulation launch
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch)
    )

    # Data Logger Node (conditional)
    data_logger_node = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_logger')),
        parameters=[
            LaunchConfiguration('config_file'),
            {'output_dir': LaunchConfiguration('output_dir')}
        ]
    )

    # Camera viewer (conditional)
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('crane_x7_log'),
        'config', 'camera_viewer.rviz'
    ])
    camera_viewer = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_viewer'))
    )

    return LaunchDescription([
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        declare_config_file,
        gazebo_sim,
        data_logger_node,
        camera_viewer,
    ])
