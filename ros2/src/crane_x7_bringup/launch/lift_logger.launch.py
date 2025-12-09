#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Liftシミュレーション + データロガーのbringup launchファイル。

引数:
  - output_dir (default: /workspace/data/tfrecord_logs): ログ保存先
  - episode_length (default: 200): エピソードあたりのステップ数
  - simulator (default: maniskill): シミュレータバックエンド
  - backend (default: gpu): シミュレーションバックエンド
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch lift simulation with data logger."""

    lift_pkg = get_package_share_directory('crane_x7_lift')
    log_pkg = get_package_share_directory('crane_x7_log')

    lift_config = os.path.join(lift_pkg, 'config', 'lift_config.yaml')
    logger_config = os.path.join(log_pkg, 'config', 'logger_config.yaml')

    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value='/workspace/data/tfrecord_logs',
        description='Directory to save logged data'
    )

    declare_episode_length = DeclareLaunchArgument(
        'episode_length',
        default_value='200',
        description='Steps per episode'
    )

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

    lift_sim_node = Node(
        package='crane_x7_lift',
        executable='lift_sim_node',
        name='lift_sim_node',
        output='screen',
        parameters=[
            lift_config,
            {
                'auto_reset': False,
                'simulator': LaunchConfiguration('simulator'),
                'backend': LaunchConfiguration('backend'),
            }
        ]
    )

    data_logger_node = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        parameters=[
            logger_config,
            {
                'output_dir': LaunchConfiguration('output_dir'),
                'episode_length': LaunchConfiguration('episode_length'),
            }
        ]
    )

    return LaunchDescription([
        declare_output_dir,
        declare_episode_length,
        declare_simulator,
        declare_backend,
        lift_sim_node,
        data_logger_node,
    ])
