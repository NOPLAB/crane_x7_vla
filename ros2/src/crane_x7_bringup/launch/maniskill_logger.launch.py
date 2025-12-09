#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
ManiSkillシミュレーション + データロガーのbringup launchファイル。

引数:
  - output_dir (default: /workspace/data/tfrecord_logs): ログ保存先
  - episode_length (default: 200): エピソードあたりのステップ数
  - sim_backend (default: gpu): シミュレーションバックエンド
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch ManiSkill simulation with data logger."""

    maniskill_pkg = get_package_share_directory('crane_x7_sim_maniskill')
    log_pkg = get_package_share_directory('crane_x7_log')

    maniskill_config = os.path.join(maniskill_pkg, 'config', 'maniskill_config.yaml')
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

    declare_sim_backend = DeclareLaunchArgument(
        'sim_backend',
        default_value='gpu',
        description='Simulation backend (cpu or gpu)'
    )

    maniskill_sim_node = Node(
        package='crane_x7_sim_maniskill',
        executable='maniskill_sim_node',
        name='maniskill_sim_node',
        output='screen',
        parameters=[
            maniskill_config,
            {
                'auto_reset': False,
                'sim_backend': LaunchConfiguration('sim_backend'),
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
        declare_sim_backend,
        maniskill_sim_node,
        data_logger_node,
    ])
