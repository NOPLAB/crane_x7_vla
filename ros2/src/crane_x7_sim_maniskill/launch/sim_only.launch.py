#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch ManiSkill simulation only."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description."""

    pkg_dir = get_package_share_directory('crane_x7_sim_maniskill')
    config_file = os.path.join(pkg_dir, 'config', 'maniskill_config.yaml')

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

    declare_render_mode = DeclareLaunchArgument(
        'render_mode',
        default_value='rgb_array',
        description='Render mode (rgb_array, human, none)'
    )

    maniskill_sim_node = Node(
        package='crane_x7_sim_maniskill',
        executable='maniskill_sim_node',
        name='maniskill_sim_node',
        output='screen',
        parameters=[
            config_file,
            {
                'sim_backend': LaunchConfiguration('sim_backend'),
                'sim_rate': LaunchConfiguration('sim_rate'),
                'env_id': LaunchConfiguration('env_id'),
                'auto_reset': LaunchConfiguration('auto_reset'),
                'render_mode': LaunchConfiguration('render_mode'),
            }
        ]
    )

    return LaunchDescription([
        declare_sim_backend,
        declare_sim_rate,
        declare_env_id,
        declare_auto_reset,
        declare_render_mode,
        maniskill_sim_node,
    ])
