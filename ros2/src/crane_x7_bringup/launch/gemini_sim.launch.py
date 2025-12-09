#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Gemini API統合（シミュレーション）のbringup launchファイル。

crane_x7_gemini/gemini_pick_and_place_sim.launch.pyをラップする。
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gemini API integration with Gazebo simulation."""
    # Include gemini_pick_and_place_sim.launch.py
    gemini_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_gemini'),
                'launch',
                'gemini_pick_and_place_sim.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        gemini_sim_launch,
    ])
