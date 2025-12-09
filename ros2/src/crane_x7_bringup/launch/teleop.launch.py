#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7テレオペレーションのbringup launchファイル。

リーダーとフォロワーの両方のロボットを単一プロセスで起動する。
リーダーはトルクOFFで人間が操作、フォロワーはトルクONでリーダーを追従。

カメラ+ロガーが必要な場合は、data_collection.launch.pyと併用する。

引数:
  - leader_port (default: /dev/ttyUSB0): リーダーロボットのUSBポート
  - follower_port (default: /dev/ttyUSB1): フォロワーロボットのUSBポート
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch teleoperation with leader and follower robots."""
    # Declare launch arguments
    declare_leader_port = DeclareLaunchArgument(
        'leader_port',
        default_value='/dev/ttyUSB0',
        description='USB port for leader robot'
    )

    declare_follower_port = DeclareLaunchArgument(
        'follower_port',
        default_value='/dev/ttyUSB1',
        description='USB port for follower robot'
    )

    # Teleop leader node
    teleop_leader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_teleop'),
                'launch',
                'teleop_leader.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('leader_port'),
        }.items()
    )

    # Teleop follower node
    teleop_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_teleop'),
                'launch',
                'teleop_follower.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('follower_port'),
        }.items()
    )

    return LaunchDescription([
        declare_leader_port,
        declare_follower_port,
        teleop_leader,
        teleop_follower,
    ])
