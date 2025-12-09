#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7実機 + rosbridge のbringup launchファイル。

リモートGPUサーバーからのVLA推論に対応。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7のUSBポート
  - use_d435 (default: true): RealSense D435使用
  - use_viewer (default: false): カメラビューア表示
  - auto_execute (default: true): VLAアクション自動実行
  - rosbridge_port (default: 9090): rosbridgeポート
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch real robot + rosbridge + robot_controller."""
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port for CRANE-X7'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='true',
        description='Use RealSense D435 camera'
    )

    declare_use_viewer = DeclareLaunchArgument(
        'use_viewer',
        default_value='false',
        description='Display RViz camera viewer'
    )

    declare_auto_execute = DeclareLaunchArgument(
        'auto_execute',
        default_value='true',
        description='Automatically execute VLA actions'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('crane_x7_vla'),
            'config',
            'vla_config.yaml'
        ]),
        description='Path to VLA config file'
    )

    declare_rosbridge_port = DeclareLaunchArgument(
        'rosbridge_port',
        default_value='9090',
        description='rosbridge WebSocket port'
    )

    # CRANE-X7 demo launch
    crane_x7_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_examples'),
                'launch',
                'demo.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name'),
            'use_d435': LaunchConfiguration('use_d435'),
        }.items()
    )

    # rosbridge WebSocket server
    rosbridge_server = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('rosbridge_port'),
            'address': '',
        }],
    )

    # Image transport republisher
    image_republisher = Node(
        package='image_transport',
        executable='republish',
        name='image_republisher',
        output='screen',
        arguments=[
            'raw',
            'compressed',
            '--ros-args',
            '-r', 'in:=/camera/color/image_raw',
            '-r', 'out/compressed:=/camera/color/image_raw/compressed',
        ],
    )

    # Robot controller node
    robot_controller_node = Node(
        package='crane_x7_vla',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'auto_execute': LaunchConfiguration('auto_execute'),
            }
        ],
    )

    # Camera viewer
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('crane_x7_log'),
        'config',
        'camera_viewer.rviz'
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
        declare_port_name,
        declare_use_d435,
        declare_use_viewer,
        declare_auto_execute,
        declare_config_file,
        declare_rosbridge_port,
        crane_x7_demo,
        rosbridge_server,
        image_republisher,
        robot_controller_node,
        camera_viewer,
    ])
