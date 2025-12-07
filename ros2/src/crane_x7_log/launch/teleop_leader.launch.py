#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7テレオペ・リーダーモードの統合launchファイル。

リーダーロボットはトルクOFFで手動教示が可能。
データロガーでデモンストレーションを記録。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7 LeaderロボットのUSBポート名
  - use_d435 (default: false): RealSense D435カメラを使用
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
    """Launch teleop leader with optional logger and viewer."""

    # Get package directories
    crane_x7_log_dir = get_package_share_directory('crane_x7_log')

    # Paths
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7 Leader robot'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='false',
        description='Use RealSense D435 camera for visual observations'
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
        description='Directory to save logged demonstration data'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=logger_config,
        description='Path to data logger config file'
    )

    # Include teleop_leader.launch.py from crane_x7_teleop package
    teleop_leader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_teleop'),
                'launch', 'teleop_leader.launch.py'
            ])
        ]),
        launch_arguments={
            'port_name': LaunchConfiguration('port_name')
        }.items()
    )

    # RealSense D435 camera node (conditional)
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('use_d435')),
        launch_arguments={
            'camera_namespace': '',
            'device_type': 'd435',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
        }.items()
    )

    # Data logger node (conditional)
    data_logger = Node(
        package='crane_x7_log',
        executable='data_logger',
        name='data_logger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_logger')),
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'output_dir': LaunchConfiguration('output_dir'),
                'joint_states_topic': '/joint_states'
            }
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
        declare_port_name,
        declare_use_d435,
        declare_use_logger,
        declare_use_viewer,
        declare_output_dir,
        declare_config_file,
        teleop_leader,
        realsense_node,
        data_logger,
        camera_viewer,
    ])
