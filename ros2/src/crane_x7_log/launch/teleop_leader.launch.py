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
  - camera_serial (default: ''): プライマリカメラのシリアル番号（空の場合は自動選択）
  - camera2_serial (default: ''): セカンダリカメラのシリアル番号（空の場合は無効）
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, AndSubstitution, NotEqualsSubstitution
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

    declare_camera_serial = DeclareLaunchArgument(
        'camera_serial',
        default_value='',
        description='Primary camera serial number (empty for auto-select)'
    )

    declare_camera2_serial = DeclareLaunchArgument(
        'camera2_serial',
        default_value='',
        description='Secondary camera serial number (empty to disable)'
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

    # RealSense D435 primary camera node (conditional)
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
            'camera_name': 'camera',
            'device_type': 'd435',
            'serial_no': LaunchConfiguration('camera_serial'),
            'pointcloud.enable': 'false',
            'align_depth.enable': 'false',
            'rgb_camera.profile': '640x480x30',
        }.items()
    )

    # RealSense D435 secondary camera node (conditional - only if camera2_serial is set)
    realsense_node2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py'
            ])
        ]),
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration('use_d435'),
                NotEqualsSubstitution(LaunchConfiguration('camera2_serial'), '')
            )
        ),
        launch_arguments={
            'camera_namespace': '',
            'camera_name': 'camera2',
            'device_type': 'd435',
            'serial_no': LaunchConfiguration('camera2_serial'),
            'pointcloud.enable': 'false',
            'align_depth.enable': 'false',
            'rgb_camera.profile': '640x480x30',
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
        declare_camera_serial,
        declare_camera2_serial,
        teleop_leader,
        realsense_node,
        realsense_node2,
        data_logger,
        camera_viewer,
    ])
