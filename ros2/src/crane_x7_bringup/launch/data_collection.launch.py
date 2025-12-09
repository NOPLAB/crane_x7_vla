#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
データ収集用launchファイル（カメラ + ロガー）。

テレオペと組み合わせて使用する。
docker compose --profile teleop --profile log up

引数:
  - use_viewer (default: false): カメラビューア(rviz2)を表示
  - output_dir: ログデータの保存先
  - camera_serial (default: ''): プライマリカメラのシリアル番号
  - camera2_serial (default: ''): セカンダリカメラのシリアル番号
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, AndSubstitution, NotEqualsSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _create_data_logger(context, *args, **kwargs):
    """Create data logger node with dynamic camera configuration."""
    camera2_serial = LaunchConfiguration('camera2_serial').perform(context)
    use_dual_camera = camera2_serial.strip() != ''

    if use_dual_camera:
        camera_names = ['primary', 'secondary']
    else:
        camera_names = ['primary']

    return [
        Node(
            package='crane_x7_log',
            executable='data_logger',
            name='data_logger',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'output_dir': LaunchConfiguration('output_dir'),
                    'joint_states_topic': '/joint_states',
                    'camera_names': camera_names,
                }
            ]
        )
    ]


def generate_launch_description():
    """Launch camera and data logger for data collection."""

    crane_x7_log_dir = get_package_share_directory('crane_x7_log')
    logger_config = os.path.join(crane_x7_log_dir, 'config', 'logger_config.yaml')

    # Launch arguments
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

    declare_camera_serial = DeclareLaunchArgument(
        'camera_serial',
        default_value='',
        description='Primary camera serial number'
    )

    declare_camera2_serial = DeclareLaunchArgument(
        'camera2_serial',
        default_value='',
        description='Secondary camera serial number'
    )

    # RealSense D435 primary camera
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py'
            ])
        ]),
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

    # RealSense D435 secondary camera (conditional)
    realsense_node2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py'
            ])
        ]),
        condition=IfCondition(
            NotEqualsSubstitution(LaunchConfiguration('camera2_serial'), '')
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

    # Data logger
    data_logger = OpaqueFunction(function=_create_data_logger)

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
        declare_use_viewer,
        declare_output_dir,
        declare_config_file,
        declare_camera_serial,
        declare_camera2_serial,
        realsense_node,
        realsense_node2,
        data_logger,
        camera_viewer,
    ])
