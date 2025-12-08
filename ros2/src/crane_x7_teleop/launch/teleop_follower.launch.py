#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
CRANE-X7テレオペ・フォロワーモードの統合launchファイル。

フォロワーロボットはトルクONでリーダーの動きを追従。

引数:
  - port_name (default: /dev/ttyUSB0): CRANE-X7 FollowerロボットのUSBポート名
  - use_d435 (default: false): RealSense D435カメラを使用
  - use_viewer (default: false): カメラビューア(rviz2)を表示
  - camera_serial (default: ''): プライマリカメラのシリアル番号（空の場合は自動選択）
  - camera2_serial (default: ''): セカンダリカメラのシリアル番号（空の場合は無効）
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


def generate_launch_description():
    # Launch arguments
    declare_port_name = DeclareLaunchArgument(
        'port_name',
        default_value='/dev/ttyUSB0',
        description='USB port name for CRANE-X7 Follower robot'
    )

    declare_use_d435 = DeclareLaunchArgument(
        'use_d435',
        default_value='false',
        description='Use RealSense D435 camera'
    )

    declare_use_viewer = DeclareLaunchArgument(
        'use_viewer',
        default_value='false',
        description='Display camera viewer (rviz2)'
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

    # Config files
    crane_x7_control_dir = get_package_share_directory('crane_x7_control')
    config_file = PathJoinSubstitution([
        FindPackageShare('crane_x7_teleop'),
        'config', 'teleop_config.yaml'
    ])

    # teleop_hardware_node (Follower mode)
    teleop_hardware = Node(
        package='crane_x7_teleop',
        executable='teleop_hardware_node',
        name='teleop_hardware_node',
        output='screen',
        parameters=[
            config_file,
            {
                'mode': 'follower',
                'port_name': LaunchConfiguration('port_name'),
                'config_file': os.path.join(crane_x7_control_dir, 'config', 'manipulator_config.yaml'),
                'links_file': os.path.join(crane_x7_control_dir, 'config', 'manipulator_links.csv')
            }
        ]
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
        declare_use_viewer,
        declare_camera_serial,
        declare_camera2_serial,
        teleop_hardware,
        realsense_node,
        realsense_node2,
        camera_viewer,
    ])
