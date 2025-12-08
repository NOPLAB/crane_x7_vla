#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch CRANE-X7 real robot with rosbridge for remote VLA inference.

This launch file starts:
- CRANE-X7 hardware control (MoveIt2)
- RealSense D435 camera
- rosbridge_server (WebSocket port 9090)
- image_transport republish (raw -> compressed)
- robot_controller node

The VLA inference runs on a remote GPU server and communicates via rosbridge
WebSocket over Tailscale VPN.

Usage:
  ros2 launch crane_x7_vla rosbridge_robot.launch.py

Architecture:
  [Local: This launch file]              [Remote: Vast.ai GPU]
  ┌─────────────────────────┐           ┌─────────────────────┐
  │ RealSense D435          │           │ vla_inference       │
  │ /camera/color/image_raw │           │ _rosbridge.py       │
  │         │               │           │                     │
  │         ▼               │           │                     │
  │ image_transport         │           │                     │
  │ /camera/color/          │           │                     │
  │ image_raw/compressed    │           │                     │
  │         │               │           │                     │
  │         ▼               │           │                     │
  │ rosbridge_server ◄──────┼─WebSocket─┼───► roslibpy        │
  │ (port 9090)             │           │                     │
  │         │               │           │                     │
  │         ▼               │           │                     │
  │ robot_controller ◄──────┼───────────┼──── /vla/predicted  │
  │         │               │           │      _action        │
  │         ▼               │           │                     │
  │ CRANE-X7 Hardware       │           │                     │
  └─────────────────────────┘           └─────────────────────┘
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
    # Declare launch arguments
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
        description='Automatically execute received VLA actions'
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

    # Include CRANE-X7 demo launch (MoveIt2 + hardware control + camera)
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
    # This allows remote clients to subscribe/publish ROS topics via WebSocket
    rosbridge_server = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('rosbridge_port'),
            'address': '',  # Bind to all interfaces (0.0.0.0)
        }],
    )

    # Image transport: republish raw image as compressed JPEG
    # This reduces bandwidth for remote inference over VPN
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

    # Robot controller node (receives actions from remote VLA inference)
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

    # Camera viewer (optional RViz)
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
