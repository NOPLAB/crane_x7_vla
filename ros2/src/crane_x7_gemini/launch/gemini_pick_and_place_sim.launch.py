#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for Gemini + CRANE-X7 simulation with pick and place task.

This launch file starts:
- CRANE-X7 Gazebo simulation with table
- Gemini object detection node
- Optional pick and place example task
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for Gemini + simulation pick and place."""

    # Declare launch arguments
    api_key_arg = DeclareLaunchArgument(
        'api_key',
        default_value=os.environ.get('GEMINI_API_KEY', ''),
        description='Google Gemini API key'
    )

    model_id_arg = DeclareLaunchArgument(
        'model_id',
        default_value='gemini-robotics-er-1.5-preview',
        description='Gemini model ID'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Input camera image topic'
    )

    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/gemini/detections',
        description='Output detection topic'
    )

    temperature_arg = DeclareLaunchArgument(
        'temperature',
        default_value='0.5',
        description='Model temperature (0.0-1.0)'
    )

    thinking_budget_arg = DeclareLaunchArgument(
        'thinking_budget',
        default_value='0',
        description='Thinking budget for model reasoning'
    )

    max_objects_arg = DeclareLaunchArgument(
        'max_objects',
        default_value='10',
        description='Maximum number of objects to detect'
    )

    auto_start_pick_place_arg = DeclareLaunchArgument(
        'auto_start_pick_place',
        default_value='false',
        description='Automatically start pick and place task after launch'
    )

    example_type_arg = DeclareLaunchArgument(
        'example',
        default_value='pick_and_place',
        description='Pick and place example type: [pick_and_place, pick_and_place_tf]'
    )

    # Include CRANE-X7 Gazebo simulation (pick and place environment)
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_sim_gazebo'),
                'launch',
                'pick_and_place.launch.py'
            ])
        ]),
        launch_arguments={
            'use_d435': 'true',  # Use RealSense camera for Gemini vision
        }.items()
    )

    # Gemini object detection node
    # Note: API key must be provided via GEMINI_API_KEY environment variable
    # or api_key launch argument. The node will fail if API key is empty.
    gemini_node = Node(
        package='crane_x7_gemini',
        executable='gemini_node',
        name='gemini_node',
        output='screen',
        parameters=[{
            'api_key': LaunchConfiguration('api_key'),
            'model_id': LaunchConfiguration('model_id'),
            'image_topic': LaunchConfiguration('image_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'temperature': LaunchConfiguration('temperature'),
            'thinking_budget': LaunchConfiguration('thinking_budget'),
            'max_objects': LaunchConfiguration('max_objects'),
        }]
    )

    # Pick and place example node (optional auto-start)
    pick_place_example = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('crane_x7_examples'),
                'launch',
                'example.launch.py'
            ])
        ]),
        launch_arguments={
            'example': LaunchConfiguration('example'),
            'use_sim_time': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('auto_start_pick_place'))
    )

    return LaunchDescription([
        api_key_arg,
        model_id_arg,
        image_topic_arg,
        output_topic_arg,
        temperature_arg,
        thinking_budget_arg,
        max_objects_arg,
        auto_start_pick_place_arg,
        example_type_arg,
        gazebo_sim,
        gemini_node,
        pick_place_example,
    ])
