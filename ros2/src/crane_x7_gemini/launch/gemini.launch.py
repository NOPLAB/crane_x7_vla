#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for Gemini Robotics-ER node.

This launch file starts the Gemini node for object detection and vision-based reasoning.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Gemini node."""

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
        description='Thinking budget for model reasoning (0 = minimal, higher = more reasoning)'
    )

    max_objects_arg = DeclareLaunchArgument(
        'max_objects',
        default_value='10',
        description='Maximum number of objects to detect'
    )

    # Gemini node
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

    return LaunchDescription([
        api_key_arg,
        model_id_arg,
        image_topic_arg,
        output_topic_arg,
        temperature_arg,
        thinking_budget_arg,
        max_objects_arg,
        gemini_node,
    ])
