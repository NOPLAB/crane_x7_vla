#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for Gemini object detection service.

This launch file starts the object detector service for on-demand detection.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for object detector service."""

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
        default_value='25',
        description='Maximum number of objects to detect'
    )

    # Object detector service node
    detector_node = Node(
        package='crane_x7_gemini',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[{
            'api_key': LaunchConfiguration('api_key'),
            'model_id': LaunchConfiguration('model_id'),
            'image_topic': LaunchConfiguration('image_topic'),
            'temperature': LaunchConfiguration('temperature'),
            'thinking_budget': LaunchConfiguration('thinking_budget'),
            'max_objects': LaunchConfiguration('max_objects'),
        }]
    )

    return LaunchDescription([
        api_key_arg,
        model_id_arg,
        image_topic_arg,
        temperature_arg,
        thinking_budget_arg,
        max_objects_arg,
        detector_node,
    ])
