#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""
Launch file for RealSense camera viewer.
Displays camera image streams using rqt_image_view.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch camera viewer for RealSense D435."""

    # Launch arguments
    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Image topic to display'
    )

    declare_use_rqt = DeclareLaunchArgument(
        'use_rqt',
        default_value='true',
        description='Use rqt_image_view (true) or image_view (false)'
    )

    # rqt_image_view (GUI tool with topic selection)
    rqt_image_view = ExecuteProcess(
        cmd=['rqt_image_view', LaunchConfiguration('image_topic')],
        output='screen',
        name='rqt_image_view'
    )

    return LaunchDescription([
        declare_image_topic,
        declare_use_rqt,
        rqt_image_view,
    ])
