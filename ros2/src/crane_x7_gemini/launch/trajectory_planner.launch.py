#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Launch file for Gemini trajectory planner with MoveIt integration.

This launch file starts the trajectory planner that receives task prompts
and generates + executes trajectories using Gemini and MoveIt.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for trajectory planner."""

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
        description='Input RGB camera image topic'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/aligned_depth_to_color/image_raw',
        description='Input depth image topic'
    )

    prompt_topic_arg = DeclareLaunchArgument(
        'prompt_topic',
        default_value='/gemini/task_prompt',
        description='Topic for receiving task prompts'
    )

    move_group_arg = DeclareLaunchArgument(
        'move_group',
        default_value='arm',
        description='MoveIt move group name'
    )

    end_effector_link_arg = DeclareLaunchArgument(
        'end_effector_link',
        default_value='crane_x7_gripper_base_link',
        description='End effector link name'
    )

    planning_time_arg = DeclareLaunchArgument(
        'planning_time',
        default_value='5.0',
        description='MoveIt planning time in seconds'
    )

    execute_trajectory_arg = DeclareLaunchArgument(
        'execute_trajectory',
        default_value='true',
        description='Whether to execute planned trajectories'
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

    # Trajectory planner node
    trajectory_planner_node = Node(
        package='crane_x7_gemini',
        executable='trajectory_planner',
        name='trajectory_planner',
        output='screen',
        parameters=[{
            'api_key': LaunchConfiguration('api_key'),
            'model_id': LaunchConfiguration('model_id'),
            'image_topic': LaunchConfiguration('image_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'prompt_topic': LaunchConfiguration('prompt_topic'),
            'move_group': LaunchConfiguration('move_group'),
            'end_effector_link': LaunchConfiguration('end_effector_link'),
            'planning_time': LaunchConfiguration('planning_time'),
            'execute_trajectory': LaunchConfiguration('execute_trajectory'),
            'temperature': LaunchConfiguration('temperature'),
            'thinking_budget': LaunchConfiguration('thinking_budget'),
        }]
    )

    return LaunchDescription([
        api_key_arg,
        model_id_arg,
        image_topic_arg,
        depth_topic_arg,
        prompt_topic_arg,
        move_group_arg,
        end_effector_link_arg,
        planning_time_arg,
        execute_trajectory_arg,
        temperature_arg,
        thinking_budget_arg,
        trajectory_planner_node,
    ])
