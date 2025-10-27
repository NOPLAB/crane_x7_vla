#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""Configuration management for data logger."""

from typing import List
from dataclasses import dataclass
from rclpy.node import Node


@dataclass
class LoggerConfig:
    """Configuration dataclass for DataLogger."""

    # Output settings
    output_dir: str
    save_format: str

    # Episode settings
    episode_length: int
    auto_start_recording: bool
    inter_episode_delay: float
    collection_rate: float

    # Camera settings
    use_camera: bool
    use_depth: bool
    image_width: int
    image_height: int
    image_quality: int

    # Joint configuration
    arm_joint_names: List[str]
    gripper_joint_names: List[str]

    # Topic names
    joint_states_topic: str
    rgb_image_topic: str
    depth_image_topic: str
    camera_info_topic: str

    # RLDS (Open X-Embodiment) settings
    dataset_name: str
    language_instruction_topic: str
    default_language_instruction: str

    # Dataset statistics
    compute_dataset_statistics: bool
    statistics_output_path: str

    # Voice notification settings
    enable_voice_notifications: bool
    voice_language: str
    voice_speed: int
    notify_on_episode_start: bool
    notify_on_episode_complete: bool
    notify_time_remaining: bool
    time_notification_intervals: List[int]


class ConfigManager:
    """Manages ROS 2 parameters for DataLogger."""

    @staticmethod
    def declare_parameters(node: Node) -> None:
        """Declare all ROS 2 parameters with defaults."""
        node.declare_parameter('output_dir', '/workspace/data/oxe_logs')
        node.declare_parameter('episode_length', 100)
        node.declare_parameter('auto_start_recording', True)
        node.declare_parameter('inter_episode_delay', 5.0)
        node.declare_parameter('collection_rate', 10.0)
        node.declare_parameter('use_camera', True)
        node.declare_parameter('use_depth', False)
        node.declare_parameter('image_width', 0)
        node.declare_parameter('image_height', 0)
        node.declare_parameter('image_quality', 90)
        node.declare_parameter('arm_joint_names', [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
        ])
        node.declare_parameter('gripper_joint_names', ['crane_x7_gripper_finger_a_joint'])
        node.declare_parameter('joint_states_topic', '/joint_states')
        node.declare_parameter('rgb_image_topic', '/camera/color/image_raw')
        node.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        node.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        node.declare_parameter('save_format', 'tfrecord')
        # RLDS settings
        node.declare_parameter('dataset_name', 'crane_x7')
        node.declare_parameter('language_instruction_topic', '/task/language_instruction')
        node.declare_parameter('default_language_instruction', 'manipulate the object')
        # Statistics settings
        node.declare_parameter('compute_dataset_statistics', True)
        node.declare_parameter('statistics_output_path', '/workspace/data/tfrecord_logs/dataset_statistics.json')
        # Voice notification settings
        node.declare_parameter('enable_voice_notifications', True)
        node.declare_parameter('voice_language', 'en')
        node.declare_parameter('voice_speed', 150)
        node.declare_parameter('notify_on_episode_start', True)
        node.declare_parameter('notify_on_episode_complete', True)
        node.declare_parameter('notify_time_remaining', True)
        node.declare_parameter('time_notification_intervals', [60, 30, 10])

    @staticmethod
    def load_config(node: Node) -> LoggerConfig:
        """Load configuration from ROS 2 parameters."""
        return LoggerConfig(
            output_dir=node.get_parameter('output_dir').value,
            episode_length=node.get_parameter('episode_length').value,
            auto_start_recording=node.get_parameter('auto_start_recording').value,
            inter_episode_delay=node.get_parameter('inter_episode_delay').value,
            collection_rate=node.get_parameter('collection_rate').value,
            use_camera=node.get_parameter('use_camera').value,
            use_depth=node.get_parameter('use_depth').value,
            image_width=node.get_parameter('image_width').value,
            image_height=node.get_parameter('image_height').value,
            image_quality=node.get_parameter('image_quality').value,
            arm_joint_names=node.get_parameter('arm_joint_names').value,
            gripper_joint_names=node.get_parameter('gripper_joint_names').value,
            joint_states_topic=node.get_parameter('joint_states_topic').value,
            rgb_image_topic=node.get_parameter('rgb_image_topic').value,
            depth_image_topic=node.get_parameter('depth_image_topic').value,
            camera_info_topic=node.get_parameter('camera_info_topic').value,
            save_format=node.get_parameter('save_format').value,
            # RLDS settings
            dataset_name=node.get_parameter('dataset_name').value,
            language_instruction_topic=node.get_parameter('language_instruction_topic').value,
            default_language_instruction=node.get_parameter('default_language_instruction').value,
            # Statistics settings
            compute_dataset_statistics=node.get_parameter('compute_dataset_statistics').value,
            statistics_output_path=node.get_parameter('statistics_output_path').value,
            # Voice notification settings
            enable_voice_notifications=node.get_parameter('enable_voice_notifications').value,
            voice_language=node.get_parameter('voice_language').value,
            voice_speed=node.get_parameter('voice_speed').value,
            notify_on_episode_start=node.get_parameter('notify_on_episode_start').value,
            notify_on_episode_complete=node.get_parameter('notify_on_episode_complete').value,
            notify_time_remaining=node.get_parameter('notify_time_remaining').value,
            time_notification_intervals=node.get_parameter('time_notification_intervals').value,
        )
