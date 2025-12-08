#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Configuration management for data logger."""

from typing import List
from dataclasses import dataclass, field
from rclpy.node import Node


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    name: str  # "primary", "secondary", "wrist"
    rgb_topic: str  # RGB image topic
    camera_info_topic: str  # Camera info topic
    width: int = 0  # Resize width (0 = original)
    height: int = 0  # Resize height (0 = original)
    quality: int = 90  # JPEG quality


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

    # Joint configuration
    arm_joint_names: List[str]
    gripper_joint_names: List[str]

    # Topic names
    joint_states_topic: str

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

    # Multiple camera settings
    cameras: List[CameraConfig] = field(default_factory=list)


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
        # Multiple camera settings
        node.declare_parameter('camera_names', ['primary'])

    @staticmethod
    def load_config(node: Node) -> LoggerConfig:
        """Load configuration from ROS 2 parameters."""
        # Load camera configurations
        cameras = ConfigManager._load_camera_configs(node)

        return LoggerConfig(
            output_dir=node.get_parameter('output_dir').value,
            episode_length=node.get_parameter('episode_length').value,
            auto_start_recording=node.get_parameter('auto_start_recording').value,
            inter_episode_delay=node.get_parameter('inter_episode_delay').value,
            collection_rate=node.get_parameter('collection_rate').value,
            arm_joint_names=node.get_parameter('arm_joint_names').value,
            gripper_joint_names=node.get_parameter('gripper_joint_names').value,
            joint_states_topic=node.get_parameter('joint_states_topic').value,
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
            cameras=cameras,
        )

    @staticmethod
    def _load_camera_configs(node: Node) -> List[CameraConfig]:
        """Load camera configurations from ROS 2 parameters."""
        camera_names = node.get_parameter('camera_names').value
        if not camera_names:
            node.get_logger().warning("No cameras configured (camera_names is empty)")
            return []

        cameras = []
        for name in camera_names:
            # Declare parameters for each camera dynamically
            prefix = f'camera.{name}'
            node.declare_parameter(f'{prefix}.rgb_topic', '')
            node.declare_parameter(f'{prefix}.camera_info_topic', '')
            node.declare_parameter(f'{prefix}.width', 0)
            node.declare_parameter(f'{prefix}.height', 0)
            node.declare_parameter(f'{prefix}.quality', 90)

            rgb_topic = node.get_parameter(f'{prefix}.rgb_topic').value
            if not rgb_topic:
                node.get_logger().warning(
                    f"Camera '{name}' has no rgb_topic configured, skipping"
                )
                continue

            cameras.append(CameraConfig(
                name=name,
                rgb_topic=rgb_topic,
                camera_info_topic=node.get_parameter(f'{prefix}.camera_info_topic').value,
                width=node.get_parameter(f'{prefix}.width').value,
                height=node.get_parameter(f'{prefix}.height').value,
                quality=node.get_parameter(f'{prefix}.quality').value,
            ))

        return cameras
