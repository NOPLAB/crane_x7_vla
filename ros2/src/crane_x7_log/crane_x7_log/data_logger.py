#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""ROS 2 node for logging CRANE-X7 data in NPZ or TFRecord format for VLA fine-tuning."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import String
import numpy as np
from threading import Lock
from typing import Optional, Dict, Any, List

from .config_manager import ConfigManager, LoggerConfig
from .image_processor import ImageProcessor
from .episode_saver import EpisodeSaver
from .voice_notifier import VoiceNotifier


class DataLogger(Node):
    """ROS 2 node for logging CRANE-X7 data."""

    def __init__(self):
        super().__init__('data_logger')

        # Load configuration
        ConfigManager.declare_parameters(self)
        self.config: LoggerConfig = ConfigManager.load_config(self)

        # Initialize components
        self.image_processor = ImageProcessor(
            self.config.image_width,
            self.config.image_height
        )
        self.episode_saver = EpisodeSaver(
            self.config.output_dir,
            self.config.save_format,
            self.get_logger(),
            dataset_name=self.config.dataset_name,
            compute_statistics=self.config.compute_dataset_statistics,
            statistics_output_path=self.config.statistics_output_path
        )
        self.voice_notifier = VoiceNotifier(
            logger=self,
            enabled=self.config.enable_voice_notifications,
            language=self.config.voice_language,
            speed=self.config.voice_speed
        )

        # Recording state
        self.is_recording = self.config.auto_start_recording
        self.is_waiting_between_episodes = False
        self.episode_count = 0
        self.step_count = 0
        self.resume_timer = None
        self.notified_time_marks = set()  # Track which time notifications have been sent

        # Data buffers
        self.lock = Lock()
        self.current_episode: List[Dict[str, Any]] = []

        # Latest data cache
        self.latest_joint_state: Optional[JointState] = None
        self.latest_rgb_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_language_instruction: Optional[str] = None

        # Setup subscriptions and timers
        self._setup_subscriptions()
        self._setup_timers()

        # Log initialization
        self._log_initialization()

        # Voice notification for first episode if auto-start
        if self.config.auto_start_recording and self.config.notify_on_episode_start:
            self.voice_notifier.notify_episode_start(self.episode_count)

    def _setup_subscriptions(self) -> None:
        """Setup ROS 2 topic subscriptions."""
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.config.joint_states_topic,
            self._joint_state_callback,
            10
        )

        # Language instruction subscription
        self.language_instruction_sub = self.create_subscription(
            String,
            self.config.language_instruction_topic,
            self._language_instruction_callback,
            10
        )

        if self.config.use_camera:
            self.rgb_image_sub = self.create_subscription(
                Image,
                self.config.rgb_image_topic,
                self._rgb_image_callback,
                10
            )
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.config.camera_info_topic,
                self._camera_info_callback,
                10
            )

            if self.config.use_depth:
                self.depth_image_sub = self.create_subscription(
                    Image,
                    self.config.depth_image_topic,
                    self._depth_image_callback,
                    10
                )

    def _setup_timers(self) -> None:
        """Setup periodic timers."""
        timer_period = 1.0 / self.config.collection_rate
        self.timer = self.create_timer(timer_period, self._collect_step)
        self.status_timer = self.create_timer(2.0, self._report_status)

    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.get_logger().info('Data Logger initialized')
        self.get_logger().info(f'  Output: {self.config.output_dir}')
        self.get_logger().info(f'  Format: {self.config.save_format}')
        self.get_logger().info(f'  Episode length: {self.config.episode_length}')
        self.get_logger().info(f'  Inter-episode delay: {self.config.inter_episode_delay}s')
        self.get_logger().info(f'  Collection rate: {self.config.collection_rate} Hz')
        self.get_logger().info(f'  Camera: {self.config.use_camera}, Depth: {self.config.use_depth}')
        self.get_logger().info(f'  Auto-start: {self.config.auto_start_recording}')
        self.get_logger().info(f'  Recording: {self.is_recording}')

    # Callbacks
    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint states topic."""
        with self.lock:
            self.latest_joint_state = msg

    def _rgb_image_callback(self, msg: Image) -> None:
        """Callback for RGB image topic."""
        with self.lock:
            self.latest_rgb_image = msg

    def _depth_image_callback(self, msg: Image) -> None:
        """Callback for depth image topic."""
        with self.lock:
            self.latest_depth_image = msg

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback for camera info topic."""
        with self.lock:
            self.latest_camera_info = msg

    def _language_instruction_callback(self, msg: String) -> None:
        """Callback for language instruction topic."""
        with self.lock:
            self.latest_language_instruction = msg.data
            self.get_logger().info(f'Received language instruction: {msg.data}')

    # Data collection
    def _extract_joint_positions(
        self,
        joint_state: JointState
    ) -> Optional[np.ndarray]:
        """
        Extract arm and gripper joint positions in correct order.

        Args:
            joint_state: JointState message

        Returns:
            Combined joint positions array, or None on error
        """
        # Create mapping from joint name to position
        joint_dict = dict(zip(joint_state.name, joint_state.position))

        # Extract arm joints
        arm_positions = []
        for name in self.config.arm_joint_names:
            if name not in joint_dict:
                self.get_logger().warn(f'Joint {name} not found in joint_states')
                return None
            arm_positions.append(joint_dict[name])

        # Extract gripper joint
        gripper_position = None
        if self.config.gripper_joint_names[0] in joint_dict:
            gripper_position = joint_dict[self.config.gripper_joint_names[0]]

        # Combine positions
        if gripper_position is not None:
            return np.array(arm_positions + [gripper_position], dtype=np.float32)
        else:
            return np.array(arm_positions, dtype=np.float32)

    def _collect_step(self) -> None:
        """Collect one step of data (timer callback)."""
        with self.lock:
            if not self.is_recording or self.is_waiting_between_episodes:
                return

            # Check minimum required data
            if not self._has_required_data():
                return

            # Extract joint positions
            joint_positions = self._extract_joint_positions(self.latest_joint_state)
            if joint_positions is None:
                return

            # Process images
            rgb_array = self._process_rgb_image()
            depth_array = self._process_depth_image()

            # Create step data
            step_data = self._create_step_data(joint_positions, rgb_array, depth_array)

            # Add to episode
            self.current_episode.append(step_data)
            self.step_count += 1

            # Check for time remaining notifications
            if self.config.notify_time_remaining:
                self._check_time_notifications()

            # Save episode if complete
            if self.step_count >= self.config.episode_length:
                self._save_current_episode()

    def _check_time_notifications(self) -> None:
        """Check if we should announce time remaining."""
        steps_remaining = self.config.episode_length - self.step_count
        # Convert steps to seconds
        seconds_remaining = int(steps_remaining / self.config.collection_rate)

        for threshold in self.config.time_notification_intervals:
            # Notify if we just crossed this threshold and haven't notified yet
            if (seconds_remaining <= threshold and
                threshold not in self.notified_time_marks and
                seconds_remaining > 0):
                self.voice_notifier.notify_time_remaining(threshold)
                self.notified_time_marks.add(threshold)

    def _has_required_data(self) -> bool:
        """Check if we have minimum required data to record a step."""
        if self.latest_joint_state is None:
            return False
        if self.config.use_camera and self.latest_rgb_image is None:
            return False
        return True

    def _process_rgb_image(self) -> Optional[np.ndarray]:
        """Process RGB image if available."""
        if self.config.use_camera and self.latest_rgb_image is not None:
            rgb_array = self.image_processor.process_rgb_image(self.latest_rgb_image)
            if rgb_array is None:
                self.get_logger().error('Failed to convert RGB image')
            return rgb_array
        return None

    def _process_depth_image(self) -> Optional[np.ndarray]:
        """Process depth image if available."""
        if self.config.use_depth and self.latest_depth_image is not None:
            depth_array = self.image_processor.process_depth_image(self.latest_depth_image)
            if depth_array is None:
                self.get_logger().error('Failed to convert depth image')
            return depth_array
        return None

    def _create_step_data(
        self,
        joint_positions: np.ndarray,
        rgb_array: Optional[np.ndarray],
        depth_array: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Create step data dictionary."""
        step_data = {
            'observation': {
                'state': joint_positions,
                'timestamp': (
                    self.latest_joint_state.header.stamp.sec +
                    self.latest_joint_state.header.stamp.nanosec * 1e-9
                )
            },
            'action': joint_positions,  # Placeholder, will be updated when saving
        }

        if rgb_array is not None:
            step_data['observation']['image'] = rgb_array

        if depth_array is not None:
            step_data['observation']['depth'] = depth_array

        return step_data

    def _save_current_episode(self) -> None:
        """Save current episode and reset counters."""
        # Use current language instruction or default
        language_instruction = (
            self.latest_language_instruction
            if self.latest_language_instruction is not None
            else self.config.default_language_instruction
        )

        self.episode_saver.save(
            self.current_episode,
            self.episode_count,
            language_instruction=language_instruction
        )

        # Voice notification for episode completion
        if self.config.notify_on_episode_complete:
            self.voice_notifier.notify_episode_complete(self.episode_count)

        self.current_episode = []
        self.step_count = 0
        self.episode_count += 1
        self.notified_time_marks.clear()  # Reset time notifications for next episode

        # Enter waiting state between episodes
        if self.config.inter_episode_delay > 0:
            self.is_waiting_between_episodes = True
            self.get_logger().info(
                f'Episode {self.episode_count - 1} saved. '
                f'Waiting {self.config.inter_episode_delay}s before next episode...'
            )

            # Voice notification about delay
            self.voice_notifier.notify_resuming(
                self.episode_count,
                self.config.inter_episode_delay
            )

            # Cancel previous timer if exists
            if self.resume_timer is not None:
                self.resume_timer.cancel()

            # Create one-shot timer to resume recording
            self.resume_timer = self.create_timer(
                self.config.inter_episode_delay,
                self._resume_recording
            )

    def _resume_recording(self) -> None:
        """Resume recording after inter-episode delay."""
        self.is_waiting_between_episodes = False
        self.get_logger().info(f'Resuming recording for episode {self.episode_count}')

        # Voice notification for episode start
        if self.config.notify_on_episode_start:
            self.voice_notifier.notify_episode_start(self.episode_count)

        # Cancel and clean up the one-shot timer
        if self.resume_timer is not None:
            self.resume_timer.cancel()
            self.resume_timer = None

    # Status reporting
    def _report_status(self) -> None:
        """Report connection and recording status periodically."""
        connected_count = self._count_connected_topics()
        total_count = self._count_total_required_topics()

        self.get_logger().info(
            f'Connection Status: {connected_count}/{total_count} topics receiving data'
        )
        self._log_topic_publishers()

        if self.is_waiting_between_episodes:
            self.get_logger().info(
                f'  Status: Waiting between episodes (next: Episode {self.episode_count})'
            )
        elif self.is_recording:
            self.get_logger().info(
                f'  Recording: Episode {self.episode_count}, '
                f'Step {self.step_count}/{self.config.episode_length}'
            )

    def _count_connected_topics(self) -> int:
        """Count how many required topics are receiving data."""
        count = 0
        if self.latest_joint_state is not None:
            count += 1
        if self.config.use_camera:
            if self.latest_rgb_image is not None:
                count += 1
            if self.config.use_depth and self.latest_depth_image is not None:
                count += 1
        return count

    def _count_total_required_topics(self) -> int:
        """Count total number of required topics."""
        count = 1  # joint_states
        if self.config.use_camera:
            count += 1  # rgb_image
            if self.config.use_depth:
                count += 1  # depth_image
        return count

    def _log_topic_publishers(self) -> None:
        """Log publisher counts for each topic."""
        topics = [self.config.joint_states_topic]
        if self.config.use_camera:
            topics.append(self.config.rgb_image_topic)
            if self.config.use_depth:
                topics.append(self.config.depth_image_topic)

        for topic in topics:
            count = self.count_publishers(topic)
            self.get_logger().info(f'  {topic}: {count} publishers')

    # Shutdown
    def shutdown(self) -> None:
        """Save any remaining data before shutdown."""
        # Cancel resume timer if active
        if self.resume_timer is not None:
            self.resume_timer.cancel()
            self.resume_timer = None

        if len(self.current_episode) > 0:
            self.get_logger().info('Saving partial episode before shutdown...')
            language_instruction = (
                self.latest_language_instruction
                if self.latest_language_instruction is not None
                else self.config.default_language_instruction
            )
            self.episode_saver.save(
                self.current_episode,
                self.episode_count,
                language_instruction=language_instruction
            )

        # Compute and save dataset statistics
        if self.config.compute_dataset_statistics:
            self.get_logger().info('Computing dataset statistics...')
            self.episode_saver.compute_and_save_statistics()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = DataLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
