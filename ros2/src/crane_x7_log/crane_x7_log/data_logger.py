#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for logging CRANE-X7 data in NPZ or TFRecord format for VLA fine-tuning."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import String
import numpy as np
from threading import Lock
from typing import Optional, Dict, Any, List

from .config_manager import ConfigManager, LoggerConfig, CameraConfig
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

        # Initialize image processors for each camera
        self.image_processors: Dict[str, ImageProcessor] = {}
        for camera in self.config.cameras:
            self.image_processors[camera.name] = ImageProcessor(
                camera.width,
                camera.height
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
        self.latest_language_instruction: Optional[str] = None

        # Multiple camera data cache (camera_name -> Image)
        self.latest_rgb_images: Dict[str, Optional[Image]] = {
            cam.name: None for cam in self.config.cameras
        }
        self.latest_camera_infos: Dict[str, Optional[CameraInfo]] = {
            cam.name: None for cam in self.config.cameras
        }

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

        # Multiple camera subscriptions
        self.rgb_image_subs: Dict[str, Any] = {}
        self.camera_info_subs: Dict[str, Any] = {}

        for camera in self.config.cameras:
            # RGB image subscription
            self.rgb_image_subs[camera.name] = self.create_subscription(
                Image,
                camera.rgb_topic,
                lambda msg, name=camera.name: self._rgb_image_callback(msg, name),
                10
            )

            # Camera info subscription (if topic is specified)
            if camera.camera_info_topic:
                self.camera_info_subs[camera.name] = self.create_subscription(
                    CameraInfo,
                    camera.camera_info_topic,
                    lambda msg, name=camera.name: self._camera_info_callback(msg, name),
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
        self.get_logger().info(f'  Cameras: {len(self.config.cameras)}')
        for camera in self.config.cameras:
            self.get_logger().info(f'    - {camera.name}: {camera.rgb_topic}')
        self.get_logger().info(f'  Auto-start: {self.config.auto_start_recording}')
        self.get_logger().info(f'  Recording: {self.is_recording}')

    # Callbacks
    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint states topic."""
        with self.lock:
            self.latest_joint_state = msg

    def _rgb_image_callback(self, msg: Image, camera_name: str) -> None:
        """Callback for RGB image topic."""
        with self.lock:
            self.latest_rgb_images[camera_name] = msg

    def _camera_info_callback(self, msg: CameraInfo, camera_name: str) -> None:
        """Callback for camera info topic."""
        with self.lock:
            self.latest_camera_infos[camera_name] = msg

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

            # Process images from all cameras
            rgb_arrays = self._process_all_rgb_images()

            # Create step data
            step_data = self._create_step_data(joint_positions, rgb_arrays)

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
        # Check if all cameras have received data
        for camera in self.config.cameras:
            if self.latest_rgb_images.get(camera.name) is None:
                return False
        return True

    def _process_all_rgb_images(self) -> Dict[str, Optional[np.ndarray]]:
        """Process RGB images from all cameras."""
        rgb_arrays: Dict[str, Optional[np.ndarray]] = {}

        for camera in self.config.cameras:
            image_msg = self.latest_rgb_images.get(camera.name)
            if image_msg is not None:
                processor = self.image_processors.get(camera.name)
                if processor:
                    rgb_array = processor.process_rgb_image(image_msg)
                    if rgb_array is None:
                        self.get_logger().error(
                            f'Failed to convert RGB image from camera {camera.name}'
                        )
                    rgb_arrays[camera.name] = rgb_array
                else:
                    rgb_arrays[camera.name] = None
            else:
                rgb_arrays[camera.name] = None

        return rgb_arrays

    def _create_step_data(
        self,
        joint_positions: np.ndarray,
        rgb_arrays: Dict[str, Optional[np.ndarray]]
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

        # Add images from each camera with RLDS naming convention
        for camera in self.config.cameras:
            image_key = f'image_{camera.name}'  # e.g., image_primary, image_secondary
            if camera.name in rgb_arrays and rgb_arrays[camera.name] is not None:
                step_data['observation'][image_key] = rgb_arrays[camera.name]

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
        # Count connected cameras
        for camera in self.config.cameras:
            if self.latest_rgb_images.get(camera.name) is not None:
                count += 1
        return count

    def _count_total_required_topics(self) -> int:
        """Count total number of required topics."""
        count = 1  # joint_states
        count += len(self.config.cameras)  # One RGB topic per camera
        return count

    def _log_topic_publishers(self) -> None:
        """Log publisher counts for each topic."""
        # Joint states topic
        joint_count = self.count_publishers(self.config.joint_states_topic)
        self.get_logger().info(
            f'  {self.config.joint_states_topic}: {joint_count} publishers'
        )

        # Camera topics
        for camera in self.config.cameras:
            count = self.count_publishers(camera.rgb_topic)
            self.get_logger().info(
                f'  {camera.rgb_topic} ({camera.name}): {count} publishers'
            )

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
