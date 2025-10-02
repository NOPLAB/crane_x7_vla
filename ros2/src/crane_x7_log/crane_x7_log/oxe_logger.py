#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from threading import Lock
import os
from datetime import datetime

class OXELogger(Node):
    """ROS 2 node for logging CRANE-X7 data in OXE format for VLA fine-tuning."""

    def __init__(self):
        super().__init__('oxe_logger')

        # Declare all parameters with defaults
        self.declare_parameter('output_dir', '/workspace/data/oxe_logs')
        self.declare_parameter('episode_length', 100)
        self.declare_parameter('auto_start_recording', True)
        self.declare_parameter('collection_rate', 10.0)
        self.declare_parameter('use_camera', True)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('image_width', 0)
        self.declare_parameter('image_height', 0)
        self.declare_parameter('image_quality', 90)
        self.declare_parameter('arm_joint_names', [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
        ])
        self.declare_parameter('gripper_joint_names', ['crane_x7_gripper_finger_a_joint'])
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('rgb_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('save_format', 'npz')
        self.declare_parameter('auto_convert_tfrecord', False)

        # Get parameters
        self.output_dir = self.get_parameter('output_dir').value
        self.episode_length = self.get_parameter('episode_length').value
        self.auto_start_recording = self.get_parameter('auto_start_recording').value
        self.collection_rate = self.get_parameter('collection_rate').value
        self.use_camera = self.get_parameter('use_camera').value
        self.use_depth = self.get_parameter('use_depth').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.image_quality = self.get_parameter('image_quality').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.gripper_joint_names = self.get_parameter('gripper_joint_names').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.rgb_image_topic = self.get_parameter('rgb_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.save_format = self.get_parameter('save_format').value
        self.auto_convert_tfrecord = self.get_parameter('auto_convert_tfrecord').value

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Recording state
        self.is_recording = self.auto_start_recording

        # Data buffers
        self.lock = Lock()
        self.current_episode = []
        self.episode_count = 0
        self.step_count = 0

        # Latest data cache
        self.latest_joint_state = None
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10
        )

        if self.use_camera:
            self.rgb_image_sub = self.create_subscription(
                Image,
                self.rgb_image_topic,
                self.rgb_image_callback,
                10
            )

            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                10
            )

            if self.use_depth:
                self.depth_image_sub = self.create_subscription(
                    Image,
                    self.depth_image_topic,
                    self.depth_image_callback,
                    10
                )

        # Timer for periodic data collection
        timer_period = 1.0 / self.collection_rate
        self.timer = self.create_timer(timer_period, self.collect_step)

        # Timer for connection status reporting
        self.status_timer = self.create_timer(2.0, self.report_connection_status)

        self.get_logger().info(f'OXE Logger initialized')
        self.get_logger().info(f'  Output: {self.output_dir}')
        self.get_logger().info(f'  Episode length: {self.episode_length}')
        self.get_logger().info(f'  Collection rate: {self.collection_rate} Hz')
        self.get_logger().info(f'  Camera: {self.use_camera}, Depth: {self.use_depth}')
        self.get_logger().info(f'  Auto-start: {self.auto_start_recording}')
        self.get_logger().info(f'  Recording: {self.is_recording}')

    def report_connection_status(self):
        """Report topic connection status periodically."""
        required_topics = [self.joint_states_topic]
        if self.use_camera:
            required_topics.append(self.rgb_image_topic)
            if self.use_depth:
                required_topics.append(self.depth_image_topic)

        connected_count = 0
        total_count = len(required_topics)

        # Check joint states
        if self.latest_joint_state is not None:
            connected_count += 1

        # Check RGB camera
        if self.use_camera:
            if self.latest_rgb_image is not None:
                connected_count += 1
            if self.use_depth and self.latest_depth_image is not None:
                connected_count += 1

        # Get topic publisher counts
        topic_info = []
        topic_info.append(f"{self.joint_states_topic}: {self._get_publisher_count(self.joint_states_topic)} publishers")

        if self.use_camera:
            topic_info.append(f"{self.rgb_image_topic}: {self._get_publisher_count(self.rgb_image_topic)} publishers")
            if self.use_depth:
                topic_info.append(f"{self.depth_image_topic}: {self._get_publisher_count(self.depth_image_topic)} publishers")

        status_msg = f"Connection Status: {connected_count}/{total_count} topics receiving data"
        self.get_logger().info(status_msg)
        for info in topic_info:
            self.get_logger().info(f"  {info}")

        if self.is_recording:
            self.get_logger().info(f"  Recording: Episode {self.episode_count}, Step {self.step_count}/{self.episode_length}")

    def _get_publisher_count(self, topic_name):
        """Get number of publishers for a topic."""
        topic_names_and_types = self.get_topic_names_and_types()
        for name, _ in topic_names_and_types:
            if name == topic_name:
                return self.count_publishers(topic_name)
        return 0

    def joint_state_callback(self, msg):
        """Callback for /joint_states topic."""
        with self.lock:
            self.latest_joint_state = msg

    def rgb_image_callback(self, msg):
        """Callback for RGB image topic."""
        with self.lock:
            self.latest_rgb_image = msg

    def depth_image_callback(self, msg):
        """Callback for depth image topic."""
        with self.lock:
            self.latest_depth_image = msg

    def camera_info_callback(self, msg):
        """Callback for camera info topic."""
        with self.lock:
            self.latest_camera_info = msg

    def extract_joint_positions(self, joint_state):
        """Extract arm and gripper joint positions in correct order."""
        if joint_state is None:
            return None, None

        # Create mapping from joint name to position
        joint_dict = dict(zip(joint_state.name, joint_state.position))

        # Extract arm joints
        arm_positions = []
        for name in self.arm_joint_names:
            if name in joint_dict:
                arm_positions.append(joint_dict[name])
            else:
                self.get_logger().warn(f'Joint {name} not found in joint_states')
                return None, None

        # Extract gripper joint
        gripper_position = None
        if self.gripper_joint_names[0] in joint_dict:
            gripper_position = joint_dict[self.gripper_joint_names[0]]

        return np.array(arm_positions, dtype=np.float32), gripper_position

    def collect_step(self):
        """Collect one step of data."""
        with self.lock:
            # Skip if not recording
            if not self.is_recording:
                return

            # Check if we have minimum required data
            if self.latest_joint_state is None:
                return

            if self.use_camera and self.latest_rgb_image is None:
                return

            # Extract joint positions
            arm_pos, gripper_pos = self.extract_joint_positions(self.latest_joint_state)
            if arm_pos is None:
                return

            # Create state vector (7 arm joints + 1 gripper)
            if gripper_pos is not None:
                state = np.concatenate([arm_pos, [gripper_pos]])
            else:
                state = arm_pos

            # Convert images
            rgb_array = None
            depth_array = None

            if self.use_camera and self.latest_rgb_image is not None:
                try:
                    rgb_array = self.bridge.imgmsg_to_cv2(
                        self.latest_rgb_image, desired_encoding='rgb8'
                    )
                    # Resize if specified
                    if self.image_width > 0 and self.image_height > 0:
                        import cv2
                        rgb_array = cv2.resize(rgb_array, (self.image_width, self.image_height))
                except Exception as e:
                    self.get_logger().error(f'Failed to convert RGB image: {e}')
                    return

            if self.use_depth and self.latest_depth_image is not None:
                try:
                    depth_array = self.bridge.imgmsg_to_cv2(
                        self.latest_depth_image, desired_encoding='passthrough'
                    )
                except Exception as e:
                    self.get_logger().error(f'Failed to convert depth image: {e}')

            # Create step data
            step_data = {
                'observation': {
                    'state': state,
                    'timestamp': self.latest_joint_state.header.stamp.sec +
                                self.latest_joint_state.header.stamp.nanosec * 1e-9
                },
                'action': state,  # For now, use current state as action (will be updated)
            }

            if rgb_array is not None:
                step_data['observation']['image'] = rgb_array

            if depth_array is not None:
                step_data['observation']['depth'] = depth_array

            # Add to current episode
            self.current_episode.append(step_data)
            self.step_count += 1

            # Save episode when reaching episode length
            if self.step_count >= self.episode_length:
                self.save_episode()
                self.current_episode = []
                self.step_count = 0
                self.episode_count += 1

    def save_episode(self):
        """Save current episode to disk."""
        if len(self.current_episode) == 0:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        episode_dir = os.path.join(
            self.output_dir,
            f'episode_{self.episode_count:04d}_{timestamp}'
        )
        os.makedirs(episode_dir, exist_ok=True)

        # Update actions: action[t] = state[t+1]
        for i in range(len(self.current_episode) - 1):
            self.current_episode[i]['action'] = self.current_episode[i + 1]['observation']['state']

        # Save as numpy arrays for now (will convert to TFRecord later)
        episode_path = os.path.join(episode_dir, 'episode_data.npz')

        # Prepare data for saving
        states = np.array([step['observation']['state'] for step in self.current_episode])
        actions = np.array([step['action'] for step in self.current_episode])
        timestamps = np.array([step['observation']['timestamp'] for step in self.current_episode])

        save_dict = {
            'states': states,
            'actions': actions,
            'timestamps': timestamps,
        }

        # Add images if available
        if 'image' in self.current_episode[0]['observation']:
            images = np.array([step['observation']['image'] for step in self.current_episode])
            save_dict['images'] = images

        if 'depth' in self.current_episode[0]['observation']:
            depths = np.array([step['observation']['depth'] for step in self.current_episode])
            save_dict['depths'] = depths

        np.savez_compressed(episode_path, **save_dict)

        self.get_logger().info(
            f'Saved episode {self.episode_count} with {len(self.current_episode)} steps to {episode_path}'
        )

    def shutdown(self):
        """Save any remaining data before shutdown."""
        if len(self.current_episode) > 0:
            self.get_logger().info('Saving partial episode before shutdown...')
            self.save_episode()


def main(args=None):
    rclpy.init(args=args)
    node = OXELogger()

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
