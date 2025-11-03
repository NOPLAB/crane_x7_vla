#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Trajectory planning node using Gemini API and MoveIt.

This node receives task prompts via a topic, generates trajectories using Gemini,
and executes them using MoveIt2.
"""

import json
from typing import List, Optional, Dict, Any
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge
import cv2

# MoveIt imports
try:
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint
    from moveit_msgs.msg import RobotTrajectory
except ImportError:
    print("WARNING: MoveIt messages not available. Install moveit_msgs.")

from crane_x7_gemini.gemini_node import GeminiNode
from crane_x7_gemini.coordinate_transformer import CoordinateTransformer


class TrajectoryPlanner(GeminiNode):
    """Trajectory planning node with Gemini and MoveIt integration."""

    def __init__(self):
        super().__init__()

        self.get_logger().info('Initializing TrajectoryPlanner')

        # Declare additional parameters
        self.declare_parameter('prompt_topic', '/gemini/task_prompt')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('move_group', 'arm')
        self.declare_parameter('end_effector_link', 'crane_x7_gripper_base_link')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('execute_trajectory', True)

        # Get parameters
        prompt_topic = self.get_parameter('prompt_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.move_group_name = self.get_parameter('move_group').value
        self.end_effector_link = self.get_parameter('end_effector_link').value
        self.planning_time = self.get_parameter('planning_time').value
        self.execute_trajectory = self.get_parameter('execute_trajectory').value

        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer()

        # Subscribe to task prompt topic
        self.prompt_sub = self.create_subscription(
            String,
            prompt_topic,
            self.prompt_callback,
            10
        )

        # Subscribe to depth image
        self.depth_image: Optional[np.ndarray] = None
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )

        # Publisher for trajectory status
        self.status_pub = self.create_publisher(
            String,
            '/gemini/trajectory_status',
            10
        )

        # Publisher for planned waypoints
        self.waypoints_pub = self.create_publisher(
            String,
            '/gemini/waypoints',
            10
        )

        self.get_logger().info(f'Subscribing to prompts on: {prompt_topic}')
        self.get_logger().info(f'Subscribing to depth on: {depth_topic}')
        self.get_logger().info('TrajectoryPlanner ready')

    def depth_callback(self, msg: Image) -> None:
        """Callback for receiving depth images."""
        try:
            # Convert depth image (16UC1 or 32FC1)
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Convert to meters (RealSense depth is in millimeters)
                self.depth_image = depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
                return

            self.get_logger().debug('Received depth image')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def prompt_callback(self, msg: String) -> None:
        """Callback for receiving task prompts."""
        prompt = msg.data
        self.get_logger().info(f'Received task prompt: {prompt}')

        # Update status
        self.publish_status('processing', f'Processing prompt: {prompt}')

        # Generate and execute trajectory
        success = self.process_task_prompt(prompt)

        if success:
            self.publish_status('completed', 'Task completed successfully')
        else:
            self.publish_status('failed', 'Task execution failed')

    def publish_status(self, status: str, message: str) -> None:
        """Publish trajectory execution status."""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'message': message
        })
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Status: {status} - {message}')

    def process_task_prompt(self, prompt: str) -> bool:
        """
        Process a task prompt and execute the trajectory.

        Args:
            prompt: Natural language task description

        Returns:
            True if successful, False otherwise
        """
        # Check prerequisites
        if self.latest_image is None:
            self.get_logger().error('No RGB image available')
            return False

        if self.depth_image is None:
            self.get_logger().error('No depth image available')
            return False

        # Generate trajectory using Gemini
        trajectory_2d = self.generate_trajectory(prompt)

        if trajectory_2d is None or len(trajectory_2d) == 0:
            self.get_logger().error('Failed to generate trajectory')
            return False

        # Convert 2D trajectory to 3D waypoints
        waypoints_3d = self.trajectory_2d_to_3d(trajectory_2d)

        if waypoints_3d is None or len(waypoints_3d) == 0:
            self.get_logger().error('Failed to convert trajectory to 3D')
            return False

        # Publish waypoints for visualization
        self.publish_waypoints(waypoints_3d)

        # Execute trajectory with MoveIt
        if self.execute_trajectory:
            return self.execute_moveit_trajectory(waypoints_3d)
        else:
            self.get_logger().info('Trajectory planning completed (execution disabled)')
            return True

    def generate_trajectory(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generate trajectory points using Gemini API.

        Args:
            prompt: Natural language task description

        Returns:
            List of trajectory points with 'point' and 'label' keys
        """
        # Construct Gemini prompt for trajectory generation
        gemini_prompt = f"""
        {prompt}

        Generate a trajectory as a sequence of points from start to end.
        Include at least 10-20 intermediate points for a smooth motion.
        Label each point by its order in the trajectory from '0' (start) to 'n' (end).

        The answer should follow the json format:
        [{{"point": [y, x], "label": "0"}}, {{"point": [y, x], "label": "1"}}, ...]
        The points are in [y, x] format normalized to 0-1000.
        """

        # Call Gemini API
        trajectory = self.detect_objects(prompt=gemini_prompt)

        if trajectory is None:
            self.get_logger().error('Gemini API call failed')
            return None

        # Sort by label to ensure correct order
        try:
            trajectory_sorted = sorted(trajectory, key=lambda x: int(x['label']))
            self.get_logger().info(f'Generated trajectory with {len(trajectory_sorted)} points')
            return trajectory_sorted
        except (KeyError, ValueError) as e:
            self.get_logger().error(f'Failed to parse trajectory response: {e}')
            return None

    def trajectory_2d_to_3d(
        self,
        trajectory_2d: List[Dict[str, Any]]
    ) -> Optional[List[Pose]]:
        """
        Convert 2D trajectory points to 3D poses in robot base frame.

        Args:
            trajectory_2d: List of 2D points from Gemini

        Returns:
            List of Pose messages in robot base frame
        """
        waypoints = []

        for point_data in trajectory_2d:
            try:
                # Extract normalized coordinates
                point_2d = point_data['point']
                y_norm, x_norm = point_2d[0], point_2d[1]

                # Convert to pixel coordinates
                pixel_x, pixel_y = self.transformer.normalized_to_pixel(y_norm, x_norm)

                # Get depth value
                if (pixel_y >= self.depth_image.shape[0] or
                    pixel_x >= self.depth_image.shape[1] or
                    pixel_y < 0 or pixel_x < 0):
                    self.get_logger().warn(f'Point outside image bounds: ({pixel_x}, {pixel_y})')
                    continue

                depth = self.depth_image[pixel_y, pixel_x]

                # Check for invalid depth
                if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                    self.get_logger().warn(f'Invalid depth at ({pixel_x}, {pixel_y}): {depth}')
                    # Use average depth from surrounding pixels
                    depth = self.get_average_depth(pixel_x, pixel_y)
                    if depth <= 0:
                        continue

                # Convert to 3D base coordinates
                point_3d = self.transformer.normalized_to_base_coords(y_norm, x_norm, depth)

                # Create Pose message
                pose = Pose()
                pose.position = point_3d

                # Set default orientation (pointing down)
                pose.orientation.x = 0.0
                pose.orientation.y = 0.707  # 90 degrees around Y axis
                pose.orientation.z = 0.0
                pose.orientation.w = 0.707

                waypoints.append(pose)

                self.get_logger().debug(
                    f'Waypoint: pixel=({pixel_x},{pixel_y}), depth={depth:.3f}m, '
                    f'base=({point_3d.x:.3f}, {point_3d.y:.3f}, {point_3d.z:.3f})'
                )

            except (KeyError, IndexError) as e:
                self.get_logger().error(f'Failed to process point: {e}')
                continue

        self.get_logger().info(f'Converted {len(waypoints)} waypoints to 3D')
        return waypoints

    def get_average_depth(self, x: int, y: int, window_size: int = 5) -> float:
        """Get average depth from surrounding pixels."""
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(self.depth_image.shape[0], y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(self.depth_image.shape[1], x + half_window + 1)

        region = self.depth_image[y_min:y_max, x_min:x_max]
        valid_depths = region[(region > 0) & ~np.isnan(region) & ~np.isinf(region)]

        if len(valid_depths) > 0:
            return float(np.mean(valid_depths))
        else:
            return 0.0

    def publish_waypoints(self, waypoints: List[Pose]) -> None:
        """Publish waypoints as JSON for visualization."""
        waypoints_data = []
        for i, pose in enumerate(waypoints):
            waypoints_data.append({
                'index': i,
                'position': {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z
                },
                'orientation': {
                    'x': pose.orientation.x,
                    'y': pose.orientation.y,
                    'z': pose.orientation.z,
                    'w': pose.orientation.w
                }
            })

        msg = String()
        msg.data = json.dumps(waypoints_data, indent=2)
        self.waypoints_pub.publish(msg)

    def execute_moveit_trajectory(self, waypoints: List[Pose]) -> bool:
        """
        Execute trajectory using MoveIt Cartesian path.

        Args:
            waypoints: List of Pose messages

        Returns:
            True if execution successful, False otherwise
        """
        self.get_logger().info(f'Planning Cartesian path with {len(waypoints)} waypoints')

        try:
            # Import MoveIt Python interface
            from moveit_commander import MoveGroupCommander, PlanningSceneInterface
            import moveit_commander

            # Initialize moveit_commander
            moveit_commander.roscpp_initialize([])

            # Create move group
            move_group = MoveGroupCommander(self.move_group_name)
            move_group.set_end_effector_link(self.end_effector_link)
            move_group.set_planning_time(self.planning_time)

            self.get_logger().info(f'Move group: {self.move_group_name}')
            self.get_logger().info(f'End effector: {self.end_effector_link}')

            # Compute Cartesian path
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints,
                0.01,  # eef_step: 1cm
                0.0    # jump_threshold: disabled
            )

            self.get_logger().info(f'Cartesian path planned: {fraction*100:.1f}% achieved')

            if fraction < 0.9:
                self.get_logger().warn('Cartesian path planning incomplete')
                return False

            # Execute trajectory
            self.get_logger().info('Executing trajectory...')
            result = move_group.execute(plan, wait=True)

            move_group.stop()
            move_group.clear_pose_targets()

            if result:
                self.get_logger().info('Trajectory execution completed')
                return True
            else:
                self.get_logger().error('Trajectory execution failed')
                return False

        except ImportError:
            self.get_logger().error(
                'moveit_commander not available. Install: sudo apt install ros-humble-moveit-commander'
            )
            return False
        except Exception as e:
            self.get_logger().error(f'MoveIt execution error: {e}')
            return False


def main(args=None):
    """Main entry point for trajectory planner node."""
    rclpy.init(args=args)

    try:
        node = TrajectoryPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
