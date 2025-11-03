#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for controlling CRANE-X7 with VLA-predicted actions."""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from typing import Optional, List


class RobotController(Node):
    """ROS 2 node for executing VLA-predicted actions on CRANE-X7."""

    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters
        self.declare_parameter('action_topic', '/vla/predicted_action')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('controller_name', '/crane_x7_arm_controller/follow_joint_trajectory')
        self.declare_parameter('execution_time', 1.0)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('auto_execute', True)
        self.declare_parameter('joint_names', [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
            'crane_x7_gripper_finger_a_joint'
        ])

        # Get parameters
        self.action_topic = self.get_parameter('action_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.controller_name = self.get_parameter('controller_name').value
        self.execution_time = self.get_parameter('execution_time').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.auto_execute = self.get_parameter('auto_execute').value
        self.joint_names = self.get_parameter('joint_names').value

        # State
        self.current_joint_state: Optional[JointState] = None
        self.latest_action: Optional[np.ndarray] = None
        self.is_executing = False

        # Setup action client for trajectory execution
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.controller_name
        )

        # Setup subscriptions
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            self.action_topic,
            self._action_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_state_callback,
            10
        )

        # Wait for action server
        self.get_logger().info(f'Waiting for action server: {self.controller_name}')
        self.trajectory_client.wait_for_server()
        self.get_logger().info('Action server connected')

        self.get_logger().info('Robot Controller initialized')
        self.get_logger().info(f'Auto execute: {self.auto_execute}')

    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for current joint states."""
        self.current_joint_state = msg

    def _action_callback(self, msg: Float32MultiArray) -> None:
        """Callback for VLA-predicted actions."""
        action = np.array(msg.data)
        self.latest_action = action

        if self.auto_execute and not self.is_executing:
            self._execute_action(action)

    def _execute_action(self, action: np.ndarray) -> None:
        """Execute the predicted action on the robot."""
        if self.current_joint_state is None:
            self.get_logger().warn('No joint state received yet')
            return

        if len(action) != len(self.joint_names):
            self.get_logger().error(
                f'Action dimension mismatch: expected {len(self.joint_names)}, got {len(action)}'
            )
            return

        self.is_executing = True

        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        point.time_from_start.sec = int(self.execution_time)
        point.time_from_start.nanosec = int((self.execution_time % 1) * 1e9)

        goal_msg.trajectory.points = [point]

        # Set tolerances
        from control_msgs.msg import JointTolerance
        goal_msg.goal_tolerance = []
        for joint_name in self.joint_names:
            tolerance = JointTolerance()
            tolerance.name = joint_name
            tolerance.position = self.position_tolerance
            goal_msg.goal_tolerance.append(tolerance)

        # Send goal
        self.get_logger().info(f'Executing action: {action}')
        send_goal_future = self.trajectory_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        """Callback when goal is accepted or rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            self.is_executing = False
            return

        self.get_logger().info('Goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _feedback_callback(self, feedback_msg) -> None:
        """Callback for execution feedback."""
        # Can log feedback here if needed
        pass

    def _get_result_callback(self, future) -> None:
        """Callback when execution is complete."""
        result = future.result().result
        self.is_executing = False

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Action executed successfully')
        else:
            self.get_logger().error(f'Action execution failed with error code: {result.error_code}')

    def execute_latest_action(self) -> None:
        """Manually execute the latest received action."""
        if self.latest_action is not None and not self.is_executing:
            self._execute_action(self.latest_action)
        elif self.is_executing:
            self.get_logger().warn('Already executing an action')
        else:
            self.get_logger().warn('No action received yet')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
