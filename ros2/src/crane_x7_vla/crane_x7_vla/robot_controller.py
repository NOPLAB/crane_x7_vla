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
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from typing import Optional, List


class RobotController(Node):
    """ROS 2 node for executing VLA-predicted actions on CRANE-X7."""

    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters
        self.declare_parameter('action_topic', '/vla/predicted_action')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('arm_controller_name', '/crane_x7_arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_controller_name', '/crane_x7_gripper_controller/gripper_cmd')
        self.declare_parameter('execution_time', 1.0)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('auto_execute', True)
        # Arm joints only (7 joints)
        self.declare_parameter('arm_joint_names', [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
        ])

        # Get parameters
        self.action_topic = self.get_parameter('action_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.arm_controller_name = self.get_parameter('arm_controller_name').value
        self.gripper_controller_name = self.get_parameter('gripper_controller_name').value
        self.execution_time = self.get_parameter('execution_time').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.auto_execute = self.get_parameter('auto_execute').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value

        # State
        self.current_joint_state: Optional[JointState] = None
        self.latest_action: Optional[np.ndarray] = None
        self.is_executing = False

        # Setup action client for arm trajectory execution
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.arm_controller_name
        )

        # Setup action client for gripper control
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            self.gripper_controller_name
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

        # Wait for action servers
        self.get_logger().info(f'Waiting for arm action server: {self.arm_controller_name}')
        self.arm_client.wait_for_server()
        self.get_logger().info('Arm action server connected')

        self.get_logger().info(f'Waiting for gripper action server: {self.gripper_controller_name}')
        self.gripper_client.wait_for_server()
        self.get_logger().info('Gripper action server connected')

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

        # Expected: 8 dimensions (7 arm + 1 gripper)
        if len(action) != 8:
            self.get_logger().error(
                f'Action dimension mismatch: expected 8 (7 arm + 1 gripper), got {len(action)}'
            )
            return

        self.is_executing = True

        # Split action into arm (7 joints) and gripper (1 joint)
        arm_action = action[:7]
        gripper_action = action[7]

        self.get_logger().info(f'Executing arm action: {arm_action}, gripper: {gripper_action}')

        # Execute arm trajectory
        self._execute_arm_action(arm_action)

        # Execute gripper action
        self._execute_gripper_action(gripper_action)

    def _execute_arm_action(self, arm_action: np.ndarray) -> None:
        """Execute arm trajectory."""
        from control_msgs.msg import JointTolerance

        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.arm_joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = arm_action.tolist()
        point.time_from_start.sec = int(self.execution_time)
        point.time_from_start.nanosec = int((self.execution_time % 1) * 1e9)

        goal_msg.trajectory.points = [point]

        # Set tolerances
        goal_msg.goal_tolerance = []
        for joint_name in self.arm_joint_names:
            tolerance = JointTolerance()
            tolerance.name = joint_name
            tolerance.position = self.position_tolerance
            goal_msg.goal_tolerance.append(tolerance)

        # Send goal
        send_goal_future = self.arm_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._arm_goal_response_callback)

    def _execute_gripper_action(self, gripper_position: float) -> None:
        """Execute gripper action."""
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(gripper_position)
        goal_msg.command.max_effort = 1.0  # Default max effort

        # Send goal
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._gripper_goal_response_callback)

    def _arm_goal_response_callback(self, future) -> None:
        """Callback when arm goal is accepted or rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected by action server')
            self.is_executing = False
            return

        self.get_logger().debug('Arm goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._arm_result_callback)

    def _gripper_goal_response_callback(self, future) -> None:
        """Callback when gripper goal is accepted or rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected by action server')
            return

        self.get_logger().debug('Gripper goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_result_callback)

    def _feedback_callback(self, feedback_msg) -> None:
        """Callback for execution feedback."""
        # Can log feedback here if needed
        pass

    def _arm_result_callback(self, future) -> None:
        """Callback when arm execution is complete."""
        result = future.result().result
        self.is_executing = False

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().debug('Arm action executed successfully')
        else:
            self.get_logger().error(f'Arm action failed with error code: {result.error_code}')

        # Execute next action if available (continuous execution)
        if self.auto_execute and self.latest_action is not None:
            self.get_logger().info('Arm completed, executing next action...')
            self._execute_action(self.latest_action)

    def _gripper_result_callback(self, future) -> None:
        """Callback when gripper execution is complete."""
        result = future.result().result
        # GripperCommand result has different structure
        self.get_logger().debug(f'Gripper action completed: position={result.position}, reached_goal={result.reached_goal}')

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
