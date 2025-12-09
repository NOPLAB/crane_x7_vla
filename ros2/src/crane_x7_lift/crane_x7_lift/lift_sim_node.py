#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for unified simulator abstraction (lift)."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Bool, String
from std_srvs.srv import Trigger, SetBool
from cv_bridge import CvBridge
from typing import Optional


class LiftSimNode(Node):
    """ROS 2 node managing simulation via lift abstraction."""

    def __init__(self):
        super().__init__('lift_sim_node')

        self._declare_parameters()
        self._load_parameters()

        self.bridge = CvBridge()
        self.simulator = None
        self.latest_action: Optional[np.ndarray] = None
        self.episode_step = 0
        self.is_running = False

        self._init_simulator()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()

        timer_period = 1.0 / self.sim_rate
        self.sim_timer = self.create_timer(timer_period, self._sim_step_callback)

        self.get_logger().info('Lift Simulation Node initialized')
        self.get_logger().info(f'Simulator: {self.simulator_name}')
        self.get_logger().info(f'Environment: {self.env_id}')
        self.get_logger().info(f'Simulation rate: {self.sim_rate} Hz')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('simulator', 'maniskill')
        self.declare_parameter('env_id', 'PickPlace-CRANE-X7')
        self.declare_parameter('backend', 'gpu')
        self.declare_parameter('render_mode', 'rgb_array')
        self.declare_parameter('control_mode', 'pd_joint_pos')
        self.declare_parameter('sim_rate', 30.0)
        self.declare_parameter('max_episode_steps', 200)
        self.declare_parameter('auto_reset', True)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('action_topic', '/vla/predicted_action')
        self.declare_parameter('episode_done_topic', '/lift/episode_done')
        self.declare_parameter('robot_init_qpos_noise', 0.02)

    def _load_parameters(self):
        """Load parameters from ROS."""
        self.simulator_name = self.get_parameter('simulator').value
        self.env_id = self.get_parameter('env_id').value
        self.backend = self.get_parameter('backend').value
        self.render_mode = self.get_parameter('render_mode').value
        self.control_mode = self.get_parameter('control_mode').value
        self.sim_rate = self.get_parameter('sim_rate').value
        self.max_episode_steps = self.get_parameter('max_episode_steps').value
        self.auto_reset = self.get_parameter('auto_reset').value
        self.image_topic = self.get_parameter('image_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.action_topic = self.get_parameter('action_topic').value
        self.episode_done_topic = self.get_parameter('episode_done_topic').value
        self.robot_init_qpos_noise = self.get_parameter('robot_init_qpos_noise').value

    def _init_simulator(self):
        """Initialize simulator via lift factory."""
        try:
            import sys
            sys.path.insert(0, '/workspace/sim/src')

            from lift import create_simulator, SimulatorConfig

            # Import adapters to register them
            if self.simulator_name == 'maniskill':
                from lift_maniskill import ManiSkillSimulator  # noqa: F401
            elif self.simulator_name == 'genesis':
                from lift_genesis import GenesisSimulator  # noqa: F401
            elif self.simulator_name == 'isaacsim':
                from lift_isaacsim import IsaacSimSimulator  # noqa: F401

            config = SimulatorConfig(
                env_id=self.env_id,
                backend=self.backend,
                render_mode=self.render_mode,
                control_mode=self.control_mode,
                sim_rate=self.sim_rate,
                max_episode_steps=self.max_episode_steps,
                robot_init_qpos_noise=self.robot_init_qpos_noise,
            )

            self.simulator = create_simulator(self.simulator_name, config)
            self.is_running = True

            self.get_logger().info(f'{self.simulator_name} simulator initialized successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize simulator: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _setup_publishers(self):
        """Setup ROS 2 publishers."""
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.joint_state_pub = self.create_publisher(
            JointState, self.joint_states_topic, 10
        )
        self.episode_done_pub = self.create_publisher(Bool, self.episode_done_topic, 10)
        self.task_info_pub = self.create_publisher(String, '/lift/task_info', 10)

    def _setup_subscribers(self):
        """Setup ROS 2 subscribers."""
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            self.action_topic,
            self._action_callback,
            10
        )

    def _setup_services(self):
        """Setup ROS 2 services."""
        self.reset_srv = self.create_service(
            Trigger, '/lift/reset', self._reset_callback
        )
        self.pause_srv = self.create_service(
            SetBool, '/lift/pause', self._pause_callback
        )

    def _action_callback(self, msg: Float32MultiArray):
        """Callback for receiving action commands."""
        self.latest_action = np.array(msg.data, dtype=np.float32)
        self.get_logger().debug(f'Received action: {self.latest_action}')

    def _reset_callback(self, request, response):
        """Service callback for resetting the environment."""
        try:
            obs, info = self.simulator.reset()
            self.episode_step = 0
            self.latest_action = None
            response.success = True
            response.message = 'Environment reset successfully'
            self.get_logger().info('Environment reset')
        except Exception as e:
            response.success = False
            response.message = f'Reset failed: {e}'
        return response

    def _pause_callback(self, request, response):
        """Service callback for pausing/resuming simulation."""
        self.is_running = not request.data
        response.success = True
        response.message = f'Simulation {"paused" if not self.is_running else "resumed"}'
        return response

    def _sim_step_callback(self):
        """Main simulation step callback."""
        if not self.is_running or self.simulator is None:
            return

        self._publish_observations()

        if self.latest_action is not None:
            self._execute_step()

    def _publish_observations(self):
        """Publish camera image and joint states."""
        if self.simulator is None:
            return

        try:
            self._publish_image()
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

        try:
            self._publish_joint_states()
        except Exception as e:
            self.get_logger().error(f'Failed to publish joint states: {e}')

    def _publish_image(self):
        """Publish hand camera image."""
        obs = self.simulator.get_observation()
        if obs.rgb_image is None:
            return

        rgb = obs.rgb_image
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'hand_camera_link'
        self.image_pub.publish(img_msg)

    def _publish_joint_states(self):
        """Publish joint states."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.simulator.all_joint_names

        qpos = self.simulator.get_qpos()
        joint_msg.position = qpos.tolist()
        joint_msg.velocity = [0.0] * len(joint_msg.name)
        joint_msg.effort = [0.0] * len(joint_msg.name)

        self.joint_state_pub.publish(joint_msg)

    def _execute_step(self):
        """Execute one simulation step with the latest action."""
        if self.latest_action is None:
            return

        try:
            action = self.latest_action

            result = self.simulator.step(action)
            self.episode_step += 1

            done = result.terminated or result.truncated or (
                self.episode_step >= self.max_episode_steps
            )

            self.get_logger().debug(
                f'Step {self.episode_step}: reward={result.reward:.4f}, done={done}'
            )

            done_msg = Bool()
            done_msg.data = bool(done)
            self.episode_done_pub.publish(done_msg)

            if done:
                self._handle_episode_end(result.info)

        except Exception as e:
            self.get_logger().error(f'Step execution failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _handle_episode_end(self, info: dict):
        """Handle end of episode."""
        success = info.get('success', False)
        if hasattr(success, 'item'):
            success = success.item()
        self.get_logger().info(
            f'Episode ended at step {self.episode_step}, success={success}'
        )

        task_msg = String()
        task_msg.data = f'Episode complete: steps={self.episode_step}, success={success}'
        self.task_info_pub.publish(task_msg)

        if self.auto_reset:
            self.simulator.reset()
            self.episode_step = 0
            self.latest_action = None
            self.get_logger().info('Auto-reset: new episode started')

    def destroy_node(self):
        """Cleanup on node destruction."""
        if self.simulator is not None:
            self.simulator.close()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = LiftSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
