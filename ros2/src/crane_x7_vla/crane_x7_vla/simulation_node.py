"""Application model/controller configuration on the shared ROS bridge."""

import rclpy
from crane_x7_vla_robot.simulation import create_simulator, simulator_config
from usim.bridges.simulation_ros2 import UsimSimNode


class CraneX7SimNode(UsimSimNode):
    """Keep robot task registration outside the generic bridge."""

    def _init_simulator(self):
        fields = (
            "backend", "render_mode", "control_mode", "sim_rate",
            "max_episode_steps", "robot_init_qpos_noise", "camera_uid",
            "robot_uid", "obs_mode", "fixed_base",
        )
        settings = {name: self.get_parameter(name).value for name in fields}
        for name in (
            "arm_joint_names", "gripper_joint_names", "joint_names",
            "velocity_joint_names", "initial_qpos", "joint_stiffness",
            "joint_damping", "joint_force_limits",
        ):
            value = self.get_parameter(name).value
            if value:
                settings[name] = tuple(value)
        path = self.get_parameter("robot_path").value
        if path:
            settings["robot_path"] = path
        config = simulator_config(self.env_id, **settings)
        self.simulator = create_simulator(self.simulator_name, config)
        self.is_running = True


def main(args=None):
    rclpy.init(args=args)
    node = CraneX7SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
