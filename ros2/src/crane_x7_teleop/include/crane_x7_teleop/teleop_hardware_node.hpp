// Copyright 2025
// Licensed under the MIT License

#ifndef CRANE_X7_TELEOP__TELEOP_HARDWARE_NODE_HPP_
#define CRANE_X7_TELEOP__TELEOP_HARDWARE_NODE_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "rt_manipulators_cpp/hardware.hpp"

namespace crane_x7_teleop
{

class TeleopHardwareNode : public rclcpp::Node
{
public:
  explicit TeleopHardwareNode(const rclcpp::NodeOptions & options);
  ~TeleopHardwareNode();

  // Explicit shutdown method for signal handling
  void shutdown();

private:
  // rt_manipulators_cpp Hardware interface
  std::shared_ptr<rt_manipulators_cpp::Hardware> hardware_;

  // Parameters
  std::string mode_;  // "leader" or "follower"
  std::string port_name_;
  int baudrate_;
  std::string config_file_path_;
  std::string links_file_path_;
  double publish_rate_;
  std::vector<std::string> joint_names_;
  std::vector<double> joint_limits_min_;
  std::vector<double> joint_limits_max_;

  // PID gains
  struct PIDGains {
    int p;
    int i;
    int d;
  };
  PIDGains torque_on_gains_;
  PIDGains torque_off_gains_;

  // Topic names
  std::string joint_states_topic_;
  std::string leader_state_topic_;

  // Group name for Dynamixel
  static constexpr auto GROUP_NAME = "arm";

  // Leader mode members
  void setup_leader_mode();
  void read_and_publish_leader();
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr leader_state_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr torque_service_;
  rclcpp::TimerBase::SharedPtr read_timer_;

  // Follower mode members
  void setup_follower_mode();
  void leader_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void read_and_publish_follower();
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr leader_state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr follower_state_pub_;
  rclcpp::TimerBase::SharedPtr feedback_timer_;

  // Last received leader state (for follower mode)
  sensor_msgs::msg::JointState::SharedPtr last_leader_state_;
  rclcpp::Time last_leader_msg_time_;
  static constexpr double LEADER_TIMEOUT_SEC = 1.0;

  // Common methods
  void set_torque_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  bool check_joint_limits(const std::vector<double> & positions);
  bool initialize_hardware();
  void set_torque(bool enable);
  void free_servos();

  // Shutdown flag to prevent double-shutdown
  std::atomic<bool> is_shutdown_{false};
};

}  // namespace crane_x7_teleop

#endif  // CRANE_X7_TELEOP__TELEOP_HARDWARE_NODE_HPP_
