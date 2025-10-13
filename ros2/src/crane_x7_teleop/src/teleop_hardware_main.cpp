// Copyright 2025
// Licensed under the MIT License

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "crane_x7_teleop/teleop_hardware_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<crane_x7_teleop::TeleopHardwareNode>(rclcpp::NodeOptions());
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("teleop_hardware_main"), "Exception: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
