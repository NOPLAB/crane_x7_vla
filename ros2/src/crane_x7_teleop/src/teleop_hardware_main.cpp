// Copyright 2025
// Licensed under the MIT License

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "crane_x7_teleop/teleop_hardware_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::shared_ptr<crane_x7_teleop::TeleopHardwareNode> node;

  try {
    node = std::make_shared<crane_x7_teleop::TeleopHardwareNode>(rclcpp::NodeOptions());

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();

  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("teleop_hardware_main"), "Exception: %s", e.what());
    if (node) {
      node->shutdown();
    }
    rclcpp::shutdown();
    return 1;
  }

  // Explicit shutdown on normal exit
  if (node) {
    node->shutdown();
  }

  rclcpp::shutdown();
  return 0;
}
