// Copyright 2025
// Licensed under the MIT License

#include "crane_x7_teleop/teleop_hardware_node.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace crane_x7_teleop
{

TeleopHardwareNode::TeleopHardwareNode(const rclcpp::NodeOptions & options)
: Node("teleop_hardware_node", options)
{
  // Declare and get parameters
  this->declare_parameter<std::string>("mode", "leader");
  this->declare_parameter<std::string>("port_name", "/dev/ttyUSB0");
  this->declare_parameter<int>("baudrate", 3000000);
  this->declare_parameter<std::string>("config_file", "");
  this->declare_parameter<std::string>("links_file", "");
  this->declare_parameter<double>("publish_rate", 30.0);
  this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
  this->declare_parameter<std::string>("leader_state_topic", "/teleop/leader/state");
  this->declare_parameter<std::vector<std::string>>("joint_names", std::vector<std::string>());
  this->declare_parameter<std::vector<double>>("joint_limits.min", std::vector<double>());
  this->declare_parameter<std::vector<double>>("joint_limits.max", std::vector<double>());
  this->declare_parameter<int>("torque_on_gains.p", 800);
  this->declare_parameter<int>("torque_on_gains.i", 0);
  this->declare_parameter<int>("torque_on_gains.d", 0);
  this->declare_parameter<int>("torque_off_gains.p", 5);
  this->declare_parameter<int>("torque_off_gains.i", 0);
  this->declare_parameter<int>("torque_off_gains.d", 0);

  mode_ = this->get_parameter("mode").as_string();
  port_name_ = this->get_parameter("port_name").as_string();
  baudrate_ = this->get_parameter("baudrate").as_int();
  config_file_path_ = this->get_parameter("config_file").as_string();
  links_file_path_ = this->get_parameter("links_file").as_string();
  publish_rate_ = this->get_parameter("publish_rate").as_double();
  joint_states_topic_ = this->get_parameter("joint_states_topic").as_string();
  leader_state_topic_ = this->get_parameter("leader_state_topic").as_string();
  joint_names_ = this->get_parameter("joint_names").as_string_array();
  joint_limits_min_ = this->get_parameter("joint_limits.min").as_double_array();
  joint_limits_max_ = this->get_parameter("joint_limits.max").as_double_array();

  torque_on_gains_.p = this->get_parameter("torque_on_gains.p").as_int();
  torque_on_gains_.i = this->get_parameter("torque_on_gains.i").as_int();
  torque_on_gains_.d = this->get_parameter("torque_on_gains.d").as_int();
  torque_off_gains_.p = this->get_parameter("torque_off_gains.p").as_int();
  torque_off_gains_.i = this->get_parameter("torque_off_gains.i").as_int();
  torque_off_gains_.d = this->get_parameter("torque_off_gains.d").as_int();

  // Validate mode
  if (mode_ != "leader" && mode_ != "follower") {
    RCLCPP_ERROR(this->get_logger(), "Invalid mode: %s. Must be 'leader' or 'follower'", mode_.c_str());
    throw std::runtime_error("Invalid mode parameter");
  }

  RCLCPP_INFO(this->get_logger(), "Starting TeleopHardwareNode in %s mode", mode_.c_str());

  // Initialize hardware
  if (!initialize_hardware()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize hardware");
    throw std::runtime_error("Hardware initialization failed");
  }

  // Setup mode-specific functionality
  if (mode_ == "leader") {
    setup_leader_mode();
  } else {
    setup_follower_mode();
  }

  RCLCPP_INFO(this->get_logger(), "TeleopHardwareNode initialized successfully");
}

TeleopHardwareNode::~TeleopHardwareNode()
{
  shutdown();
}

void TeleopHardwareNode::shutdown()
{
  // Prevent double-shutdown
  if (is_shutdown_.exchange(true)) {
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Shutting down %s mode", mode_.c_str());

  free_servos();

  if (hardware_) {
    hardware_->disconnect();
    RCLCPP_INFO(this->get_logger(), "Hardware disconnected");
  }
}

void TeleopHardwareNode::free_servos()
{
  if (!hardware_) {
    return;
  }

  // Set low PID gains before disabling torque
  if (!hardware_->write_position_pid_gain_to_group(
      GROUP_NAME, torque_off_gains_.p, torque_off_gains_.i, torque_off_gains_.d))
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to set PID gains during shutdown");
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Disable torque to free servos (especially important for follower mode)
  if (!hardware_->torque_off(GROUP_NAME)) {
    RCLCPP_WARN(this->get_logger(), "Failed to disable torque, retrying...");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!hardware_->torque_off(GROUP_NAME)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to disable torque - servos may not be free");
    } else {
      RCLCPP_INFO(this->get_logger(), "Servos freed");
    }
  } else {
    RCLCPP_INFO(this->get_logger(), "Servos freed");
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

bool TeleopHardwareNode::initialize_hardware()
{
  // Create hardware interface
  hardware_ = std::make_shared<rt_manipulators_cpp::Hardware>(port_name_);

  // Connect to Dynamixel servos
  if (!hardware_->connect(baudrate_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to connect to hardware on port %s", port_name_.c_str());
    return false;
  }
  RCLCPP_INFO(this->get_logger(), "Connected to hardware on port %s", port_name_.c_str());

  // Load config file
  if (!config_file_path_.empty()) {
    if (!hardware_->load_config_file(config_file_path_)) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to load config file: %s. Continuing without hardware config.",
                  config_file_path_.c_str());
      // Continue anyway - we'll try to work with manual joint configuration
    } else {
      RCLCPP_INFO(this->get_logger(), "Loaded config file: %s", config_file_path_.c_str());
    }
  } else {
    RCLCPP_INFO(this->get_logger(), "No config file specified, using manual joint configuration");
  }

  return true;
}

void TeleopHardwareNode::setup_leader_mode()
{
  RCLCPP_INFO(this->get_logger(), "Setting up Leader mode");

  // Read current position for initialization
  if (!hardware_->sync_read(GROUP_NAME)) {
    RCLCPP_WARN(this->get_logger(), "Failed to read initial positions in leader mode");
  }

  // Set torque OFF (low PID gains for manual teaching)
  set_torque(false);

  // Create publishers
  joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10);
  leader_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
    leader_state_topic_, 10);

  // Create service for torque control
  torque_service_ = this->create_service<std_srvs::srv::SetBool>(
    "~/set_torque",
    std::bind(&TeleopHardwareNode::set_torque_callback, this,
              std::placeholders::_1, std::placeholders::_2));

  // Create timer for reading and publishing joint states
  auto timer_period = std::chrono::duration<double>(1.0 / publish_rate_);
  read_timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
    std::bind(&TeleopHardwareNode::read_and_publish_leader, this));

  RCLCPP_INFO(this->get_logger(), "Leader mode setup complete. Torque is OFF (manual teaching enabled)");
}

void TeleopHardwareNode::setup_follower_mode()
{
  RCLCPP_INFO(this->get_logger(), "Setting up Follower mode");

  // Read current positions and set as initial command (for safe startup)
  if (!hardware_->sync_read(GROUP_NAME)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to read initial positions");
    throw std::runtime_error("Failed to read initial positions in follower mode");
  }

  std::vector<double> current_positions;
  hardware_->get_positions(GROUP_NAME, current_positions);

  if (current_positions.size() == joint_names_.size()) {
    // Clamp to joint limits
    for (size_t i = 0; i < current_positions.size(); ++i) {
      current_positions[i] = std::clamp(
        current_positions[i], joint_limits_min_[i], joint_limits_max_[i]);
    }
    hardware_->set_positions(GROUP_NAME, current_positions);
    hardware_->sync_write(GROUP_NAME);
    RCLCPP_INFO(this->get_logger(), "Set initial position commands from current state");
  } else {
    RCLCPP_WARN(this->get_logger(), "Initial position size mismatch, skipping initial command");
  }

  // Set torque ON (position control mode)
  set_torque(true);

  // Create subscriber for leader state
  leader_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    leader_state_topic_, 10,
    std::bind(&TeleopHardwareNode::leader_state_callback, this, std::placeholders::_1));

  // Create publisher for follower feedback
  follower_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
    joint_states_topic_, 10);

  // Create timer for reading and publishing feedback
  auto timer_period = std::chrono::duration<double>(1.0 / publish_rate_);
  feedback_timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
    std::bind(&TeleopHardwareNode::read_and_publish_follower, this));

  // Initialize last message time
  last_leader_msg_time_ = this->now();

  RCLCPP_INFO(this->get_logger(), "Follower mode setup complete. Torque is ON (waiting for leader commands)");
}

void TeleopHardwareNode::read_and_publish_leader()
{
  // Read joint positions from hardware
  if (!hardware_->sync_read(GROUP_NAME)) {
    RCLCPP_WARN(this->get_logger(), "Failed to sync read from hardware");
    return;
  }

  std::vector<double> positions;
  hardware_->get_positions(GROUP_NAME, positions);

  if (positions.size() != joint_names_.size()) {
    RCLCPP_WARN(this->get_logger(), "Position data size mismatch: expected %zu, got %zu",
                joint_names_.size(), positions.size());
    return;
  }

  // Read current values (torque)
  std::vector<double> currents;
  hardware_->get_currents(GROUP_NAME, currents);

  // Create JointState message
  auto joint_state_msg = sensor_msgs::msg::JointState();
  joint_state_msg.header.stamp = this->now();
  joint_state_msg.name = joint_names_;
  joint_state_msg.position = positions;

  // Add effort (torque) data if available
  if (currents.size() == joint_names_.size()) {
    joint_state_msg.effort = currents;  // Using current as effort approximation
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Current data size mismatch: expected %zu, got %zu",
                         joint_names_.size(), currents.size());
  }

  // Publish to both topics
  joint_state_pub_->publish(joint_state_msg);
  leader_state_pub_->publish(joint_state_msg);
}

void TeleopHardwareNode::read_and_publish_follower()
{
  // Check for leader message timeout
  auto time_since_last_msg = (this->now() - last_leader_msg_time_).seconds();
  if (time_since_last_msg > LEADER_TIMEOUT_SEC) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "No leader messages received for %.1f seconds", time_since_last_msg);
  }

  // Read current joint positions from hardware
  if (!hardware_->sync_read(GROUP_NAME)) {
    RCLCPP_WARN(this->get_logger(), "Failed to sync read from hardware");
    return;
  }

  std::vector<double> positions;
  hardware_->get_positions(GROUP_NAME, positions);

  if (positions.size() != joint_names_.size()) {
    RCLCPP_WARN(this->get_logger(), "Position data size mismatch: expected %zu, got %zu",
                joint_names_.size(), positions.size());
    return;
  }

  // Read current values (torque)
  std::vector<double> currents;
  hardware_->get_currents(GROUP_NAME, currents);

  // Create and publish JointState message (for data_logger)
  auto joint_state_msg = sensor_msgs::msg::JointState();
  joint_state_msg.header.stamp = this->now();
  joint_state_msg.name = joint_names_;
  joint_state_msg.position = positions;

  // Add effort (torque) data if available
  if (currents.size() == joint_names_.size()) {
    joint_state_msg.effort = currents;  // Using current as effort approximation
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Current data size mismatch: expected %zu, got %zu",
                         joint_names_.size(), currents.size());
  }

  follower_state_pub_->publish(joint_state_msg);
}

void TeleopHardwareNode::leader_state_callback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  // Update last message time
  last_leader_msg_time_ = this->now();

  // Store the last leader state
  last_leader_state_ = msg;

  // Validate message content
  if (msg->name.size() != joint_names_.size()) {
    RCLCPP_WARN(this->get_logger(), "Received joint state with wrong number of joints: expected %zu, got %zu",
                joint_names_.size(), msg->name.size());
    return;
  }

  if (msg->position.size() != joint_names_.size()) {
    RCLCPP_WARN(this->get_logger(), "Received joint state with wrong number of positions: expected %zu, got %zu",
                joint_names_.size(), msg->position.size());
    return;
  }

  // Check joint limits
  if (!check_joint_limits(msg->position)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Received joint positions outside limits, skipping");
    return;
  }

  // Set positions to hardware
  hardware_->set_positions(GROUP_NAME, msg->position);

  // Write to Dynamixel servos
  if (!hardware_->sync_write(GROUP_NAME)) {
    RCLCPP_WARN(this->get_logger(), "Failed to sync write to hardware");
  }
}

void TeleopHardwareNode::set_torque_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  set_torque(request->data);
  response->success = true;
  response->message = request->data ? "Torque ON" : "Torque OFF";
  RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
}

void TeleopHardwareNode::set_torque(bool enable)
{
  if (enable) {
    // Set high PID gains for position control
    if (!hardware_->write_position_pid_gain_to_group(
        GROUP_NAME, torque_on_gains_.p, torque_on_gains_.i, torque_on_gains_.d))
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to set torque ON PID gains");
      return;
    }

    // Enable torque
    if (!hardware_->torque_on(GROUP_NAME)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to enable torque");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Torque enabled (P=%d, I=%d, D=%d)",
                torque_on_gains_.p, torque_on_gains_.i, torque_on_gains_.d);
  } else {
    // Set low PID gains for manual teaching
    if (!hardware_->write_position_pid_gain_to_group(
        GROUP_NAME, torque_off_gains_.p, torque_off_gains_.i, torque_off_gains_.d))
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to set torque OFF PID gains");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Torque disabled (manual teaching mode, P=%d, I=%d, D=%d)",
                torque_off_gains_.p, torque_off_gains_.i, torque_off_gains_.d);
  }
}

bool TeleopHardwareNode::check_joint_limits(const std::vector<double> & positions)
{
  if (positions.size() != joint_limits_min_.size() ||
      positions.size() != joint_limits_max_.size())
  {
    RCLCPP_ERROR(this->get_logger(), "Joint limits configuration mismatch");
    return false;
  }

  for (size_t i = 0; i < positions.size(); ++i) {
    if (positions[i] < joint_limits_min_[i] || positions[i] > joint_limits_max_[i]) {
      RCLCPP_WARN(this->get_logger(), "Joint %zu position %.3f outside limits [%.3f, %.3f]",
                  i, positions[i], joint_limits_min_[i], joint_limits_max_[i]);
      return false;
    }
  }

  return true;
}

}  // namespace crane_x7_teleop
