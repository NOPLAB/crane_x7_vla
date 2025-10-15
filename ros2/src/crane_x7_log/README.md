# crane_x7_log

Data logging package for CRANE-X7 robotic arm in OXE (Open X-Embodiment) format for VLA fine-tuning.

## Overview

This package provides ROS 2 nodes to record CRANE-X7 manipulation data in a format compatible with the Open X-Embodiment dataset specification, suitable for fine-tuning Vision-Language-Action (VLA) models like OpenVLA.

## Features

- **Multi-modal data logging**: Joint states, RGB images, depth images (optional)
- **OXE format compatibility**: TFRecord output format for VLA training
- **Configurable episodes**: Adjustable episode length and data collection rate
- **Automatic saving**: Episodes saved automatically when reaching target length

## Nodes

### oxe_logger

Main data collection node that subscribes to robot and camera topics.

**Subscribed Topics:**
- `/joint_states` (sensor_msgs/JointState): Robot joint positions
- `/camera/color/image_raw` (sensor_msgs/Image): RGB camera feed
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): Camera calibration
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image): Depth image (optional)

**Parameters:**
- `output_dir` (string, default: `/workspace/data/oxe_logs`): Output directory
- `episode_length` (int, default: 100): Steps per episode
- `use_camera` (bool, default: true): Enable camera logging
- `use_depth` (bool, default: false): Enable depth logging

## Usage

### Building

```bash
cd /workspace/ros2
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### Running with real robot

```bash
# Launch robot control with data logger
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

Or separately:

```bash
# Start robot control
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true

# In another terminal, start logger
ros2 launch crane_x7_log data_logger.launch.py
```

### Running with teleoperation (kinesthetic teaching)

#### Leader Mode (Manual Teaching)

Launch the Leader robot (torque OFF for manual movement) with data logger:

```bash
ros2 launch crane_x7_log teleop_leader_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  output_dir:=/workspace/data/teleop_demonstrations
```

The Leader robot can be manually moved to demonstrate tasks. Joint states are published to `/joint_states` and `/teleop/leader/state` and automatically recorded by the data logger.

#### Follower Mode (Imitation Recording)

If you have a second CRANE-X7 that follows the Leader's movements:

```bash
# This launch file is in crane_x7_teleop package
ros2 launch crane_x7_teleop teleop_with_logger.launch.py \
  port_name:=/dev/ttyUSB1 \
  output_dir:=/workspace/data/follower_demonstrations
```

### Running in simulation

```bash
# Launch Gazebo simulation with data logger
ros2 launch crane_x7_log demo_with_logger.launch.py
```

Or separately:

```bash
# Start Gazebo simulation
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py

# In another terminal, start logger (without camera)
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/sim_logs
```

### Custom parameters

```bash
ros2 launch crane_x7_log real_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  output_dir:=/path/to/data \
  use_d435:=true
```

All launch files accept these common parameters:
- `output_dir`: Directory to save logged data (default: `/workspace/data/tfrecord_logs`)
- `config_file`: Path to logger config file (optional)

For teleop leader mode:
- `port_name`: USB port for Leader robot (default: `/dev/ttyUSB0`)
- `use_d435`: Enable RealSense D435 camera (default: `false`)

## Data Format

### NPZ Format (Intermediate)

Episodes are initially saved as compressed NumPy archives (`.npz`):

```
episode_0000_20250102_120000/
  └── episode_data.npz
      ├── states: (N, 8) - 7 arm joints + 1 gripper
      ├── actions: (N, 8) - next state (shifted by 1)
      ├── timestamps: (N,) - UNIX timestamps
      ├── images: (N, H, W, 3) - RGB images (if enabled)
      └── depths: (N, H, W) - depth images (if enabled)
```

### TFRecord Format (OXE Compatible)

Convert NPZ to TFRecord:

```bash
python3 -m crane_x7_log.oxe_writer episode_data.npz episode_data.tfrecord
```

TFRecord features:
- `observation/state`: float32 joint positions
- `observation/image`: JPEG-encoded RGB image
- `observation/depth`: float32 depth array
- `observation/timestamp`: float32 timestamp
- `action`: float32 target joint positions

## Dependencies

- ROS 2 Humble
- Python packages:
  - rclpy
  - sensor_msgs
  - cv_bridge
  - numpy
  - opencv-python
  - tensorflow

## License

MIT License
