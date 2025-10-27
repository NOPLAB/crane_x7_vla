# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains ROS 2 Humble code for controlling the CRANE-X7 robotic arm, along with an OpenVLA (Vision-Language-Action) integration for vision-based manipulation tasks. The project supports both real hardware and Gazebo simulation.

## Architecture

### Main Components

1. **ROS 2 Workspace** (`ros2/`)
   - `crane_x7_ros/`: RT Corporation's official ROS 2 packages for CRANE-X7
     - `crane_x7_control`: Hardware control interface and USB communication
     - `crane_x7_examples`: Sample programs demonstrating robot capabilities
     - `crane_x7_gazebo`: Gazebo simulation environment
     - `crane_x7_moveit_config`: MoveIt2 configuration for motion planning
   - `crane_x7_description/`: URDF/xacro robot model definitions
   - `crane_x7_log/`: Data logging package for VLA training
     - Collects robot manipulation episodes in OXE-compatible format
     - Supports both NPZ and TFRecord output formats
     - Captures joint states, RGB images, and optional depth data

2. **OpenVLA** (`vla/openvla/`)
   - Vision-Language-Action model for robotic manipulation
   - Based on Prismatic VLMs
   - Supports fine-tuning and deployment for embodied AI tasks
   - Trained on Open X-Embodiment dataset mixtures

3. **Docker Environment**
   - Multi-stage Dockerfile with `base` (production) and `dev` (development) targets
   - ROS Humble base image (Ubuntu 22.04)
   - X11 forwarding for GUI applications (RViz, Gazebo)

### Development Modes

The repository supports multiple execution modes controlled via docker-compose profiles:
- **real**: Connect to physical CRANE-X7 via USB (`/dev/ttyUSB0`)
- **real-viewer**: Real robot with camera viewer (displays RealSense D435 stream)
- **sim**: Run Gazebo simulation without hardware

## Common Commands

### Docker Development

Build the Docker image:
```bash
./ros2/scripts/build.sh
```

Run interactive development container (with real robot hardware):
```bash
./ros2/scripts/run.sh real
```

Or run with simulation:
```bash
./ros2/scripts/run.sh sim
```

Inside the container, build ROS packages:
```bash
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash
```

Build specific package:
```bash
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### Docker Compose (Quick Start)

Create `.env` from `.env.template` in the `ros2/` directory and configure:
- `USB_DEVICE`: USB device path (default: `/dev/ttyUSB0`)
- `USB_DEVICE_FOLLOWER`: USB device path for follower robot (default: `/dev/ttyUSB1`)
- `DISPLAY`: X11 display (default: `:0`)

```bash
# Create .env file from template
cd ros2
cp .env.template .env
# Edit .env as needed
```

Run with real robot:
```bash
docker compose -f ros2/docker-compose.yml --profile real up
```

Run in simulation:
```bash
docker compose -f ros2/docker-compose.yml --profile sim up
```

Run with real robot and camera viewer (displays RealSense D435 stream):
```bash
docker compose -f ros2/docker-compose.yml --profile real-viewer up
```

Run with teleoperation (kinesthetic teaching):
```bash
# Leader mode only (manual teaching without recording)
docker compose -f ros2/docker-compose.yml --profile teleop-leader up

# Leader mode with data logger (manual teaching with recording)
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up

# Leader mode with data logger and camera viewer (manual teaching with recording and video display)
docker compose -f ros2/docker-compose.yml --profile teleop-leader-viewer up

# Follower mode only (requires 2 robots)
docker compose -f ros2/docker-compose.yml --profile teleop-follower up

# Follower mode with camera viewer (follower robot with video display, requires 2 robots)
docker compose -f ros2/docker-compose.yml --profile teleop-follower-viewer up

# Follower mode with data logger (imitation recording, requires 2 robots)
docker compose -f ros2/docker-compose.yml --profile teleop-follower-logger up

# Both leader and follower simultaneously
docker compose -f ros2/docker-compose.yml --profile teleop up

# Both leader and follower with data logger
docker compose -f ros2/docker-compose.yml --profile teleop-logger up

# Follower with camera viewer (follower side camera display)
docker compose -f ros2/docker-compose.yml --profile teleop-viewer up
```

### ROS 2 Launch Commands

Launch demo with real robot (inside container):
```bash
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0
```

Launch Gazebo simulation:
```bash
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py
```

Run example programs (in separate terminal):
```bash
ros2 launch crane_x7_examples example.launch.py example:='gripper_control'
```

Display robot model in RViz:
```bash
ros2 launch crane_x7_description display.launch.py
```

For RealSense D435 camera mount, add `use_d435:=true` to launch commands.

Display RealSense camera stream (standalone):
```bash
ros2 launch crane_x7_log camera_viewer.launch.py
```

Display RealSense camera stream with custom topic:
```bash
ros2 launch crane_x7_log camera_viewer.launch.py image_topic:=/camera/depth/image_rect_raw
```

### Data Logging

Launch robot control with data logger (real robot):
```bash
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

Launch robot control with data logger (simulation):
```bash
ros2 launch crane_x7_log demo_with_logger.launch.py
```

Standalone data logger (requires robot already running):
```bash
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/tfrecord_logs
```

Convert NPZ episode to TFRecord format:
```bash
python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
```

### ROS 2 Build System

The workspace uses colcon build system:
- `colcon build --symlink-install`: Build all packages with symlinks (recommended for development)
- `colcon build --packages-select <package_name>`: Build specific package
- `source install/setup.bash`: Source the workspace after building

## Key Architecture Details

### Launch Flow

**Real Robot**:
- Docker Compose: Runs `crane_x7_log/real_with_logger.launch.py`
  - Includes `crane_x7_examples/demo.launch.py` (MoveIt2 + hardware control)
  - Adds data logger node for OXE data collection
- Manual: `ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0`
  - Starts MoveIt2 (`crane_x7_moveit_config`) and hardware controllers (`crane_x7_control`)

**Simulation**:
- Docker Compose: Runs `crane_x7_log/demo_with_logger.launch.py`
  - Includes Gazebo simulation with data logger
- Manual: `ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py`
  - Starts Gazebo with robot model and MoveIt2

### USB Device Access

The real robot requires USB access to Dynamixel servos. The docker-compose setup:
- Maps host device `$USB_DEVICE` to `/dev/ttyUSB0` inside container
- See `crane_x7_control/README.md` for USB permission setup

### X11 Display

Both WSL and native Linux are supported via different volume mounts:
- WSL: Mounts `/tmp/.X11-unix` and `/mnt/wslg`
- Linux: Mounts `/tmp/.X11-unix` (requires `xhost +`)

### Data Logging Architecture

The `crane_x7_log` package implements an OXE-compatible data collection pipeline:

**Data Flow**:
1. **Subscriptions**: `data_logger` node subscribes to:
   - `/joint_states`: 7 arm joints + 1 gripper state
   - `/camera/color/image_raw`: RGB camera feed (optional)
   - `/camera/aligned_depth_to_color/image_raw`: Depth image (optional)
2. **Buffering**: Steps are buffered in memory until episode length is reached
3. **Action Assignment**: `action[t] = state[t+1]` (next-state prediction format)
4. **Saving**: Episodes saved to disk in NPZ or TFRecord format

**Key Components**:
- `data_logger.py`: Main ROS 2 node that collects multi-modal data
- `episode_saver.py`: Handles episode persistence (NPZ/TFRecord)
- `tfrecord_writer.py`: Converts episode data to TFRecord format for VLA training
- `image_processor.py`: Image encoding and processing utilities
- `config_manager.py`: Configuration loading and validation

**Output Format** (NPZ):
```
episode_0000_YYYYMMDD_HHMMSS/
  └── episode_data.npz
      ├── states: (N, 8)      # Joint positions
      ├── actions: (N, 8)     # Next state (shifted by 1)
      ├── timestamps: (N,)    # UNIX timestamps
      ├── images: (N, H, W, 3) # RGB images (optional)
      └── depths: (N, H, W)   # Depth images (optional)
```

**Output Format** (TFRecord):
- `observation/state`: Joint positions (float32)
- `observation/image`: JPEG-encoded RGB image (bytes)
- `observation/depth`: Depth array (bytes, float32)
- `observation/timestamp`: UNIX timestamp (float32)
- `action`: Target joint positions (float32)

## VLA Training Workflow

1. **Data Collection**: Use `crane_x7_log` to collect demonstration episodes
   - Run robot with logger: `docker compose -f ros2/docker-compose.yml --profile real up`
   - Episodes saved automatically to `/workspace/data/tfrecord_logs`
   - Configure episode length, save format (NPZ/TFRecord) via launch parameters

2. **Data Format Conversion** (if using NPZ):
   ```bash
   python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
   ```

3. **Fine-Tuning OpenVLA**:
   - Located in `vla/openvla/` directory
   - Use collected TFRecord data to fine-tune pretrained OpenVLA models
   - Supports LoRA and full fine-tuning via HuggingFace PEFT
   - See `vla/openvla/README.md` for detailed fine-tuning instructions

4. **Deployment**:
   - Deploy fine-tuned model via REST API (`vla-scripts/deploy.py`)
   - Integrate with ROS 2 control stack for closed-loop manipulation

## Licensing Notes

### This Repository (Original Code)

- **Project root and original code**: MIT License (Copyright 2025 nop)
- **crane_x7_log**: MIT License
- **crane_x7_vla**: MIT License
- **crane_x7_teleop**: MIT License
- **VLA fine-tuning scripts**: MIT License

### External/Third-Party Packages (Git Submodules)

- **crane_x7_ros** (RT Corporation): Apache License 2.0
- **crane_x7_description** (RT Corporation): RT Corporation non-commercial license
  - Research and internal use only
  - Commercial use requires prior permission from RT Corporation
- **OpenVLA**: MIT License (code)
  - Pretrained models may have additional restrictions (e.g., Llama-2 license)

**Important**: The RT Corporation packages (`crane_x7_ros`, `crane_x7_description`) have different licenses from this repository's original code. Please review their respective LICENSE files before use.

## References

- RT Corporation CRANE-X7 resources:
  - https://github.com/rt-net/crane_x7
  - https://github.com/rt-net/crane_x7_ros
  - https://github.com/rt-net/crane_x7_description
- OpenVLA project: https://openvla.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
