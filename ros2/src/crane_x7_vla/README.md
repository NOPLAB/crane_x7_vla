# crane_x7_vla

VLA (Vision-Language-Action) based control package for CRANE-X7 robotic arm using fine-tuned OpenVLA models.

## Overview

This package enables autonomous control of the CRANE-X7 robotic arm using OpenVLA models fine-tuned on CRANE-X7 demonstration data. The package provides:

- **VLA Inference Node**: Runs OpenVLA model inference on camera images and language instructions to predict robot actions
- **Robot Controller**: Executes VLA-predicted joint actions on the real robot or in simulation
- **Integrated Launch Files**: Easy deployment with existing CRANE-X7 control infrastructure

## Architecture

```
┌─────────────────┐
│  Camera Image   │
└────────┬────────┘
         │
         v
┌─────────────────────────────┐
│  VLA Inference Node         │
│  - Load fine-tuned model    │
│  - Process image + task     │
│  - Predict joint actions    │
└────────┬────────────────────┘
         │ /vla/predicted_action
         v
┌─────────────────────────────┐
│  Robot Controller           │
│  - Receive action           │
│  - Execute trajectory       │
└─────────────────────────────┘
         │
         v
┌─────────────────┐
│  CRANE-X7 Robot │
└─────────────────┘
```

## Prerequisites

1. **Fine-tuned VLA Model**: Train a model using `vla/finetune.py` on CRANE-X7 data
2. **ROS 2 Humble**: With CRANE-X7 packages installed
3. **Python Dependencies**: PyTorch, Transformers, OpenCV, etc.

## Installation

This package is already in the workspace. Build it with:

```bash
cd /workspace/ros2
colcon build --packages-select crane_x7_vla --symlink-install
source install/setup.bash
```

## Usage

### Configuration

Edit `config/vla_config.yaml` to set:
- `model_path`: Path to your fine-tuned VLA model
- `task_instruction`: Default task description
- `device`: 'cuda' or 'cpu'
- Other inference and control parameters

### Running with Real Robot

Launch CRANE-X7 with VLA control:

```bash
ros2 launch crane_x7_vla real_with_vla.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_d435:=true \
  model_path:=/workspace/vla/models/crane_x7_finetuned \
  task_instruction:='pick up the object'
```

### Running in Simulation

Launch Gazebo simulation with VLA control:

```bash
ros2 launch crane_x7_vla sim_with_vla.launch.py \
  model_path:=/workspace/vla/models/crane_x7_finetuned \
  task_instruction:='pick up the object'
```

### Running Components Separately

For more control, launch components individually:

1. **Start robot control** (real or sim):
   ```bash
   # Real robot
   ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true

   # Or simulation
   ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py
   ```

2. **Launch VLA control nodes**:
   ```bash
   ros2 launch crane_x7_vla vla_control.launch.py \
     model_path:=/workspace/vla/models/crane_x7_finetuned \
     task_instruction:='pick up the object'
   ```

### Updating Task Instruction at Runtime

Change the task instruction while the system is running:

```bash
ros2 topic pub /vla/update_instruction std_msgs/msg/String "data: 'place the object on the table'" --once
```

### Monitoring

View predicted actions:

```bash
ros2 topic echo /vla/predicted_action
```

View robot joint states:

```bash
ros2 topic echo /joint_states
```

## Launch Files

- **`vla_control.launch.py`**: Launch VLA inference and robot controller nodes only
- **`real_with_vla.launch.py`**: Complete launch for real CRANE-X7 with VLA control
- **`sim_with_vla.launch.py`**: Complete launch for Gazebo simulation with VLA control

## Nodes

### vla_inference_node

Runs OpenVLA model inference to predict robot actions.

**Subscribed Topics:**
- `/camera/color/image_raw` (sensor_msgs/Image): RGB camera feed
- `/joint_states` (sensor_msgs/JointState): Current robot joint states
- `/vla/update_instruction` (std_msgs/String): Update task instruction

**Published Topics:**
- `/vla/predicted_action` (std_msgs/Float32MultiArray): Predicted joint positions (8D: 7 arm joints + 1 gripper)

**Parameters:**
- `model_path`: Path to fine-tuned VLA model checkpoint
- `task_instruction`: Task description (e.g., "pick up the object")
- `device`: Inference device ('cuda' or 'cpu')
- `inference_rate`: Prediction frequency in Hz (default: 10.0)
- `use_flash_attention`: Enable Flash Attention 2 (default: false)
- `unnorm_key`: Dataset key for action unnormalization (default: 'crane_x7')

### robot_controller

Executes VLA-predicted actions on the robot.

**Subscribed Topics:**
- `/vla/predicted_action` (std_msgs/Float32MultiArray): VLA-predicted joint positions
- `/joint_states` (sensor_msgs/JointState): Current robot joint states

**Action Clients:**
- `/crane_x7_arm_controller/follow_joint_trajectory` (control_msgs/FollowJointTrajectory)

**Parameters:**
- `auto_execute`: Automatically execute received actions (default: true)
- `execution_time`: Time to execute each action in seconds (default: 1.0)
- `position_tolerance`: Goal position tolerance in radians (default: 0.01)

## Workflow

1. **Collect demonstration data** using `crane_x7_log` package
2. **Fine-tune OpenVLA model** using `vla/finetune.py`
3. **Deploy fine-tuned model** with this package
4. **Monitor and adjust** task instructions as needed

## Integration with Data Collection

This package works seamlessly with the data logging workflow:

```bash
# 1. Collect demonstrations
ros2 launch crane_x7_log real_with_logger.launch.py

# 2. Fine-tune VLA model
cd /workspace/vla
python finetune.py --data_root /workspace/data/tfrecord_logs

# 3. Deploy fine-tuned model
ros2 launch crane_x7_vla real_with_vla.launch.py \
  model_path:=/workspace/vla/models/crane_x7_finetuned
```

## Troubleshooting

### Model Loading Issues

- Ensure `model_path` points to a valid checkpoint directory
- Check that the model contains `dataset_statistics.json` for proper action unnormalization
- Verify PyTorch and Transformers are installed correctly

### Inference Performance

- Use `device:=cuda` for GPU acceleration
- Reduce `inference_rate` if predictions are slow
- Enable `use_flash_attention:=true` on compatible GPUs (Ampere or newer)

### Robot Control Issues

- Verify robot controller is running: `ros2 topic list | grep follow_joint_trajectory`
- Check joint state messages: `ros2 topic echo /joint_states`
- Ensure `auto_execute:=true` or manually trigger execution

### Camera Not Detected

- Confirm camera topic: `ros2 topic list | grep camera`
- Update `image_topic` parameter in config if needed
- For RealSense D435, ensure `use_d435:=true` in launch

## License

MIT License

## References

- OpenVLA: https://openvla.github.io/
- CRANE-X7 ROS 2: https://github.com/rt-net/crane_x7_ros
- Open X-Embodiment: https://robotics-transformer-x.github.io/
