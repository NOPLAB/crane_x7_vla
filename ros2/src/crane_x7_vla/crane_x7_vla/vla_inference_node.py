#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""ROS 2 node for VLA model inference."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import torch
from PIL import Image as PILImage
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add VLA directory to path
VLA_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "vla"
sys.path.insert(0, str(VLA_PATH))


class VLAInferenceNode(Node):
    """ROS 2 node for OpenVLA model inference."""

    def __init__(self):
        super().__init__('vla_inference_node')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_base_name', 'openvla')
        self.declare_parameter('task_instruction', 'pick up the object')
        self.declare_parameter('use_flash_attention', False)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('action_topic', '/vla/predicted_action')
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('center_crop', False)
        self.declare_parameter('unnorm_key', 'crane_x7')

        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.model_base_name = self.get_parameter('model_base_name').value
        self.task_instruction = self.get_parameter('task_instruction').value
        self.use_flash_attention = self.get_parameter('use_flash_attention').value
        self.device_name = self.get_parameter('device').value
        self.image_topic = self.get_parameter('image_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.action_topic = self.get_parameter('action_topic').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.center_crop = self.get_parameter('center_crop').value
        self.unnorm_key = self.get_parameter('unnorm_key').value

        # Initialize
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_joint_state: Optional[JointState] = None
        self.model = None
        self.processor = None

        # Setup device
        if torch.cuda.is_available() and self.device_name == 'cuda':
            self.device = torch.device('cuda')
            self.get_logger().info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('Using CPU device')

        # Load model
        self._load_model()

        # Setup subscriptions
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._joint_state_callback,
            10
        )

        # Setup publishers
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.action_topic,
            10
        )

        # Setup service for updating task instruction
        from std_srvs.srv import SetBool
        from example_interfaces.srv import SetString

        # Create custom service callback for updating instruction
        self.update_instruction_sub = self.create_subscription(
            String,
            '/vla/update_instruction',
            self._update_instruction_callback,
            10
        )

        # Setup inference timer
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self._inference_callback)

        self.get_logger().info('VLA Inference Node initialized')
        self.get_logger().info(f'Model: {self.model_path}')
        self.get_logger().info(f'Task: {self.task_instruction}')
        self.get_logger().info(f'Inference rate: {self.inference_rate} Hz')

    def _load_model(self) -> None:
        """Load VLA model and processor."""
        if not self.model_path:
            self.get_logger().error('Model path not specified!')
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            self.get_logger().error(f'Model path does not exist: {model_path}')
            return

        self.get_logger().info(f'Loading VLA model from {model_path}...')

        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            self.get_logger().info('Processor loaded successfully')

            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.get_logger().info('Using Flash Attention 2')

            self.model = AutoModelForVision2Seq.from_pretrained(
                str(model_path),
                **model_kwargs
            )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load normalization statistics if available
            norm_stats_path = model_path / "dataset_statistics.json"
            if norm_stats_path.exists():
                import json
                with open(norm_stats_path, 'r') as f:
                    self.model.norm_stats = json.load(f)
                self.get_logger().info('Loaded dataset normalization statistics')
            else:
                self.get_logger().warn('No dataset_statistics.json found - using default normalization')

            self.get_logger().info('VLA model loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _image_callback(self, msg: Image) -> None:
        """Callback for RGB image."""
        try:
            # Convert ROS Image to numpy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint states."""
        self.latest_joint_state = msg

    def _update_instruction_callback(self, msg: String) -> None:
        """Callback for updating task instruction."""
        self.task_instruction = msg.data
        self.get_logger().info(f'Updated task instruction: {self.task_instruction}')

    def _inference_callback(self) -> None:
        """Perform VLA inference and publish action."""
        if self.model is None or self.processor is None:
            return

        if self.latest_image is None:
            self.get_logger().warn('No image received yet', throttle_duration_sec=5.0)
            return

        try:
            # Prepare image
            image = PILImage.fromarray(self.latest_image)
            image = image.convert("RGB")

            # Build prompt based on model version
            if "openvla-v01" in self.model_base_name:
                # OpenVLA v0.1 format
                system_prompt = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                )
                prompt = f"{system_prompt} USER: What action should the robot take to {self.task_instruction.lower()}? ASSISTANT:"
            else:
                # OpenVLA format
                prompt = f"In: What action should the robot take to {self.task_instruction.lower()}?\nOut:"

            # Process inputs
            inputs = self.processor(prompt, image)
            inputs = {k: v.to(self.device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)

            # Convert to numpy and publish
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Ensure action is 1D array
            action = np.asarray(action).flatten()

            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = action.tolist()
            self.action_pub.publish(action_msg)

            self.get_logger().debug(f'Published action: {action}')

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = VLAInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
