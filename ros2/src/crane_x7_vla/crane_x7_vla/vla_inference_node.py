#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ROS 2 node for VLA model inference."""

import json
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
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

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

        # Subscribe to instruction updates
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
            self.get_logger().error(
                'Model path not specified! '
                'Set VLA_MODEL_PATH in ros2/.env file. '
                'Example: VLA_MODEL_PATH=/workspace/vla/outputs/<your_model_dir>'
            )
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            self.get_logger().error(f'Model path does not exist: {model_path}')
            # List available models in outputs directory
            outputs_dir = Path('/workspace/vla/outputs')
            if outputs_dir.exists():
                available = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
                if available:
                    self.get_logger().info(f'Available models in {outputs_dir}: {available}')
            return

        self.get_logger().info(f'Loading VLA model from {model_path}...')

        try:
            # Register custom OpenVLA classes for HuggingFace Auto classes
            # These are needed to properly load models fine-tuned from openvla-7b
            try:
                from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as HFOpenVLAConfig
                from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
                from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
                from transformers import AutoImageProcessor

                AutoConfig.register("openvla", HFOpenVLAConfig)
                AutoImageProcessor.register(HFOpenVLAConfig, PrismaticImageProcessor)
                AutoProcessor.register(HFOpenVLAConfig, PrismaticProcessor)
                AutoModelForVision2Seq.register(HFOpenVLAConfig, OpenVLAForActionPrediction)
                self.get_logger().info('Registered OpenVLA custom classes')
            except ImportError:
                self.get_logger().warn(
                    'Could not import prismatic classes - relying on trust_remote_code'
                )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            self.get_logger().info('Processor loaded successfully')

            # Check if this is a LoRA checkpoint (has lora_adapters subdirectory)
            lora_adapter_path = model_path / "lora_adapters"
            is_lora_checkpoint = lora_adapter_path.exists()

            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager",  # Use eager attention for compatibility
            }

            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.get_logger().info('Using Flash Attention 2')

            if is_lora_checkpoint:
                # Load base model first, then apply LoRA adapter
                self.get_logger().info('Detected LoRA checkpoint, loading base model + adapter...')

                # Read adapter config to get base model path
                import json
                adapter_config_path = lora_adapter_path / "adapter_config.json"
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path", "openvla/openvla-7b")
                self.get_logger().info(f'Base model: {base_model_path}')

                # Load base model
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_model_path,
                    **model_kwargs
                )
                self.get_logger().info('Base model loaded')

                # Load and apply LoRA adapter using PEFT
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(lora_adapter_path),
                    is_trainable=False,
                )
                self.get_logger().info('LoRA adapter applied')

                # Merge LoRA weights for faster inference
                self.model = self.model.merge_and_unload()
                self.get_logger().info('LoRA weights merged')
            else:
                # Load merged model directly
                self.model = AutoModelForVision2Seq.from_pretrained(
                    str(model_path),
                    **model_kwargs
                )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load normalization statistics
            # Always try to load from checkpoint's dataset_statistics.json first
            # This contains the fine-tuned dataset statistics (e.g., crane_x7)
            norm_stats_path = model_path / "dataset_statistics.json"
            if norm_stats_path.exists():
                with open(norm_stats_path, 'r') as f:
                    checkpoint_stats = json.load(f)
                # Merge with existing norm_stats or create new
                if not hasattr(self.model, 'norm_stats') or self.model.norm_stats is None:
                    self.model.norm_stats = {}
                self.model.norm_stats.update(checkpoint_stats)
                self.get_logger().info(
                    f'Loaded checkpoint statistics: {list(checkpoint_stats.keys())}'
                )
            elif hasattr(self.model, 'norm_stats') and self.model.norm_stats:
                self.get_logger().info(
                    f'Using model norm_stats: {list(self.model.norm_stats.keys())}'
                )
            else:
                self.get_logger().warn(
                    'No dataset_statistics.json found - using default normalization'
                )

            # Verify unnorm_key is available
            if hasattr(self.model, 'norm_stats') and self.model.norm_stats:
                if self.unnorm_key not in self.model.norm_stats:
                    available_keys = list(self.model.norm_stats.keys())
                    self.get_logger().warn(
                        f'unnorm_key "{self.unnorm_key}" not in norm_stats. '
                        f'Available keys: {available_keys}'
                    )
                    # Select a fallback key - prefer bridge_orig (common) or first available
                    if 'bridge_orig' in available_keys:
                        self.unnorm_key = 'bridge_orig'
                    elif available_keys:
                        self.unnorm_key = available_keys[0]
                    self.get_logger().info(f'Using fallback unnorm_key: {self.unnorm_key}')

                action_dim = len(self.model.norm_stats[self.unnorm_key]["action"]["q01"])
                self.get_logger().info(
                    f'Using unnorm_key: {self.unnorm_key} (action_dim={action_dim})'
                )

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
            # Debug: log image info periodically
            self.get_logger().debug(
                f'Image received: shape={cv_image.shape}, '
                f'dtype={cv_image.dtype}, '
                f'mean={cv_image.mean():.2f}',
                throttle_duration_sec=2.0
            )
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

            # Debug: log image hash to detect if image is changing
            image_hash = hash(self.latest_image.tobytes())
            self.get_logger().info(
                f'Inference input: image_size={image.size}, '
                f'image_hash={image_hash % 10000:04d}, '
                f'mean_pixel={self.latest_image.mean():.2f}'
            )

            # Build prompt based on model version
            if "openvla-v01" in self.model_base_name or "v01" in self.model_base_name:
                # OpenVLA v0.1 format (VicunaV15ChatPromptBuilder)
                system_prompt = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                )
                prompt = (
                    f"{system_prompt} USER: What action should the robot take to "
                    f"{self.task_instruction.lower()}? ASSISTANT:"
                )
            else:
                # OpenVLA format (PurePromptBuilder)
                prompt = f"In: What action should the robot take to {self.task_instruction.lower()}?\nOut:"

            # Process inputs using the Prismatic processor
            # The processor expects (text, images) and returns input_ids, attention_mask, pixel_values
            inputs = self.processor(prompt, image)

            # Move tensors to device with appropriate dtype
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]

            # Debug: log pixel_values statistics
            if isinstance(pixel_values, torch.Tensor):
                pv_mean = pixel_values.float().mean().item()
                pv_std = pixel_values.float().std().item()
                self.get_logger().info(
                    f'pixel_values: shape={list(pixel_values.shape)}, '
                    f'mean={pv_mean:.4f}, std={pv_std:.4f}'
                )

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(self.device)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

            # Run inference using predict_action method
            # predict_action expects input_ids and passes other kwargs to generate()
            with torch.no_grad():
                action_dim = self.model.get_action_dim(self.unnorm_key)

                # Add special empty token if needed (same as predict_action)
                # IMPORTANT: Also update attention_mask to match input_ids length
                if not torch.all(input_ids[:, -1] == 29871):
                    input_ids = torch.cat(
                        (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                    )
                    # Update attention_mask to include the new token
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
                        )

                # Run generate once and decode tokens manually
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=action_dim,
                    do_sample=False,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )

                # Extract predicted action tokens
                predicted_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
                self.get_logger().info(f'Generated token IDs: {predicted_token_ids}')

                # Decode tokens to actions (same logic as predict_action)
                # vocab_size - token_id gives the bin index
                vocab_size = self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of
                discretized_actions = vocab_size - predicted_token_ids
                discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1)
                normalized_actions = self.model.bin_centers[discretized_actions]

                # Unnormalize actions
                action_norm_stats = self.model.get_action_stats(self.unnorm_key)
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                action = np.where(
                    mask,
                    0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                    normalized_actions,
                )

            # Convert to numpy and publish
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Ensure action is 1D array
            action = np.asarray(action).flatten()

            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = action.tolist()
            self.action_pub.publish(action_msg)

            self.get_logger().info(f'Published action: {action}')

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
