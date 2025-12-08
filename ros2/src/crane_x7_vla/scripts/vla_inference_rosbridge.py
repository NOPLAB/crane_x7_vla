#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""VLA inference client using rosbridge (WebSocket).

This script runs on a remote GPU server (Vast.ai/Runpod) and communicates
with the local robot via rosbridge_server over WebSocket.

No ROS 2 installation required - uses roslibpy (pure Python).

Usage:
    python vla_inference_rosbridge.py \
        --rosbridge-host crane-x7-local \
        --rosbridge-port 9090 \
        --model-path sikip/openvla-7b-finetuned-crane-x7 \
        --task-instruction "pick up the object"

Environment variables:
    ROSBRIDGE_HOST: rosbridge server hostname (default: crane-x7-local)
    ROSBRIDGE_PORT: rosbridge server port (default: 9090)
    VLA_MODEL_PATH: Path to VLA model (required)
    VLA_TASK_INSTRUCTION: Task instruction (default: pick up the object)
    VLA_DEVICE: Inference device (default: cuda)
    VLA_INFERENCE_RATE: Inference rate in Hz (default: 10.0)
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add parent directory to path for vla_inference_core
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import roslibpy
except ImportError:
    print("ERROR: roslibpy not installed. Install with: pip install roslibpy")
    sys.exit(1)

from crane_x7_vla.vla_inference_core import VLAInferenceCore


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VLARosbridgeClient:
    """VLA inference client using rosbridge WebSocket connection.

    This client connects to a rosbridge_server running on the local robot,
    subscribes to camera images, runs VLA inference on GPU, and publishes
    predicted actions back to the robot.

    Attributes:
        rosbridge_host: Hostname of rosbridge server
        rosbridge_port: Port of rosbridge server
        vla_core: VLA inference core instance
        task_instruction: Current task instruction
        inference_rate: Inference rate in Hz
    """

    def __init__(
        self,
        rosbridge_host: str,
        rosbridge_port: int,
        model_path: str,
        task_instruction: str = "pick up the object",
        device: str = "cuda",
        unnorm_key: str = "crane_x7",
        inference_rate: float = 10.0,
    ):
        """Initialize VLA rosbridge client.

        Args:
            rosbridge_host: Hostname of rosbridge server
            rosbridge_port: Port of rosbridge server
            model_path: Path to VLA model
            task_instruction: Task instruction for the robot
            device: Inference device ('cuda' or 'cpu')
            unnorm_key: Key for action normalization statistics
            inference_rate: Inference rate in Hz
        """
        self.rosbridge_host = rosbridge_host
        self.rosbridge_port = rosbridge_port
        self.task_instruction = task_instruction
        self.inference_rate = inference_rate

        # State
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_time: float = 0
        self.running = False
        self._lock = threading.Lock()

        # rosbridge client
        self.client: Optional[roslibpy.Ros] = None

        # Topics
        self.image_subscriber: Optional[roslibpy.Topic] = None
        self.instruction_subscriber: Optional[roslibpy.Topic] = None
        self.action_publisher: Optional[roslibpy.Topic] = None

        # Initialize VLA inference core
        self.vla_core = VLAInferenceCore(
            model_path=model_path,
            device=device,
            unnorm_key=unnorm_key,
            logger=logger,
        )

    def connect(self) -> bool:
        """Connect to rosbridge server.

        Returns:
            True if connected successfully
        """
        logger.info(f"Connecting to rosbridge at {self.rosbridge_host}:{self.rosbridge_port}")

        try:
            self.client = roslibpy.Ros(
                host=self.rosbridge_host,
                port=self.rosbridge_port
            )
            self.client.run()

            # Wait for connection
            timeout = 30
            start_time = time.time()
            while not self.client.is_connected:
                if time.time() - start_time > timeout:
                    logger.error(f"Connection timeout after {timeout}s")
                    return False
                time.sleep(0.1)

            logger.info("Connected to rosbridge server")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to rosbridge: {e}")
            return False

    def setup_topics(self) -> None:
        """Setup ROS topics for subscription and publication."""
        # Subscribe to compressed image
        # Using compressed image for bandwidth efficiency over VPN
        self.image_subscriber = roslibpy.Topic(
            self.client,
            '/camera/color/image_raw/compressed',
            'sensor_msgs/CompressedImage'
        )
        self.image_subscriber.subscribe(self._image_callback)
        logger.info("Subscribed to /camera/color/image_raw/compressed")

        # Subscribe to instruction updates
        self.instruction_subscriber = roslibpy.Topic(
            self.client,
            '/vla/update_instruction',
            'std_msgs/String'
        )
        self.instruction_subscriber.subscribe(self._instruction_callback)
        logger.info("Subscribed to /vla/update_instruction")

        # Publisher for predicted actions
        self.action_publisher = roslibpy.Topic(
            self.client,
            '/vla/predicted_action',
            'std_msgs/Float32MultiArray'
        )
        self.action_publisher.advertise()
        logger.info("Advertising /vla/predicted_action")

    def _image_callback(self, message: dict) -> None:
        """Callback for compressed image messages.

        Args:
            message: rosbridge message dict containing compressed image
        """
        try:
            # Decode compressed image (JPEG)
            # rosbridge sends data as base64-encoded string
            image_data = base64.b64decode(message['data'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))

            with self._lock:
                self.latest_image = image_array
                self.latest_image_time = time.time()

            logger.debug(
                f"Image received: shape={image_array.shape}, "
                f"mean={image_array.mean():.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to decode image: {e}")

    def _instruction_callback(self, message: dict) -> None:
        """Callback for instruction update messages.

        Args:
            message: rosbridge message dict containing instruction string
        """
        new_instruction = message.get('data', '')
        if new_instruction:
            self.task_instruction = new_instruction
            logger.info(f"Updated task instruction: {self.task_instruction}")

    def publish_action(self, action: np.ndarray) -> None:
        """Publish predicted action to rosbridge.

        Args:
            action: Action array to publish
        """
        if self.action_publisher is None:
            return

        # Format as Float32MultiArray
        message = {
            'layout': {
                'dim': [],
                'data_offset': 0
            },
            'data': action.tolist()
        }
        self.action_publisher.publish(roslibpy.Message(message))
        logger.info(f"Published action: {action}")

    def inference_loop(self) -> None:
        """Main inference loop."""
        logger.info(f"Starting inference loop at {self.inference_rate} Hz")

        period = 1.0 / self.inference_rate
        last_inference_time = 0

        while self.running:
            current_time = time.time()

            # Check if enough time has passed
            if current_time - last_inference_time < period:
                time.sleep(0.001)  # Small sleep to avoid busy waiting
                continue

            last_inference_time = current_time

            # Check if we have an image
            with self._lock:
                if self.latest_image is None:
                    logger.warning("No image received yet", extra={'throttle': 5.0})
                    continue
                image = self.latest_image.copy()
                image_age = current_time - self.latest_image_time

            # Check image freshness (warn if older than 1 second)
            if image_age > 1.0:
                logger.warning(f"Image is {image_age:.1f}s old")

            # Run inference
            action = self.vla_core.predict_action(
                image=image,
                instruction=self.task_instruction,
                log_callback=lambda msg: logger.info(msg)
            )

            if action is not None:
                self.publish_action(action)

    def run(self) -> None:
        """Run the VLA rosbridge client."""
        # Load model
        logger.info("Loading VLA model...")
        if not self.vla_core.load_model():
            logger.error("Failed to load VLA model")
            return

        # Connect to rosbridge
        if not self.connect():
            logger.error("Failed to connect to rosbridge server")
            return

        # Setup topics
        self.setup_topics()

        # Start inference loop
        self.running = True

        try:
            self.inference_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the client."""
        logger.info("Shutting down...")
        self.running = False

        if self.image_subscriber:
            self.image_subscriber.unsubscribe()
        if self.instruction_subscriber:
            self.instruction_subscriber.unsubscribe()
        if self.action_publisher:
            self.action_publisher.unadvertise()
        if self.client:
            self.client.terminate()

        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLA inference client using rosbridge WebSocket"
    )
    parser.add_argument(
        '--rosbridge-host',
        type=str,
        default=os.environ.get('ROSBRIDGE_HOST', 'crane-x7-local'),
        help='Rosbridge server hostname (default: crane-x7-local)'
    )
    parser.add_argument(
        '--rosbridge-port',
        type=int,
        default=int(os.environ.get('ROSBRIDGE_PORT', '9090')),
        help='Rosbridge server port (default: 9090)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.environ.get('VLA_MODEL_PATH', ''),
        help='Path to VLA model (local or HuggingFace Hub ID)'
    )
    parser.add_argument(
        '--task-instruction',
        type=str,
        default=os.environ.get('VLA_TASK_INSTRUCTION', 'pick up the object'),
        help='Task instruction for the robot'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=os.environ.get('VLA_DEVICE', 'cuda'),
        choices=['cuda', 'cpu'],
        help='Inference device (default: cuda)'
    )
    parser.add_argument(
        '--unnorm-key',
        type=str,
        default=os.environ.get('VLA_UNNORM_KEY', 'crane_x7'),
        help='Key for action normalization statistics'
    )
    parser.add_argument(
        '--inference-rate',
        type=float,
        default=float(os.environ.get('VLA_INFERENCE_RATE', '10.0')),
        help='Inference rate in Hz (default: 10.0)'
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model_path:
        logger.error(
            "Model path not specified! "
            "Set VLA_MODEL_PATH environment variable or use --model-path"
        )
        sys.exit(1)

    # Print configuration
    logger.info("=" * 50)
    logger.info("VLA Rosbridge Client Configuration")
    logger.info("=" * 50)
    logger.info(f"  Rosbridge host: {args.rosbridge_host}")
    logger.info(f"  Rosbridge port: {args.rosbridge_port}")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Task instruction: {args.task_instruction}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Inference rate: {args.inference_rate} Hz")
    logger.info("=" * 50)

    # Create and run client
    client = VLARosbridgeClient(
        rosbridge_host=args.rosbridge_host,
        rosbridge_port=args.rosbridge_port,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        unnorm_key=args.unnorm_key,
        inference_rate=args.inference_rate,
    )

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        client.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run client
    client.run()


if __name__ == '__main__':
    main()
