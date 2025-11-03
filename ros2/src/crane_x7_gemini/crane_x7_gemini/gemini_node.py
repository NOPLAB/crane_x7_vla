#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Gemini Robotics-ER API client node for CRANE-X7.

This node provides a ROS 2 interface to Google's Gemini Robotics-ER 1.5 model,
enabling vision-based perception and reasoning for robot manipulation tasks.
"""

import os
import json
from typing import Optional, List, Dict, Any
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not installed. Please run: pip install google-genai")
    exit(1)


class GeminiNode(Node):
    """ROS 2 node for Gemini Robotics-ER API integration."""

    def __init__(self):
        super().__init__('gemini_node')

        # Declare parameters
        self.declare_parameter('api_key', '')
        self.declare_parameter('model_id', 'gemini-robotics-er-1.5-preview')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/gemini/detections')
        self.declare_parameter('temperature', 0.5)
        self.declare_parameter('thinking_budget', 0)
        self.declare_parameter('max_objects', 10)

        # Get parameters
        api_key = self.get_parameter('api_key').value
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY', '')

        if not api_key:
            self.get_logger().error(
                'No API key provided! Set api_key parameter or GEMINI_API_KEY environment variable.'
            )
            raise ValueError('GEMINI_API_KEY is required')

        self.model_id = self.get_parameter('model_id').value
        self.temperature = self.get_parameter('temperature').value
        self.thinking_budget = self.get_parameter('thinking_budget').value
        self.max_objects = self.get_parameter('max_objects').value

        # Initialize Gemini client
        try:
            self.client = genai.Client(api_key=api_key)
            self.get_logger().info(f'Initialized Gemini client with model: {self.model_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Gemini client: {e}')
            raise

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribe to camera image topic
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Publisher for detection results
        output_topic = self.get_parameter('output_topic').value
        self.detection_pub = self.create_publisher(String, output_topic, 10)

        # Store latest image
        self.latest_image: Optional[np.ndarray] = None
        self.image_timestamp = None

        self.get_logger().info('Gemini node initialized successfully')
        self.get_logger().info(f'Subscribing to: {image_topic}')
        self.get_logger().info(f'Publishing to: {output_topic}')

    def image_callback(self, msg: Image) -> None:
        """Callback for receiving camera images."""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            self.image_timestamp = msg.header.stamp

            self.get_logger().debug(
                f'Received image: {cv_image.shape[1]}x{cv_image.shape[0]}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def detect_objects(
        self,
        image: Optional[np.ndarray] = None,
        prompt: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Detect objects in an image using Gemini API.

        Args:
            image: OpenCV image (BGR format). If None, uses latest received image.
            prompt: Custom prompt for detection. If None, uses default detection prompt.

        Returns:
            List of detected objects with 'point' and 'label' keys, or None on error.
        """
        if image is None:
            if self.latest_image is None:
                self.get_logger().warn('No image available for detection')
                return None
            image = self.latest_image

        # Default prompt for object detection
        if prompt is None:
            prompt = f"""
            Point to no more than {self.max_objects} items in the image. The label returned
            should be an identifying name for the object detected.
            The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...].
            The points are in [y, x] format normalized to 0-1000.
            """

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Encode image as JPEG
            success, encoded_image = cv2.imencode('.jpg', rgb_image)
            if not success:
                self.get_logger().error('Failed to encode image')
                return None

            image_bytes = encoded_image.tobytes()

            # Call Gemini API
            self.get_logger().info('Calling Gemini API for object detection...')
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                )
            )

            # Parse response
            response_text = response.text.strip()
            self.get_logger().info(f'Gemini response: {response_text}')

            # Try to parse JSON response
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            detections = json.loads(response_text)

            # Publish detections
            detection_msg = String()
            detection_msg.data = json.dumps(detections, indent=2)
            self.detection_pub.publish(detection_msg)

            self.get_logger().info(f'Detected {len(detections)} objects')
            return detections

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse Gemini response as JSON: {e}')
            self.get_logger().error(f'Response text: {response_text}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error during object detection: {e}')
            return None

    def detect_specific_objects(
        self,
        object_names: List[str],
        image: Optional[np.ndarray] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Detect specific objects by name.

        Args:
            object_names: List of object names to detect.
            image: OpenCV image (BGR format). If None, uses latest received image.

        Returns:
            List of detected objects with 'point' and 'label' keys, or None on error.
        """
        prompt = f"""
        Get all points matching the following objects: {', '.join(object_names)}.
        The label returned should be an identifying name for the object detected.
        The answer should follow the json format:
        [{{"point": <point>, "label": <label>}}, ...].
        The points are in [y, x] format normalized to 0-1000.
        """

        return self.detect_objects(image=image, prompt=prompt)

    def get_bounding_boxes(
        self,
        image: Optional[np.ndarray] = None,
        max_objects: int = 25
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get 2D bounding boxes for objects in the image.

        Args:
            image: OpenCV image (BGR format). If None, uses latest received image.
            max_objects: Maximum number of objects to detect.

        Returns:
            List of objects with 'box_2d' and 'label' keys, or None on error.
        """
        prompt = f"""
        Return bounding boxes as a JSON array with labels. Never return masks
        or code fencing. Limit to {max_objects} objects. Include as many objects as you
        can identify in the scene.
        If an object is present multiple times, name them according to their
        unique characteristic (colors, size, position, unique characteristics, etc..).
        The format should be as follows: [{{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>}}]
        normalized to 0-1000. The values in box_2d must only be integers.
        """

        return self.detect_objects(image=image, prompt=prompt)


def main(args=None):
    """Main entry point for the Gemini node."""
    rclpy.init(args=args)

    try:
        node = GeminiNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
