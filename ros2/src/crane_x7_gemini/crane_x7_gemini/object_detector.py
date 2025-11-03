#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Object detection service using Gemini Robotics-ER API.

This node provides a service interface for on-demand object detection.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import json

from crane_x7_gemini.gemini_node import GeminiNode


class ObjectDetectorService(GeminiNode):
    """Service-based object detector using Gemini API."""

    def __init__(self):
        super().__init__()

        # Override node name
        self.get_logger().info('Initializing ObjectDetectorService')

        # Create service for triggering detection
        self.detect_service = self.create_service(
            Trigger,
            'gemini/detect_objects',
            self.detect_service_callback
        )

        self.get_logger().info('Object detection service ready')
        self.get_logger().info('Call service: ros2 service call /gemini/detect_objects std_srvs/srv/Trigger')

    def detect_service_callback(self, request, response):
        """Service callback for object detection."""
        if self.latest_image is None:
            response.success = False
            response.message = 'No image available'
            self.get_logger().warn('Detection service called but no image available')
            return response

        # Perform detection
        detections = self.detect_objects()

        if detections is not None:
            response.success = True
            response.message = json.dumps(detections, indent=2)
            self.get_logger().info(f'Detection service returned {len(detections)} objects')
        else:
            response.success = False
            response.message = 'Detection failed'
            self.get_logger().error('Detection service failed')

        return response


def main(args=None):
    """Main entry point for the object detector service."""
    rclpy.init(args=args)

    try:
        node = ObjectDetectorService()
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
