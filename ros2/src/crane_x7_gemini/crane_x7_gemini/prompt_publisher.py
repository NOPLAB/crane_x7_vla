#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Simple node for publishing task prompts to the trajectory planner.

This is a utility node for testing and demonstration purposes.
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PromptPublisher(Node):
    """Node for publishing task prompts."""

    def __init__(self, prompt: str):
        super().__init__('prompt_publisher')

        self.declare_parameter('prompt_topic', '/gemini/task_prompt')
        prompt_topic = self.get_parameter('prompt_topic').value

        self.publisher = self.create_publisher(String, prompt_topic, 10)

        # Wait for connections
        self.get_logger().info('Waiting for subscribers...')
        import time
        time.sleep(1.0)

        # Publish prompt
        msg = String()
        msg.data = prompt
        self.publisher.publish(msg)

        self.get_logger().info(f'Published prompt: {prompt}')
        self.get_logger().info(f'Topic: {prompt_topic}')


def main(args=None):
    """Main entry point for prompt publisher."""
    rclpy.init(args=args)

    # Get prompt from command line
    if len(sys.argv) < 2:
        print('Usage: ros2 run crane_x7_gemini prompt_publisher "<your task prompt>"')
        print('Example: ros2 run crane_x7_gemini prompt_publisher "Pick up the red pen and move it to the organizer"')
        return

    prompt = ' '.join(sys.argv[1:])

    try:
        node = PromptPublisher(prompt)
        # Keep node alive briefly to ensure message is sent
        rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
