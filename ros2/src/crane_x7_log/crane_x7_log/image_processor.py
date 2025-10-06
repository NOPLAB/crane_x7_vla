#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""Image processing utilities for data logger."""

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from typing import Optional, Tuple


class ImageProcessor:
    """Handles image conversion and processing."""

    def __init__(self, target_width: int = 0, target_height: int = 0):
        """
        Initialize image processor.

        Args:
            target_width: Target width for resizing (0 = no resize)
            target_height: Target height for resizing (0 = no resize)
        """
        self.bridge = CvBridge()
        self.target_width = target_width
        self.target_height = target_height
        self.should_resize = target_width > 0 and target_height > 0

    def process_rgb_image(self, msg: Image) -> Optional[np.ndarray]:
        """
        Convert and process RGB image message.

        Args:
            msg: ROS Image message

        Returns:
            Processed RGB image as numpy array, or None on error
        """
        try:
            rgb_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if self.should_resize:
                rgb_array = cv2.resize(rgb_array, (self.target_width, self.target_height))
            return rgb_array
        except Exception as e:
            return None

    def process_depth_image(self, msg: Image) -> Optional[np.ndarray]:
        """
        Convert depth image message.

        Args:
            msg: ROS Image message

        Returns:
            Depth image as numpy array, or None on error
        """
        try:
            depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.should_resize:
                depth_array = cv2.resize(depth_array, (self.target_width, self.target_height))
            return depth_array
        except Exception as e:
            return None

    def get_resize_dimensions(self) -> Tuple[int, int]:
        """
        Get target resize dimensions.

        Returns:
            (width, height) tuple
        """
        return (self.target_width, self.target_height)
