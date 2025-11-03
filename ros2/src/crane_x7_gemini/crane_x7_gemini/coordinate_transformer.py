#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Coordinate transformation utilities for converting between camera and robot coordinates.
"""

import numpy as np
from typing import Tuple, Optional
from geometry_msgs.msg import Point


class CoordinateTransformer:
    """Transform coordinates between camera frame and robot base frame."""

    def __init__(
        self,
        camera_intrinsics: Optional[dict] = None,
        camera_to_base_transform: Optional[np.ndarray] = None
    ):
        """
        Initialize coordinate transformer.

        Args:
            camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy' keys
            camera_to_base_transform: 4x4 transformation matrix from camera to robot base
        """
        # Default RealSense D435 intrinsics (640x480)
        if camera_intrinsics is None:
            camera_intrinsics = {
                'fx': 615.0,  # Focal length x
                'fy': 615.0,  # Focal length y
                'cx': 320.0,  # Principal point x
                'cy': 240.0,  # Principal point y
                'width': 640,
                'height': 480,
            }

        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']
        self.width = camera_intrinsics.get('width', 640)
        self.height = camera_intrinsics.get('height', 480)

        # Default camera to base transform (adjust based on your robot setup)
        # This is a placeholder - you need to calibrate this for your setup
        if camera_to_base_transform is None:
            # Example: Camera is 0.5m in front, 0.3m above robot base, looking down
            camera_to_base_transform = np.array([
                [0, -1, 0, 0.0],    # Camera X -> -Robot Y
                [0, 0, -1, 0.0],    # Camera Y -> -Robot Z
                [1, 0, 0, 0.5],     # Camera Z -> Robot X
                [0, 0, 0, 1]
            ])

        self.camera_to_base = camera_to_base_transform

    def normalized_to_pixel(self, y_norm: float, x_norm: float) -> Tuple[int, int]:
        """
        Convert normalized coordinates (0-1000) to pixel coordinates.

        Args:
            y_norm: Normalized y coordinate (0-1000)
            x_norm: Normalized x coordinate (0-1000)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        pixel_x = int((x_norm / 1000.0) * self.width)
        pixel_y = int((y_norm / 1000.0) * self.height)
        return pixel_x, pixel_y

    def pixel_to_camera_coords(
        self,
        pixel_x: int,
        pixel_y: int,
        depth: float
    ) -> np.ndarray:
        """
        Convert pixel coordinates to 3D camera coordinates.

        Args:
            pixel_x: Pixel x coordinate
            pixel_y: Pixel y coordinate
            depth: Depth value in meters

        Returns:
            3D point in camera frame [x, y, z]
        """
        # Convert pixel to camera coordinates using pinhole camera model
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth

        return np.array([x, y, z])

    def camera_to_base_coords(self, camera_point: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to robot base frame.

        Args:
            camera_point: 3D point in camera frame [x, y, z]

        Returns:
            3D point in robot base frame [x, y, z]
        """
        # Convert to homogeneous coordinates
        camera_point_h = np.append(camera_point, 1)

        # Apply transformation
        base_point_h = self.camera_to_base @ camera_point_h

        # Convert back to 3D
        return base_point_h[:3]

    def normalized_to_base_coords(
        self,
        y_norm: float,
        x_norm: float,
        depth: float
    ) -> Point:
        """
        Convert normalized 2D coordinates to 3D robot base coordinates.

        Args:
            y_norm: Normalized y coordinate (0-1000)
            x_norm: Normalized x coordinate (0-1000)
            depth: Depth value in meters

        Returns:
            geometry_msgs/Point in robot base frame
        """
        # Convert normalized to pixel
        pixel_x, pixel_y = self.normalized_to_pixel(y_norm, x_norm)

        # Convert pixel to camera 3D
        camera_point = self.pixel_to_camera_coords(pixel_x, pixel_y, depth)

        # Convert camera to base
        base_point = self.camera_to_base_coords(camera_point)

        # Create Point message
        point = Point()
        point.x = float(base_point[0])
        point.y = float(base_point[1])
        point.z = float(base_point[2])

        return point

    def set_camera_to_base_transform(self, transform: np.ndarray):
        """Update camera to base transformation matrix."""
        self.camera_to_base = transform

    def set_camera_intrinsics(self, intrinsics: dict):
        """Update camera intrinsics."""
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.width = intrinsics.get('width', 640)
        self.height = intrinsics.get('height', 480)
