# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Multi-camera management system.

Handles multiple camera streams, synchronization, and calibration.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera information and calibration data."""

    name: str
    """Camera name (e.g., 'primary', 'wrist_left', 'wrist_right')"""

    topic: str
    """ROS topic for this camera"""

    width: int = 640
    """Image width"""

    height: int = 480
    """Image height"""

    fps: int = 30
    """Frame rate"""

    enabled: bool = True
    """Whether this camera is enabled"""

    # Calibration (optional)
    intrinsic_matrix: Optional[np.ndarray] = None
    """Camera intrinsic matrix [3x3]"""

    distortion_coeffs: Optional[np.ndarray] = None
    """Distortion coefficients"""

    extrinsic_matrix: Optional[np.ndarray] = None
    """Camera extrinsic matrix [4x4] (transform from robot base)"""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert numpy arrays to lists for YAML serialization
        if self.intrinsic_matrix is not None:
            d['intrinsic_matrix'] = self.intrinsic_matrix.tolist()
        if self.distortion_coeffs is not None:
            d['distortion_coeffs'] = self.distortion_coeffs.tolist()
        if self.extrinsic_matrix is not None:
            d['extrinsic_matrix'] = self.extrinsic_matrix.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraInfo':
        """Create from dictionary."""
        # Convert lists back to numpy arrays
        if 'intrinsic_matrix' in d and d['intrinsic_matrix'] is not None:
            d['intrinsic_matrix'] = np.array(d['intrinsic_matrix'])
        if 'distortion_coeffs' in d and d['distortion_coeffs'] is not None:
            d['distortion_coeffs'] = np.array(d['distortion_coeffs'])
        if 'extrinsic_matrix' in d and d['extrinsic_matrix'] is not None:
            d['extrinsic_matrix'] = np.array(d['extrinsic_matrix'])
        return cls(**d)


class CameraManager:
    """
    Manages multiple camera streams.

    Handles camera configuration, synchronization, and provides
    unified access to multi-camera data.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize camera manager.

        Args:
            config_path: Path to camera configuration YAML file
        """
        self.cameras: Dict[str, CameraInfo] = {}
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.latest_timestamps: Dict[str, float] = {}

        if config_path is not None:
            self.load_config(config_path)

    def add_camera(
        self,
        name: str,
        topic: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enabled: bool = True
    ) -> None:
        """
        Add a camera to the manager.

        Args:
            name: Camera name
            topic: ROS topic
            width: Image width
            height: Image height
            fps: Frame rate
            enabled: Whether camera is enabled
        """
        camera_info = CameraInfo(
            name=name,
            topic=topic,
            width=width,
            height=height,
            fps=fps,
            enabled=enabled
        )
        self.cameras[name] = camera_info
        logger.info(f"Added camera: {name} (topic: {topic})")

    def remove_camera(self, name: str) -> None:
        """Remove a camera from the manager."""
        if name in self.cameras:
            del self.cameras[name]
            if name in self.latest_frames:
                del self.latest_frames[name]
            if name in self.latest_timestamps:
                del self.latest_timestamps[name]
            logger.info(f"Removed camera: {name}")

    def update_frame(
        self,
        camera_name: str,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update the latest frame for a camera.

        Args:
            camera_name: Name of the camera
            frame: Image frame
            timestamp: Frame timestamp (optional)
        """
        if camera_name not in self.cameras:
            logger.warning(f"Camera {camera_name} not registered")
            return

        self.latest_frames[camera_name] = frame

        if timestamp is not None:
            self.latest_timestamps[camera_name] = timestamp

    def get_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from a camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Latest frame or None if not available
        """
        return self.latest_frames.get(camera_name)

    def get_all_frames(
        self,
        only_enabled: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get latest frames from all cameras.

        Args:
            only_enabled: Only return frames from enabled cameras

        Returns:
            Dictionary mapping camera names to frames
        """
        if only_enabled:
            camera_names = [
                name for name, cam in self.cameras.items()
                if cam.enabled
            ]
        else:
            camera_names = list(self.cameras.keys())

        return {
            name: self.latest_frames[name]
            for name in camera_names
            if name in self.latest_frames
        }

    def is_synchronized(
        self,
        max_time_diff: float = 0.1
    ) -> bool:
        """
        Check if all camera frames are synchronized.

        Args:
            max_time_diff: Maximum allowed time difference in seconds

        Returns:
            True if all frames are within max_time_diff
        """
        if not self.latest_timestamps:
            return False

        enabled_cameras = [
            name for name, cam in self.cameras.items()
            if cam.enabled
        ]

        # Check if we have timestamps for all enabled cameras
        if not all(name in self.latest_timestamps for name in enabled_cameras):
            return False

        timestamps = [self.latest_timestamps[name] for name in enabled_cameras]
        time_diff = max(timestamps) - min(timestamps)

        return time_diff <= max_time_diff

    def get_synchronized_frames(
        self,
        max_time_diff: float = 0.1,
        pad_missing: bool = True
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get synchronized frames from all cameras.

        Args:
            max_time_diff: Maximum allowed time difference in seconds
            pad_missing: If True, pad missing cameras with zeros

        Returns:
            Dictionary of synchronized frames or None if not synchronized
        """
        if not pad_missing and not self.is_synchronized(max_time_diff):
            return None

        frames = self.get_all_frames(only_enabled=True)

        if pad_missing:
            # Add zero-padded frames for missing cameras
            for name, cam in self.cameras.items():
                if cam.enabled and name not in frames:
                    frames[name] = np.zeros(
                        (cam.height, cam.width, 3),
                        dtype=np.uint8
                    )

        return frames

    def set_calibration(
        self,
        camera_name: str,
        intrinsic_matrix: Optional[np.ndarray] = None,
        distortion_coeffs: Optional[np.ndarray] = None,
        extrinsic_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Set calibration data for a camera.

        Args:
            camera_name: Name of the camera
            intrinsic_matrix: Camera intrinsic matrix [3x3]
            distortion_coeffs: Distortion coefficients
            extrinsic_matrix: Camera extrinsic matrix [4x4]
        """
        if camera_name not in self.cameras:
            logger.warning(f"Camera {camera_name} not registered")
            return

        cam = self.cameras[camera_name]

        if intrinsic_matrix is not None:
            cam.intrinsic_matrix = intrinsic_matrix
        if distortion_coeffs is not None:
            cam.distortion_coeffs = distortion_coeffs
        if extrinsic_matrix is not None:
            cam.extrinsic_matrix = extrinsic_matrix

        logger.info(f"Updated calibration for camera: {camera_name}")

    def save_config(self, config_path: Path) -> None:
        """
        Save camera configuration to YAML file.

        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            'cameras': [cam.to_dict() for cam in self.cameras.values()]
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved camera configuration to {config_path}")

    def load_config(self, config_path: Path) -> None:
        """
        Load camera configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Camera config file not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'cameras' in config:
            for cam_dict in config['cameras']:
                cam_info = CameraInfo.from_dict(cam_dict)
                self.cameras[cam_info.name] = cam_info

        logger.info(f"Loaded camera configuration from {config_path}")
        logger.info(f"Registered {len(self.cameras)} cameras")

    def get_camera_info(self, camera_name: str) -> Optional[CameraInfo]:
        """Get camera information."""
        return self.cameras.get(camera_name)

    def list_cameras(self) -> List[str]:
        """Get list of registered camera names."""
        return list(self.cameras.keys())

    def list_enabled_cameras(self) -> List[str]:
        """Get list of enabled camera names."""
        return [name for name, cam in self.cameras.items() if cam.enabled]

    def __len__(self) -> int:
        """Get number of registered cameras."""
        return len(self.cameras)

    def __repr__(self) -> str:
        """String representation."""
        enabled = len(self.list_enabled_cameras())
        return f"CameraManager({len(self.cameras)} cameras, {enabled} enabled)"
