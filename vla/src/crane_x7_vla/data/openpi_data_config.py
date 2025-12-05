# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenPI Data Configuration Factory for CRANE-X7.

This module provides a DataConfigFactory for training OpenPI models
on CRANE-X7 robot data.
"""

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import override

import numpy as np

# Import OpenPI modules
from openpi.models import model as _model
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    AssetsConfig,
    ModelTransformFactory,
)
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class CraneX7Inputs(_transforms.DataTransformFn):
    """
    CRANE-X7 input transformation.

    Transforms CRANE-X7 specific input format to OpenPI expected format:
    - Pads 8 DOF state to 32 DOF
    - Maps single camera to OpenPI's 3-camera format
    - Sets image masks for missing cameras
    """

    # Source action/state dimension (CRANE-X7 = 8)
    source_dim: int = 8
    # Target action/state dimension (OpenPI = 32)
    target_dim: int = 32
    # Camera names expected by OpenPI
    camera_names: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # Primary camera name in source data
    primary_camera: str = "primary"

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        # Handle state padding (8 -> 32)
        if "state" in data:
            state = data["state"]
            if state.shape[-1] < self.target_dim:
                data["state"] = _transforms.pad_to_dim(state, self.target_dim, axis=-1)

        # Handle image/images key mapping
        if "image" not in data and "images" not in data:
            # Check for observation/image key
            if "observation/image" in data:
                data["image"] = {self.camera_names[0]: data.pop("observation/image")}

        # Ensure images dict exists
        if "image" not in data:
            if "images" in data:
                data["image"] = data.pop("images")
            else:
                data["image"] = {}

        # Map primary camera to base_0_rgb if needed
        if self.primary_camera in data["image"] and self.camera_names[0] not in data["image"]:
            data["image"][self.camera_names[0]] = data["image"].pop(self.primary_camera)

        # Create image_mask for available cameras
        if "image_mask" not in data:
            data["image_mask"] = {}

        # Ensure all expected cameras exist (pad with zeros if missing)
        for cam_name in self.camera_names:
            if cam_name not in data["image"]:
                # Create zero image (will be resized later)
                data["image"][cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)
                data["image_mask"][cam_name] = False
            else:
                data["image_mask"][cam_name] = True

        return data


@dataclasses.dataclass(frozen=True)
class CraneX7Outputs(_transforms.DataTransformFn):
    """
    CRANE-X7 output transformation.

    Transforms OpenPI output to CRANE-X7 format:
    - Truncates 32 DOF actions to 8 DOF
    """

    # Target action dimension (CRANE-X7 = 8)
    target_dim: int = 8

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        # Truncate actions from 32 -> 8
        if "actions" in data:
            actions = data["actions"]
            if actions.shape[-1] > self.target_dim:
                data["actions"] = actions[..., :self.target_dim]

        return data


@dataclasses.dataclass(frozen=True)
class CraneX7DataConfigFactory(DataConfigFactory):
    """
    CRANE-X7 Data Configuration Factory for OpenPI.

    Creates DataConfig for training OpenPI models on CRANE-X7 robot data.
    Handles the conversion from CRANE-X7's 8 DOF format to OpenPI's 32 DOF format.
    """

    # LeRobot format dataset path or repo ID
    repo_id: str = "crane_x7_vla"

    # Assets configuration
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)

    # Default prompt for manipulation tasks
    default_prompt: str | None = "manipulate objects"

    # Whether to use delta actions (action = next_state - current_state)
    use_delta_joint_actions: bool = False

    # Source dimension (CRANE-X7)
    source_dim: int = 8

    # Target dimension (OpenPI)
    target_dim: int = 32

    # Action sequence keys in the dataset
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        Create DataConfig for CRANE-X7 data.

        Args:
            assets_dirs: Base directory for assets (norm stats, etc.)
            model_config: Model configuration

        Returns:
            DataConfig configured for CRANE-X7 data
        """
        # Repack transform: Map TFRecord/LeRobot keys to OpenPI expected keys
        # This transforms the dataset format to match what OpenPI expects
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    # Map observation keys
                    "image": {
                        "base_0_rgb": "observation/image",
                    },
                    "state": "observation/state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        )

        # Data transforms: CRANE-X7 specific transformations
        data_transforms = _transforms.Group(
            inputs=[
                CraneX7Inputs(
                    source_dim=self.source_dim,
                    target_dim=self.target_dim,
                ),
            ],
            outputs=[
                CraneX7Outputs(target_dim=self.source_dim),
            ],
        )

        # Add delta action transform if requested
        if self.use_delta_joint_actions:
            # Create mask: 7 joints use delta, gripper (1) uses absolute
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms: Model-specific transformations
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # Create base config with norm stats
        base_config = self.create_base_config(assets_dirs, model_config)

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class CraneX7LeRobotDataConfig(DataConfigFactory):
    """
    Alternative CRANE-X7 Data Configuration for LeRobot format data.

    Use this when the data has already been converted to LeRobot format
    with standard key naming conventions.
    """

    # LeRobot format dataset path or repo ID
    repo_id: str = "crane_x7_vla"

    # Assets configuration
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)

    # Default prompt
    default_prompt: str | None = "manipulate objects"

    # Use delta actions
    use_delta_joint_actions: bool = False

    # Source/target dimensions
    source_dim: int = 8
    target_dim: int = 32

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create DataConfig for LeRobot format CRANE-X7 data."""

        # Repack transform for LeRobot standard format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "image": {
                        "base_0_rgb": "observation.images.primary",
                    },
                    "state": "observation.state",
                    "actions": "action",
                })
            ]
        )

        # Data transforms
        data_transforms = _transforms.Group(
            inputs=[
                CraneX7Inputs(
                    source_dim=self.source_dim,
                    target_dim=self.target_dim,
                    primary_camera="primary",
                ),
            ],
            outputs=[
                CraneX7Outputs(target_dim=self.source_dim),
            ],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            prompt_from_task=True,  # Use task from LeRobot dataset
        )
