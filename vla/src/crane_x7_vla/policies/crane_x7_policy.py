# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Policy for OpenPI.

Provides a Policy wrapper for running inference on CRANE-X7 robot
using OpenPI models (π₀-FAST, π₀, etc.).
"""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Import OpenPI modules
from openpi.models import model as _model
from openpi.policies.policy import Policy
import openpi.transforms as _transforms
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class CraneX7InputTransform(_transforms.DataTransformFn):
    """
    Transform CRANE-X7 observation format to OpenPI input format.

    This transform handles:
    - State padding from 8 DOF to 32 DOF
    - Image format conversion (single camera to multi-camera)
    - Image mask creation for missing cameras
    """

    # Source state dimension (CRANE-X7)
    source_dim: int = 8
    # Target state dimension (OpenPI)
    target_dim: int = 32
    # Expected camera names by OpenPI
    camera_names: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # Image size for padding
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        # Pad state from 8 -> 32
        if "state" in data:
            state = np.asarray(data["state"])
            if state.shape[-1] < self.target_dim:
                data["state"] = _transforms.pad_to_dim(state, self.target_dim, axis=-1)

        # Ensure images dict exists and is properly formatted
        if "image" not in data:
            data["image"] = {}

        # If only a single image is provided, map to primary camera
        if isinstance(data.get("image"), np.ndarray):
            data["image"] = {self.camera_names[0]: data["image"]}

        # Create image mask
        if "image_mask" not in data:
            data["image_mask"] = {}

        # Ensure all expected cameras exist
        for cam_name in self.camera_names:
            if cam_name not in data["image"]:
                # Create zero-padded image
                data["image"][cam_name] = np.zeros(
                    (*self.image_size, 3), dtype=np.uint8
                )
                data["image_mask"][cam_name] = False
            else:
                data["image_mask"][cam_name] = True

        return data


@dataclasses.dataclass(frozen=True)
class CraneX7OutputTransform(_transforms.DataTransformFn):
    """
    Transform OpenPI output format to CRANE-X7 action format.

    This transform handles:
    - Action truncation from 32 DOF to 8 DOF
    - Optional action chunk selection
    """

    # Target action dimension (CRANE-X7)
    target_dim: int = 8
    # Whether to return only the first action from the chunk
    first_action_only: bool = False

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" in data:
            actions = np.asarray(data["actions"])

            # Truncate from 32 -> 8 DOF
            if actions.shape[-1] > self.target_dim:
                actions = actions[..., :self.target_dim]

            # Return only first action if requested
            if self.first_action_only and actions.ndim > 1:
                actions = actions[0]

            data["actions"] = actions

        return data


class CraneX7Policy(Policy):
    """
    Policy wrapper for CRANE-X7 robot with OpenPI models.

    Extends the base OpenPI Policy with CRANE-X7 specific transformations.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        source_dim: int = 8,
        target_dim: int = 32,
        first_action_only: bool = True,
    ):
        """
        Initialize CRANE-X7 Policy.

        Args:
            model: OpenPI model for action prediction
            rng: JAX random key (for JAX models)
            transforms: Additional input transforms
            output_transforms: Additional output transforms
            sample_kwargs: Additional kwargs for model.sample_actions
            metadata: Policy metadata
            pytorch_device: Device for PyTorch models
            is_pytorch: Whether using PyTorch model
            source_dim: CRANE-X7 action dimension (8)
            target_dim: OpenPI action dimension (32)
            first_action_only: Return only first action from chunk
        """
        # Add CRANE-X7 specific transforms
        crane_x7_input = CraneX7InputTransform(
            source_dim=source_dim,
            target_dim=target_dim,
        )
        crane_x7_output = CraneX7OutputTransform(
            target_dim=source_dim,
            first_action_only=first_action_only,
        )

        all_transforms = (crane_x7_input, *transforms)
        all_output_transforms = (crane_x7_output, *output_transforms)

        # Add CRANE-X7 metadata
        crane_x7_metadata = {
            "robot": "crane_x7",
            "source_dim": source_dim,
            "target_dim": target_dim,
            **(metadata or {}),
        }

        super().__init__(
            model,
            rng=rng,
            transforms=all_transforms,
            output_transforms=all_output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=crane_x7_metadata,
            pytorch_device=pytorch_device,
            is_pytorch=is_pytorch,
        )

    def predict_action(
        self,
        state: np.ndarray,
        image: np.ndarray,
        prompt: str = "manipulate objects",
        *,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict action for CRANE-X7 robot.

        Convenience method that wraps infer() with CRANE-X7 specific formatting.

        Args:
            state: Robot state [8] (joint positions)
            image: RGB image [H, W, 3]
            prompt: Task instruction
            noise: Optional noise for action sampling

        Returns:
            Predicted action [8] or action chunk [horizon, 8]
        """
        obs = {
            "state": state,
            "image": {
                "base_0_rgb": image,
            },
            "prompt": prompt,
        }

        result = self.infer(obs, noise=noise)
        return result["actions"]


def create_crane_x7_policy(
    model: _model.BaseModel,
    *,
    transforms: Sequence[_transforms.DataTransformFn] = (),
    output_transforms: Sequence[_transforms.DataTransformFn] = (),
    rng: at.KeyArrayLike | None = None,
    is_pytorch: bool = False,
    pytorch_device: str = "cuda:0",
    first_action_only: bool = True,
    default_prompt: str = "manipulate objects",
) -> CraneX7Policy:
    """
    Factory function to create a CRANE-X7 policy.

    Args:
        model: OpenPI model
        transforms: Additional input transforms (applied after CRANE-X7 transform)
        output_transforms: Additional output transforms (applied before CRANE-X7 transform)
        rng: JAX random key
        is_pytorch: Whether model is PyTorch
        pytorch_device: Device for PyTorch models
        first_action_only: Return only first action
        default_prompt: Default task instruction

    Returns:
        Configured CraneX7Policy instance
    """
    # Add default prompt injection
    prompt_transform = _transforms.InjectDefaultPrompt(default_prompt)

    all_transforms = (prompt_transform, *transforms)

    return CraneX7Policy(
        model,
        rng=rng,
        transforms=all_transforms,
        output_transforms=output_transforms,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device,
        first_action_only=first_action_only,
    )
