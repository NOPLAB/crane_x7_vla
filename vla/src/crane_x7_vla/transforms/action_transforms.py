"""
Action transformation utilities.

Handles action padding, chunking, and normalization for different VLA backends.
"""

import numpy as np
from typing import Literal, Optional, Tuple


class ActionPadder:
    """
    Pads actions from CRANE-X7's native dimension to target dimension.

    For OpenPI: pads 8-dim actions to 32-dim by zero-padding.
    """

    def __init__(self, source_dim: int = 8, target_dim: int = 32):
        """
        Initialize action padder.

        Args:
            source_dim: Source action dimension (CRANE-X7 has 8 DOF)
            target_dim: Target action dimension (e.g., 32 for OpenPI)
        """
        self.source_dim = source_dim
        self.target_dim = target_dim

        if target_dim < source_dim:
            raise ValueError(
                f"Target dimension ({target_dim}) must be >= source dimension ({source_dim})"
            )

    def pad(self, action: np.ndarray) -> np.ndarray:
        """
        Pad action to target dimension.

        Args:
            action: Action array of shape (..., source_dim)

        Returns:
            Padded action array of shape (..., target_dim)
        """
        if action.shape[-1] != self.source_dim:
            raise ValueError(
                f"Expected action dimension {self.source_dim}, got {action.shape[-1]}"
            )

        if self.target_dim == self.source_dim:
            return action

        # Zero-pad to target dimension
        pad_width = [(0, 0)] * (action.ndim - 1) + [(0, self.target_dim - self.source_dim)]
        return np.pad(action, pad_width, mode='constant', constant_values=0)

    def unpad(self, action: np.ndarray) -> np.ndarray:
        """
        Unpad action back to source dimension.

        Args:
            action: Action array of shape (..., target_dim)

        Returns:
            Unpadded action array of shape (..., source_dim)
        """
        if action.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected action dimension {self.target_dim}, got {action.shape[-1]}"
            )

        # Take only the first source_dim elements
        return action[..., :self.source_dim]


class ActionChunker:
    """
    Converts single-step actions to action chunks.

    For OpenPI: converts 1-step actions to 50-step action horizons.
    """

    def __init__(
        self,
        action_horizon: int = 50,
        interpolation: Literal["repeat", "linear"] = "linear"
    ):
        """
        Initialize action chunker.

        Args:
            action_horizon: Number of future actions to predict
            interpolation: How to generate chunks:
                - 'repeat': Repeat the same action
                - 'linear': Linear interpolation from current to next action
        """
        self.action_horizon = action_horizon
        self.interpolation = interpolation

    def chunk_single_action(
        self,
        action: np.ndarray,
        next_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert a single action to an action chunk.

        Args:
            action: Current action of shape (action_dim,)
            next_action: Next action for interpolation (optional)

        Returns:
            Action chunk of shape (action_horizon, action_dim)
        """
        if self.action_horizon == 1:
            return action[np.newaxis, :]

        if self.interpolation == "repeat":
            # Simply repeat the action
            return np.repeat(action[np.newaxis, :], self.action_horizon, axis=0)

        elif self.interpolation == "linear":
            if next_action is None:
                # If no next action, repeat the current action
                return np.repeat(action[np.newaxis, :], self.action_horizon, axis=0)

            # Linear interpolation from action to next_action
            alphas = np.linspace(0, 1, self.action_horizon)[:, np.newaxis]
            chunk = (1 - alphas) * action + alphas * next_action
            return chunk

        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpolation}")

    def chunk_trajectory(
        self,
        actions: np.ndarray,
        pad_last: bool = True
    ) -> np.ndarray:
        """
        Convert a trajectory of actions to action chunks.

        Args:
            actions: Action trajectory of shape (T, action_dim)
            pad_last: If True, pad the last chunk by repeating the last action

        Returns:
            Action chunks of shape (T, action_horizon, action_dim)
        """
        T = len(actions)
        chunks = []

        for t in range(T):
            if t < T - 1:
                chunk = self.chunk_single_action(actions[t], actions[t + 1])
            else:
                if pad_last:
                    chunk = self.chunk_single_action(actions[t], None)
                else:
                    chunk = self.chunk_single_action(actions[t], actions[t])

            chunks.append(chunk)

        return np.stack(chunks, axis=0)


class ActionNormalizer:
    """
    Normalizes and denormalizes actions.

    Supports both quantile-based and z-score normalization.
    """

    def __init__(
        self,
        mode: Literal["quantile", "zscore"] = "quantile",
        quantile_low: float = 0.01,
        quantile_high: float = 0.99
    ):
        """
        Initialize action normalizer.

        Args:
            mode: Normalization mode ('quantile' or 'zscore')
            quantile_low: Lower quantile for quantile normalization
            quantile_high: Upper quantile for quantile normalization
        """
        self.mode = mode
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high

        # Statistics (to be computed from data)
        self.stats = {}

    def fit(self, actions: np.ndarray) -> None:
        """
        Compute normalization statistics from data.

        Args:
            actions: Action array of shape (N, action_dim) or (N, T, action_dim)
        """
        # Flatten to (N * T, action_dim) if needed
        if actions.ndim == 3:
            actions = actions.reshape(-1, actions.shape[-1])

        if self.mode == "quantile":
            # Compute quantiles along the batch dimension
            self.stats['q_low'] = np.quantile(actions, self.quantile_low, axis=0)
            self.stats['q_high'] = np.quantile(actions, self.quantile_high, axis=0)
            self.stats['range'] = self.stats['q_high'] - self.stats['q_low']
            # Avoid division by zero
            self.stats['range'] = np.where(
                self.stats['range'] < 1e-6,
                1.0,
                self.stats['range']
            )

        elif self.mode == "zscore":
            # Compute mean and std along the batch dimension
            self.stats['mean'] = np.mean(actions, axis=0)
            self.stats['std'] = np.std(actions, axis=0)
            # Avoid division by zero
            self.stats['std'] = np.where(
                self.stats['std'] < 1e-6,
                1.0,
                self.stats['std']
            )

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalize actions to approximately [-1, 1] range.

        Args:
            actions: Action array of any shape (..., action_dim)

        Returns:
            Normalized actions
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.mode == "quantile":
            normalized = (actions - self.stats['q_low']) / self.stats['range']
            # Scale to [-1, 1]
            normalized = 2 * normalized - 1
            # Clip to [-1, 1]
            normalized = np.clip(normalized, -1, 1)

        elif self.mode == "zscore":
            normalized = (actions - self.stats['mean']) / self.stats['std']
            # Clip to reasonable range
            normalized = np.clip(normalized, -10, 10)

        return normalized

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize actions back to original range.

        Args:
            actions: Normalized action array of any shape (..., action_dim)

        Returns:
            Denormalized actions
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.mode == "quantile":
            # Scale from [-1, 1] to [0, 1]
            denormalized = (actions + 1) / 2
            # Scale to original range
            denormalized = denormalized * self.stats['range'] + self.stats['q_low']

        elif self.mode == "zscore":
            denormalized = actions * self.stats['std'] + self.stats['mean']

        return denormalized

    def save_stats(self, path: str) -> None:
        """Save normalization statistics to file."""
        np.savez(path, **self.stats, mode=self.mode)

    def load_stats(self, path: str) -> None:
        """Load normalization statistics from file."""
        data = np.load(path, allow_pickle=True)
        self.stats = {k: data[k] for k in data.files if k != 'mode'}
        if 'mode' in data:
            self.mode = str(data['mode'])
