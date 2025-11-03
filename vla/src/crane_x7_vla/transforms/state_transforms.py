"""
State transformation utilities.

Handles state normalization and padding.
"""

import numpy as np
from typing import Literal, Optional


class StateNormalizer:
    """
    Normalizes robot states (joint positions, velocities, etc.).

    Similar to ActionNormalizer but specifically for proprioceptive states.
    """

    def __init__(
        self,
        mode: Literal["quantile", "zscore", "minmax"] = "quantile",
        quantile_low: float = 0.01,
        quantile_high: float = 0.99
    ):
        """
        Initialize state normalizer.

        Args:
            mode: Normalization mode ('quantile', 'zscore', or 'minmax')
            quantile_low: Lower quantile for quantile normalization
            quantile_high: Upper quantile for quantile normalization
        """
        self.mode = mode
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high

        # Statistics (to be computed from data)
        self.stats = {}

    def fit(self, states: np.ndarray) -> None:
        """
        Compute normalization statistics from data.

        Args:
            states: State array of shape (N, state_dim) or (N, T, state_dim)
        """
        # Flatten to (N * T, state_dim) if needed
        if states.ndim == 3:
            states = states.reshape(-1, states.shape[-1])

        if self.mode == "quantile":
            # Compute quantiles along the batch dimension
            self.stats['q_low'] = np.quantile(states, self.quantile_low, axis=0)
            self.stats['q_high'] = np.quantile(states, self.quantile_high, axis=0)
            self.stats['range'] = self.stats['q_high'] - self.stats['q_low']
            # Avoid division by zero
            self.stats['range'] = np.where(
                self.stats['range'] < 1e-6,
                1.0,
                self.stats['range']
            )

        elif self.mode == "zscore":
            # Compute mean and std along the batch dimension
            self.stats['mean'] = np.mean(states, axis=0)
            self.stats['std'] = np.std(states, axis=0)
            # Avoid division by zero
            self.stats['std'] = np.where(
                self.stats['std'] < 1e-6,
                1.0,
                self.stats['std']
            )

        elif self.mode == "minmax":
            # Compute min and max
            self.stats['min'] = np.min(states, axis=0)
            self.stats['max'] = np.max(states, axis=0)
            self.stats['range'] = self.stats['max'] - self.stats['min']
            # Avoid division by zero
            self.stats['range'] = np.where(
                self.stats['range'] < 1e-6,
                1.0,
                self.stats['range']
            )

    def normalize(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize states.

        Args:
            states: State array of any shape (..., state_dim)

        Returns:
            Normalized states
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.mode == "quantile":
            normalized = (states - self.stats['q_low']) / self.stats['range']
            # Scale to [-1, 1]
            normalized = 2 * normalized - 1
            # Clip to [-1, 1]
            normalized = np.clip(normalized, -1, 1)

        elif self.mode == "zscore":
            normalized = (states - self.stats['mean']) / self.stats['std']
            # Clip to reasonable range
            normalized = np.clip(normalized, -10, 10)

        elif self.mode == "minmax":
            normalized = (states - self.stats['min']) / self.stats['range']
            # Scale to [-1, 1]
            normalized = 2 * normalized - 1
            # Clip to [-1, 1]
            normalized = np.clip(normalized, -1, 1)

        return normalized

    def denormalize(self, states: np.ndarray) -> np.ndarray:
        """
        Denormalize states back to original range.

        Args:
            states: Normalized state array of any shape (..., state_dim)

        Returns:
            Denormalized states
        """
        if not self.stats:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if self.mode == "quantile":
            # Scale from [-1, 1] to [0, 1]
            denormalized = (states + 1) / 2
            # Scale to original range
            denormalized = denormalized * self.stats['range'] + self.stats['q_low']

        elif self.mode == "zscore":
            denormalized = states * self.stats['std'] + self.stats['mean']

        elif self.mode == "minmax":
            # Scale from [-1, 1] to [0, 1]
            denormalized = (states + 1) / 2
            # Scale to original range
            denormalized = denormalized * self.stats['range'] + self.stats['min']

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


class StatePadder:
    """
    Pads states to match the model's expected dimension.

    Similar to ActionPadder but for states.
    """

    def __init__(self, source_dim: int = 8, target_dim: int = 32):
        """
        Initialize state padder.

        Args:
            source_dim: Source state dimension
            target_dim: Target state dimension
        """
        self.source_dim = source_dim
        self.target_dim = target_dim

        if target_dim < source_dim:
            raise ValueError(
                f"Target dimension ({target_dim}) must be >= source dimension ({source_dim})"
            )

    def pad(self, state: np.ndarray) -> np.ndarray:
        """
        Pad state to target dimension.

        Args:
            state: State array of shape (..., source_dim)

        Returns:
            Padded state array of shape (..., target_dim)
        """
        if state.shape[-1] != self.source_dim:
            raise ValueError(
                f"Expected state dimension {self.source_dim}, got {state.shape[-1]}"
            )

        if self.target_dim == self.source_dim:
            return state

        # Zero-pad to target dimension
        pad_width = [(0, 0)] * (state.ndim - 1) + [(0, self.target_dim - self.source_dim)]
        return np.pad(state, pad_width, mode='constant', constant_values=0)

    def unpad(self, state: np.ndarray) -> np.ndarray:
        """
        Unpad state back to source dimension.

        Args:
            state: State array of shape (..., target_dim)

        Returns:
            Unpadded state array of shape (..., source_dim)
        """
        if state.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected state dimension {self.target_dim}, got {state.shape[-1]}"
            )

        # Take only the first source_dim elements
        return state[..., :self.source_dim]
