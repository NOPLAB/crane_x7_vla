#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Compute normalization statistics for CRANE-X7 data.

This script computes the mean, std, and quantiles for state and action data
from CRANE-X7 TFRecord files. The statistics are saved in OpenPI-compatible
format for use during training.

Usage:
    python -m crane_x7_vla.scripts.compute_crane_x7_norm_stats \
        --data_dir /path/to/tfrecord_logs \
        --output_dir /path/to/assets/crane_x7_vla

The output norm_stats.json can be used directly with OpenPI training.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Import CRANE-X7 data adapter
from crane_x7_vla.data.adapters import CraneX7DataAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunningStats:
    """
    Compute running statistics of a batch of vectors.

    Compatible with OpenPI's normalize.RunningStats.
    """

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch of vectors."""
        batch = np.atleast_2d(batch)
        if batch.ndim > 2:
            batch = batch.reshape(-1, batch.shape[-1])

        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    f"Vector length mismatch: expected {self._mean.size}, got {vector_length}"
                )

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )

            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[min(idx, len(edges) - 1)])
            results.append(np.array(q_values))
        return results

    def get_statistics(self) -> dict:
        """Return computed statistics in OpenPI-compatible format."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])

        return {
            "mean": self._mean.tolist(),
            "std": stddev.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def compute_statistics(
    data_dir: Path,
    output_dir: Path,
    max_steps: Optional[int] = None,
    pad_to_dim: int = 32,
) -> dict:
    """
    Compute normalization statistics from CRANE-X7 TFRecord data.

    Args:
        data_dir: Directory containing TFRecord files
        output_dir: Directory to save norm_stats.json
        max_steps: Maximum number of steps to process (None for all)
        pad_to_dim: Dimension to pad state/action to (OpenPI uses 32)

    Returns:
        Dictionary containing computed statistics
    """
    logger.info(f"Loading data from {data_dir}")

    # Create data adapter
    adapter = CraneX7DataAdapter(data_dir, split="train", shuffle=False)

    # Initialize running stats for state and action
    state_stats = RunningStats()
    action_stats = RunningStats()

    step_count = 0

    for episode in adapter.iterate_episodes():
        # Get state and action
        state = episode.get("observation/state")
        action = episode.get("action")

        if state is None or action is None:
            continue

        # Pad to target dimension if needed
        if state.shape[-1] < pad_to_dim:
            state = np.pad(
                state, (0, pad_to_dim - state.shape[-1]), mode="constant"
            )
        if action.shape[-1] < pad_to_dim:
            action = np.pad(
                action, (0, pad_to_dim - action.shape[-1]), mode="constant"
            )

        # Update statistics
        state_stats.update(state.reshape(1, -1))
        action_stats.update(action.reshape(1, -1))

        step_count += 1

        if step_count % 1000 == 0:
            logger.info(f"Processed {step_count} steps")

        if max_steps is not None and step_count >= max_steps:
            break

    logger.info(f"Processed {step_count} total steps")

    # Get final statistics
    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    # Save to output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "norm_stats.json"
    with open(output_file, "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)

    logger.info(f"Saved statistics to {output_file}")

    return norm_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for CRANE-X7 data"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing TFRecord files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save norm_stats.json",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps to process (default: all)",
    )
    parser.add_argument(
        "--pad_to_dim",
        type=int,
        default=32,
        help="Dimension to pad state/action to (default: 32 for OpenPI)",
    )

    args = parser.parse_args()

    compute_statistics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        pad_to_dim=args.pad_to_dim,
    )


if __name__ == "__main__":
    main()
