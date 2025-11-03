#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Episode saving functionality for data logger."""

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from rclpy.impl.rcutils_logger import RcutilsLogger


class EpisodeSaver:
    """Handles saving episode data to disk in RLDS-compatible format."""

    def __init__(
        self,
        output_dir: str,
        save_format: str,
        logger: RcutilsLogger,
        dataset_name: str = "crane_x7",
        compute_statistics: bool = True,
        statistics_output_path: str = None
    ):
        """
        Initialize episode saver.

        Args:
            output_dir: Directory to save episodes
            save_format: Format to save ('npz' or 'tfrecord')
            logger: ROS logger instance
            dataset_name: Dataset identifier for RLDS
            compute_statistics: Whether to compute and save dataset statistics
            statistics_output_path: Path to save statistics JSON
        """
        self.output_dir = output_dir
        self.save_format = save_format.lower()
        self.logger = logger
        self.dataset_name = dataset_name
        self.compute_statistics = compute_statistics
        self.statistics_output_path = statistics_output_path

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Validate format
        if self.save_format not in ['npz', 'tfrecord']:
            raise ValueError(f"Invalid save format: {save_format}. Must be 'npz' or 'tfrecord'")

        # Statistics accumulator
        self.action_statistics = [] if compute_statistics else None

    def save(
        self,
        episode_data: List[Dict[str, Any]],
        episode_number: int,
        language_instruction: str = None
    ) -> str:
        """
        Save episode to disk in RLDS format.

        Args:
            episode_data: List of step dictionaries
            episode_number: Episode number for naming
            language_instruction: Language task description for the episode

        Returns:
            Path to saved file
        """
        if len(episode_data) == 0:
            self.logger.warn('Cannot save empty episode')
            return ''

        # Use default instruction if not provided
        if language_instruction is None:
            language_instruction = "manipulate the object"

        # Update actions: action[t] = state[t+1]
        self._update_actions(episode_data)

        # Accumulate action statistics if enabled
        if self.compute_statistics:
            self._accumulate_action_statistics(episode_data)

        # Create episode directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        episode_dir = os.path.join(
            self.output_dir,
            f'episode_{episode_number:04d}_{timestamp}'
        )
        os.makedirs(episode_dir, exist_ok=True)

        # Save based on format
        if self.save_format == 'tfrecord':
            return self._save_tfrecord(
                episode_data, episode_dir, episode_number, language_instruction
            )
        else:
            return self._save_npz(episode_data, episode_dir, episode_number)

    def _update_actions(self, episode_data: List[Dict[str, Any]]) -> None:
        """
        Update actions in episode data: action[t] = state[t+1].

        Args:
            episode_data: List of step dictionaries (modified in-place)
        """
        for i in range(len(episode_data) - 1):
            episode_data[i]['action'] = episode_data[i + 1]['observation']['state']

    def _save_tfrecord(
        self,
        episode_data: List[Dict[str, Any]],
        episode_dir: str,
        episode_number: int,
        language_instruction: str
    ) -> str:
        """Save episode as RLDS-formatted TFRecord."""
        from .tfrecord_writer import TFRecordWriter

        tfrecord_path = os.path.join(episode_dir, 'episode_data.tfrecord')
        writer = TFRecordWriter(
            tfrecord_path,
            dataset_name=self.dataset_name,
            language_instruction=language_instruction
        )
        writer.write_episode(episode_data)
        writer.close()

        self.logger.info(
            f'Saved RLDS episode {episode_number} with {len(episode_data)} steps '
            f'(instruction: "{language_instruction[:50]}...") to {tfrecord_path}'
        )
        return tfrecord_path

    def _save_npz(
        self,
        episode_data: List[Dict[str, Any]],
        episode_dir: str,
        episode_number: int
    ) -> str:
        """Save episode as compressed NPZ."""
        episode_path = os.path.join(episode_dir, 'episode_data.npz')

        # Prepare data arrays
        states = np.array([step['observation']['state'] for step in episode_data])
        actions = np.array([step['action'] for step in episode_data])
        timestamps = np.array([step['observation']['timestamp'] for step in episode_data])

        save_dict = {
            'states': states,
            'actions': actions,
            'timestamps': timestamps,
        }

        # Add optional data
        if 'image' in episode_data[0]['observation']:
            images = np.array([step['observation']['image'] for step in episode_data])
            save_dict['images'] = images

        if 'depth' in episode_data[0]['observation']:
            depths = np.array([step['observation']['depth'] for step in episode_data])
            save_dict['depths'] = depths

        np.savez_compressed(episode_path, **save_dict)

        self.logger.info(
            f'Saved episode {episode_number} with {len(episode_data)} steps to {episode_path}'
        )
        return episode_path

    def _accumulate_action_statistics(self, episode_data: List[Dict[str, Any]]) -> None:
        """Accumulate action values for computing dataset statistics."""
        if self.action_statistics is None:
            return

        for step in episode_data:
            if 'action' in step:
                self.action_statistics.append(step['action'])

    def compute_and_save_statistics(self) -> None:
        """Compute and save dataset-wide action statistics."""
        if self.action_statistics is None or len(self.action_statistics) == 0:
            self.logger.warn('No action data to compute statistics')
            return

        if self.statistics_output_path is None:
            self.logger.warn('No statistics output path configured')
            return

        import json

        # Convert to numpy array
        all_actions = np.array(self.action_statistics)  # (num_steps, action_dim)

        # Compute statistics
        stats = {
            'dataset_name': self.dataset_name,
            'num_trajectories': 0,  # Will be updated by caller if needed
            'num_transitions': len(all_actions),
            'action': {
                'mean': all_actions.mean(axis=0).tolist(),
                'std': all_actions.std(axis=0).tolist(),
                'min': all_actions.min(axis=0).tolist(),
                'max': all_actions.max(axis=0).tolist(),
                'q01': np.percentile(all_actions, 1, axis=0).tolist(),
                'q99': np.percentile(all_actions, 99, axis=0).tolist(),
            }
        }

        # Ensure directory exists
        stats_dir = os.path.dirname(self.statistics_output_path)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)

        # Save to JSON
        with open(self.statistics_output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f'Saved dataset statistics to {self.statistics_output_path}')
        self.logger.info(f'  Total transitions: {stats["num_transitions"]}')
