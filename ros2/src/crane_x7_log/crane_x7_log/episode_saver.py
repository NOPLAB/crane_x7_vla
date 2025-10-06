#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""Episode saving functionality for data logger."""

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from rclpy.logging import Logger


class EpisodeSaver:
    """Handles saving episode data to disk."""

    def __init__(
        self,
        output_dir: str,
        save_format: str,
        logger: Logger
    ):
        """
        Initialize episode saver.

        Args:
            output_dir: Directory to save episodes
            save_format: Format to save ('npz' or 'tfrecord')
            logger: ROS logger instance
        """
        self.output_dir = output_dir
        self.save_format = save_format.lower()
        self.logger = logger

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Validate format
        if self.save_format not in ['npz', 'tfrecord']:
            raise ValueError(f"Invalid save format: {save_format}. Must be 'npz' or 'tfrecord'")

    def save(self, episode_data: List[Dict[str, Any]], episode_number: int) -> str:
        """
        Save episode to disk.

        Args:
            episode_data: List of step dictionaries
            episode_number: Episode number for naming

        Returns:
            Path to saved file
        """
        if len(episode_data) == 0:
            self.logger.warn('Cannot save empty episode')
            return ''

        # Update actions: action[t] = state[t+1]
        self._update_actions(episode_data)

        # Create episode directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        episode_dir = os.path.join(
            self.output_dir,
            f'episode_{episode_number:04d}_{timestamp}'
        )
        os.makedirs(episode_dir, exist_ok=True)

        # Save based on format
        if self.save_format == 'tfrecord':
            return self._save_tfrecord(episode_data, episode_dir, episode_number)
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
        episode_number: int
    ) -> str:
        """Save episode as TFRecord."""
        from .tfrecord_writer import TFRecordWriter

        tfrecord_path = os.path.join(episode_dir, 'episode_data.tfrecord')
        writer = TFRecordWriter(tfrecord_path)
        writer.write_episode(episode_data)
        writer.close()

        self.logger.info(
            f'Saved episode {episode_number} with {len(episode_data)} steps to {tfrecord_path}'
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
