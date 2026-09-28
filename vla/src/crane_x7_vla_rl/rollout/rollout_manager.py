# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Rollout manager for VLA-RL training."""

import logging
from typing import Callable

import numpy as np
import torch

from crane_x7_vla_rl.config.rollout_config import RolloutConfig
from crane_x7_vla_rl.environments.parallel_envs import ParallelLiftEnvironments
from crane_x7_vla_rl.rollout.trajectory_buffer import Trajectory, TrajectoryBuffer

logger = logging.getLogger(__name__)


class RolloutManager:
    """Manages VLA rollout data collection.

    Orchestrates parallel environment execution and trajectory collection
    for PPO training.
    """

    def __init__(
        self,
        config: RolloutConfig,
        policy_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, float, float]],
        device: torch.device | str = "cuda",
    ):
        """Initialize rollout manager.

        Args:
            config: Rollout configuration.
            policy_fn: Policy function that takes observations and returns
                       (action, log_prob, value).
            device: Device for inference.
        """
        self.config = config
        self.policy_fn = policy_fn
        self.device = torch.device(device) if isinstance(device, str) else device

        # Create parallel environments
        self.envs = ParallelLiftEnvironments(
            num_envs=config.num_parallel_envs,
            config=config,
        )

        # Trajectory buffer
        self.buffer = TrajectoryBuffer()

        # Statistics
        self.total_steps = 0
        self.total_episodes = 0

    def collect_rollouts(
        self,
        num_rollouts: int | None = None,
        seed: int | None = None,
    ) -> TrajectoryBuffer:
        """Collect rollouts from parallel environments.

        Args:
            num_rollouts: Number of complete episodes to collect
                         (defaults to config.num_rollouts_per_update).
            seed: Random seed for environment reset.

        Returns:
            TrajectoryBuffer containing collected trajectories.
        """
        if num_rollouts is None:
            num_rollouts = self.config.num_rollouts_per_update

        self.buffer.clear()
        completed_rollouts = 0

        # Initialize trajectories for each env
        active_trajectories = [Trajectory() for _ in range(self.envs.num_envs)]

        # Reset all environments
        batch_obs = self.envs.reset_all(seed=seed)

        while completed_rollouts < num_rollouts:
            # Get observations as dict
            obs_dict = {
                "image": batch_obs.images,
                "state": batch_obs.states,
            }

            # Get actions from policy
            actions, log_probs, values = self._get_actions(obs_dict)

            # Step environments
            result = self.envs.step_all(actions)

            # Store transitions
            for i in range(self.envs.num_envs):
                active_trajectories[i].append(
                    observation={
                        "image": batch_obs.images[i],
                        "state": batch_obs.states[i],
                    },
                    action=actions[i],
                    reward=result.rewards[i],
                    log_prob=log_probs[i],
                    value=values[i],
                    done=result.terminateds[i] or result.truncateds[i],
                    info=result.infos[i],
                )

                # Check if episode ended
                if result.terminateds[i] or result.truncateds[i]:
                    # Save completed trajectory
                    self.buffer.add(active_trajectories[i])
                    completed_rollouts += 1
                    self.total_episodes += 1

                    # Log progress
                    if completed_rollouts % 10 == 0:
                        logger.debug(
                            f"Collected {completed_rollouts}/{num_rollouts} rollouts"
                        )

                    # Start new trajectory
                    active_trajectories[i] = Trajectory()

                    # Early exit if we have enough
                    if completed_rollouts >= num_rollouts:
                        break

            # Update observation
            batch_obs = result.observations
            self.total_steps += self.envs.num_envs

            # Reset done environments
            dones = result.terminateds | result.truncateds
            if np.any(dones):
                self.envs.reset_done(dones, seed=seed)

        # Log statistics
        stats = self.buffer.get_statistics()
        logger.info(
            f"Rollout complete: {stats['num_trajectories']} episodes, "
            f"success_rate={stats['success_rate']:.2%}, "
            f"mean_reward={stats['mean_reward']:.2f}, "
            f"mean_length={stats['mean_length']:.1f}"
        )

        return self.buffer

    def _get_actions(
        self,
        observations: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get actions from policy for batched observations.

        Args:
            observations: Dict with batched observations.

        Returns:
            Tuple of (actions, log_probs, values) arrays.
        """
        batch_size = observations["image"].shape[0]

        actions_list = []
        log_probs_list = []
        values_list = []

        # Process each observation through policy
        for i in range(batch_size):
            obs_i = {
                "image": observations["image"][i],
                "state": observations["state"][i],
            }

            action, log_prob, value = self.policy_fn(obs_i)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

        return (
            np.stack(actions_list),
            np.array(log_probs_list, dtype=np.float32),
            np.array(values_list, dtype=np.float32),
        )

    def evaluate(
        self,
        num_episodes: int = 10,
        seed: int | None = None,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """Evaluate policy performance.

        Args:
            num_episodes: Number of episodes to evaluate.
            seed: Random seed.
            deterministic: Whether to use deterministic actions.

        Returns:
            Dict with evaluation metrics.
        """
        # Use single env for evaluation to ensure sequential execution
        from crane_x7_vla_rl.environments.lift_wrapper import LiftRolloutEnvironment

        eval_env = LiftRolloutEnvironment.from_config(
            env_id=self.config.env_id,
            simulator_name=self.config.simulator,
            backend=self.config.backend,
            render_mode=self.config.render_mode,
            max_episode_steps=self.config.max_steps,
            use_binary_reward=self.config.use_binary_reward,
        )

        successes = []
        rewards = []
        lengths = []

        for ep in range(num_episodes):
            ep_seed = seed + ep if seed is not None else None
            obs, _ = eval_env.reset(seed=ep_seed)

            episode_reward = 0.0
            episode_length = 0
            episode_success = False

            for step in range(self.config.max_steps):
                obs_dict = {
                    "image": obs.image,
                    "state": obs.state,
                }

                action, _, _ = self.policy_fn(obs_dict)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_reward += reward
                episode_length += 1

                if info.get("success", False):
                    episode_success = True

                if terminated or truncated:
                    break

            successes.append(episode_success)
            rewards.append(episode_reward)
            lengths.append(episode_length)

        eval_env.close()

        return {
            "eval_success_rate": np.mean(successes),
            "eval_mean_reward": np.mean(rewards),
            "eval_mean_length": np.mean(lengths),
            "eval_num_episodes": num_episodes,
        }

    def close(self) -> None:
        """Release resources."""
        self.envs.close()

    def __enter__(self) -> "RolloutManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
