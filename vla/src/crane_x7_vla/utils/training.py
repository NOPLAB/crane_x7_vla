# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Common training utilities for VLA backends."""

from typing import List

import numpy as np

from crane_x7_vla.types import OverfitMetrics


def compute_overfit_metrics(
    losses: List[float],
    action_accuracies: List[float],
    l1_losses: List[float],
) -> OverfitMetrics:
    """
    Compute averaged overfitting detection metrics.

    This function is shared across backends to ensure consistent
    metric calculation for overfitting detection.

    Args:
        losses: List of loss values from overfitting check batches
        action_accuracies: List of action accuracy values
        l1_losses: List of L1 loss values

    Returns:
        OverfitMetrics containing averaged metrics
    """
    return OverfitMetrics(
        overfit_loss=float(np.mean(losses)) if losses else 0.0,
        overfit_action_accuracy=float(np.mean(action_accuracies)) if action_accuracies else 0.0,
        overfit_l1_loss=float(np.mean(l1_losses)) if l1_losses else 0.0,
    )


def format_training_progress(
    step: int,
    max_steps: int,
    loss: float,
    lr: float | None = None,
    additional_metrics: dict[str, float] | None = None,
) -> str:
    """
    Format training progress message.

    Args:
        step: Current training step
        max_steps: Maximum training steps
        loss: Current loss value
        lr: Learning rate (optional)
        additional_metrics: Additional metrics to display (optional)

    Returns:
        Formatted progress string
    """
    parts = [f"Step {step}/{max_steps}", f"Loss: {loss:.4f}"]

    if lr is not None:
        parts.append(f"LR: {lr:.2e}")

    if additional_metrics:
        for name, value in additional_metrics.items():
            parts.append(f"{name}: {value:.4f}")

    return " | ".join(parts)


def format_overfit_metrics(
    step: int,
    metrics: OverfitMetrics,
) -> str:
    """
    Format overfitting check metrics message.

    Args:
        step: Current training step
        metrics: Overfitting metrics

    Returns:
        Formatted metrics string
    """
    return (
        f"[Step {step}] Overfit Loss: {metrics['overfit_loss']:.4f}, "
        f"Overfit Accuracy: {metrics['overfit_action_accuracy']:.4f}, "
        f"Overfit L1: {metrics['overfit_l1_loss']:.4f}"
    )


def compute_gradient_norm(parameters) -> float:
    """
    Compute total gradient norm across all parameters.

    Args:
        parameters: Iterable of model parameters

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
