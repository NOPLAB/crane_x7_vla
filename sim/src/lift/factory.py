# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Factory for creating simulator instances."""

from typing import Callable, Type

from lift.interface import Simulator
from lift.types import SimulatorConfig

_SIMULATORS: dict[str, Type[Simulator]] = {}


def register_simulator(name: str) -> Callable[[Type[Simulator]], Type[Simulator]]:
    """Decorator to register a simulator implementation.

    Args:
        name: Name to register the simulator under.

    Returns:
        Decorator function.

    Example:
        @register_simulator("maniskill")
        class ManiSkillSimulator(Simulator):
            ...
    """

    def decorator(cls: Type[Simulator]) -> Type[Simulator]:
        if name in _SIMULATORS:
            raise ValueError(f"Simulator '{name}' is already registered")
        _SIMULATORS[name] = cls
        return cls

    return decorator


def create_simulator(name: str, config: SimulatorConfig) -> Simulator:
    """Create a simulator instance by name.

    Args:
        name: Name of the registered simulator.
        config: Configuration for the simulator.

    Returns:
        Simulator instance.

    Raises:
        ValueError: If simulator name is not registered.
    """
    if name not in _SIMULATORS:
        available = ", ".join(_SIMULATORS.keys()) if _SIMULATORS else "none"
        raise ValueError(f"Unknown simulator: '{name}'. Available: {available}")
    return _SIMULATORS[name](config)


def list_simulators() -> list[str]:
    """List all registered simulator names.

    Returns:
        List of registered simulator names.
    """
    return list(_SIMULATORS.keys())
