"""Single-environment policy observations from native batched engine results."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from usim import Observation, StepResult


@pytest.fixture
def wrapper_module(monkeypatch):
    path = Path(__file__).parents[1] / "src/crane_x7_vla_rl/environments/usim_wrapper.py"
    spec = importlib.util.spec_from_file_location("usim_wrapper_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batched", [False, True])
def test_single_native_result_preserves_policy_shapes_and_reward(wrapper_module, batched):
    # Given equivalent single-environment outputs from scalar and batched engines.
    image = np.full((8, 12, 3), 64, dtype=np.uint8)
    state = np.arange(9, dtype=np.float32)
    observation = Observation(
        rgb_image=image[None] if batched else image,
        qpos=state[None] if batched else state,
    )
    result = StepResult(
        observation=observation,
        reward=np.array([2.0]) if batched else 2.0,
        terminated=np.array([True]) if batched else True,
        truncated=np.array([False]) if batched else False,
        info=[{"success": True}] if batched else {"success": True},
    )
    simulator = SimpleNamespace(
        config=SimpleNamespace(n_envs=1),
        step=lambda action: result,
        reset=lambda seed=None: (observation, {}),
    )
    environment = wrapper_module.UsimRolloutEnvironment(simulator, dense_reward_weight=0.5)
    # When a native episode step is consumed by the policy wrapper.
    wrapped, reward, terminated, truncated, info = environment.step(np.zeros(8))
    # Then the policy sees the same HWC image, joint vector and scalar episode values.
    np.testing.assert_array_equal(wrapped.image, image)
    np.testing.assert_array_equal(wrapped.state, state)
    assert reward == 2.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["episode_reward"] == 2.0


def test_single_policy_wrapper_rejects_multi_environment_engine(wrapper_module):
    # Given a native engine with two simultaneous environments.
    simulator = SimpleNamespace(config=SimpleNamespace(n_envs=2))
    # When using the scalar rollout interface.
    # Then environments cannot be silently dropped from the episode.
    with pytest.raises(ValueError, match="single environment"):
        wrapper_module.UsimRolloutEnvironment(simulator)
