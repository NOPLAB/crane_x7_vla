"""Exercise the configuration files used by each VLA training entry point."""

from argparse import Namespace
from pathlib import Path

import pytest

from crane_x7_vla.core.config.base import UnifiedVLAConfig
from crane_x7_vla.training.cli import _apply_lora_args_to_config, _get_backend_configs, create_default_config


@pytest.mark.parametrize("backend", ["openvla", "openvla-oft", "minivla", "pi0", "pi0.5"])
def test_generated_config_roundtrip(backend: str, tmp_path: Path):
    config = create_default_config(backend, tmp_path / "data", tmp_path / "outputs", "smoke")
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    assert "!!python/" not in path.read_text()
    assert UnifiedVLAConfig.from_yaml(path).backend == backend

    loaded = _get_backend_configs(backend)[0].from_yaml(path)
    assert loaded.backend == backend
    assert getattr(loaded, "pi0" if backend.startswith("pi0") else backend.replace("-", "_")) == getattr(
        config, "pi0" if backend.startswith("pi0") else backend.replace("-", "_")
    )


@pytest.mark.parametrize("backend", ["openvla", "openvla-oft", "minivla", "pi0", "pi0.5"])
def test_backend_settings_survive_roundtrip(backend: str, tmp_path: Path):
    config = create_default_config(backend, tmp_path / "data", tmp_path / "outputs", "smoke")
    name = "pi0" if backend.startswith("pi0") else backend.replace("-", "_")
    specific = getattr(config, name)
    specific.image_size = (128, 160)
    config.training.max_steps = 3
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    loaded = _get_backend_configs(backend)[0].from_yaml(path)
    assert loaded.training.max_steps == 3
    assert getattr(loaded, name).image_size == (128, 160)


@pytest.mark.parametrize(
    ("filename", "backend"),
    [
        ("openvla_default.yaml", "openvla"),
        ("minivla_default.yaml", "minivla"),
        ("pi0_default.yaml", "pi0"),
        ("pi05_default.yaml", "pi0.5"),
    ],
)
def test_repository_example_configs(filename: str, backend: str):
    path = Path(__file__).parents[1] / "configs" / filename
    loaded = _get_backend_configs(backend)[0].from_yaml(path)
    assert loaded.backend == backend
    if backend == "minivla":
        assert loaded.minivla.vq.action_horizon == 8
        assert loaded.minivla.multi_image.use_wrist_camera
    elif backend == "pi0.5":
        assert loaded.pi0.max_token_len == 200
        assert loaded.pi0.discrete_state_input


@pytest.mark.parametrize("backend", ["openvla", "openvla-oft", "minivla", "pi0", "pi0.5"])
def test_lora_cli_overrides_reach_backend(backend: str, tmp_path: Path):
    config = create_default_config(backend, tmp_path / "data", tmp_path / "outputs", "smoke")
    _apply_lora_args_to_config(
        Namespace(lora_enabled=False, lora_rank=4, lora_alpha=8, lora_dropout=0.1,
                  lora_target_modules=["q_proj"], lora_skip_merge_on_save=False),
        config,
    )
    specific = getattr(config, "pi0" if backend.startswith("pi0") else backend.replace("-", "_"))
    assert specific.use_lora is False
    assert specific.lora_dropout == 0.1
    if backend.startswith("pi0"):
        assert specific.expert_lora_rank == specific.vlm_lora_rank == 4
        assert specific.expert_lora_alpha == specific.vlm_lora_alpha == 8
        assert specific.lora_target_modules == ["q_proj"]
    else:
        assert specific.lora_rank == 4
        assert specific.lora_alpha == 8
        assert specific.skip_merge_on_save is False
