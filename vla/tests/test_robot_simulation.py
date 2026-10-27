"""CPU contracts for application-owned robot injection and profile mapping."""

import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from usim import factory
from usim.model import ModelSource
from usim.specs import AgentSpec, ControlSpec

from crane_x7_vla_robot.crane_x7 import CraneX7Config, get_mjcf_path
from crane_x7_vla_robot.simulation import create_simulator, simulator_config


def test_downstream_mjcf_mesh_resources_are_present():
    model = Path(get_mjcf_path())
    meshes = ElementTree.parse(model).getroot().findall("./asset/mesh")
    assert meshes
    for mesh in meshes:
        assert (model.parent / mesh.attrib["file"]).is_file()


def test_servo_profile_expands_in_explicit_joint_order():
    config = simulator_config(actuator_profile="servo_envelope", gripper_force_limit=1.5)
    assert config.robot_uid == "CRANE-X7"
    assert config.robot_path is None
    assert config.joint_names == tuple(CraneX7Config.ALL_JOINT_NAMES)
    assert config.joint_force_limits == (10, 10, 4, 4, 4, 4, 4, 1.5, 1.5)
    assert config.joint_stiffness == (1000,) * 9
    assert config.joint_damping == (100,) * 9
    assert config.initial_qpos == tuple(CraneX7Config.REST_QPOS)


def test_file_model_can_be_injected_as_agent_spec():
    config = simulator_config("JointControl")
    agent = AgentSpec(
        name="crane",
        model=ModelSource.from_path(get_mjcf_path()),
        control=ControlSpec(
            joint_names=config.joint_names,
            position_joints=config.joint_names,
            initial_qpos=config.initial_qpos,
            joint_stiffness=config.joint_stiffness,
            joint_damping=config.joint_damping,
            joint_force_limits=config.joint_force_limits,
        ),
    )
    assert config.robot_path == str(agent.model.path)
    assert agent.control.joint_names == tuple(CraneX7Config.ALL_JOINT_NAMES)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_application_force_override_is_rejected(limit):
    with pytest.raises(ValueError, match="Gripper force limit"):
        simulator_config(gripper_force_limit=limit)


def test_registered_agent_receives_numeric_settings_outside_generic_engine(monkeypatch):
    class Agent:
        pass

    class Task:
        pass

    registration = SimpleNamespace(CraneX7=Agent, PickPlace=Task)
    monkeypatch.setitem(sys.modules, "crane_x7_vla_robot.maniskill", registration)
    calls = []
    monkeypatch.setattr(factory, "create_simulator", lambda name, config: calls.append((name, config)))
    config = simulator_config(
        actuator_profile="servo_envelope", camera_uid="scene_camera",
        robot_init_qpos_noise=0.04,
    )
    create_simulator("maniskill", config)
    assert Agent.arm_force_limit == (10, 10, 4, 4, 4, 4, 4)
    assert Agent.gripper_force_limit == (2, 2)
    assert Agent.arm_stiffness == (1000,) * 7
    assert Task.use_scene_camera is True
    assert Task.robot_init_qpos_noise_default == 0.04
    assert calls[0][0] == "maniskill"
    native = calls[0][1]
    assert native.joint_force_limits is None
    assert native.joint_stiffness is None
    assert native.joint_damping is None
    assert config.joint_force_limits is not None


def test_joint_control_engine_does_not_claim_robot_task_support():
    with pytest.raises(ValueError, match="pick-place rollouts require ManiSkill"):
        create_simulator("genesis", simulator_config())
