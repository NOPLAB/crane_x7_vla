"""Application-owned CRANE-X7 configuration for robot-neutral usim engines."""

from dataclasses import replace
from math import isfinite

from usim.types import SimulatorConfig

from .crane_x7 import CraneX7Config, get_mjcf_path


def simulator_config(
    env_id: str = "PickPlace-CRANE-X7",
    *,
    actuator_profile: str = "legacy",
    gripper_force_limit: float | None = None,
    **kwargs,
) -> SimulatorConfig:
    """Resolve application servo profiles to explicit native joint settings."""
    arm, gripper = CraneX7Config.controller_force_limits(actuator_profile)
    if gripper_force_limit is not None:
        if not isfinite(gripper_force_limit) or gripper_force_limit <= 0:
            raise ValueError("Gripper force limit must be finite and positive")
        gripper = gripper_force_limit
    arm_limits = (float(arm),) * 7 if isinstance(arm, (float, int)) else arm
    defaults = {
        # Registered ManiSkill agents own their model path. File-based
        # JointControl engines receive the explicit model instead.
        "robot_path": get_mjcf_path() if env_id == "JointControl" else None,
        "robot_uid": "CRANE-X7",
        "joint_names": tuple(CraneX7Config.ALL_JOINT_NAMES),
        "arm_joint_names": tuple(CraneX7Config.ARM_JOINT_NAMES),
        "gripper_joint_names": tuple(CraneX7Config.GRIPPER_JOINT_NAMES),
        "initial_qpos": tuple(float(q) for q in CraneX7Config.REST_QPOS),
        "joint_stiffness": (CraneX7Config.ARM_STIFFNESS,) * 7
        + (CraneX7Config.GRIPPER_STIFFNESS,) * 2,
        "joint_damping": (CraneX7Config.ARM_DAMPING,) * 7
        + (CraneX7Config.GRIPPER_DAMPING,) * 2,
        "joint_force_limits": tuple(arm_limits) + (gripper,) * 2,
    }
    defaults.update(kwargs)
    return SimulatorConfig(env_id=env_id, **defaults)


def create_simulator(name: str, config: SimulatorConfig):
    """Register application tasks explicitly before entering the generic engine."""
    from usim.factory import create_simulator as create

    if name == "maniskill":
        from .maniskill import CraneX7, PickPlace

        names = config.joint_names or config.arm_joint_names + config.gripper_joint_names
        for values, attribute in (
            (config.joint_stiffness, "stiffness"),
            (config.joint_damping, "damping"),
            (config.joint_force_limits, "force_limit"),
        ):
            if values is not None:
                by_name = dict(zip(names, values))
                setattr(CraneX7, f"arm_{attribute}", tuple(
                    by_name[name] for name in CraneX7Config.ARM_JOINT_NAMES
                ))
                setattr(CraneX7, f"gripper_{attribute}", tuple(
                    by_name[name] for name in CraneX7Config.GRIPPER_JOINT_NAMES
                ))

        if config.env_id == "PickPlace-CRANE-X7":
            if config.camera_uid not in ("hand_camera", "scene_camera"):
                raise ValueError(f"Unsupported CRANE pick-place camera: {config.camera_uid}")
            PickPlace.use_scene_camera = config.camera_uid == "scene_camera"
            PickPlace.robot_init_qpos_noise_default = config.robot_init_qpos_noise
        config = replace(
            config, joint_stiffness=None, joint_damping=None, joint_force_limits=None,
        )
    elif config.env_id != "JointControl":
        raise ValueError(
            "CRANE-X7 pick-place rollouts require ManiSkill; Genesis/Isaac use "
            "JointControl with application-owned task orchestration"
        )
    return create(name, config)
