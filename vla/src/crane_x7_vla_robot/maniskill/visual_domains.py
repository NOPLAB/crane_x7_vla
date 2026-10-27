"""Explicit visual domains for simulator data collection, separate from paired RL runs."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from usim.types import SimulatorConfig
from crane_x7_vla_robot.simulation import create_simulator
from crane_x7_vla_robot.maniskill.pick_place import PickPlace


@dataclass(frozen=True)
class VisualDomain:
    shape: str = 'cube'
    half_size: float = 0.02
    rgb: tuple[float, float, float] = (12 / 255, 42 / 255, 160 / 255)
    camera_eye: tuple[float, float, float] = (0.32, -0.12, 0.42)
    camera_target: tuple[float, float, float] = (0.15, 0.02, 0.10)
    camera_fov_degrees: float = 75.0

    def __post_init__(self):
        for name in ('rgb', 'camera_eye', 'camera_target'):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f'{name} needs three finite values')
            object.__setattr__(self, name, values)
        if self.shape not in ('cube', 'box', 'cylinder'):
            raise ValueError('Unsupported object shape')
        if not math.isfinite(self.half_size) or self.half_size <= 0:
            raise ValueError('Object half size must be finite and positive')
        if not all(0 <= value <= 1 for value in self.rgb):
            raise ValueError('RGB values must be in [0, 1]')
        if not math.isfinite(self.camera_fov_degrees) or not 1 < self.camera_fov_degrees < 179:
            raise ValueError('Camera field of view must be between 1 and 179 degrees')
        direction = np.asarray(self.camera_target) - self.camera_eye
        if np.linalg.norm(direction) < 1e-6 or np.linalg.norm(direction[:2]) < 1e-6:
            raise ValueError('Camera direction must be nonzero and not parallel to world up')


_registered_domains: dict[str, VisualDomain] = {}


def create_visual_simulator(config: SimulatorConfig, domain: VisualDomain):
    """Build an isolated registered task for each immutable visual domain."""
    if config.env_id != 'PickPlace-CRANE-X7' or config.camera_uid != 'scene_camera':
        raise ValueError('Visual domains require PickPlace-CRANE-X7 and scene_camera')
    identity = hashlib.sha256(json.dumps(asdict(domain), sort_keys=True).encode()).hexdigest()
    env_id = f'PickPlace-CRANE-X7-visual-{identity}'
    if env_id not in _registered_domains:

        class DomainPickPlace(PickPlace):
            def __init__(self, *args, **kwargs):
                self.object_shape = domain.shape
                self.cube_half_size = domain.half_size
                self.use_scene_camera = True
                super().__init__(*args, **kwargs)

            def _load_scene(self, options):
                self.table_scene = TableSceneBuilder(
                    env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
                )
                self.table_scene.build()
                object_args = dict(
                    scene=self.scene,
                    color=np.array([*domain.rgb, 1.0]),
                    name='cube',
                    body_type='dynamic',
                    initial_pose=sapien.Pose(p=[0, 0, domain.half_size]),
                )
                if domain.shape == 'cube':
                    self.obj = actors.build_cube(half_size=domain.half_size, **object_args)
                elif domain.shape == 'box':
                    self.obj = actors.build_box(
                        half_sizes=np.array([1, 0.75, 1]) * domain.half_size, **object_args
                    )
                else:
                    self.obj = actors.build_cylinder(
                        radius=domain.half_size, half_length=domain.half_size, **object_args
                    )

            @property
            def _default_sensor_configs(self):
                pose = sapien_utils.look_at(domain.camera_eye, domain.camera_target)
                return [
                    CameraConfig(
                        'scene_camera', pose, 640, 480,
                        np.deg2rad(domain.camera_fov_degrees), 0.01, 10,
                    )
                ]

        register_env(env_id, max_episode_steps=200)(DomainPickPlace)
        _registered_domains[env_id] = domain
    return create_simulator('maniskill', replace(config, env_id=env_id))
