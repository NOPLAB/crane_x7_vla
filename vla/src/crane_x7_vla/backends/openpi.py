# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenPI JAX/Flax backend implementation.

Integrates OpenPI's JAX training pipeline with the unified VLA backend interface.
Supports π₀-FAST model with action chunking for CRANE-X7 robot.
"""

import dataclasses
import functools
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Add OpenPI source path
openpi_path = Path(__file__).parent.parent.parent / "openpi" / "src"
if str(openpi_path) not in sys.path:
    sys.path.insert(0, str(openpi_path))

# Try to import JAX and OpenPI modules
try:
    import jax
    import jax.numpy as jnp
    from flax.training import common_utils
    import flax.nnx as nnx
    import tqdm.auto as tqdm
    import wandb

    # OpenPI imports
    import openpi.models.model as _model
    import openpi.models.pi0_fast as pi0_fast
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.checkpoints as _checkpoints
    import openpi.training.data_loader as _data_loader
    import openpi.training.optimizer as _optimizer
    import openpi.training.sharding as sharding
    import openpi.training.utils as training_utils
    import openpi.training.weight_loaders as _weight_loaders
    import openpi.transforms as _transforms
    import openpi.shared.array_typing as at
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.policies.policy import Policy

    OPENPI_AVAILABLE = True
except ImportError as e:
    OPENPI_AVAILABLE = False
    logger.warning(f"OpenPI JAX modules not available: {e}. OpenPI backend will not work.")

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openpi_config import OpenPIConfig


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.handlers:
        root_logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    """Initialize Weights & Biases logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resuming:
        wandb_id_file = ckpt_dir / "wandb_id.txt"
        if wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


class OpenPIBackend(VLABackend):
    """
    OpenPI JAX/Flax backend implementation.

    Integrates OpenPI's JAX training pipeline for CRANE-X7 data.
    Uses π₀-FAST model with action chunking.
    """

    def __init__(self, config: OpenPIConfig):
        """
        Initialize OpenPI backend.

        Args:
            config: OpenPI configuration
        """
        if not OPENPI_AVAILABLE:
            raise ImportError(
                "OpenPI JAX modules not available. Please ensure OpenPI and JAX are properly installed."
            )

        super().__init__(config)
        self.openpi_config = config
        self._action_dim = config.openpi.action_dim
        self._action_horizon = config.openpi.action_horizon
        self._image_size = config.openpi.image_size

        # Model and policy (loaded later)
        self.model = None
        self.policy = None
        self._train_state = None
        self._mesh = None

    def _create_model_config(self) -> _model.BaseModelConfig:
        """Create OpenPI model configuration."""
        model_type = self.openpi_config.openpi.model_type.lower()

        if model_type == "pi0_fast":
            return pi0_fast.Pi0FASTConfig(
                dtype="bfloat16" if self.openpi_config.training.mixed_precision == "bf16" else "float32",
                action_dim=self._action_dim,
                action_horizon=self._action_horizon,
                max_token_len=self.openpi_config.openpi.max_token_len,
            )
        elif model_type == "pi0":
            return pi0_config.Pi0Config(
                dtype="bfloat16" if self.openpi_config.training.mixed_precision == "bf16" else "float32",
                action_dim=self._action_dim,
                action_horizon=self._action_horizon,
                max_token_len=self.openpi_config.openpi.max_token_len,
            )
        elif model_type == "pi05":
            return pi0_config.Pi0Config(
                dtype="bfloat16" if self.openpi_config.training.mixed_precision == "bf16" else "float32",
                action_dim=self._action_dim,
                action_horizon=self._action_horizon,
                max_token_len=self.openpi_config.openpi.max_token_len,
                pi05=True,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_data_config_factory(self) -> _config.DataConfigFactory:
        """Create data configuration factory for CRANE-X7."""
        from crane_x7_vla.data.openpi_data_config import CraneX7DataConfigFactory

        return CraneX7DataConfigFactory(
            repo_id=str(self.openpi_config.data.data_root),
            default_prompt=self.openpi_config.openpi.default_prompt,
            use_delta_joint_actions=self.openpi_config.openpi.use_delta_actions,
            source_dim=self.openpi_config.crane_x7_action_dim,
            target_dim=self._action_dim,
        )

    def _create_train_config(self) -> _config.TrainConfig:
        """Create OpenPI TrainConfig from our configuration."""
        model_config = self._create_model_config()
        data_config_factory = self._create_data_config_factory()

        # Determine weight loader
        if self.openpi_config.openpi.pretrained_model_path:
            weight_loader = _weight_loaders.CheckpointWeightLoader(
                self.openpi_config.openpi.pretrained_model_path
            )
        else:
            # Default to base model
            weight_loader = _weight_loaders.HuggingFaceWeightLoader(
                repo_id="physical-intelligence/fast",
                path="params",
            )

        # Freeze filter for LoRA
        if self.openpi_config.openpi.use_lora:
            freeze_filter = model_config.get_freeze_filter()
        else:
            freeze_filter = nnx.Nothing

        # Create LR schedule
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps=self.openpi_config.training.warmup_steps,
            peak_lr=self.openpi_config.training.learning_rate,
            decay_steps=self.openpi_config.training.max_steps,
            decay_lr=self.openpi_config.training.learning_rate * 0.1,
        )

        # Create optimizer
        optimizer = _optimizer.AdamW(
            weight_decay=self.openpi_config.training.weight_decay,
            clip_gradient_norm=self.openpi_config.training.max_grad_norm,
        )

        return _config.TrainConfig(
            name="crane_x7_pi0_fast",
            project_name=self.openpi_config.wandb_project or "crane_x7_vla",
            exp_name=self.openpi_config.experiment_name,
            model=model_config,
            weight_loader=weight_loader,
            lr_schedule=lr_schedule,
            optimizer=optimizer,
            ema_decay=None if self.openpi_config.openpi.use_lora else 0.99,
            freeze_filter=freeze_filter,
            data=data_config_factory,
            assets_base_dir=str(self.openpi_config.output_dir / "assets"),
            checkpoint_base_dir=str(self.openpi_config.output_dir / "checkpoints"),
            seed=self.openpi_config.seed,
            batch_size=self.openpi_config.training.batch_size,
            num_workers=self.openpi_config.data.num_workers,
            num_train_steps=self.openpi_config.training.max_steps,
            log_interval=self.openpi_config.training.log_interval,
            save_interval=self.openpi_config.training.save_interval,
            keep_period=self.openpi_config.openpi.keep_period,
            overwrite=False,
            resume=self.openpi_config.resume_from_checkpoint is not None,
            wandb_enabled=self.openpi_config.use_wandb,
            fsdp_devices=self.openpi_config.openpi.fsdp_devices,
        )

    def _init_train_state(
        self,
        config: _config.TrainConfig,
        init_rng: jax.Array,
        mesh: jax.sharding.Mesh,
        *,
        resume: bool,
    ) -> tuple:
        """Initialize training state."""
        tx = _optimizer.create_optimizer(
            config.optimizer, config.lr_schedule, weight_decay_mask=None
        )

        def init(rng, partial_params=None):
            rng, model_rng = jax.random.split(rng)
            model = config.model.create(model_rng)

            if partial_params is not None:
                graphdef, state = nnx.split(model)
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            params = nnx_utils.state_map(
                params,
                config.freeze_filter,
                lambda p: p.replace(p.value.astype(jnp.bfloat16)),
            )

            return training_utils.TrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=tx,
                opt_state=tx.init(params.filter(config.trainable_filter)),
                ema_decay=config.ema_decay,
                ema_params=None if config.ema_decay is None else params,
            )

        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

        if resume:
            return train_state_shape, state_sharding

        # Load pretrained weights
        partial_params = self._load_weights(
            config.weight_loader, train_state_shape.params.to_pure_dict()
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )

        train_state = jax.jit(
            init,
            donate_argnums=(1,),
            in_shardings=replicated_sharding,
            out_shardings=state_sharding,
        )(init_rng, partial_params)

        return train_state, state_sharding

    def _load_weights(self, loader, params_shape):
        """Load and validate weights."""
        import flax.traverse_util as traverse_util

        loaded_params = loader.load(params_shape)
        at.check_pytree_equality(
            expected=params_shape,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )

        return traverse_util.unflatten_dict({
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        })

    def _train_step(
        self,
        config: _config.TrainConfig,
        rng: jax.Array,
        state: training_utils.TrainState,
        batch: tuple,
    ) -> tuple:
        """Execute a single training step."""
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, observation, actions):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        diff_state = nnx.DiffState(0, config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model, train_rng, observation, actions
        )

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )

        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(
            state, step=state.step + 1, params=new_params, opt_state=new_opt_state
        )

        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        info = {
            "loss": loss,
            "grad_norm": jnp.sqrt(
                sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
            ),
        }
        return new_state, info

    def train(self) -> Dict[str, Any]:
        """
        Execute the OpenPI training loop.

        Returns:
            Dictionary containing training results
        """
        init_logging()
        logger.info(f"Running on: {platform.node()}")

        # Create training config
        train_config = self._create_train_config()

        # Validate batch size
        if train_config.batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Batch size {train_config.batch_size} must be divisible by "
                f"device count {jax.device_count()}"
            )

        # Configure JAX
        jax.config.update(
            "jax_compilation_cache_dir",
            str(Path("~/.cache/jax").expanduser()),
        )

        # Initialize RNG
        rng = jax.random.key(train_config.seed)
        train_rng, init_rng = jax.random.split(rng)

        # Create mesh and sharding
        mesh = sharding.make_mesh(train_config.fsdp_devices)
        self._mesh = mesh
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )

        # Initialize checkpoint manager
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            train_config.checkpoint_dir,
            keep_period=train_config.keep_period,
            overwrite=train_config.overwrite,
            resume=train_config.resume,
        )

        # Initialize W&B
        init_wandb(
            train_config, resuming=resuming, enabled=train_config.wandb_enabled
        )

        # Create data loader
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )
        data_loader = _data_loader.create_data_loader(
            train_config,
            sharding=data_sharding,
            shuffle=True,
        )
        data_iter = iter(data_loader)
        batch = next(data_iter)
        logger.info(
            f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
        )

        # Log sample images
        if train_config.wandb_enabled:
            images_to_log = [
                wandb.Image(
                    np.concatenate(
                        [np.array(img[i]) for img in batch[0].images.values()],
                        axis=1,
                    )
                )
                for i in range(min(5, len(next(iter(batch[0].images.values())))))
            ]
            wandb.log({"camera_views": images_to_log}, step=0)

        # Initialize training state
        train_state, train_state_sharding = self._init_train_state(
            train_config, init_rng, mesh, resume=resuming
        )
        jax.block_until_ready(train_state)
        logger.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
        )

        if resuming:
            train_state = _checkpoints.restore_state(
                checkpoint_manager, train_state, data_loader
            )

        self._train_state = train_state

        # JIT compile training step
        ptrain_step = jax.jit(
            functools.partial(self._train_step, train_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

        # Training loop
        start_step = int(train_state.step)
        pbar = tqdm.tqdm(
            range(start_step, train_config.num_train_steps),
            initial=start_step,
            total=train_config.num_train_steps,
            dynamic_ncols=True,
        )

        infos = []
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)

            if step % train_config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                infos = []

            batch = next(data_iter)

            if (
                step % train_config.save_interval == 0 and step > start_step
            ) or step == train_config.num_train_steps - 1:
                _checkpoints.save_state(
                    checkpoint_manager, train_state, data_loader, step
                )

        self._train_state = train_state
        logger.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()

        return {
            "final_step": int(train_state.step),
            "checkpoint_dir": str(train_config.checkpoint_dir),
        }

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError("Evaluation not yet implemented for OpenPI backend")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Dictionary containing:
                - 'state': Robot state [8]
                - 'image': RGB image [H, W, 3]
            language_instruction: Task instruction

        Returns:
            Predicted action [8] (first action from chunk)
        """
        if self.policy is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # Prepare observation
        state = observation.get("state", np.zeros(8, dtype=np.float32))
        image = observation.get("image")

        if image is None:
            raise ValueError("Image is required for inference")

        # Prepare input dict
        obs_dict = {
            "state": state,
            "image": {"base_0_rgb": image},
            "prompt": language_instruction or self.openpi_config.openpi.default_prompt,
        }

        # Run inference through policy
        result = self.policy.infer(obs_dict)

        # Return first action from chunk (truncated to CRANE-X7 dim)
        actions = result["actions"]
        if actions.ndim > 1:
            actions = actions[0]  # First action from chunk

        # Truncate to CRANE-X7 dimension
        return actions[: self.openpi_config.crane_x7_action_dim]

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._train_state is None:
            raise ValueError("No training state to save")

        # Save using OpenPI checkpoint system
        _checkpoints.save_train_state(
            path / "params",
            self._train_state,
        )

        # Save config
        import json

        config_dict = {
            "model_type": self.openpi_config.openpi.model_type,
            "action_dim": self._action_dim,
            "action_horizon": self._action_horizon,
            "crane_x7_dim": self.openpi_config.crane_x7_action_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint directory
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")

        logger.info(f"Loading OpenPI model from {path}")

        # Create model config
        model_config = self._create_model_config()

        # Initialize model
        rng = jax.random.key(0)
        model = model_config.create(rng)

        # Load parameters
        params_path = path / "params" if (path / "params").exists() else path
        params = _checkpoints.restore_params(params_path)

        # Apply parameters to model
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        self.model = nnx.merge(graphdef, state)

        # Create policy for inference
        from crane_x7_vla.policies.crane_x7_policy import create_crane_x7_policy

        self.policy = create_crane_x7_policy(
            self.model,
            rng=jax.random.key(0),
            is_pytorch=False,
            first_action_only=True,
            default_prompt=self.openpi_config.openpi.default_prompt,
        )

        logger.info("Model loaded successfully")

    @property
    def action_dim(self) -> int:
        """Get the action dimension (CRANE-X7 native dimension)."""
        return self.openpi_config.crane_x7_action_dim

    @property
    def action_horizon(self) -> int:
        """Get the action horizon (OpenPI uses action chunking)."""
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        """Get the expected image size for the model."""
        return self._image_size
