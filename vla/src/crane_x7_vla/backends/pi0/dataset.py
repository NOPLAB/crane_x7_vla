# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Dataset for Pi0/Pi0.5 Training.

This dataset loader is designed for training Pi0 models with:
- Multi-camera support (base, left_wrist, right_wrist)
- Action chunk prediction with flow matching
- Language instruction tokenization
- Support for both Pi0 (continuous state) and Pi0.5 (discrete state) formats
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import sentencepiece
import torch
from torch.utils.data import IterableDataset

from crane_x7_vla.core.data.image_utils import decode_jpeg, resize_image
from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files
from crane_x7_vla.core.transforms.action_transforms import (
    ActionChunker,
    ActionNormalizer,
    ActionPadder,
)


logger = logging.getLogger(__name__)


class PaliGemmaTokenizer:
    """
    OpenPI準拠のPaliGemma SentencePieceトークナイザ。

    Pi0では状態は連続値として別途処理され、Pi0.5では状態が256ビンに離散化されて
    言語トークンに統合されます。
    """

    # PaliGemmaトークナイザのダウンロードURL
    TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

    def __init__(self, max_len: int = 48):
        """
        PaliGemmaトークナイザを初期化。

        Args:
            max_len: 最大トークン長。Pi0=48、Pi0.5=200
        """
        self._max_len = max_len
        self._tokenizer = None

    @property
    def tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        """遅延ロードでトークナイザを取得。"""
        if self._tokenizer is None:
            tokenizer_path = self._download_tokenizer()
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        return self._tokenizer

    def _download_tokenizer(self) -> Path:
        """PaliGemmaトークナイザをダウンロード。"""
        cache_dir = Path.home() / ".cache" / "crane_x7_vla" / "tokenizers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = cache_dir / "paligemma_tokenizer.model"

        if not tokenizer_path.exists():
            logger.info(f"Downloading PaliGemma tokenizer to {tokenizer_path}")
            import urllib.request

            urllib.request.urlretrieve(self.TOKENIZER_URL, tokenizer_path)
            logger.info("PaliGemma tokenizer downloaded successfully")

        return tokenizer_path

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        プロンプトをトークン化。

        Args:
            prompt: 言語指示
            state: ロボット状態 (Pi0.5の場合のみ使用)。[-1, 1]に正規化済みと想定。

        Returns:
            (token_ids, token_mask) のタプル
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            # Pi0.5フォーマット: 状態を256ビンに離散化して言語トークンに統合
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            discretized_state = np.clip(discretized_state, 0, 255)  # 範囲を保証
            state_str = " ".join(map(str, discretized_state.astype(int)))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self.tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0フォーマット: 状態は連続値として別途処理
            # "\n"を別途トークン化して"start of answer"トークンとする
            tokens = self.tokenizer.encode(cleaned_text, add_bos=True) + self.tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            # パディング
            padding = [0] * (self._max_len - tokens_len)  # 0 = pad token
            mask = [True] * tokens_len + [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
        else:
            # トランケーション
            if tokens_len > self._max_len:
                logger.warning(
                    f"Token length ({tokens_len}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing max_token_len if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)


class CraneX7Pi0Dataset(IterableDataset):
    """
    PyTorch IterableDataset for CRANE-X7 robot data with Pi0/Pi0.5 format.

    This dataset:
    - Loads TFRecord data in episode format
    - Pads 8 DOF actions/states to 32 DOF for Pi0 compatibility
    - Returns action chunks (50 future actions) for flow matching training
    - Supports multi-camera views
    - Provides tokenized language instructions
    """

    # TFRecord feature description for CRANE-X7 format (tfrecord library format)
    FEATURE_DESCRIPTION: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "observation/timestep": "int",
        "action": "float",
        "task/language_instruction": "byte",
        "dataset_name": "byte",
    }

    # Minimal feature description for backward compatibility
    FEATURE_DESCRIPTION_MINIMAL: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "action": "float",
    }

    # Camera names following OpenPI convention
    CAMERA_NAMES: ClassVar[list[str]] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    def __init__(
        self,
        data_root_dir: Path,
        action_horizon: int = 50,
        source_action_dim: int = 8,
        target_action_dim: int = 32,
        max_token_len: int = 48,
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        image_aug: bool = False,
        normalize_actions: bool = True,
        normalization_mode: str = "quantile",
        default_prompt: str = "manipulate objects",
        camera_names: list[str] | None = None,
        discrete_state_input: bool = False,
        overfit_split_ratio: float = 0.0,
        split: str = "train",
        split_seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize Pi0 action chunk dataset.

        Args:
            data_root_dir: Root directory containing TFRecord episode files
            action_horizon: Number of future actions to predict (chunk size)
            source_action_dim: Source action dimension (CRANE-X7 has 8 DOF)
            target_action_dim: Target action dimension (Pi0 uses 32)
            max_token_len: Maximum token length for language instructions
            resize_resolution: Target image resolution (height, width)
            shuffle_buffer_size: Buffer size for shuffling episodes
            train: Whether this is training set
            image_aug: Whether to apply image augmentation
            normalize_actions: Whether to normalize actions
            normalization_mode: Normalization mode ("quantile" or "zscore")
            default_prompt: Default language instruction
            camera_names: List of camera names to use
            discrete_state_input: Whether to use discrete state (Pi0.5)
            overfit_split_ratio: Ratio of steps for overfitting detection
            split: Dataset split ("train" or "overfit")
            split_seed: Random seed for splitting
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.action_horizon = action_horizon
        self.source_action_dim = source_action_dim
        self.target_action_dim = target_action_dim
        self.max_token_len = max_token_len
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.normalize_actions = normalize_actions
        self.normalization_mode = normalization_mode
        self.default_prompt = default_prompt
        self.camera_names = camera_names or self.CAMERA_NAMES[:1]  # Default: only base camera
        self.discrete_state_input = discrete_state_input
        self.overfit_split_ratio = overfit_split_ratio
        self.split = split
        self.split_seed = split_seed
        self.rank = rank
        self.world_size = world_size

        # Initialize transforms
        self.action_padder = ActionPadder(source_action_dim, target_action_dim)
        self.action_chunker = ActionChunker(action_horizon, interpolation="linear")
        self.action_normalizer = ActionNormalizer(mode=normalization_mode)

        # Initialize PaliGemma tokenizer (OpenPI準拠)
        self._paligemma_tokenizer = PaliGemmaTokenizer(max_len=max_token_len)

        # Find all TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()
        logger.info(f"[Rank {self.rank}/{self.world_size}] Found {len(self.tfrecord_files)} TFRecord files")

        # Load or compute statistics
        self.dataset_statistics = self._get_dataset_statistics()

        # Fit normalizer with statistics
        if self.normalize_actions and self.dataset_statistics:
            self._fit_normalizer()

        # Load episodes into memory for efficient chunking
        self.episodes = self._load_episodes()
        logger.info(f"[Rank {self.rank}/{self.world_size}] Loaded {len(self.episodes)} episodes")

        # Compute total number of valid samples
        self.total_samples = sum(max(0, len(ep) - self.action_horizon) for ep in self.episodes)
        logger.info(f"[Rank {self.rank}/{self.world_size}] Total samples: {self.total_samples}")

    @property
    def paligemma_tokenizer(self) -> PaliGemmaTokenizer:
        """PaliGemmaトークナイザを取得。"""
        return self._paligemma_tokenizer

    def _find_tfrecord_files(self) -> list[Path]:
        """Find all TFRecord files in the data directory."""
        return find_tfrecord_files(self.data_root_dir)

    def _get_dataset_statistics(self) -> dict[str, Any]:
        """Load or compute dataset statistics."""
        stats_path = self.data_root_dir / "dataset_statistics.json"
        if stats_path.exists():
            with stats_path.open() as f:
                stats = json.load(f)
                if "crane_x7" in stats:
                    return stats["crane_x7"]
                return stats
        return {}

    def _fit_normalizer(self) -> None:
        """Fit action normalizer with dataset statistics."""
        if not self.dataset_statistics:
            return

        action_stats = self.dataset_statistics.get("action", {})

        if self.normalization_mode == "zscore":
            # Use mean and std for z-score normalization
            if "mean" in action_stats and "std" in action_stats:
                std = np.array(action_stats["std"])
                self.action_normalizer.stats = {
                    "mean": np.array(action_stats["mean"]),
                    "std": np.where(std < 1e-6, 1.0, std),
                }
            else:
                logger.warning("zscore normalization requested but mean/std not in statistics")
        else:
            # Use quantiles for quantile normalization
            if "q01" in action_stats and "q99" in action_stats:
                self.action_normalizer.stats = {
                    "q_low": np.array(action_stats["q01"]),
                    "q_high": np.array(action_stats["q99"]),
                    "range": np.array(action_stats["q99"]) - np.array(action_stats["q01"]),
                }
                self.action_normalizer.stats["range"] = np.where(
                    self.action_normalizer.stats["range"] < 1e-6, 1.0, self.action_normalizer.stats["range"]
                )
            else:
                logger.warning("quantile normalization requested but q01/q99 not in statistics")

    def _load_episodes(self) -> list[list[dict[str, Any]]]:
        """Load all episodes from TFRecord files."""
        episodes = []

        # Shard files for distributed training
        files_to_load = self.tfrecord_files
        if self.world_size > 1:
            files_to_load = [f for i, f in enumerate(self.tfrecord_files) if i % self.world_size == self.rank]

        for tfrecord_file in files_to_load:
            episode = []

            # Try full feature set first, then minimal
            try:
                reader = TFRecordReader([tfrecord_file], feature_spec=self.FEATURE_DESCRIPTION)
                for example in reader:
                    step = self._parse_example(example)
                    episode.append(step)
            except Exception:
                # Fallback to minimal features
                reader = TFRecordReader([tfrecord_file], feature_spec=self.FEATURE_DESCRIPTION_MINIMAL)
                for example in reader:
                    step = self._parse_example(example)
                    episode.append(step)

            if len(episode) >= self.action_horizon:
                episodes.append(episode)

        return episodes

    def _parse_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Parse a single TFRecord example into step format."""
        # Get state (proprio)
        state = example.get("observation/proprio")
        state = np.array(state, dtype=np.float32).reshape(8) if state is not None else np.zeros(8, dtype=np.float32)

        # Get action
        action = example.get("action")
        action = np.array(action, dtype=np.float32).reshape(8) if action is not None else np.zeros(8, dtype=np.float32)

        # Get image bytes
        image_bytes = example.get("observation/image_primary", b"")

        # Get prompt
        prompt = example.get("task/language_instruction", self.default_prompt.encode())
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8", errors="replace")

        return {
            "state": state,
            "action": action,
            "image_bytes": image_bytes,
            "prompt": prompt,
        }

    def _decode_and_process_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode and process image."""
        image = decode_jpeg(image_bytes, channels=3)
        image = resize_image(image, self.resize_resolution)

        if self.image_aug and self.train:
            image = self._apply_image_augmentation(image)

        return image

    def _apply_image_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply image augmentation using NumPy/OpenCV (CPU-only)."""
        h, w = image.shape[:2]

        # Random resized crop (90% of original size)
        crop_h, crop_w = int(h * 0.9), int(w * 0.9)
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        image = image[top : top + crop_h, left : left + crop_w]
        image = cv2.resize(image, (self.resize_resolution[1], self.resize_resolution[0]))

        # Convert to float32 for color augmentation
        image = image.astype(np.float32)

        # Random brightness (±20%)
        brightness_delta = np.random.uniform(-0.2, 0.2) * 255
        image = image + brightness_delta

        # Random contrast (0.8-1.2x)
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = image.mean()
        image = (image - mean) * contrast_factor + mean

        # Random saturation (0.8-1.2x) - convert to HSV
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation_factor = np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Random hue shift (±5% of 180 degrees)
        hue_delta = np.random.uniform(-0.05, 0.05) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image

    def _tokenize_prompt(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize language prompt with optional state integration.

        OpenPI準拠のトークン化:
        - Pi0: 状態は別途連続値として処理 (state=None)
        - Pi0.5: 状態を256ビンに離散化してプロンプトに統合

        Args:
            prompt: Language instruction string
            state: Robot state (normalized to [-1, 1]). Only used for Pi0.5.

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        if self.discrete_state_input and state is not None:
            # Pi0.5: 状態をプロンプトに統合
            return self.paligemma_tokenizer.tokenize(prompt, state)
        else:
            # Pi0: 状態なし
            return self.paligemma_tokenizer.tokenize(prompt, state=None)

    def _should_include_step(self, episode_idx: int, step_idx: int) -> bool:
        """Determine if a step should be included based on split."""
        if self.overfit_split_ratio <= 0:
            return True

        hash_value = hash((episode_idx, step_idx, self.split_seed))
        mod_value = abs(hash_value) % 1000
        threshold = int(self.overfit_split_ratio * 1000)

        if self.split == "overfit":
            return mod_value < threshold
        else:
            return mod_value >= threshold

    def __iter__(self):
        """Iterate over dataset yielding Pi0 format samples."""
        # Create list of (episode_idx, step_idx) pairs
        samples = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode) - self.action_horizon):
                if self._should_include_step(ep_idx, step_idx):
                    samples.append((ep_idx, step_idx))

        # Shuffle if training
        if self.train:
            np.random.shuffle(samples)

        for ep_idx, step_idx in samples:
            episode = self.episodes[ep_idx]
            current_step = episode[step_idx]

            # Decode and process image (use base camera as primary)
            image = self._decode_and_process_image(current_step["image_bytes"])

            # Normalize to [-1, 1] for Pi0
            image_normalized = (image.astype(np.float32) / 127.5) - 1.0

            # Create image dict for all cameras (duplicate if only one camera)
            images = {}
            image_masks = {}
            for camera_name in self.camera_names:
                images[camera_name] = torch.tensor(image_normalized, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
                image_masks[camera_name] = torch.tensor(True)

            # Current state (pad to target dim)
            state = current_step["state"]
            if self.normalize_actions and self.action_normalizer.stats:
                state = self.action_normalizer.normalize(state)
            state_padded = self.action_padder.pad(state)

            # Action chunk (future actions)
            action_chunk = []
            for i in range(self.action_horizon):
                action = episode[step_idx + i]["action"]
                if self.normalize_actions and self.action_normalizer.stats:
                    action = self.action_normalizer.normalize(action)
                action_padded = self.action_padder.pad(action)
                action_chunk.append(action_padded)
            action_chunk = np.stack(action_chunk, axis=0)

            # Tokenize prompt (with state for Pi0.5)
            prompt = current_step.get("prompt", self.default_prompt)
            if self.discrete_state_input:
                # Pi0.5: 状態をトークンに統合
                token_ids, token_mask = self._tokenize_prompt(prompt, state_padded)
                # 状態はトークンに含まれるため、出力はNone
                output_state = None
            else:
                # Pi0: 状態は連続値として別途処理
                token_ids, token_mask = self._tokenize_prompt(prompt, state=None)
                output_state = torch.tensor(state_padded, dtype=torch.float32)

            yield {
                "images": images,
                "image_masks": image_masks,
                "state": output_state,
                "actions": torch.tensor(action_chunk, dtype=torch.float32),
                "lang_tokens": torch.tensor(token_ids, dtype=torch.long),
                "lang_masks": torch.tensor(token_mask, dtype=torch.bool),
                "prompt": prompt,
            }

    def __len__(self) -> int:
        return self.total_samples


def collate_pi0_batch(batch: list[dict]) -> dict[str, Any]:
    """
    Collate function for Pi0 dataset.

    Args:
        batch: List of samples from CraneX7Pi0Dataset

    Returns:
        Collated batch with batched tensors
    """
    # Collate images per camera
    images = {}
    image_masks = {}
    camera_names = list(batch[0]["images"].keys())

    for camera_name in camera_names:
        images[camera_name] = torch.stack([b["images"][camera_name] for b in batch])
        image_masks[camera_name] = torch.stack([b["image_masks"][camera_name] for b in batch])

    # Pi0.5モードではstateがNone (状態は言語トークンに統合)
    states = torch.stack([b["state"] for b in batch]) if batch[0]["state"] is not None else None

    actions = torch.stack([b["actions"] for b in batch])
    lang_tokens = torch.stack([b["lang_tokens"] for b in batch])
    lang_masks = torch.stack([b["lang_masks"] for b in batch])
    prompts = [b["prompt"] for b in batch]

    return {
        "images": images,
        "image_masks": image_masks,
        "state": states,
        "actions": actions,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "prompts": prompts,
    }


# Alias for backward compatibility
CraneX7ActionChunkDataset = CraneX7Pi0Dataset
collate_action_chunk_batch = collate_pi0_batch
