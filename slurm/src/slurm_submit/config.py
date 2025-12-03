# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""設定管理モジュール.

.envファイルから設定を読み込み、pydanticでバリデーションを行う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SSHConfig(BaseModel):
    """SSH接続設定."""

    host: str = Field(..., description="SSHホスト名またはIPアドレス")
    user: str = Field(..., description="SSHユーザー名")
    port: int = Field(default=22, description="SSHポート")
    auth: Literal["password", "key"] = Field(
        default="password", description="認証方式 (password または key)"
    )
    key_path: Path | None = Field(default=None, description="SSH秘密鍵ファイルパス")

    @field_validator("key_path", mode="before")
    @classmethod
    def expand_home(cls, v: str | Path | None) -> Path | None:
        """~をホームディレクトリに展開."""
        if v is None or v == "":
            return None
        return Path(v).expanduser()

    @model_validator(mode="after")
    def validate_key_auth(self) -> SSHConfig:
        """key認証の場合、key_pathが必須."""
        if self.auth == "key" and self.key_path is None:
            raise ValueError("key認証を使用する場合、key_pathを指定してください")
        return self


class SlurmConfig(BaseModel):
    """Slurm設定."""

    remote_workdir: Path = Field(..., description="リモートサーバー上の作業ディレクトリ")
    partition: str = Field(default="gpu", description="Slurmパーティション名")
    gpus: int = Field(default=1, ge=0, description="GPU数")
    gpu_type: str | None = Field(default=None, description="GPUタイプ (例: a100, v100)")
    time: str = Field(default="24:00:00", description="実行時間 (HH:MM:SS形式)")
    mem: str = Field(default="32G", description="メモリ (例: 32G)")
    cpus: int = Field(default=8, ge=1, description="CPU数")
    job_prefix: str = Field(default="job", description="ジョブ名のプレフィックス")
    container: str | None = Field(default=None, description="コンテナイメージ (Pyxis/Enroot用)")

    @field_validator("remote_workdir", mode="before")
    @classmethod
    def expand_home_workdir(cls, v: str | Path) -> Path:
        """~をホームディレクトリに展開."""
        return Path(str(v).replace("~", "$HOME"))

    @field_validator("gpu_type", "container", mode="before")
    @classmethod
    def empty_to_none(cls, v: str | None) -> str | None:
        """空文字列をNoneに変換."""
        if v == "":
            return None
        return v

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """時間フォーマットを検証."""
        parts = v.split(":")
        if len(parts) == 3:
            try:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                if 0 <= m < 60 and 0 <= s < 60:
                    return v
            except ValueError:
                pass
        # D-HH:MM:SS形式も許容
        if "-" in v:
            return v
        raise ValueError(f"無効な時間フォーマット: {v} (HH:MM:SS形式を使用)")


class WandbConfig(BaseModel):
    """Weights & Biases設定."""

    api_key: str | None = Field(default=None, description="W&B APIキー")
    entity: str | None = Field(default=None, description="W&Bエンティティ (チーム名/ユーザー名)")
    project: str | None = Field(default=None, description="W&Bプロジェクト名")

    @field_validator("api_key", "entity", "project", mode="before")
    @classmethod
    def empty_to_none(cls, v: str | None) -> str | None:
        """空文字列をNoneに変換."""
        if v == "":
            return None
        return v


class TrainingConfig(BaseModel):
    """トレーニング設定 (Sweep用)."""

    data_root: Path = Field(default=Path("/data"), description="データディレクトリ")
    output_dir: Path = Field(default=Path("/output"), description="出力ディレクトリ")
    num_epochs: int = Field(default=10, ge=1, description="エポック数")
    save_interval: int = Field(default=500, ge=1, description="チェックポイント保存間隔")
    eval_interval: int = Field(default=100, ge=1, description="評価間隔")


class Settings(BaseSettings):
    """全体設定.

    .envファイルから環境変数を読み込み、各設定クラスにマッピング。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SSH設定
    slurm_ssh_host: str = Field(alias="SLURM_SSH_HOST")
    slurm_ssh_user: str = Field(alias="SLURM_SSH_USER")
    slurm_ssh_port: int = Field(default=22, alias="SLURM_SSH_PORT")
    slurm_ssh_auth: Literal["password", "key"] = Field(default="password", alias="SLURM_SSH_AUTH")
    slurm_ssh_key: str | None = Field(default=None, alias="SLURM_SSH_KEY")

    # Slurm設定
    slurm_remote_workdir: str = Field(alias="SLURM_REMOTE_WORKDIR")
    slurm_partition: str = Field(default="gpu", alias="SLURM_PARTITION")
    slurm_gpus: int = Field(default=1, alias="SLURM_GPUS")
    slurm_gpu_type: str | None = Field(default=None, alias="SLURM_GPU_TYPE")
    slurm_time: str = Field(default="24:00:00", alias="SLURM_TIME")
    slurm_mem: str = Field(default="32G", alias="SLURM_MEM")
    slurm_cpus: int = Field(default=8, alias="SLURM_CPUS")
    slurm_job_prefix: str = Field(default="job", alias="SLURM_JOB_PREFIX")
    slurm_container: str | None = Field(default=None, alias="SLURM_CONTAINER")

    # W&B設定
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")
    wandb_entity: str | None = Field(default=None, alias="WANDB_ENTITY")
    wandb_project: str | None = Field(default=None, alias="WANDB_PROJECT")

    # トレーニング設定
    data_root: str = Field(default="/data", alias="DATA_ROOT")
    output_dir: str = Field(default="/output", alias="OUTPUT_DIR")
    num_epochs: int = Field(default=10, alias="NUM_EPOCHS")
    save_interval: int = Field(default=500, alias="SAVE_INTERVAL")
    eval_interval: int = Field(default=100, alias="EVAL_INTERVAL")

    @property
    def ssh(self) -> SSHConfig:
        """SSH設定を取得."""
        return SSHConfig(
            host=self.slurm_ssh_host,
            user=self.slurm_ssh_user,
            port=self.slurm_ssh_port,
            auth=self.slurm_ssh_auth,
            key_path=Path(self.slurm_ssh_key) if self.slurm_ssh_key else None,
        )

    @property
    def slurm(self) -> SlurmConfig:
        """Slurm設定を取得."""
        return SlurmConfig(
            remote_workdir=Path(self.slurm_remote_workdir),
            partition=self.slurm_partition,
            gpus=self.slurm_gpus,
            gpu_type=self.slurm_gpu_type if self.slurm_gpu_type else None,
            time=self.slurm_time,
            mem=self.slurm_mem,
            cpus=self.slurm_cpus,
            job_prefix=self.slurm_job_prefix,
            container=self.slurm_container if self.slurm_container else None,
        )

    @property
    def wandb(self) -> WandbConfig:
        """W&B設定を取得."""
        return WandbConfig(
            api_key=self.wandb_api_key if self.wandb_api_key else None,
            entity=self.wandb_entity if self.wandb_entity else None,
            project=self.wandb_project if self.wandb_project else None,
        )

    @property
    def training(self) -> TrainingConfig:
        """トレーニング設定を取得."""
        return TrainingConfig(
            data_root=Path(self.data_root),
            output_dir=Path(self.output_dir),
            num_epochs=self.num_epochs,
            save_interval=self.save_interval,
            eval_interval=self.eval_interval,
        )


def load_settings(env_file: Path | str = ".env") -> Settings:
    """設定を読み込む.

    Args:
        env_file: .envファイルのパス

    Returns:
        Settings: 読み込まれた設定

    Raises:
        ValidationError: 設定のバリデーションに失敗した場合
    """
    return Settings(_env_file=str(env_file))
