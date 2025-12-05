# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""W&B Sweep APIクライアント.

wandbを使用してSweepの作成と状態管理を行う。

新しいアーキテクチャでは、パラメータの取得とRunの作成は
Slurmジョブ内のwandb.agent()が行います。
このクライアントはSweepの作成と状態確認のみを担当します。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from slurm_submit.config import WandbConfig

console = Console()


class WandbSweepError(Exception):
    """W&B Sweep操作に関するエラー."""


class WandbSweepClient:
    """W&B Sweep APIのラッパー.

    Sweepの作成と状態確認のみを担当。
    パラメータ取得とRun作成はSlurmジョブ内のwandb.agent()が行う。
    """

    def __init__(self, config: WandbConfig):
        """クライアントを初期化.

        Args:
            config: W&B設定
        """
        self.config = config
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """W&B APIが初期化されていることを確認."""
        if self._initialized:
            return

        # APIキーを環境変数に設定
        if self.config.api_key:
            os.environ["WANDB_API_KEY"] = self.config.api_key

        # wandbをインポート (遅延インポート)
        try:
            import wandb
            self._wandb = wandb
        except ImportError as e:
            raise WandbSweepError(
                "wandbがインストールされていません。pip install wandb を実行してください"
            ) from e

        # デフォルトentityを取得（configで未設定の場合）
        if not self.config.entity:
            try:
                api = wandb.Api()
                self._default_entity = api.default_entity
                console.print(f"[dim]W&Bデフォルトentity: {self._default_entity}[/dim]")
            except Exception:
                self._default_entity = None
        else:
            self._default_entity = None

        self._initialized = True

    @property
    def effective_entity(self) -> str | None:
        """実効entity（config設定値またはデフォルト）を取得."""
        self._ensure_initialized()
        return self.config.entity or self._default_entity

    def create_sweep(
        self,
        config_path: Path,
        entity: str | None = None,
        project: str | None = None,
    ) -> str:
        """新規Sweepを作成.

        Args:
            config_path: Sweep設定YAMLファイルのパス
            entity: W&Bエンティティ (省略時は設定から)
            project: W&Bプロジェクト (省略時は設定から)

        Returns:
            作成されたSweepのID

        Raises:
            WandbSweepError: Sweep作成に失敗した場合
        """
        self._ensure_initialized()

        if not config_path.exists():
            raise WandbSweepError(f"Sweep設定ファイルが見つかりません: {config_path}")

        # 設定を読み込み
        try:
            with open(config_path) as f:
                sweep_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise WandbSweepError(f"Sweep設定の読み込みに失敗しました: {e}") from e

        # エンティティとプロジェクトを決定
        entity = entity or self.effective_entity
        project = project or self.config.project

        if not project:
            raise WandbSweepError("W&Bプロジェクト名が指定されていません")

        try:
            sweep_id = self._wandb.sweep(
                sweep=sweep_config,
                entity=entity,
                project=project,
            )
            console.print(f"[green]Sweepを作成しました: {sweep_id}[/green]")
            return sweep_id
        except Exception as e:
            raise WandbSweepError(f"Sweep作成に失敗しました: {e}") from e

    def get_sweep_state(
        self,
        sweep_id: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> str:
        """Sweepの状態を取得.

        Args:
            sweep_id: SweepのID
            entity: W&Bエンティティ
            project: W&Bプロジェクト

        Returns:
            Sweepの状態 (RUNNING, FINISHED, など)
        """
        self._ensure_initialized()

        entity = entity or self.effective_entity
        project = project or self.config.project

        try:
            api = self._wandb.Api()
            sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
            sweep = api.sweep(sweep_path)
            return sweep.state
        except Exception as e:
            console.print(f"[yellow]Sweep状態の取得に失敗: {e}[/yellow]")
            return "UNKNOWN"

    def get_sweep_url(
        self,
        sweep_id: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> str:
        """SweepのURLを取得.

        Args:
            sweep_id: SweepのID
            entity: W&Bエンティティ
            project: W&Bプロジェクト

        Returns:
            SweepのURL
        """
        self._ensure_initialized()
        entity = entity or self.effective_entity or "unknown"
        project = project or self.config.project or "unknown"
        return f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
