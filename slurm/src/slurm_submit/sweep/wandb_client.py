# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""W&B Sweep APIクライアント.

wandbを使用してSweepの作成と管理を行う。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from rich.console import Console

from slurm_submit.config import WandbConfig

console = Console()


class WandbSweepError(Exception):
    """W&B Sweep操作に関するエラー."""


class WandbSweepClient:
    """W&B Sweep APIのラッパー."""

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

        self._initialized = True

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
        entity = entity or self.config.entity
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

    def get_next_run(
        self,
        sweep_id: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any] | None:
        """次の実行パラメータを取得.

        Args:
            sweep_id: SweepのID
            entity: W&Bエンティティ
            project: W&Bプロジェクト

        Returns:
            パラメータ辞書、または取得できない場合はNone
        """
        self._ensure_initialized()

        entity = entity or self.config.entity
        project = project or self.config.project

        if not project:
            raise WandbSweepError("W&Bプロジェクト名が指定されていません")

        try:
            # Sweep Agentとして次のパラメータを取得
            api = self._wandb.Api()
            sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
            sweep = api.sweep(sweep_path)

            # Sweepの状態を確認
            if sweep.state == "FINISHED":
                console.print("[yellow]Sweepは終了しています[/yellow]")
                return None

            # 新しいrunを初期化してパラメータを取得
            run = self._wandb.init(
                entity=entity,
                project=project,
                group=f"sweep-{sweep_id}",
                reinit=True,
            )

            if run is None:
                return None

            # パラメータを取得
            params = dict(run.config)
            run_id = run.id

            # runを終了
            run.finish()

            if not params:
                return None

            return {
                "run_id": run_id,
                "params": params,
            }

        except Exception as e:
            console.print(f"[yellow]パラメータ取得に失敗しました: {e}[/yellow]")
            return None

    def init_sweep_agent_run(
        self,
        sweep_id: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Sweep Agentとしてrunを初期化してパラメータを取得.

        W&B Sweep Agentのようにパラメータを取得する。
        取得後、実際のジョブ実行前にfinishする必要がある。

        Args:
            sweep_id: SweepのID
            entity: W&Bエンティティ
            project: W&Bプロジェクト

        Returns:
            (run_id, params)のタプル、または取得できない場合はNone
        """
        self._ensure_initialized()

        entity = entity or self.config.entity
        project = project or self.config.project

        if not project:
            raise WandbSweepError("W&Bプロジェクト名が指定されていません")

        sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"

        try:
            # Sweep Agentとして動作するため環境変数を設定
            os.environ["WANDB_SWEEP_ID"] = sweep_id
            if entity:
                os.environ["WANDB_ENTITY"] = entity
            if project:
                os.environ["WANDB_PROJECT"] = project

            # Sweep Agentとしてrunを初期化
            run = self._wandb.init(
                entity=entity,
                project=project,
                reinit=True,
            )

            if run is None:
                return None

            # パラメータを取得
            params = dict(run.config)
            run_id = run.id

            console.print(f"[dim]W&B Run: {run_id}[/dim]")
            console.print(f"[dim]パラメータ: {params}[/dim]")

            return run_id, params

        except self._wandb.errors.CommError as e:
            # Sweepが終了している場合など
            console.print(f"[yellow]Sweepエージェント初期化に失敗: {e}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]予期せぬエラー: {e}[/red]")
            return None

    def finish_run(
        self,
        exit_code: int = 0,
    ) -> None:
        """現在のrunを終了.

        Args:
            exit_code: 終了コード (0=成功, 1=失敗)
        """
        self._ensure_initialized()

        try:
            if self._wandb.run is not None:
                self._wandb.finish(exit_code=exit_code)
        except Exception as e:
            console.print(f"[yellow]Run終了に失敗: {e}[/yellow]")

    def report_run_result(
        self,
        sweep_id: str,
        run_id: str,
        status: Literal["finished", "failed", "crashed"],
        entity: str | None = None,
        project: str | None = None,
    ) -> None:
        """実行結果を報告.

        Args:
            sweep_id: SweepのID
            run_id: RunのID
            status: 実行結果
            entity: W&Bエンティティ
            project: W&Bプロジェクト
        """
        self._ensure_initialized()

        entity = entity or self.config.entity
        project = project or self.config.project

        # W&B APIで状態を更新
        try:
            api = self._wandb.Api()
            run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(run_path)

            # 状態に応じたサマリを記録
            if status == "finished":
                console.print(f"[green]Run {run_id} を完了として記録[/green]")
            elif status == "failed":
                console.print(f"[red]Run {run_id} を失敗として記録[/red]")
            else:
                console.print(f"[yellow]Run {run_id} をクラッシュとして記録[/yellow]")

        except Exception as e:
            console.print(f"[yellow]結果報告に失敗しました: {e}[/yellow]")

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

        entity = entity or self.config.entity
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
        entity = entity or self.config.entity or "unknown"
        project = project or self.config.project or "unknown"
        return f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
