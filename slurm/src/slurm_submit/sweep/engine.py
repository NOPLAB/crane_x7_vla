# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep実行エンジン.

W&B Sweepのパラメータを取得し、Slurmジョブとして実行するエンジン。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from slurm_submit.config import Settings
from slurm_submit.job_script import JobScriptBuilder, SlurmDirectives
from slurm_submit.slurm_client import SlurmClient, SlurmError
from slurm_submit.sweep.wandb_client import WandbSweepClient, WandbSweepError
from slurm_submit.utils import generate_timestamp

console = Console()


class JobGenerator(Protocol):
    """ジョブスクリプト生成プロトコル."""

    def __call__(self, params: dict[str, Any], run_id: str) -> str:
        """パラメータからジョブスクリプトを生成.

        Args:
            params: Sweepから取得したパラメータ
            run_id: W&B Run ID

        Returns:
            ジョブスクリプトの内容
        """
        ...


class SweepEngine:
    """Sweep実行エンジン."""

    def __init__(
        self,
        slurm: SlurmClient,
        wandb: WandbSweepClient,
        settings: Settings,
        job_generator: JobGenerator | None = None,
    ):
        """エンジンを初期化.

        Args:
            slurm: Slurmクライアント
            wandb: W&B Sweepクライアント
            settings: 設定
            job_generator: ジョブスクリプト生成関数 (省略時はデフォルト生成)
        """
        self.slurm = slurm
        self.wandb = wandb
        self.settings = settings
        self.job_generator = job_generator or self._default_job_generator

        # 状態ディレクトリ
        self._state_dir = Path(".sweep_state")

    def _default_job_generator(self, params: dict[str, Any], run_id: str) -> str:
        """デフォルトのジョブスクリプト生成.

        Args:
            params: Sweepパラメータ
            run_id: W&B Run ID

        Returns:
            ジョブスクリプトの内容
        """
        slurm_config = self.settings.slurm
        training_config = self.settings.training

        directives = SlurmDirectives(
            job_name=f"{slurm_config.job_prefix}_sweep_{run_id[:8]}",
            partition=slurm_config.partition,
            cpus_per_task=slurm_config.cpus,
            mem=slurm_config.mem,
            gpus=slurm_config.gpus,
            gpu_type=slurm_config.gpu_type,
            time=slurm_config.time,
            container=slurm_config.container,
        )

        builder = JobScriptBuilder(directives)

        # 環境変数
        builder.add_env("PYTHONUNBUFFERED", "1")
        builder.add_env("WANDB_RUN_ID", run_id)
        if self.settings.wandb.api_key:
            builder.add_env("WANDB_API_KEY", self.settings.wandb.api_key)

        # パラメータをJSON形式で環境変数に設定
        builder.add_env("SWEEP_PARAMS", json.dumps(params))

        # セットアップ
        builder.add_setup(f"cd {slurm_config.remote_workdir}")
        builder.add_setup("echo 'Starting sweep job...'")
        builder.add_setup(f"echo 'Run ID: {run_id}'")
        builder.add_setup(f"echo 'Parameters: {json.dumps(params)}'")

        # メインコマンド (ユーザーがカスタマイズすべき)
        builder.add_comment("TODO: 実際のトレーニングコマンドに置き換えてください")
        builder.add_command("echo 'Please customize the job generator for your training command'")
        builder.add_command("echo $SWEEP_PARAMS")

        return builder.build()

    def _save_state(self, sweep_id: str, run_id: str, job_id: str) -> None:
        """Sweep状態を保存.

        Args:
            sweep_id: Sweep ID
            run_id: W&B Run ID
            job_id: Slurm Job ID
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self._state_dir / f"{sweep_id}.json"

        # 既存の状態を読み込み
        state: dict[str, Any] = {}
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

        # 新しい実行を追加
        if "runs" not in state:
            state["runs"] = []

        state["runs"].append({
            "run_id": run_id,
            "job_id": job_id,
            "timestamp": generate_timestamp(),
        })

        # 保存
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def run(
        self,
        sweep_id: str,
        max_runs: int = 10,
        poll_interval: int = 300,
        dry_run: bool = False,
    ) -> None:
        """Sweepを実行.

        Args:
            sweep_id: SweepのID
            max_runs: 最大実行数
            poll_interval: ジョブ完了待機のポーリング間隔 (秒)
            dry_run: ドライランモード
        """
        console.print(
            Panel(
                f"Sweep: {sweep_id}\n"
                f"最大実行数: {max_runs}\n"
                f"ポーリング間隔: {poll_interval}秒\n"
                f"URL: {self.wandb.get_sweep_url(sweep_id)}",
                title="Sweep実行開始",
                border_style="green",
            )
        )

        if dry_run:
            console.print("[yellow]ドライランモード: 実際にはジョブを投下しません[/yellow]")

        completed_runs = 0

        while completed_runs < max_runs:
            console.print(f"\n[bold]Run {completed_runs + 1}/{max_runs}[/bold]")

            # Sweepの状態を確認
            sweep_state = self.wandb.get_sweep_state(sweep_id)
            if sweep_state == "FINISHED":
                console.print("[green]Sweepが終了しました[/green]")
                break

            # 次のパラメータを取得
            console.print("[dim]次のパラメータを取得中...[/dim]")
            result = self.wandb.init_sweep_agent_run(sweep_id)

            if result is None:
                console.print("[yellow]パラメータを取得できませんでした。Sweepが終了した可能性があります。[/yellow]")
                break

            run_id, params = result
            console.print(f"[cyan]Run ID: {run_id}[/cyan]")
            console.print(f"[dim]パラメータ: {json.dumps(params, indent=2)}[/dim]")

            # ジョブスクリプトを生成
            script_content = self.job_generator(params, run_id)

            if dry_run:
                console.print("\n[yellow]生成されるジョブスクリプト:[/yellow]")
                console.print("-" * 40)
                console.print(script_content)
                console.print("-" * 40)
                self.wandb.finish_run(exit_code=0)
                completed_runs += 1
                continue

            # W&Bのrunを一旦終了 (Slurmジョブ内で再開する)
            self.wandb.finish_run(exit_code=0)

            # ジョブを投下
            try:
                script_name = f"sweep_{run_id[:8]}_{generate_timestamp()}.sh"
                job_id = self.slurm.submit_script_content(script_content, script_name)

                # 状態を保存
                self._save_state(sweep_id, run_id, job_id)

                console.print(f"[green]ジョブを投下しました: {job_id}[/green]")

            except SlurmError as e:
                console.print(f"[red]ジョブ投下に失敗しました: {e}[/red]")
                self.wandb.report_run_result(sweep_id, run_id, "crashed")
                continue

            # ジョブ完了を待機
            console.print(f"[dim]ジョブ {job_id} の完了を待機中...[/dim]")

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Job {job_id} 実行中...", total=None)

                    def update_progress(job_info: Any, state: str | None) -> None:
                        if state:
                            progress.update(task, description=f"Job {job_id}: {state}")

                    final_state = self.slurm.wait_for_completion(
                        job_id,
                        poll_interval=poll_interval,
                        callback=update_progress,
                    )

                # 結果を報告
                if final_state == "COMPLETED":
                    self.wandb.report_run_result(sweep_id, run_id, "finished")
                else:
                    self.wandb.report_run_result(sweep_id, run_id, "failed")

            except SlurmError as e:
                console.print(f"[red]ジョブ待機に失敗しました: {e}[/red]")
                self.wandb.report_run_result(sweep_id, run_id, "crashed")

            completed_runs += 1

        console.print(f"\n[bold green]Sweep完了: {completed_runs}回実行しました[/bold green]")


def create_custom_job_generator(
    template_path: Path | None = None,
    command_template: str | None = None,
    settings: Settings | None = None,
) -> JobGenerator:
    """カスタムジョブ生成関数を作成.

    Args:
        template_path: ジョブテンプレートファイルのパス
        command_template: コマンドテンプレート文字列
        settings: 設定

    Returns:
        ジョブ生成関数
    """
    if template_path and template_path.exists():
        template_content = template_path.read_text()
    else:
        template_content = None

    def generator(params: dict[str, Any], run_id: str) -> str:
        if template_content:
            # テンプレートのプレースホルダを置換
            script = template_content
            script = script.replace("{{RUN_ID}}", run_id)
            script = script.replace("{{PARAMS_JSON}}", json.dumps(params))

            # 個別パラメータも置換
            for key, value in params.items():
                script = script.replace(f"{{{{{key}}}}}", str(value))

            return script

        elif command_template and settings:
            slurm_config = settings.slurm

            directives = SlurmDirectives(
                job_name=f"{slurm_config.job_prefix}_sweep_{run_id[:8]}",
                partition=slurm_config.partition,
                cpus_per_task=slurm_config.cpus,
                mem=slurm_config.mem,
                gpus=slurm_config.gpus,
                gpu_type=slurm_config.gpu_type,
                time=slurm_config.time,
                container=slurm_config.container,
            )

            builder = JobScriptBuilder(directives)
            builder.add_env("PYTHONUNBUFFERED", "1")
            builder.add_env("WANDB_RUN_ID", run_id)

            # コマンドテンプレートを展開
            cmd = command_template
            cmd = cmd.replace("{{RUN_ID}}", run_id)
            for key, value in params.items():
                cmd = cmd.replace(f"{{{{{key}}}}}", str(value))

            builder.add_command(cmd)
            return builder.build()

        else:
            raise WandbSweepError(
                "template_path または command_template を指定してください"
            )

    return generator
