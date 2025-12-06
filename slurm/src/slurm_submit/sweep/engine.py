# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep実行エンジン.

W&B Sweepを作成し、Slurmジョブとして実行するエンジン。

新しいアーキテクチャでは、パラメータの取得とRunの作成は
Slurmジョブ内のwandb.agent()が行います。これにより、
RunがSweepに正しく関連付けられます。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.panel import Panel

from slurm_submit.clients import SlurmClient, SlurmError
from slurm_submit.config import Settings
from slurm_submit.core.console import console
from slurm_submit.job_script import JobScriptBuilder, SlurmDirectives
from slurm_submit.sweep.template import JobGenerator
from slurm_submit.sweep.wandb_client import WandbSweepClient
from slurm_submit.utils import generate_timestamp


class SweepEngine:
    """Sweep実行エンジン.

    新しいアーキテクチャでは、パラメータの取得とRunの作成は
    Slurmジョブ内のwandb.agent()が行います。
    エンジンはSweepの作成とジョブの投下・監視のみを担当します。
    """

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

    def _default_job_generator(self, sweep_id: str, run_number: int) -> str:
        """デフォルトのジョブスクリプト生成.

        Note: 実際の使用ではテンプレートベースのジェネレータを使用することを推奨。
        このデフォルト実装は crane_x7_vla.training.cli の agent コマンドを呼び出します。

        Args:
            sweep_id: W&B Sweep ID
            run_number: このSweep内での実行番号

        Returns:
            ジョブスクリプトの内容
        """
        slurm_config = self.settings.slurm
        wandb_config = self.settings.wandb

        directives = SlurmDirectives(
            job_name=f"{slurm_config.job_prefix}_sweep_{sweep_id[:8]}_{run_number:03d}",
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
        if wandb_config.api_key:
            builder.add_env("WANDB_API_KEY", wandb_config.api_key)
        if wandb_config.entity:
            builder.add_env("WANDB_ENTITY", wandb_config.entity)
        if wandb_config.project:
            builder.add_env("WANDB_PROJECT", wandb_config.project)

        # セットアップ
        builder.add_setup(f"cd {slurm_config.remote_workdir}")
        builder.add_setup("echo '=== Starting Sweep Agent Job ==='")
        builder.add_setup(f"echo 'Sweep ID: {sweep_id}'")
        builder.add_setup(f"echo 'Run Number: {run_number}'")

        # agent コマンドを実行
        # wandb.agent()が自動的にパラメータを取得し、RunをSweepに関連付ける
        builder.add_comment("crane_x7_vla agent コマンドでSweepからパラメータを取得")
        builder.add_command(
            f"python -m crane_x7_vla.training.cli agent openvla "
            f"--sweep-id {sweep_id} "
            f"--entity {wandb_config.entity or ''} "
            f"--project {wandb_config.project or 'crane_x7'} "
            f"--data-root /data "
            f"--output-dir /output "
            f"--count 1"
        )

        return builder.build()

    def _save_state(self, sweep_id: str, run_number: int, job_id: str) -> None:
        """Sweep状態を保存.

        Args:
            sweep_id: Sweep ID
            run_number: 実行番号
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
        if "jobs" not in state:
            state["jobs"] = []

        state["jobs"].append({
            "run_number": run_number,
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
        max_concurrent_jobs: int = 1,
        poll_interval: int = 60,
        log_poll_interval: int = 5,
        dry_run: bool = False,
    ) -> None:
        """Sweepを実行.

        新しいアーキテクチャでは、各ジョブがwandb.agent()を使用して
        パラメータを取得します。エンジンはジョブの投下と監視のみを行います。

        Args:
            sweep_id: SweepのID
            max_runs: 最大実行数
            max_concurrent_jobs: 同時実行ジョブ数の上限 (1の場合は逐次実行)
            poll_interval: 状態ポーリング間隔 (秒)
            log_poll_interval: ログポーリング間隔 (秒)
            dry_run: ドライランモード
        """
        console.print(
            Panel(
                f"Sweep: {sweep_id}\n"
                f"最大実行数: {max_runs}\n"
                f"同時実行数: {max_concurrent_jobs}\n"
                f"ポーリング間隔: {poll_interval}秒\n"
                f"URL: {self.wandb.get_sweep_url(sweep_id)}",
                title="Sweep実行開始",
                border_style="green",
            )
        )

        if dry_run:
            console.print("[yellow]ドライランモード: 実際にはジョブを投下しません[/yellow]")

        if max_concurrent_jobs == 1:
            # 逐次実行モード（リアルタイムログ表示あり）
            self._run_sequential(
                sweep_id=sweep_id,
                max_runs=max_runs,
                poll_interval=poll_interval,
                log_poll_interval=log_poll_interval,
                dry_run=dry_run,
            )
        else:
            # 並列実行モード（リアルタイムログ表示なし）
            self._run_parallel(
                sweep_id=sweep_id,
                max_runs=max_runs,
                max_concurrent_jobs=max_concurrent_jobs,
                poll_interval=poll_interval,
                dry_run=dry_run,
            )

    def _run_sequential(
        self,
        sweep_id: str,
        max_runs: int,
        poll_interval: int,
        log_poll_interval: int,
        dry_run: bool,
    ) -> None:
        """逐次実行モード（リアルタイムログ表示あり）."""
        completed_runs = 0

        while completed_runs < max_runs:
            run_number = completed_runs + 1
            console.print(f"\n[bold]Run {run_number}/{max_runs}[/bold]")

            # Sweepの状態を確認
            sweep_state = self.wandb.get_sweep_state(sweep_id)
            if sweep_state == "FINISHED":
                console.print("[green]Sweepが終了しました[/green]")
                break

            # ジョブスクリプトを生成
            console.print("[dim]ジョブスクリプトを生成中...[/dim]")
            script_content = self.job_generator(sweep_id, run_number)

            if dry_run:
                console.print("\n[yellow]生成されるジョブスクリプト:[/yellow]")
                console.print("-" * 40)
                console.print(script_content)
                console.print("-" * 40)
                completed_runs += 1
                continue

            # ジョブを投下
            try:
                script_name = f"sweep_{sweep_id[:8]}_{run_number:03d}_{generate_timestamp()}.sh"
                job_id = self.slurm.submit_script_content(script_content, script_name)

                # 状態を保存
                self._save_state(sweep_id, run_number, job_id)

            except SlurmError as e:
                console.print(f"[red]ジョブ投下に失敗しました: {e}[/red]")
                continue

            # ジョブ完了を待機
            try:
                final_state = self.slurm.wait_for_completion(
                    job_id,
                    poll_interval=poll_interval,
                    log_poll_interval=log_poll_interval,
                )

                # 結果をログ
                if final_state == "COMPLETED":
                    console.print(f"[green]✓ ジョブ {job_id} が完了しました[/green]")
                else:
                    console.print(f"[red]✗ ジョブ {job_id} が失敗しました: {final_state}[/red]")

            except SlurmError as e:
                console.print(f"[red]ジョブ待機に失敗しました: {e}[/red]")

            completed_runs += 1

        console.print(f"\n[bold green]Sweep完了: {completed_runs}回実行しました[/bold green]")

    def _run_parallel(
        self,
        sweep_id: str,
        max_runs: int,
        max_concurrent_jobs: int,
        poll_interval: int,
        dry_run: bool,
    ) -> None:
        """並列実行モード（リアルタイムログ表示なし）."""
        running_jobs: dict[str, int] = {}  # {job_id: run_number}
        completed_runs = 0
        submitted_runs = 0

        console.print(f"[cyan]並列実行モード: 最大{max_concurrent_jobs}ジョブ同時実行[/cyan]")

        while completed_runs < max_runs:
            # Sweepの状態を確認
            sweep_state = self.wandb.get_sweep_state(sweep_id)
            if sweep_state == "FINISHED":
                console.print("[green]Sweepが終了しました[/green]")
                break

            # 完了したジョブを確認
            for job_id in list(running_jobs.keys()):
                state = self.slurm.get_job_state(job_id)
                if state not in ("RUNNING", "PENDING", "R", "PD"):
                    run_number = running_jobs[job_id]
                    del running_jobs[job_id]
                    completed_runs += 1
                    # 結果をログ
                    if state == "COMPLETED":
                        console.print(
                            f"[green]✓ Run {run_number} (Job {job_id}) 完了[/green]"
                        )
                    else:
                        console.print(
                            f"[red]✗ Run {run_number} (Job {job_id}) 失敗: {state}[/red]"
                        )

            # 新しいジョブを投下（上限まで）
            while (
                len(running_jobs) < max_concurrent_jobs
                and submitted_runs < max_runs
            ):
                submitted_runs += 1
                run_number = submitted_runs

                console.print(f"\n[bold]Run {run_number}/{max_runs} 投下中...[/bold]")

                script_content = self.job_generator(sweep_id, run_number)

                if dry_run:
                    console.print(f"[yellow]ドライラン: Run {run_number}[/yellow]")
                    completed_runs += 1
                    continue

                try:
                    script_name = (
                        f"sweep_{sweep_id[:8]}_{run_number:03d}_{generate_timestamp()}.sh"
                    )
                    job_id = self.slurm.submit_script_content(script_content, script_name)
                    running_jobs[job_id] = run_number
                    self._save_state(sweep_id, run_number, job_id)
                    console.print(
                        f"[dim]Run {run_number}: Job {job_id} を投下しました[/dim]"
                    )
                except SlurmError as e:
                    console.print(f"[red]Run {run_number} の投下に失敗: {e}[/red]")

            # 実行中ジョブの状態を表示
            if running_jobs and not dry_run:
                console.print(
                    f"[dim]実行中: {len(running_jobs)}件, "
                    f"完了: {completed_runs}/{max_runs}件[/dim]"
                )
                time.sleep(poll_interval)

        # 残りのジョブを待機
        if running_jobs:
            console.print(f"\n[cyan]残り{len(running_jobs)}件のジョブの完了を待機中...[/cyan]")

        while running_jobs:
            for job_id in list(running_jobs.keys()):
                state = self.slurm.get_job_state(job_id)
                if state not in ("RUNNING", "PENDING", "R", "PD"):
                    run_number = running_jobs[job_id]
                    del running_jobs[job_id]
                    completed_runs += 1
                    if state == "COMPLETED":
                        console.print(
                            f"[green]✓ Run {run_number} (Job {job_id}) 完了[/green]"
                        )
                    else:
                        console.print(
                            f"[red]✗ Run {run_number} (Job {job_id}) 失敗: {state}[/red]"
                        )
            if running_jobs:
                console.print(
                    f"[dim]残り: {len(running_jobs)}件[/dim]"
                )
                time.sleep(poll_interval)

        console.print(f"\n[bold green]Sweep完了: {completed_runs}回実行しました[/bold green]")


# 後方互換性のためのエイリアス
from slurm_submit.sweep.template import create_custom_job_generator  # noqa: E402, F401
