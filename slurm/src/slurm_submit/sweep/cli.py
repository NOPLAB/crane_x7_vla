# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep CLIサブコマンド."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel

from slurm_submit.core import console, create_clients, load_settings_with_error
from slurm_submit.sweep.engine import SweepEngine, create_custom_job_generator
from slurm_submit.sweep.wandb_client import WandbSweepClient, WandbSweepError


sweep_app = typer.Typer(
    name="sweep",
    help="W&B Sweepコマンド",
    no_args_is_help=True,
)


@sweep_app.command("start")
def sweep_start(
    config: Annotated[
        Path,
        typer.Argument(
            help="Sweep設定YAMLファイル",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
    max_runs: Annotated[
        int,
        typer.Option(
            "--max-runs",
            "-n",
            help="最大実行数",
        ),
    ] = 10,
    max_concurrent: Annotated[
        Optional[int],
        typer.Option(
            "--max-concurrent",
            "-c",
            help="同時実行ジョブ数の上限 [default: SLURM_MAX_CONCURRENT_JOBS]",
        ),
    ] = None,
    poll_interval: Annotated[
        Optional[int],
        typer.Option(
            "--poll-interval",
            "-i",
            help="状態ポーリング間隔 (秒) [default: SLURM_POLL_INTERVAL]",
        ),
    ] = None,
    log_interval: Annotated[
        Optional[int],
        typer.Option(
            "--log-interval",
            "-l",
            help="ログポーリング間隔 (秒) [default: SLURM_LOG_POLL_INTERVAL]",
        ),
    ] = None,
    template: Annotated[
        Optional[Path],
        typer.Option(
            "--template",
            "-t",
            help="ジョブテンプレートファイル",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="実際にはジョブを投下せず、動作を確認",
        ),
    ] = False,
    password: Annotated[
        Optional[str],
        typer.Option(
            "--password",
            "-p",
            help="SSHパスワード (省略時は対話的に入力)",
            hide_input=True,
        ),
    ] = None,
) -> None:
    """新規Sweepを開始."""
    settings = load_settings_with_error(env_file)

    # 設定からデフォルト値を取得
    actual_poll_interval = poll_interval if poll_interval is not None else settings.slurm.poll_interval
    actual_log_interval = log_interval if log_interval is not None else settings.slurm.log_poll_interval
    actual_max_concurrent = max_concurrent if max_concurrent is not None else settings.slurm.max_concurrent_jobs

    # W&Bクライアントを作成してSweepを作成
    wandb_client = WandbSweepClient(settings.wandb)

    try:
        sweep_id = wandb_client.create_sweep(config)
    except WandbSweepError as e:
        console.print(f"[red]Sweep作成に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e

    # W&Bの実効entityを取得（テンプレートに渡すため）
    effective_entity = wandb_client.effective_entity

    console.print(
        Panel(
            f"Sweep ID: {sweep_id}\n"
            f"Entity: {effective_entity or '(default)'}\n"
            f"URL: {wandb_client.get_sweep_url(sweep_id)}",
            title="Sweep作成完了",
            border_style="green",
        )
    )

    # SSH/Slurmクライアントを作成
    ssh, slurm = create_clients(settings, password)

    try:
        # 追加変数（実効entityを含む）
        extra_vars = {"WANDB_ENTITY": effective_entity} if effective_entity else None

        # ジョブ生成関数を作成
        if template:
            job_generator = create_custom_job_generator(
                template_path=template,
                env_file=env_file,
                extra_vars=extra_vars,
            )
        else:
            job_generator = None

        # Sweepエンジンを作成して実行
        engine = SweepEngine(
            slurm=slurm,
            wandb=wandb_client,
            settings=settings,
            job_generator=job_generator,
        )

        engine.run(
            sweep_id=sweep_id,
            max_runs=max_runs,
            max_concurrent_jobs=actual_max_concurrent,
            poll_interval=actual_poll_interval,
            log_poll_interval=actual_log_interval,
            dry_run=dry_run,
        )

    finally:
        ssh.close()


@sweep_app.command("resume")
def sweep_resume(
    sweep_id: Annotated[
        str,
        typer.Argument(help="再開するSweep ID"),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
    max_runs: Annotated[
        int,
        typer.Option(
            "--max-runs",
            "-n",
            help="最大実行数",
        ),
    ] = 10,
    max_concurrent: Annotated[
        Optional[int],
        typer.Option(
            "--max-concurrent",
            "-c",
            help="同時実行ジョブ数の上限 [default: SLURM_MAX_CONCURRENT_JOBS]",
        ),
    ] = None,
    poll_interval: Annotated[
        Optional[int],
        typer.Option(
            "--poll-interval",
            "-i",
            help="状態ポーリング間隔 (秒) [default: SLURM_POLL_INTERVAL]",
        ),
    ] = None,
    log_interval: Annotated[
        Optional[int],
        typer.Option(
            "--log-interval",
            "-l",
            help="ログポーリング間隔 (秒) [default: SLURM_LOG_POLL_INTERVAL]",
        ),
    ] = None,
    template: Annotated[
        Optional[Path],
        typer.Option(
            "--template",
            "-t",
            help="ジョブテンプレートファイル",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="実際にはジョブを投下せず、動作を確認",
        ),
    ] = False,
    password: Annotated[
        Optional[str],
        typer.Option(
            "--password",
            "-p",
            help="SSHパスワード (省略時は対話的に入力)",
            hide_input=True,
        ),
    ] = None,
) -> None:
    """既存Sweepを再開."""
    settings = load_settings_with_error(env_file)

    # 設定からデフォルト値を取得
    actual_poll_interval = poll_interval if poll_interval is not None else settings.slurm.poll_interval
    actual_log_interval = log_interval if log_interval is not None else settings.slurm.log_poll_interval
    actual_max_concurrent = max_concurrent if max_concurrent is not None else settings.slurm.max_concurrent_jobs

    # W&Bクライアントを作成
    wandb_client = WandbSweepClient(settings.wandb)

    # Sweepの状態を確認
    sweep_state = wandb_client.get_sweep_state(sweep_id)
    if sweep_state == "FINISHED":
        console.print("[yellow]このSweepは既に終了しています[/yellow]")
        raise typer.Exit(0)

    # W&Bの実効entityを取得（テンプレートに渡すため）
    effective_entity = wandb_client.effective_entity

    console.print(
        Panel(
            f"Sweep ID: {sweep_id}\n"
            f"Entity: {effective_entity or '(default)'}\n"
            f"状態: {sweep_state}\n"
            f"URL: {wandb_client.get_sweep_url(sweep_id)}",
            title="Sweep再開",
            border_style="cyan",
        )
    )

    # SSH/Slurmクライアントを作成
    ssh, slurm = create_clients(settings, password)

    try:
        # 追加変数（実効entityを含む）
        extra_vars = {"WANDB_ENTITY": effective_entity} if effective_entity else None

        # ジョブ生成関数を作成
        if template:
            job_generator = create_custom_job_generator(
                template_path=template,
                env_file=env_file,
                extra_vars=extra_vars,
            )
        else:
            job_generator = None

        # Sweepエンジンを作成して実行
        engine = SweepEngine(
            slurm=slurm,
            wandb=wandb_client,
            settings=settings,
            job_generator=job_generator,
        )

        engine.run(
            sweep_id=sweep_id,
            max_runs=max_runs,
            max_concurrent_jobs=actual_max_concurrent,
            poll_interval=actual_poll_interval,
            log_poll_interval=actual_log_interval,
            dry_run=dry_run,
        )

    finally:
        ssh.close()


@sweep_app.command("status")
def sweep_status(
    sweep_id: Annotated[
        str,
        typer.Argument(help="確認するSweep ID"),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
) -> None:
    """Sweepの状態を確認."""
    settings = load_settings_with_error(env_file)

    wandb_client = WandbSweepClient(settings.wandb)
    sweep_state = wandb_client.get_sweep_state(sweep_id)
    sweep_url = wandb_client.get_sweep_url(sweep_id)

    state_style = {
        "RUNNING": "green",
        "FINISHED": "blue",
        "PAUSED": "yellow",
        "UNKNOWN": "red",
    }.get(sweep_state, "white")

    console.print(
        Panel(
            f"Sweep ID: {sweep_id}\n"
            f"状態: [{state_style}]{sweep_state}[/{state_style}]\n"
            f"URL: {sweep_url}",
            title="Sweep状態",
            border_style=state_style,
        )
    )
