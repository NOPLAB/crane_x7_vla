# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweep CLIサブコマンド."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from slurm_submit.config import Settings, load_settings
from slurm_submit.slurm_client import SlurmClient
from slurm_submit.ssh_client import SSHClient, SSHError
from slurm_submit.sweep.engine import SweepEngine, create_custom_job_generator
from slurm_submit.sweep.wandb_client import WandbSweepClient, WandbSweepError

console = Console()

sweep_app = typer.Typer(
    name="sweep",
    help="W&B Sweepコマンド",
    no_args_is_help=True,
)


def _load_settings_with_error(env_file: Path) -> Settings:
    """設定を読み込み、エラー時はわかりやすいメッセージを表示."""
    try:
        return load_settings(env_file)
    except ValidationError as e:
        console.print("[red]設定ファイルの読み込みに失敗しました[/red]")
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            console.print(f"  [yellow]{field}[/yellow]: {msg}")
        console.print(f"\n[dim]設定ファイル: {env_file}[/dim]")
        raise typer.Exit(1) from e
    except FileNotFoundError:
        console.print(f"[red]設定ファイルが見つかりません: {env_file}[/red]")
        raise typer.Exit(1)


def _create_clients(
    settings: Settings, password: str | None = None
) -> tuple[SSHClient, SlurmClient]:
    """SSH/Slurmクライアントを作成して接続."""
    ssh = SSHClient(settings.ssh)
    try:
        ssh.connect(password=password)
    except SSHError as e:
        console.print(f"[red]SSH接続に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e

    slurm = SlurmClient(ssh, settings.slurm)
    return ssh, slurm


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
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            "-i",
            help="ジョブ完了待機のポーリング間隔 (秒)",
        ),
    ] = 300,
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
    settings = _load_settings_with_error(env_file)

    # W&Bクライアントを作成してSweepを作成
    wandb_client = WandbSweepClient(settings.wandb)

    try:
        sweep_id = wandb_client.create_sweep(config)
    except WandbSweepError as e:
        console.print(f"[red]Sweep作成に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(
        Panel(
            f"Sweep ID: {sweep_id}\n"
            f"URL: {wandb_client.get_sweep_url(sweep_id)}",
            title="Sweep作成完了",
            border_style="green",
        )
    )

    # SSH/Slurmクライアントを作成
    ssh, slurm = _create_clients(settings, password)

    try:
        # ジョブ生成関数を作成
        if template:
            job_generator = create_custom_job_generator(
                template_path=template,
                settings=settings,
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
            poll_interval=poll_interval,
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
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            "-i",
            help="ジョブ完了待機のポーリング間隔 (秒)",
        ),
    ] = 300,
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
    settings = _load_settings_with_error(env_file)

    # W&Bクライアントを作成
    wandb_client = WandbSweepClient(settings.wandb)

    # Sweepの状態を確認
    sweep_state = wandb_client.get_sweep_state(sweep_id)
    if sweep_state == "FINISHED":
        console.print("[yellow]このSweepは既に終了しています[/yellow]")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"Sweep ID: {sweep_id}\n"
            f"状態: {sweep_state}\n"
            f"URL: {wandb_client.get_sweep_url(sweep_id)}",
            title="Sweep再開",
            border_style="cyan",
        )
    )

    # SSH/Slurmクライアントを作成
    ssh, slurm = _create_clients(settings, password)

    try:
        # ジョブ生成関数を作成
        if template:
            job_generator = create_custom_job_generator(
                template_path=template,
                settings=settings,
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
            poll_interval=poll_interval,
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
    settings = _load_settings_with_error(env_file)

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
