# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""CLIエントリーポイント.

typerを使用したコマンドラインインターフェース。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from slurm_submit import __version__
from slurm_submit.config import Settings, load_settings
from slurm_submit.slurm_client import SlurmClient, SlurmError
from slurm_submit.ssh_client import SSHClient, SSHError

console = Console()

app = typer.Typer(
    name="slurm-submit",
    help="SSH経由でSlurmクラスターにジョブを投下するツール",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """バージョン表示コールバック."""
    if value:
        console.print(f"slurm-submit version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="バージョンを表示",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """SSH経由でSlurmクラスターにジョブを投下するツール."""
    pass


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
        console.print("[dim].env.templateをコピーして.envを作成してください[/dim]")
        raise typer.Exit(1)


def _create_clients(settings: Settings, password: str | None = None) -> tuple[SSHClient, SlurmClient]:
    """SSH/Slurmクライアントを作成して接続."""
    ssh = SSHClient(settings.ssh)
    try:
        ssh.connect(password=password)
    except SSHError as e:
        console.print(f"[red]SSH接続に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e

    slurm = SlurmClient(ssh, settings.slurm)
    return ssh, slurm


@app.command()
def submit(
    script: Annotated[
        Path,
        typer.Argument(
            help="ジョブスクリプトのパス",
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
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="実際には投下せず、スクリプト内容を表示",
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
    """ジョブスクリプトをSlurmクラスターに投下."""
    settings = _load_settings_with_error(env_file)

    if dry_run:
        console.print("[yellow]ドライランモード: 実際には投下しません[/yellow]")
        console.print(f"\n[bold]スクリプト: {script}[/bold]")
        console.print("-" * 40)
        console.print(script.read_text())
        console.print("-" * 40)
        console.print(f"\n[dim]接続先: {settings.ssh.user}@{settings.ssh.host}[/dim]")
        console.print(f"[dim]リモートワークディレクトリ: {settings.slurm.remote_workdir}[/dim]")
        return

    ssh, slurm = _create_clients(settings, password)
    try:
        job_id = slurm.submit(script)
        console.print(f"\n[bold green]ジョブID: {job_id}[/bold green]")
    except SlurmError as e:
        console.print(f"[red]ジョブ投下に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e
    finally:
        ssh.close()


@app.command()
def status(
    job_id: Annotated[
        Optional[str],
        typer.Argument(help="特定のジョブID (省略時は自分の全ジョブ)"),
    ] = None,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
    all_users: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="全ユーザーのジョブを表示",
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
    """ジョブキューの状態を確認."""
    settings = _load_settings_with_error(env_file)
    ssh, slurm = _create_clients(settings, password)

    try:
        user = None if all_users else settings.ssh.user
        jobs = slurm.status(job_id=job_id, user=user)
        slurm.print_status_table(jobs)
    except SlurmError as e:
        console.print(f"[red]状態取得に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e
    finally:
        ssh.close()


@app.command()
def cancel(
    job_id: Annotated[
        str,
        typer.Argument(help="キャンセルするジョブID"),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
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
    """ジョブをキャンセル."""
    settings = _load_settings_with_error(env_file)
    ssh, slurm = _create_clients(settings, password)

    try:
        slurm.cancel(job_id)
    except SlurmError as e:
        console.print(f"[red]キャンセルに失敗しました: {e}[/red]")
        raise typer.Exit(1) from e
    finally:
        ssh.close()


@app.command()
def wait(
    job_id: Annotated[
        str,
        typer.Argument(help="待機するジョブID"),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env",
            "-e",
            help="環境設定ファイル (.env)",
        ),
    ] = Path(".env"),
    poll_interval: Annotated[
        int,
        typer.Option(
            "--interval",
            "-i",
            help="ポーリング間隔 (秒)",
        ),
    ] = 60,
    timeout: Annotated[
        Optional[int],
        typer.Option(
            "--timeout",
            "-t",
            help="タイムアウト (秒)",
        ),
    ] = None,
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
    """ジョブの完了を待機."""
    settings = _load_settings_with_error(env_file)
    ssh, slurm = _create_clients(settings, password)

    try:
        final_state = slurm.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if final_state in ("COMPLETED",):
            raise typer.Exit(0)
        else:
            raise typer.Exit(1)
    except SlurmError as e:
        console.print(f"[red]待機に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e
    finally:
        ssh.close()


# Sweepサブコマンドをインポート (遅延インポートで循環参照を回避)
def _add_sweep_commands() -> None:
    """Sweepサブコマンドを追加."""
    try:
        from slurm_submit.sweep.cli import sweep_app
        app.add_typer(sweep_app, name="sweep")
    except ImportError:
        # Sweep機能が利用できない場合は無視
        pass


_add_sweep_commands()


if __name__ == "__main__":
    app()
