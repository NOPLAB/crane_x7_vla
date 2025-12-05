# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""CLIエントリーポイント.

typerを使用したコマンドラインインターフェース。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from slurm_submit import __version__
from slurm_submit.clients import SlurmError
from slurm_submit.core import console, create_clients, load_settings_with_error

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
    settings = load_settings_with_error(env_file)

    if dry_run:
        console.print("[yellow]ドライランモード: 実際には投下しません[/yellow]")
        console.print(f"\n[bold]スクリプト: {script}[/bold]")
        console.print("-" * 40)
        console.print(script.read_text())
        console.print("-" * 40)
        console.print(f"\n[dim]接続先: {settings.ssh.user}@{settings.ssh.host}[/dim]")
        console.print(f"[dim]リモートワークディレクトリ: {settings.slurm.remote_workdir}[/dim]")
        return

    ssh, slurm = create_clients(settings, password)
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
    settings = load_settings_with_error(env_file)
    ssh, slurm = create_clients(settings, password)

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
    settings = load_settings_with_error(env_file)
    ssh, slurm = create_clients(settings, password)

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
        Optional[int],
        typer.Option(
            "--interval",
            "-i",
            help="状態ポーリング間隔 (秒) [default: SLURM_POLL_INTERVAL or 60]",
        ),
    ] = None,
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
    no_log: Annotated[
        bool,
        typer.Option(
            "--no-log",
            help="ログ表示を無効化",
        ),
    ] = False,
    log_interval: Annotated[
        Optional[int],
        typer.Option(
            "--log-interval",
            "-l",
            help="ログポーリング間隔 (秒) [default: SLURM_LOG_POLL_INTERVAL or 5]",
        ),
    ] = None,
) -> None:
    """ジョブの完了を待機.

    実行時間とログを表示しながらジョブの完了を待ちます。
    ログは5行のスクロールウィンドウで表示されます。

    ポーリング間隔は.envファイルで設定可能:
      SLURM_POLL_INTERVAL=60      # 状態確認間隔 (秒)
      SLURM_LOG_POLL_INTERVAL=5   # ログ確認間隔 (秒)
    """
    settings = load_settings_with_error(env_file)
    ssh, slurm = create_clients(settings, password)

    # 設定からデフォルト値を取得（コマンドライン引数で上書き可能）
    actual_poll_interval = poll_interval if poll_interval is not None else settings.slurm.poll_interval
    actual_log_interval = log_interval if log_interval is not None else settings.slurm.log_poll_interval

    try:
        final_state = slurm.wait_for_completion(
            job_id,
            poll_interval=actual_poll_interval,
            timeout=timeout,
            show_log=not no_log,
            log_poll_interval=actual_log_interval,
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
