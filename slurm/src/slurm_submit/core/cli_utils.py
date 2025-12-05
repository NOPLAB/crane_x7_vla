# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""CLI共通ユーティリティ."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError

from slurm_submit.config import Settings, load_settings
from slurm_submit.core.console import console

if TYPE_CHECKING:
    from slurm_submit.clients import SlurmClient, SSHClient


def load_settings_with_error(env_file: Path) -> Settings:
    """設定を読み込み、エラー時はわかりやすいメッセージを表示.

    Args:
        env_file: 環境設定ファイルパス

    Returns:
        読み込まれた設定

    Raises:
        typer.Exit: 設定読み込みに失敗した場合
    """
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


def create_clients(
    settings: Settings, password: str | None = None
) -> tuple["SSHClient", "SlurmClient"]:
    """SSH/Slurmクライアントを作成して接続.

    Args:
        settings: 設定
        password: SSHパスワード（省略時は対話的に入力）

    Returns:
        (SSHClient, SlurmClient) のタプル

    Raises:
        typer.Exit: SSH接続に失敗した場合
    """
    from slurm_submit.clients import SlurmClient, SSHClient, SSHError

    ssh = SSHClient(settings.ssh)
    try:
        ssh.connect(password=password)
    except SSHError as e:
        console.print(f"[red]SSH接続に失敗しました: {e}[/red]")
        raise typer.Exit(1) from e

    slurm = SlurmClient(ssh, settings.slurm)
    return ssh, slurm
