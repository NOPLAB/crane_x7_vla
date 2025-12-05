# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""SSH/SCPクライアントモジュール.

paramikoを使用してSSH接続とファイル転送を行う。
"""

from __future__ import annotations

import getpass
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import paramiko

from slurm_submit.config import SSHConfig
from slurm_submit.core.console import console

if TYPE_CHECKING:
    from paramiko.channel import ChannelFile


class SSHError(Exception):
    """SSH操作に関するエラー."""


class SSHClient:
    """SSH/SCP操作を抽象化するクライアント."""

    def __init__(self, config: SSHConfig):
        """SSHクライアントを初期化.

        Args:
            config: SSH接続設定
        """
        self.config = config
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    @property
    def is_connected(self) -> bool:
        """接続状態を確認."""
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    def connect(self, password: str | None = None) -> None:
        """SSH接続を確立.

        Args:
            password: パスワード認証の場合のパスワード。
                      Noneの場合、パスワード認証時は対話的に入力を求める。

        Raises:
            SSHError: 接続に失敗した場合
        """
        if self.is_connected:
            return

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if self.config.auth == "key":
                # 公開鍵認証
                key_path = self.config.key_path
                if key_path is None:
                    raise SSHError("公開鍵認証を使用するには key_path を指定してください")

                expanded_path = key_path.expanduser()
                if not expanded_path.exists():
                    raise SSHError(f"秘密鍵ファイルが見つかりません: {expanded_path}")

                console.print(f"[dim]公開鍵認証で接続中: {self.config.user}@{self.config.host}[/dim]")
                self._client.connect(
                    hostname=self.config.host,
                    port=self.config.port,
                    username=self.config.user,
                    key_filename=str(expanded_path),
                    look_for_keys=False,
                    allow_agent=False,
                )
            else:
                # パスワード認証
                if password is None:
                    password = getpass.getpass(
                        f"Password for {self.config.user}@{self.config.host}: "
                    )

                console.print(f"[dim]パスワード認証で接続中: {self.config.user}@{self.config.host}[/dim]")
                self._client.connect(
                    hostname=self.config.host,
                    port=self.config.port,
                    username=self.config.user,
                    password=password,
                    look_for_keys=False,
                    allow_agent=False,
                )

            console.print("[green]SSH接続成功[/green]")

        except paramiko.AuthenticationException as e:
            self._client = None
            raise SSHError(f"認証に失敗しました: {e}") from e
        except paramiko.SSHException as e:
            self._client = None
            raise SSHError(f"SSH接続に失敗しました: {e}") from e
        except OSError as e:
            self._client = None
            raise SSHError(f"接続エラー: {e}") from e

    def execute(self, command: str, timeout: float | None = None) -> tuple[str, str, int]:
        """リモートでコマンドを実行.

        Args:
            command: 実行するコマンド
            timeout: タイムアウト秒数 (None=無制限)

        Returns:
            (stdout, stderr, exit_code) のタプル

        Raises:
            SSHError: コマンド実行に失敗した場合
        """
        if not self.is_connected or self._client is None:
            raise SSHError("SSH接続が確立されていません")

        try:
            # ログインシェルとして実行（.bashrc等を読み込む）
            wrapped_command = f"bash -l -c {shlex.quote(command)}"
            stdin, stdout, stderr = self._client.exec_command(wrapped_command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            return (
                stdout.read().decode("utf-8", errors="replace"),
                stderr.read().decode("utf-8", errors="replace"),
                exit_code,
            )
        except paramiko.SSHException as e:
            raise SSHError(f"コマンド実行に失敗しました: {e}") from e

    def _get_sftp(self) -> paramiko.SFTPClient:
        """SFTPクライアントを取得."""
        if not self.is_connected or self._client is None:
            raise SSHError("SSH接続が確立されていません")

        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def _expand_remote_path(self, remote_path: str | Path) -> str:
        """リモートパスのチルダを展開.

        SFTPはチルダ(~)を展開しないため、ホームディレクトリの
        絶対パスに変換する。

        Args:
            remote_path: リモートパス

        Returns:
            展開されたパス
        """
        path_str = str(remote_path)
        if path_str.startswith("~/") or path_str == "~":
            # ホームディレクトリを取得
            stdout, stderr, exit_code = self.execute("echo $HOME")
            if exit_code == 0:
                home_dir = stdout.strip()
                path_str = path_str.replace("~", home_dir, 1)
        return path_str

    def upload(self, local_path: Path, remote_path: str | Path) -> None:
        """ファイルをアップロード.

        Args:
            local_path: ローカルファイルパス
            remote_path: リモートファイルパス

        Raises:
            SSHError: アップロードに失敗した場合
        """
        if not local_path.exists():
            raise SSHError(f"ローカルファイルが見つかりません: {local_path}")

        try:
            sftp = self._get_sftp()
            expanded_path = self._expand_remote_path(remote_path)
            sftp.put(str(local_path), expanded_path)
        except (OSError, paramiko.SFTPError) as e:
            raise SSHError(f"ファイルのアップロードに失敗しました: {e}") from e

    def upload_string(self, content: str, remote_path: str | Path) -> None:
        """文字列をリモートファイルとしてアップロード.

        Args:
            content: ファイル内容
            remote_path: リモートファイルパス

        Raises:
            SSHError: アップロードに失敗した場合
        """
        try:
            sftp = self._get_sftp()
            expanded_path = self._expand_remote_path(remote_path)
            with sftp.file(expanded_path, "w") as f:
                f.write(content)
        except (OSError, paramiko.SFTPError) as e:
            raise SSHError(f"ファイルのアップロードに失敗しました: {e}") from e

    def download(self, remote_path: str | Path, local_path: Path) -> None:
        """ファイルをダウンロード.

        Args:
            remote_path: リモートファイルパス
            local_path: ローカルファイルパス

        Raises:
            SSHError: ダウンロードに失敗した場合
        """
        try:
            sftp = self._get_sftp()
            expanded_path = self._expand_remote_path(remote_path)
            # ローカルディレクトリが存在しない場合は作成
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(expanded_path, str(local_path))
        except (OSError, paramiko.SFTPError) as e:
            raise SSHError(f"ファイルのダウンロードに失敗しました: {e}") from e

    def makedirs(self, remote_path: str | Path) -> None:
        """リモートディレクトリを再帰的に作成.

        Args:
            remote_path: 作成するディレクトリパス

        Raises:
            SSHError: ディレクトリ作成に失敗した場合
        """
        # mkdir -p を使用
        stdout, stderr, exit_code = self.execute(f"mkdir -p {remote_path}")
        if exit_code != 0:
            raise SSHError(f"ディレクトリの作成に失敗しました: {stderr}")

    def exists(self, remote_path: str | Path) -> bool:
        """リモートパスの存在を確認.

        Args:
            remote_path: 確認するパス

        Returns:
            パスが存在する場合True
        """
        try:
            sftp = self._get_sftp()
            expanded_path = self._expand_remote_path(remote_path)
            sftp.stat(expanded_path)
            return True
        except FileNotFoundError:
            return False
        except (OSError, paramiko.SFTPError):
            return False

    def close(self) -> None:
        """接続を閉じる."""
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:  # noqa: BLE001
                pass
            self._sftp = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def __enter__(self) -> SSHClient:
        """コンテキストマネージャのエントリ."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """コンテキストマネージャの終了."""
        self.close()
