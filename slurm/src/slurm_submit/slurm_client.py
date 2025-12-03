# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Slurmコマンドラッパーモジュール.

sbatch, squeue, scancelなどのSlurmコマンドをラップする。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.table import Table

from slurm_submit.config import SlurmConfig
from slurm_submit.ssh_client import SSHClient, SSHError
from slurm_submit.utils import parse_job_id

if TYPE_CHECKING:
    pass

console = Console()


class SlurmError(Exception):
    """Slurm操作に関するエラー."""


@dataclass
class JobInfo:
    """Slurmジョブ情報."""

    job_id: str
    name: str
    user: str
    state: str
    partition: str
    time: str
    nodes: int
    nodelist: str = ""

    @property
    def is_running(self) -> bool:
        """ジョブが実行中かどうか."""
        return self.state in ("RUNNING", "R")

    @property
    def is_pending(self) -> bool:
        """ジョブが保留中かどうか."""
        return self.state in ("PENDING", "PD")

    @property
    def is_completed(self) -> bool:
        """ジョブが完了したかどうか."""
        return self.state in ("COMPLETED", "CD")

    @property
    def is_failed(self) -> bool:
        """ジョブが失敗したかどうか."""
        return self.state in ("FAILED", "F", "TIMEOUT", "TO", "CANCELLED", "CA", "NODE_FAIL", "NF")

    @property
    def is_active(self) -> bool:
        """ジョブがアクティブ（実行中または保留中）かどうか."""
        return self.is_running or self.is_pending


class SlurmClient:
    """Slurmコマンドのラッパー."""

    def __init__(self, ssh: SSHClient, config: SlurmConfig):
        """Slurmクライアントを初期化.

        Args:
            ssh: SSH接続クライアント
            config: Slurm設定
        """
        self.ssh = ssh
        self.config = config

    def submit(self, script_path: Path, remote_script_path: str | None = None) -> str:
        """ジョブスクリプトを投下.

        Args:
            script_path: ローカルのジョブスクリプトパス
            remote_script_path: リモートでのスクリプトパス (省略時は自動生成)

        Returns:
            投下されたジョブのID

        Raises:
            SlurmError: ジョブ投下に失敗した場合
        """
        if not script_path.exists():
            raise SlurmError(f"スクリプトファイルが見つかりません: {script_path}")

        # リモートパスを決定
        if remote_script_path is None:
            remote_workdir = str(self.config.remote_workdir).replace("$HOME", "~")
            remote_script_path = f"{remote_workdir}/jobs/{script_path.name}"

        # リモートディレクトリを作成
        remote_dir = str(Path(remote_script_path).parent)
        self.ssh.makedirs(remote_dir)

        # スクリプトをアップロード
        console.print(f"[dim]スクリプトをアップロード中: {remote_script_path}[/dim]")
        self.ssh.upload(script_path, remote_script_path)

        # sbatchで投下
        console.print(f"[dim]ジョブを投下中...[/dim]")
        stdout, stderr, exit_code = self.ssh.execute(f"sbatch {remote_script_path}")

        if exit_code != 0:
            raise SlurmError(f"ジョブ投下に失敗しました: {stderr}")

        # ジョブIDを抽出
        job_id = parse_job_id(stdout)
        if job_id is None:
            raise SlurmError(f"ジョブIDを抽出できませんでした: {stdout}")

        console.print(f"[green]ジョブが投下されました: {job_id}[/green]")
        return job_id

    def submit_script_content(self, script_content: str, script_name: str = "job.sh") -> str:
        """ジョブスクリプト内容を直接投下.

        Args:
            script_content: ジョブスクリプトの内容
            script_name: リモートでのスクリプト名

        Returns:
            投下されたジョブのID

        Raises:
            SlurmError: ジョブ投下に失敗した場合
        """
        remote_workdir = str(self.config.remote_workdir).replace("$HOME", "~")
        remote_script_path = f"{remote_workdir}/jobs/{script_name}"

        # リモートディレクトリを作成
        remote_dir = str(Path(remote_script_path).parent)
        self.ssh.makedirs(remote_dir)

        # スクリプトをアップロード
        console.print(f"[dim]スクリプトをアップロード中: {remote_script_path}[/dim]")
        self.ssh.upload_string(script_content, remote_script_path)

        # 実行権限を付与
        self.ssh.execute(f"chmod +x {remote_script_path}")

        # sbatchで投下
        console.print(f"[dim]ジョブを投下中...[/dim]")
        stdout, stderr, exit_code = self.ssh.execute(f"sbatch {remote_script_path}")

        if exit_code != 0:
            raise SlurmError(f"ジョブ投下に失敗しました: {stderr}")

        # ジョブIDを抽出
        job_id = parse_job_id(stdout)
        if job_id is None:
            raise SlurmError(f"ジョブIDを抽出できませんでした: {stdout}")

        console.print(f"[green]ジョブが投下されました: {job_id}[/green]")
        return job_id

    def status(self, job_id: str | None = None, user: str | None = None) -> list[JobInfo]:
        """ジョブ状態を取得.

        Args:
            job_id: 特定のジョブID (省略時は全ジョブ)
            user: 特定のユーザー (省略時は自分のジョブ)

        Returns:
            ジョブ情報のリスト
        """
        # squeueコマンドを構築
        cmd = "squeue --format='%i|%j|%u|%T|%P|%M|%D|%N' --noheader"
        if job_id:
            cmd += f" --job={job_id}"
        if user:
            cmd += f" --user={user}"
        else:
            # デフォルトは自分のジョブ
            cmd += " --me"

        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code != 0:
            # ジョブが見つからない場合は空リストを返す
            if "Invalid job id" in stderr or "not found" in stderr.lower():
                return []
            raise SlurmError(f"squeue実行に失敗しました: {stderr}")

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 7:
                jobs.append(
                    JobInfo(
                        job_id=parts[0].strip(),
                        name=parts[1].strip(),
                        user=parts[2].strip(),
                        state=parts[3].strip(),
                        partition=parts[4].strip(),
                        time=parts[5].strip(),
                        nodes=int(parts[6].strip()) if parts[6].strip().isdigit() else 1,
                        nodelist=parts[7].strip() if len(parts) > 7 else "",
                    )
                )

        return jobs

    def get_job_state(self, job_id: str) -> str | None:
        """特定ジョブの状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブの状態、見つからない場合はNone
        """
        # sacctを使用して完了済みジョブも含めて状態を取得
        cmd = f"sacct -j {job_id} --format=State --noheader --parsable2 | head -1"
        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code == 0 and stdout.strip():
            # "COMPLETED" や "FAILED" など
            state = stdout.strip().split("|")[0].strip()
            if state:
                return state

        # squeueでも確認
        jobs = self.status(job_id=job_id)
        if jobs:
            return jobs[0].state

        return None

    def cancel(self, job_id: str) -> None:
        """ジョブをキャンセル.

        Args:
            job_id: キャンセルするジョブID

        Raises:
            SlurmError: キャンセルに失敗した場合
        """
        stdout, stderr, exit_code = self.ssh.execute(f"scancel {job_id}")

        if exit_code != 0:
            raise SlurmError(f"ジョブのキャンセルに失敗しました: {stderr}")

        console.print(f"[yellow]ジョブ {job_id} をキャンセルしました[/yellow]")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int | None = None,
        callback: Callable[[JobInfo | None, str | None], None] | None = None,
    ) -> str:
        """ジョブ完了まで待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: ポーリング間隔 (秒)
            timeout: タイムアウト (秒、None=無制限)
            callback: 状態変化時に呼ばれるコールバック (job_info, state) -> None

        Returns:
            最終状態 (COMPLETED, FAILED, TIMEOUT, など)

        Raises:
            SlurmError: タイムアウトした場合
        """
        start_time = time.time()
        last_state: str | None = None

        console.print(f"[dim]ジョブ {job_id} の完了を待機中... (間隔: {poll_interval}秒)[/dim]")

        while True:
            # タイムアウトチェック
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise SlurmError(f"ジョブ {job_id} がタイムアウトしました ({timeout}秒)")

            # 状態を取得
            state = self.get_job_state(job_id)

            # 状態変化をコールバック
            if state != last_state:
                jobs = self.status(job_id=job_id)
                job_info = jobs[0] if jobs else None
                if callback:
                    callback(job_info, state)
                last_state = state

            # 完了判定
            if state is None:
                # ジョブが見つからない = 完了済み
                console.print(f"[green]ジョブ {job_id} が完了しました[/green]")
                return "COMPLETED"

            if state in ("COMPLETED", "CD"):
                console.print(f"[green]ジョブ {job_id} が完了しました[/green]")
                return "COMPLETED"

            if state in ("FAILED", "F"):
                console.print(f"[red]ジョブ {job_id} が失敗しました[/red]")
                return "FAILED"

            if state in ("TIMEOUT", "TO"):
                console.print(f"[red]ジョブ {job_id} がタイムアウトしました[/red]")
                return "TIMEOUT"

            if state in ("CANCELLED", "CA"):
                console.print(f"[yellow]ジョブ {job_id} がキャンセルされました[/yellow]")
                return "CANCELLED"

            if state in ("NODE_FAIL", "NF"):
                console.print(f"[red]ジョブ {job_id} がノード障害で失敗しました[/red]")
                return "NODE_FAIL"

            # 待機
            time.sleep(poll_interval)

    def print_status_table(self, jobs: list[JobInfo]) -> None:
        """ジョブ状態をテーブル形式で表示.

        Args:
            jobs: 表示するジョブ情報のリスト
        """
        if not jobs:
            console.print("[dim]アクティブなジョブはありません[/dim]")
            return

        table = Table(title="Slurm Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("State", style="green")
        table.add_column("Partition", style="blue")
        table.add_column("Time", style="yellow")
        table.add_column("Nodes", style="magenta")

        for job in jobs:
            state_style = "green" if job.is_running else "yellow" if job.is_pending else "red"
            table.add_row(
                job.job_id,
                job.name,
                f"[{state_style}]{job.state}[/{state_style}]",
                job.partition,
                job.time,
                str(job.nodes),
            )

        console.print(table)
