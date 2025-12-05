# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ジョブステータステーブル.

ジョブ状態をテーブル形式で表示する機能を提供する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from slurm_submit.core.console import console

if TYPE_CHECKING:
    from slurm_submit.clients.slurm import JobInfo


def print_job_status_table(jobs: list["JobInfo"]) -> None:
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
    table.add_column("Nodelist(Reason)", style="dim")

    for job in jobs:
        state_style = (
            "green"
            if job.is_running
            else "yellow"
            if job.is_pending
            else "red"
        )
        table.add_row(
            job.job_id,
            job.name,
            f"[{state_style}]{job.state}[/{state_style}]",
            job.partition,
            job.time,
            str(job.nodes),
            job.nodelist_or_reason,
        )

    console.print(table)
