# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ジョブ監視UI.

ジョブ監視状態の表示を構築する機能を提供する。
"""

from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from slurm_submit.clients.slurm import JobInfo


def get_terminal_log_lines() -> int:
    """ターミナルサイズに基づいてログ表示行数を計算.

    Returns:
        ログ表示に使用する行数
    """
    terminal_height = shutil.get_terminal_size().lines
    # ステータス行(2行) + マージン(4行) を除いた残りをログ表示に使用
    log_lines = max(5, terminal_height - 6)
    return log_lines


@dataclass
class MonitorState:
    """ジョブ監視状態.

    Attributes:
        job_id: ジョブID
        state: ジョブの状態
        slurm_time: Slurmで報告された実行時間
        elapsed_time: 監視開始からの経過時間
        log_lines: 表示するログ行
    """

    job_id: str
    state: str = "UNKNOWN"
    slurm_time: str = "0:00"
    elapsed_time: str = "0:00"
    log_lines: deque = field(default_factory=lambda: deque(maxlen=100))


class MonitorDisplayBuilder:
    """監視表示ビルダー.

    ジョブ監視のRich表示を構築する。
    """

    def __init__(self, state: MonitorState, job_info: "JobInfo | None" = None):
        """ビルダーを初期化.

        Args:
            state: 監視状態
            job_info: ジョブ情報（オプション）
        """
        self.state = state
        self.job_info = job_info

    def build_status_line(self) -> Text:
        """ステータス行を構築.

        Returns:
            ステータス行のText
        """
        state = self.state.state
        if state in ("RUNNING", "R"):
            state_text = Text("[RUNNING] ", style="bold green")
        elif state in ("PENDING", "PD"):
            state_text = Text("[PENDING] ", style="bold yellow")
        else:
            state_text = Text(f"[{state}] ", style="bold blue")

        status_line = Text()
        status_line.append(state_text)
        status_line.append(f"Job {self.state.job_id}")
        if self.job_info:
            status_line.append(f" ({self.job_info.name})")
        return status_line

    def build_time_line(self) -> Text:
        """時間情報行を構築.

        Returns:
            時間情報行のText
        """
        time_line = Text()
        time_line.append("  Elapsed: ", style="dim")
        time_line.append(self.state.elapsed_time, style="cyan")

        if self.state.slurm_time and self.state.slurm_time != "0:00":
            time_line.append(" | Slurm: ", style="dim")
            time_line.append(self.state.slurm_time, style="cyan")

        if self.job_info:
            if self.job_info.nodelist:
                time_line.append(" | Node: ", style="dim")
                time_line.append(self.job_info.nodelist, style="magenta")
            elif self.job_info.reason:
                time_line.append(" | Reason: ", style="dim")
                time_line.append(self.job_info.reason, style="yellow")

        return time_line

    def build_log_panel(self) -> Panel:
        """ログパネルを構築.

        Returns:
            ログ表示のPanel
        """
        log_lines_count = get_terminal_log_lines()
        panel_height = log_lines_count + 2  # ボーダー2行分
        terminal_width = shutil.get_terminal_size().columns
        max_line_width = max(40, terminal_width - 4)

        if self.state.log_lines:
            lines = list(self.state.log_lines)[-log_lines_count:]
            log_text = self._format_log_lines(lines, max_line_width)
            title = f"Log (last {len(lines)} lines)"
        else:
            log_text = Text("Waiting for output...", style="dim italic")
            title = "Log"

        return Panel(
            log_text,
            title=title,
            title_align="left",
            border_style="dim",
            height=panel_height,
        )

    def _format_log_lines(self, lines: list[str], max_width: int) -> Text:
        """ログ行をフォーマット.

        Args:
            lines: ログ行のリスト
            max_width: 最大幅

        Returns:
            フォーマットされたText
        """
        log_text = Text()
        for i, line in enumerate(lines):
            if len(line) > max_width:
                display_line = line[: max_width - 3] + "..."
            else:
                display_line = line
            log_text.append(display_line)
            if i < len(lines) - 1:
                log_text.append("\n")
        return log_text

    def build(self) -> Group:
        """完全な表示を構築.

        Returns:
            全要素を含むGroup
        """
        return Group(
            self.build_status_line(),
            self.build_time_line(),
            self.build_log_panel(),
        )
