# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ユーティリティ関数."""

from __future__ import annotations

import re
from datetime import datetime


def generate_timestamp() -> str:
    """タイムスタンプ文字列を生成.

    Returns:
        YYYYMMDD_HHMMSS形式のタイムスタンプ
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_job_name(name: str) -> str:
    """ジョブ名をサニタイズ.

    Slurmジョブ名に使用できない文字を置換する。

    Args:
        name: 元のジョブ名

    Returns:
        サニタイズされたジョブ名
    """
    # 英数字、アンダースコア、ハイフン以外を置換
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # 先頭が数字の場合、プレフィックスを追加
    if sanitized and sanitized[0].isdigit():
        sanitized = "job_" + sanitized
    return sanitized


def parse_job_id(sbatch_output: str) -> str | None:
    """sbatch出力からジョブIDを抽出.

    Args:
        sbatch_output: sbatchコマンドの出力

    Returns:
        ジョブID、または抽出できない場合はNone
    """
    # "Submitted batch job 12345" 形式をパース
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if match:
        return match.group(1)
    return None


def format_duration(seconds: int) -> str:
    """秒数を読みやすい形式にフォーマット.

    Args:
        seconds: 秒数

    Returns:
        "Xh Ym Zs" 形式の文字列
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def truncate_string(s: str, max_length: int = 80, suffix: str = "...") -> str:
    """文字列を最大長で切り詰め.

    Args:
        s: 元の文字列
        max_length: 最大長
        suffix: 切り詰め時に付加するサフィックス

    Returns:
        切り詰められた文字列
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix
