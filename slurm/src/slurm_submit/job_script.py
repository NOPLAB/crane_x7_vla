# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ジョブスクリプト生成モジュール.

Slurmジョブスクリプトを動的に生成するためのビルダー。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlurmDirectives:
    """Slurm #SBATCHディレクティブ."""

    job_name: str
    partition: str = "gpu"
    nodes: int = 1
    cpus_per_task: int = 8
    mem: str = "32G"
    gpus: int = 1
    gpu_type: str | None = None
    time: str = "24:00:00"
    output: str | None = None
    error: str | None = None
    container: str | None = None
    account: str | None = None
    qos: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def to_sbatch_lines(self) -> list[str]:
        """#SBATCHディレクティブ行を生成.

        Returns:
            #SBATCHディレクティブ行のリスト
        """
        lines = []
        lines.append(f"#SBATCH --job-name={self.job_name}")
        lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --nodes={self.nodes}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        lines.append(f"#SBATCH --mem={self.mem}")
        lines.append(f"#SBATCH --time={self.time}")

        if self.gpus > 0:
            if self.gpu_type:
                lines.append(f"#SBATCH --gres=gpu:{self.gpu_type}:{self.gpus}")
            else:
                lines.append(f"#SBATCH --gres=gpu:{self.gpus}")

        if self.output:
            lines.append(f"#SBATCH --output={self.output}")
        else:
            lines.append(f"#SBATCH --output={self.job_name}_%j.out")

        if self.error:
            lines.append(f"#SBATCH --error={self.error}")
        else:
            lines.append(f"#SBATCH --error={self.job_name}_%j.err")

        if self.container:
            lines.append(f"#SBATCH --container={self.container}")

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        for key, value in self.extra.items():
            lines.append(f"#SBATCH --{key}={value}")

        return lines


class JobScriptBuilder:
    """ジョブスクリプトのビルダー."""

    def __init__(self, directives: SlurmDirectives | None = None):
        """ビルダーを初期化.

        Args:
            directives: Slurmディレクティブ (省略時はデフォルト)
        """
        self.directives = directives
        self._shebang = "#!/bin/bash"
        self._env_vars: dict[str, str] = {}
        self._setup_commands: list[str] = []
        self._commands: list[str] = []
        self._comments: list[str] = []

    def set_shebang(self, shebang: str) -> JobScriptBuilder:
        """シェバングを設定.

        Args:
            shebang: シェバング行

        Returns:
            self (メソッドチェーン用)
        """
        self._shebang = shebang
        return self

    def add_comment(self, comment: str) -> JobScriptBuilder:
        """コメントを追加.

        Args:
            comment: コメント文字列 (#は自動付与)

        Returns:
            self (メソッドチェーン用)
        """
        self._comments.append(comment)
        return self

    def add_env(self, name: str, value: str) -> JobScriptBuilder:
        """環境変数を追加.

        Args:
            name: 環境変数名
            value: 値

        Returns:
            self (メソッドチェーン用)
        """
        self._env_vars[name] = value
        return self

    def add_envs(self, env_vars: dict[str, str]) -> JobScriptBuilder:
        """複数の環境変数を追加.

        Args:
            env_vars: 環境変数の辞書

        Returns:
            self (メソッドチェーン用)
        """
        self._env_vars.update(env_vars)
        return self

    def add_setup(self, command: str) -> JobScriptBuilder:
        """セットアップコマンドを追加.

        セットアップコマンドはメインコマンドの前に実行される。

        Args:
            command: 実行コマンド

        Returns:
            self (メソッドチェーン用)
        """
        self._setup_commands.append(command)
        return self

    def add_command(self, command: str) -> JobScriptBuilder:
        """実行コマンドを追加.

        Args:
            command: 実行コマンド

        Returns:
            self (メソッドチェーン用)
        """
        self._commands.append(command)
        return self

    def add_commands(self, commands: list[str]) -> JobScriptBuilder:
        """複数の実行コマンドを追加.

        Args:
            commands: 実行コマンドのリスト

        Returns:
            self (メソッドチェーン用)
        """
        self._commands.extend(commands)
        return self

    def build(self) -> str:
        """ジョブスクリプト文字列を生成.

        Returns:
            ジョブスクリプトの内容
        """
        lines: list[str] = []

        # シェバング
        lines.append(self._shebang)
        lines.append("")

        # SBATCHディレクティブ
        if self.directives:
            lines.extend(self.directives.to_sbatch_lines())
            lines.append("")

        # コメント
        if self._comments:
            for comment in self._comments:
                lines.append(f"# {comment}")
            lines.append("")

        # 環境変数
        if self._env_vars:
            lines.append("# Environment variables")
            for name, value in self._env_vars.items():
                lines.append(f"export {name}={self._quote_value(value)}")
            lines.append("")

        # セットアップコマンド
        if self._setup_commands:
            lines.append("# Setup")
            lines.extend(self._setup_commands)
            lines.append("")

        # メインコマンド
        if self._commands:
            lines.append("# Main commands")
            lines.extend(self._commands)
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """ジョブスクリプトをファイルに保存.

        Args:
            path: 保存先パス
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())

    @staticmethod
    def _quote_value(value: str) -> str:
        """値をクォート.

        Args:
            value: 値

        Returns:
            必要に応じてクォートされた値
        """
        # スペースや特殊文字が含まれる場合はダブルクォート
        if " " in value or "$" in value or '"' in value or "'" in value:
            # ダブルクォート内のダブルクォートをエスケープ
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value


def create_simple_job_script(
    job_name: str,
    command: str,
    partition: str = "gpu",
    cpus: int = 8,
    mem: str = "32G",
    gpus: int = 1,
    time: str = "24:00:00",
    env_vars: dict[str, str] | None = None,
    setup_commands: list[str] | None = None,
    **extra_sbatch: Any,
) -> str:
    """シンプルなジョブスクリプトを作成.

    Args:
        job_name: ジョブ名
        command: 実行コマンド
        partition: Slurmパーティション
        cpus: CPU数
        mem: メモリ
        gpus: GPU数
        time: 実行時間
        env_vars: 環境変数
        setup_commands: セットアップコマンド
        **extra_sbatch: 追加のSBATCHディレクティブ

    Returns:
        ジョブスクリプトの内容
    """
    directives = SlurmDirectives(
        job_name=job_name,
        partition=partition,
        cpus_per_task=cpus,
        mem=mem,
        gpus=gpus,
        time=time,
        extra=extra_sbatch,
    )

    builder = JobScriptBuilder(directives)

    if env_vars:
        builder.add_envs(env_vars)

    if setup_commands:
        for cmd in setup_commands:
            builder.add_setup(cmd)

    builder.add_command(command)

    return builder.build()
