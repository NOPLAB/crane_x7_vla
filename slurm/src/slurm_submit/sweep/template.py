# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Sweepテンプレート処理.

ジョブテンプレートのプレースホルダ展開を行う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from slurm_submit.config import load_env_vars
from slurm_submit.core.console import console


class TemplateError(Exception):
    """テンプレート処理エラー."""


class JobGenerator(Protocol):
    """ジョブスクリプト生成プロトコル."""

    def __call__(self, sweep_id: str, run_number: int) -> str:
        """Sweep IDからジョブスクリプトを生成.

        Args:
            sweep_id: W&B Sweep ID
            run_number: このSweep内での実行番号

        Returns:
            ジョブスクリプトの内容
        """
        ...


@dataclass
class TemplateContext:
    """テンプレート展開コンテキスト.

    Attributes:
        sweep_id: W&B Sweep ID
        run_number: このSweep内での実行番号
        env_vars: 環境変数辞書
    """

    sweep_id: str
    run_number: int
    env_vars: dict[str, str]


class TemplateProcessor:
    """テンプレートプロセッサ.

    {{KEY}}形式のプレースホルダを展開する。
    """

    # 自動的に提供される変数
    BUILTIN_VARS = {"SWEEP_ID", "RUN_NUMBER"}

    def __init__(
        self,
        template_content: str,
        strict: bool = False,
    ):
        """プロセッサを初期化.

        Args:
            template_content: テンプレート内容
            strict: True の場合、未展開プレースホルダでエラー
        """
        self.template_content = template_content
        self.strict = strict
        self._placeholders = self._extract_placeholders()

    def _extract_placeholders(self) -> set[str]:
        """テンプレート内のプレースホルダを抽出."""
        return set(re.findall(r"\{\{(\w+)\}\}", self.template_content))

    @property
    def placeholders(self) -> set[str]:
        """プレースホルダ一覧."""
        return self._placeholders

    def validate(self, env_vars: dict[str, str]) -> list[str]:
        """テンプレートを検証し、未定義の変数を返す.

        Args:
            env_vars: 環境変数辞書

        Returns:
            未定義変数のリスト
        """
        available = set(env_vars.keys()) | self.BUILTIN_VARS
        return list(self._placeholders - available)

    def render(self, context: TemplateContext) -> str:
        """テンプレートをレンダリング.

        Args:
            context: 展開コンテキスト

        Returns:
            展開されたスクリプト

        Raises:
            TemplateError: strict=Trueで未展開変数がある場合
        """
        script = self.template_content

        # ビルトイン変数を展開
        script = script.replace("{{SWEEP_ID}}", context.sweep_id)
        script = script.replace("{{RUN_NUMBER}}", str(context.run_number))

        # 環境変数を展開
        for key, value in context.env_vars.items():
            script = script.replace(f"{{{{{key}}}}}", value)

        # 未展開プレースホルダをチェック
        remaining = set(re.findall(r"\{\{(\w+)\}\}", script))
        if remaining:
            if self.strict:
                raise TemplateError(
                    f"未展開のプレースホルダがあります: {remaining}"
                )
            console.print(f"[yellow]警告: 未展開のプレースホルダ: {remaining}[/yellow]")
            console.print("[yellow].envファイルにこれらの変数を定義してください[/yellow]")

        return script


def create_template_job_generator(
    template_path: Path,
    env_file: Path | str = ".env",
    strict: bool = False,
    extra_vars: dict[str, str] | None = None,
) -> JobGenerator:
    """テンプレートベースのジョブ生成関数を作成.

    Args:
        template_path: テンプレートファイルパス
        env_file: 環境変数ファイルパス
        strict: 未展開変数でエラーにするか
        extra_vars: 追加の変数（.envより優先）

    Returns:
        ジョブ生成関数

    Raises:
        FileNotFoundError: テンプレートが見つからない場合
        TemplateError: strict=Trueで未定義変数がある場合
    """
    if not template_path.exists():
        raise FileNotFoundError(f"テンプレートファイルが見つかりません: {template_path}")

    template_content = template_path.read_text()

    # .envファイルから環境変数を読み込む
    env_path = Path(env_file)
    if not env_path.exists():
        console.print(f"[yellow]警告: .envファイルが見つかりません: {env_path.absolute()}[/yellow]")

    env_vars = load_env_vars(env_file)

    # 追加変数をマージ（空でない値のみ、.envより優先）
    if extra_vars:
        for key, value in extra_vars.items():
            if value:  # 空でない値のみ追加
                env_vars[key] = value

    console.print(f"[dim].envから {len(env_vars)} 個の変数を読み込みました[/dim]")

    processor = TemplateProcessor(template_content, strict=strict)
    console.print(f"[dim]テンプレート内のプレースホルダ: {processor.placeholders}[/dim]")

    # 事前検証
    undefined = processor.validate(env_vars)
    if undefined:
        msg = f"テンプレートで使用されているが、.envに定義されていない変数: {undefined}"
        if strict:
            raise TemplateError(msg)
        console.print(f"[yellow]警告: {msg}[/yellow]")

    def generator(sweep_id: str, run_number: int) -> str:
        context = TemplateContext(
            sweep_id=sweep_id,
            run_number=run_number,
            env_vars=env_vars,
        )
        return processor.render(context)

    return generator


# 後方互換性のためのエイリアス
def create_custom_job_generator(
    template_path: Path,
    env_file: Path | str = ".env",
    extra_vars: dict[str, str] | None = None,
) -> JobGenerator:
    """カスタムジョブ生成関数を作成（後方互換性のためのエイリアス）.

    新しいコードでは create_template_job_generator() を使用してください。
    """
    return create_template_job_generator(
        template_path, env_file, strict=False, extra_vars=extra_vars
    )
