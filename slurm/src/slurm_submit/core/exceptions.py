# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""共通例外クラス."""


class CLIError(Exception):
    """CLI操作に関するエラー."""

    pass


class ConfigError(Exception):
    """設定読み込みエラー."""

    pass
