# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""slurm-submit: SSH経由でSlurmクラスターにジョブを投下するツール."""

__version__ = "0.1.0"

from slurm_submit.config import Settings, SSHConfig, SlurmConfig, WandbConfig
from slurm_submit.job_script import JobScriptBuilder, SlurmDirectives
from slurm_submit.slurm_client import JobInfo, SlurmClient
from slurm_submit.ssh_client import SSHClient

__all__ = [
    "Settings",
    "SSHConfig",
    "SlurmConfig",
    "WandbConfig",
    "SSHClient",
    "SlurmClient",
    "JobInfo",
    "JobScriptBuilder",
    "SlurmDirectives",
]
