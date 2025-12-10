# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Setup script for crane_x7_vla_rl package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="crane_x7_vla_rl",
    version="0.1.0",
    description="VLA Reinforcement Learning for CRANE-X7 Robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="nop",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Deep learning
        "torch>=2.5.1",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        # RL utilities
        "numpy>=1.24.0",
        "tensordict>=0.3.0",
        # Logging
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        # Configuration
        "PyYAML>=6.0",
        # Image processing
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crane-x7-vla-rl=crane_x7_vla_rl.training.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
