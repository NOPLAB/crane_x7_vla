#!/usr/bin/env python3
"""Setup script for crane_x7_vla package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="crane_x7_vla",
    version="0.1.0",
    author="nop",
    author_email="",
    description="Unified VLA training framework for CRANE-X7 robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NOPLAB/crane_x7_vla",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.5.0",
        "tensorflow>=2.12.0",  # For TFRecord loading
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "openpi": [
            "jax[cuda12_pip]",
            "flax",
            "optax",
        ],
    },
    entry_points={
        "console_scripts": [
            "crane_x7_vla=crane_x7_vla.training.cli:main",
        ],
    },
)
