# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import setup, find_packages

setup(
    name="crane_x7_maniskill",
    version="0.1.0",
    description="CRANE-X7 robot simulation for ManiSkill",
    author="nop",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "mani-skill>=3.0.0",
        "sapien",
        "torch>=1.10.0",
        "gymnasium>=0.26.0",
        "numpy>=1.20.0",
    ],
    package_data={
        "crane_x7": [
            "*.xml",
            "meshes/collision/*.stl",
            "meshes/visual/*.stl",
        ],
    },
    include_package_data=True,
)
