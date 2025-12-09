# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import setup, find_packages

setup(
    name="crane_x7_sim",
    version="0.1.0",
    description="CRANE-X7 robot simulation with unified simulator abstraction (lift)",
    author="nop",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "maniskill": [
            "mani-skill>=3.0.0",
            "sapien",
            "torch>=1.10.0",
            "gymnasium>=0.26.0",
        ],
        "genesis": [
            "genesis-world",
            "torch>=2.0.0",
        ],
        "isaacsim": [
            # Isaac Sim dependencies are installed via Omniverse
        ],
    },
    package_data={
        "robot": [
            "assets/*.mjcf",
            "assets/meshes/collision/*.stl",
            "assets/meshes/visual/*.stl",
        ],
    },
    include_package_data=True,
)
