# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crane_x7_lift'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python-headless',
    ],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='noplab90@gmail.com',
    description='Unified simulator abstraction (lift) for CRANE-X7 VLA inference and data collection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lift_sim_node = crane_x7_lift.lift_sim_node:main',
        ],
    },
)
