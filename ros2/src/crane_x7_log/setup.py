# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crane_x7_log'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'tensorflow',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Data logging package for CRANE-X7 in NPZ/TFRecord format for VLA fine-tuning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_logger = crane_x7_log.data_logger:main',
            'tfrecord_writer = crane_x7_log.tfrecord_writer:main',
        ],
    },
)
