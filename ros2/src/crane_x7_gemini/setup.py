# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crane_x7_gemini'

setup(
    name=package_name,
    version='0.0.0',
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
        'google-genai',
        'opencv-python',
        'numpy',
        'pillow',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Google Gemini Robotics-ER API integration for CRANE-X7',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gemini_node = crane_x7_gemini.gemini_node:main',
            'object_detector = crane_x7_gemini.object_detector:main',
            'trajectory_planner = crane_x7_gemini.trajectory_planner:main',
            'prompt_publisher = crane_x7_gemini.prompt_publisher:main',
        ],
    },
)
