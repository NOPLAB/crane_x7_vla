from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crane_x7_vla'

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
        'numpy',
        'opencv-python',
        'torch',
        'transformers',
        'Pillow',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='VLA-based control package for CRANE-X7 robotic arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_inference_node = crane_x7_vla.vla_inference_node:main',
            'robot_controller = crane_x7_vla.robot_controller:main',
        ],
    },
)
