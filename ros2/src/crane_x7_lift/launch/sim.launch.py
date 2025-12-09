# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Launch file for lift simulation node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for lift simulation."""

    pkg_share = get_package_share_directory('crane_x7_lift')
    default_config = os.path.join(pkg_share, 'config', 'lift_config.yaml')

    declare_simulator = DeclareLaunchArgument(
        'simulator',
        default_value='maniskill',
        description='Simulator backend (maniskill, genesis, isaacsim)'
    )

    declare_env_id = DeclareLaunchArgument(
        'env_id',
        default_value='PickPlace-CRANE-X7',
        description='Environment ID'
    )

    declare_backend = DeclareLaunchArgument(
        'backend',
        default_value='gpu',
        description='Simulation backend (cpu, gpu)'
    )

    declare_render_mode = DeclareLaunchArgument(
        'render_mode',
        default_value='rgb_array',
        description='Render mode (rgb_array, human, none)'
    )

    declare_sim_rate = DeclareLaunchArgument(
        'sim_rate',
        default_value='30.0',
        description='Simulation rate in Hz'
    )

    declare_auto_reset = DeclareLaunchArgument(
        'auto_reset',
        default_value='true',
        description='Auto-reset environment on episode end'
    )

    lift_sim_node = Node(
        package='crane_x7_lift',
        executable='lift_sim_node',
        name='lift_sim_node',
        output='screen',
        parameters=[
            default_config,
            {
                'simulator': LaunchConfiguration('simulator'),
                'env_id': LaunchConfiguration('env_id'),
                'backend': LaunchConfiguration('backend'),
                'render_mode': LaunchConfiguration('render_mode'),
                'sim_rate': LaunchConfiguration('sim_rate'),
                'auto_reset': LaunchConfiguration('auto_reset'),
            }
        ],
    )

    return LaunchDescription([
        declare_simulator,
        declare_env_id,
        declare_backend,
        declare_render_mode,
        declare_sim_rate,
        declare_auto_reset,
        lift_sim_node,
    ])
