# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from crane_x7_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_robot_description_string(use_d435):
    """Generate robot description URDF string by running xacro directly."""
    xacro_path = os.path.join(
        get_package_share_directory('crane_x7_description'),
        'urdf',
        'crane_x7.urdf.xacro')

    # Build xacro command
    xacro_cmd = [
        'xacro', xacro_path,
        'port_name:=/dev/ttyUSB0',
        'baudrate:=3000000',
        'timeout_seconds:=1.0',
        'manipulator_config_file_path:=',
        'manipulator_links_file_path:=',
        'use_gazebo:=true',
        f'use_d435:={use_d435}',
        'gz_control_config_package:=crane_x7_control',
        'gz_control_config_file_path:=config/crane_x7_controllers.yaml',
    ]

    # Run xacro and get URDF string
    result = subprocess.run(xacro_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'xacro failed: {result.stderr}')

    base_description = result.stdout

    # If D435 is enabled, add Ignition Gazebo camera sensors
    if use_d435 == 'true':
        # Ignition Gazebo camera sensor configuration for D435
        # These sensors attach to the camera_link created by realsense2_description
        # The Sensors plugin is loaded at world level in pick_and_place.sdf
        d435_gazebo_sensors = '''
  <!-- D435 RGB Camera Sensor for Ignition Gazebo -->
  <gazebo reference="camera_link">
    <sensor name="d435_color" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
      <topic>camera/color/image_raw</topic>
      <camera>
        <horizontal_fov>1.211</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>RGB_INT8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
    </sensor>
  </gazebo>

  <!-- D435 Depth Camera Sensor for Ignition Gazebo -->
  <gazebo reference="camera_link">
    <sensor name="d435_depth" type="depth_camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <visualize>false</visualize>
      <topic>camera/depth/image_raw</topic>
      <camera>
        <horizontal_fov>1.518</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R_FLOAT32</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </camera>
    </sensor>
  </gazebo>
'''
        # Insert sensors before closing </robot> tag
        description = base_description.replace('</robot>', d435_gazebo_sensors + '</robot>')
    else:
        description = base_description

    return description


def launch_setup(context, *args, **kwargs):
    """Setup launch configuration based on arguments."""
    use_d435_config = LaunchConfiguration('use_d435')
    use_d435 = use_d435_config.perform(context)

    # Generate robot description with sensors
    description = generate_robot_description_string(use_d435)

    # Environment variables for Gazebo
    env = {
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(
            get_package_share_directory('crane_x7_description'))
    }

    world_file = os.path.join(
        get_package_share_directory('crane_x7_sim_gazebo'), 'worlds', 'pick_and_place.sdf')
    gui_config = os.path.join(
        get_package_share_directory('crane_x7_sim_gazebo'), 'gui', 'gui.config')

    # Start Ignition Gazebo
    ign_gazebo = ExecuteProcess(
        cmd=['ign gazebo -r', world_file, '--gui-config', gui_config],
        output='screen',
        additional_env=env,
        shell=True
    )

    # Spawn robot entity
    ignition_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-name', 'crane_x7',
                   '-z', '1.015',
                   '-allow_renaming', 'true'],
    )

    # MoveIt2 configuration
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('crane_x7_moveit_config'),
            '/launch/run_move_group.launch.py']),
        launch_arguments={'loaded_description': description}.items()
    )

    # Spawn controllers
    spawn_joint_state_controller = ExecuteProcess(
        cmd=['ros2 run controller_manager spawner joint_state_controller'],
        shell=True,
        output='screen',
    )

    spawn_arm_controller = ExecuteProcess(
        cmd=['ros2 run controller_manager spawner crane_x7_arm_controller'],
        shell=True,
        output='screen',
    )

    spawn_gripper_controller = ExecuteProcess(
        cmd=['ros2 run controller_manager spawner crane_x7_gripper_controller'],
        shell=True,
        output='screen',
    )

    # Bridge configuration - always bridge clock
    bridge_args = ['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']

    # Add camera bridges if D435 is enabled
    if use_d435 == 'true':
        bridge_args.extend([
            '/camera/color/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ])

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen'
    )

    return [
        ign_gazebo,
        ignition_spawn_entity,
        move_group,
        spawn_joint_state_controller,
        spawn_arm_controller,
        spawn_gripper_controller,
        bridge,
    ]


def generate_launch_description():
    # Declare launch arguments
    use_d435_arg = DeclareLaunchArgument(
        'use_d435',
        default_value='false',
        description='Set to true to use RealSense D435 camera model with Ignition Gazebo sensors'
    )

    return LaunchDescription([
        use_d435_arg,
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=launch_setup),
    ])
