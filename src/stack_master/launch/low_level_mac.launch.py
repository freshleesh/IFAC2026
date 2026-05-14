"""macOS 실차 HW + sensor 통합 low_level launch.

camera_lidar_mac.launch.py (camera + livox, use_rviz 옵션) +
vesc_all.launch.xml 의 vesc 4 노드를 한 번에 띄움. macOS 실차 bringup 전용.

기동 노드:
  - opencv_cam (see3cam 1280×720, headless=false 일 때만 / macOS GUI 세션 필요)
  - livox_ros_driver2 (Mid-360) + RViz (use_rviz 따라)
  - vesc_driver_node                 (시리얼 통신)
  - vesc_ackermann/ackermann_to_vesc_node  (ackermann → 모터/서보)
  - vesc_ackermann/vesc_to_odom_node       (VESC state → odom + TF)
  - vesc_driver_mac/teleop_joy             (joystick → ackermann)
  - joy_mac/joy_node                       (macOS GameController → /joy)

사용:
  # 기본 (camera + livox + rviz + vesc + joy 전부)
  ros2 launch stack_master low_level_mac.launch.py

  # rviz 끄고 (sensor + HW 만)
  ros2 launch stack_master low_level_mac.launch.py use_rviz:=false

  # SSH (headless — camera skip, rviz 도 끄려면 use_rviz:=false 도 같이)
  ros2 launch stack_master low_level_mac.launch.py headless:=true use_rviz:=false

  # joy 별도 터미널에서
  ros2 launch stack_master low_level_mac.launch.py joy:=false
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ── Args ──
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    camera_index = LaunchConfiguration('camera_index')
    camera_width = LaunchConfiguration('camera_width')
    camera_height = LaunchConfiguration('camera_height')
    camera_fps = LaunchConfiguration('camera_fps')
    vesc_config = LaunchConfiguration('vesc_config')
    use_joy = LaunchConfiguration('joy')

    # ── Sensor bringup (camera + livox + optional rviz; DYLD_LIBRARY_PATH 포함) ──
    cam_lidar_share = get_package_share_directory('camera_lidar_calibration')
    sensor_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cam_lidar_share, 'launch', 'camera_lidar_mac.launch.py')
        ),
        launch_arguments={
            'headless': headless,
            'use_rviz': use_rviz,
            'camera_index': camera_index,
            'camera_width': camera_width,
            'camera_height': camera_height,
            'camera_fps': camera_fps,
        }.items(),
    )

    # ── VESC HW (시리얼 통신 + ackermann 변환 + odom) ──
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        output='screen',
        parameters=[vesc_config],
    )
    ackermann_to_vesc = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        output='screen',
        parameters=[vesc_config],
    )
    vesc_to_odom = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        output='screen',
        parameters=[vesc_config],
    )

    # ── Teleop (joystick → ackermann_cmd) ──
    teleop_joy_node = Node(
        package='vesc_driver_mac',
        executable='teleop_joy',
        name='teleop_joy_node',
        output='screen',
    )

    # ── joy_mac/joy_node (macOS GameController → /joy) ──
    joy_node = Node(
        package='joy_mac',
        executable='joy_node',
        name='joy_node',
        output='screen',
        condition=IfCondition(use_joy),
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Skip camera (macOS GUI 세션 없을 때 / SSH).'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Run RViz (livox view). headless 와 독립.'),
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('camera_width', default_value='1280'),
        DeclareLaunchArgument('camera_height', default_value='720'),
        DeclareLaunchArgument('camera_fps', default_value='60'),
        DeclareLaunchArgument(
            'vesc_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('vesc_driver_mac'), 'config', 'vesc_config.yaml',
            ]),
            description='VESC config YAML (serial port, gains, etc.)',
        ),
        DeclareLaunchArgument('joy', default_value='true',
                              description='Start joy_mac/joy_node. false 면 별도 터미널에서 실행.'),

        sensor_bringup,
        vesc_driver_node,
        ackermann_to_vesc,
        vesc_to_odom,
        teleop_joy_node,
        joy_node,
    ])
