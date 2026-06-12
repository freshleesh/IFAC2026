"""Global path generation: centerline extraction → trajectory optimization.

Usage:
  ros2 launch stack_master create_path.launch.py map:=<map_name>
  ros2 launch stack_master create_path.launch.py map:=<map_name> reverse:=true
  ros2 launch stack_master create_path.launch.py map:=<map_name> optimize:=false

Args:
  map      : map folder name under IFAC2026_SH/maps/
  reverse  : reverse centerline direction (true=CW, default=CCW)
  optimize : run trajectory optimizer after centerline extraction (default=true)

Mac Mini (real car):
  On macOS, fast_livo2's real-time SLAM map directory is symlinked into the unified
  maps root so that centerline_extractor and trajectory_optimizer can find it.
  FAST_LIVO_MAP_ROOT must match the fast_livo2 map path on the Mac Mini.
"""
import os
import platform

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── Mac Mini real-car setting ─────────────────────────────────────────────────
# fast_livo2 SLAM saves maps here; a symlink is created into the unified maps root.
FAST_LIVO_MAP_ROOT = "/Users/mini/ros2_ws/src/fast_livo2/map"


def _maps_root() -> str:
    """Return the unified IFAC2026_SH/maps/ directory.

    Ubuntu: repo == colcon ws (ws/maps).  Mac mini real car: repo is cloned
    into ws/src/IFAC2026_SH, so the maps dir is ws/src/IFAC2026_SH/maps.
    """
    _sm_install = get_package_share_directory('stack_master')
    _ws = os.path.normpath(os.path.join(_sm_install, '..', '..', '..', '..'))
    for repo in (_ws, os.path.join(_ws, 'src', 'IFAC2026_SH')):
        maps = os.path.join(repo, 'maps')
        if os.path.isdir(maps):
            return maps
    return os.path.join(_ws, 'maps')


def _setup_mac_symlink(context: LaunchContext, *_):
    """On macOS only: symlink fast_livo2/<map_name> → IFAC2026_SH/maps/<map_name>."""
    if platform.system() != 'Darwin':
        return []

    map_name = LaunchConfiguration('map').perform(context)
    maps = _maps_root()
    src = os.path.join(FAST_LIVO_MAP_ROOT, map_name)
    dst = os.path.join(maps, map_name)

    os.makedirs(maps, exist_ok=True)

    if os.path.islink(dst):
        os.unlink(dst)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f'[create_path] symlinked {src} → {dst}')
    else:
        print(f'[create_path] maps dir already exists: {dst}')

    return []


def generate_launch_description():
    optimizer_yaml = os.path.join(
        get_package_share_directory('stack_master'), 'config', 'trajectory_optimizer.yaml'
    )

    map_arg      = DeclareLaunchArgument('map',      description='Map name')
    reverse_arg  = DeclareLaunchArgument('reverse',  default_value='false',
                                         description='Reverse centerline (true=CW)')
    optimize_arg = DeclareLaunchArgument('optimize', default_value='true',
                                         description='Run trajectory optimizer after extraction')

    mac_symlink = OpaqueFunction(function=_setup_mac_symlink)

    centerline_node = Node(
        package='global_planner',
        executable='centerline_extractor',
        name='centerline_extractor',
        output='screen',
        parameters=[{
            'map_name': LaunchConfiguration('map'),
            'reverse':  LaunchConfiguration('reverse'),
        }],
    )

    optimizer_node = Node(
        package='global_planner',
        executable='trajectory_optimizer',
        name='trajectory_optimizer',
        output='screen',
        parameters=[
            optimizer_yaml,
            {'map_name': LaunchConfiguration('map')},
        ],
        condition=IfCondition(LaunchConfiguration('optimize')),
    )

    run_optimizer_after_extractor = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=centerline_node,
            on_exit=[
                LogInfo(msg='Centerline extraction done — starting trajectory optimizer...'),
                optimizer_node,
            ],
        ),
        condition=IfCondition(LaunchConfiguration('optimize')),
    )

    return LaunchDescription([
        map_arg,
        reverse_arg,
        optimize_arg,
        mac_symlink,
        centerline_node,
        run_optimizer_after_extractor,
    ])
