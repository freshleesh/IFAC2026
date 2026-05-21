"""Global path generation: centerline extraction → trajectory optimization.

Usage:
  ros2 launch stack_master create_path.launch.py map:=<map_name>
  ros2 launch stack_master create_path.launch.py map:=<map_name> reverse:=true
  ros2 launch stack_master create_path.launch.py map:=<map_name> optimize:=false

Args:
  map       : map folder name. Real data lives at fast_livo2/map/<name>/.
              A symlink stack_master/maps/<name> → fast_livo2/map/<name> is
              created automatically (the global_planner nodes hardcode the
              stack_master path); this lets a single `map:=<name>` argument
              point at the fast_livo prior-map directory.
  reverse   : reverse centerline direction (true=CW, default=CCW)
  optimize  : run trajectory optimizer after centerline extraction (default=true)
"""
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# fast_livo prior-map directory is the canonical map storage. global_planner
# nodes (centerline_extractor / trajectory_optimizer / waypoint_publisher)
# resolve maps via stack_master/maps/<name> at runtime, so we symlink the
# fast_livo dir at that location instead of patching three node sources.
FAST_LIVO_MAP_ROOT = "/Users/mini/ros2_ws/src/fast_livo2/map"
STACK_MAPS_ROOT    = "/Users/mini/ros2_ws/src/IFAC2026_SH/src/stack_master/maps"


def generate_launch_description():
    map_arg      = DeclareLaunchArgument('map',      description='Map name')
    reverse_arg  = DeclareLaunchArgument('reverse',  default_value='false',
                                         description='Reverse centerline (true=CW)')
    optimize_arg = DeclareLaunchArgument('optimize', default_value='true',
                                         description='Run trajectory optimizer after extraction')

    map_name   = LaunchConfiguration('map')
    livo_path  = PathJoinSubstitution([FAST_LIVO_MAP_ROOT, map_name])
    stack_path = PathJoinSubstitution([STACK_MAPS_ROOT, map_name])

    # `ln -sfn` is safe for re-runs (replaces an existing symlink) but errors
    # out cleanly if a real directory already sits at the target — in that
    # case the user must clean it up manually so we never clobber real data.
    make_symlink = ExecuteProcess(
        cmd=['ln', '-sfn', livo_path, stack_path],
        output='log',
    )

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
        parameters=[{
            'map_name': LaunchConfiguration('map'),
        }],
        condition=IfCondition(LaunchConfiguration('optimize')),
    )

    # trajectory_optimizer starts only after centerline_extractor exits cleanly
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

    # centerline_extractor only runs after the symlink is in place so its
    # __file__-relative path resolution lands on the fast_livo map dir.
    run_extractor_after_symlink = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=make_symlink,
            on_exit=[centerline_node],
        ),
    )

    return LaunchDescription([
        map_arg,
        reverse_arg,
        optimize_arg,
        make_symlink,
        run_extractor_after_symlink,
        run_optimizer_after_extractor,
    ])
