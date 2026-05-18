"""Global path generation: centerline extraction → trajectory optimization.

Usage:
  ros2 launch stack_master create_path.launch.py map:=<map_name>
  ros2 launch stack_master create_path.launch.py map:=<map_name> reverse:=true
  ros2 launch stack_master create_path.launch.py map:=<map_name> optimize:=false

Args:
  map       : map folder name under stack_master/maps/
  reverse   : reverse centerline direction (true=CW, default=CCW)
  optimize  : run trajectory optimizer after centerline extraction (default=true)
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_arg      = DeclareLaunchArgument('map',      description='Map name')
    reverse_arg  = DeclareLaunchArgument('reverse',  default_value='false',
                                         description='Reverse centerline (true=CW)')
    optimize_arg = DeclareLaunchArgument('optimize', default_value='true',
                                         description='Run trajectory optimizer after extraction')

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

    return LaunchDescription([
        map_arg,
        reverse_arg,
        optimize_arg,
        centerline_node,
        run_optimizer_after_extractor,
    ])
