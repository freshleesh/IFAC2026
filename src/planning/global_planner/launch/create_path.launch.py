import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_TRAJ_OPT_YAML = os.path.join(
    get_package_share_directory('stack_master'),
    'config', 'trajectory_optimizer.yaml',
)


def generate_launch_description():
    map_arg         = DeclareLaunchArgument('map',                description='Map folder name under stack_master/maps/')
    extract_arg     = DeclareLaunchArgument('extract_centerline', default_value='true',         description='Extract centerline from map image')
    optimize_arg    = DeclareLaunchArgument('optimize',           default_value='false',        description='Run IQP+SP trajectory optimization')
    reverse_arg     = DeclareLaunchArgument('reverse',            default_value='false',        description='Reverse direction (true=CW)')
    output_csv_arg  = DeclareLaunchArgument('output_csv',         default_value='centerline.csv')

    map_name     = LaunchConfiguration('map')
    do_extract   = LaunchConfiguration('extract_centerline')
    do_optimize  = LaunchConfiguration('optimize')
    reverse      = LaunchConfiguration('reverse')
    output_csv   = LaunchConfiguration('output_csv')

    extractor = Node(
        package='global_planner',
        executable='centerline_extractor',
        name='centerline_extractor',
        output='screen',
        condition=IfCondition(do_extract),
        parameters=[{
            'map_name':   map_name,
            'reverse':    reverse,
            'output_csv': output_csv,
        }],
    )

    optimizer = Node(
        package='global_planner',
        executable='trajectory_optimizer',
        name='trajectory_optimizer',
        output='screen',
        condition=IfCondition(do_optimize),
        parameters=[
            _TRAJ_OPT_YAML,           # safety_width_iqp/sp and other defaults
            {
                'map_name':  map_name,
                'input_csv': output_csv,
            },
        ],
    )

    # optimizer는 extractor가 정상 종료된 후에 실행
    run_optimizer_after_extractor = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=extractor,
            on_exit=[optimizer],
        ),
        condition=IfCondition(do_optimize),
    )

    return LaunchDescription([
        map_arg,
        extract_arg,
        optimize_arg,
        reverse_arg,
        output_csv_arg,
        extractor,
        run_optimizer_after_extractor,
    ])
