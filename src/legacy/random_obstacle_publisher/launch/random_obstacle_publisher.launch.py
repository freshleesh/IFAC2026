"""ros2 launch random_obstacle_publisher random_obstacle_publisher.launch.py."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("n_obstacles", default_value="8"),
        DeclareLaunchArgument("publish_at_lookahead", default_value="false"),
        DeclareLaunchArgument("lookahead_distance", default_value="5.0"),
        DeclareLaunchArgument("rnd_seed", default_value="84"),
        DeclareLaunchArgument("obstacle_width", default_value="0.2"),
        DeclareLaunchArgument("obstacle_length", default_value="0.3"),
        DeclareLaunchArgument("obstacle_max_d_from_traj", default_value="1.0"),
        DeclareLaunchArgument("rate_hz", default_value="25.0"),
    ]

    node = Node(
        package="random_obstacle_publisher",
        executable="random_obstacle_publisher",
        name="random_obstacle_publisher",
        parameters=[{
            "n_obstacles": LaunchConfiguration("n_obstacles"),
            "publish_at_lookahead": LaunchConfiguration("publish_at_lookahead"),
            "lookahead_distance": LaunchConfiguration("lookahead_distance"),
            "rnd_seed": LaunchConfiguration("rnd_seed"),
            "obstacle_width": LaunchConfiguration("obstacle_width"),
            "obstacle_length": LaunchConfiguration("obstacle_length"),
            "obstacle_max_d_from_traj": LaunchConfiguration("obstacle_max_d_from_traj"),
            "rate_hz": LaunchConfiguration("rate_hz"),
        }],
        output="screen",
    )

    return LaunchDescription([*args, node])
