"""ros2 launch obstacle_publisher obstacle_publisher.launch.py."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("trajectory", default_value="min_curv",
                              description="centerline / min_curv / shortest_path / updated"),
        DeclareLaunchArgument("start_s", default_value="0.0"),
        DeclareLaunchArgument("speed_scaler", default_value="0.5"),
        DeclareLaunchArgument("constant_speed", default_value="false"),
    ]

    node = Node(
        package="obstacle_publisher",
        executable="obstacle_publisher",
        name="obstacle_publisher",
        parameters=[{
            "trajectory": LaunchConfiguration("trajectory"),
            "start_s": LaunchConfiguration("start_s"),
            "speed_scaler": LaunchConfiguration("speed_scaler"),
            "constant_speed": LaunchConfiguration("constant_speed"),
        }],
        output="screen",
    )

    return LaunchDescription([*args, node])
