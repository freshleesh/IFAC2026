"""ros2 launch fake_odom_publisher fake_odom.launch.py [map:=NAME]."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    map_arg = DeclareLaunchArgument(
        "map",
        default_value="gazebo_wall_2",
        description="map directory name under share/<pkg>/maps/<map>/global_waypoints.json",
    )
    rate_arg = DeclareLaunchArgument("rate", default_value="50.0")
    speed_scale_arg = DeclareLaunchArgument("speed_scale", default_value="1.0")
    waypoints_path_arg = DeclareLaunchArgument(
        "waypoints_path",
        default_value="",
        description="absolute path to global_waypoints.json (overrides map)",
    )

    node = Node(
        package="fake_odom_publisher",
        executable="fake_odom_publisher",
        name="fake_odom_publisher",
        parameters=[{
            "map": LaunchConfiguration("map"),
            "waypoints_path": LaunchConfiguration("waypoints_path"),
            "rate": LaunchConfiguration("rate"),
            "speed_scale": LaunchConfiguration("speed_scale"),
        }],
        output="screen",
    )

    return LaunchDescription([map_arg, rate_arg, speed_scale_arg, waypoints_path_arg, node])
