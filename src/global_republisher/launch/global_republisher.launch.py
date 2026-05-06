"""ros2 launch global_republisher global_republisher.launch.py [map:=NAME]."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    map_arg = DeclareLaunchArgument(
        "map",
        default_value="gazebo_wall_2",
        description="map 이름 (fallback: ROS1 ws stack_master/maps/<map>/global_waypoints.json)",
    )
    map_path_arg = DeclareLaunchArgument(
        "map_path",
        default_value="",
        description="global_waypoints.json 절대경로 (지정 시 우선)",
    )
    rate_arg = DeclareLaunchArgument("publish_rate_hz", default_value="0.5")

    node = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{
            "map": LaunchConfiguration("map"),
            "map_path": LaunchConfiguration("map_path"),
            "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
        }],
        output="screen",
    )

    return LaunchDescription([map_arg, map_path_arg, rate_arg, node])
