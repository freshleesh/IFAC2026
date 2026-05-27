"""ros2 launch frenet_conversion frenet_conversion_server.launch.py."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    perception_only_arg = DeclareLaunchArgument(
        "PerceptionOnly",
        default_value="false",
        description="true 면 service 이름에 _perception 접미사 (perception 노드 전용 서버)",
    )

    node = Node(
        package="frenet_conversion",
        executable="frenet_conversion_server",
        name="frenet_conversion_server",
        parameters=[{
            "PerceptionOnly": LaunchConfiguration("PerceptionOnly"),
        }],
        output="screen",
    )

    return LaunchDescription([perception_only_arg, node])
