"""ros2 launch frenet_odom_republisher frenet_odom_republisher.launch.py.

원본 ROS1 launch 의 remap 패턴 그대로 옮김:
  /odom              ← /car_state/odom
  /odom_frenet       → /car_state/odom_frenet
  /odom_frenet_fixed → /car_state/odom_frenet_fixed
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    node = Node(
        package="frenet_odom_republisher",
        executable="frenet_odom_republisher",
        name="frenet_odom_republisher",
        remappings=[
            ("/odom", "/car_state/odom"),
            ("/odom_frenet", "/car_state/odom_frenet"),
            ("/odom_frenet_fixed", "/car_state/odom_frenet_fixed"),
        ],
        output="screen",
    )

    return LaunchDescription([node])
