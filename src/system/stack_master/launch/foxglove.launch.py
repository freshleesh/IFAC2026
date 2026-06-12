"""foxglove_bridge 단독 launch — 원격 디버깅 (SSH Ubuntu → Mac on RC car).

사용:
  # 기본 (포트 8765, 모든 인터페이스 listen, 압축 ON)
  ros2 launch stack_master foxglove.launch.py

  # 포트 변경
  ros2 launch stack_master foxglove.launch.py port:=9000

  # 토픽 화이트리스트 (정규식)
  ros2 launch stack_master foxglove.launch.py \\
      topic_whitelist:="['/car_state/.*','/LIVO2/imu_propagate','/scan']"

Ubuntu 쪽에서 Foxglove Studio (https://foxglove.dev/download) 실행 →
"Open connection" → "Foxglove WebSocket" → ws://<mac-ip>:8765
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration("port")
    address = LaunchConfiguration("address")
    use_compression = LaunchConfiguration("use_compression")
    topic_whitelist = LaunchConfiguration("topic_whitelist")
    send_buffer_limit = LaunchConfiguration("send_buffer_limit")

    foxglove_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[
            {
                "port": port,
                "address": address,
                "tls": False,
                "use_compression": use_compression,
                # 큰 PointCloud 끊김 방지 — 기본 10MB 너무 작음
                "send_buffer_limit": send_buffer_limit,
                # WiFi 환경에서 굳이 모든 토픽 다 보내지 말도록 화이트리스트
                "topic_whitelist": topic_whitelist,
                # client 가 직접 토픽 publish 도 가능 (joy 흉내 등 디버깅 용도)
                "capabilities": [
                    "clientPublish",
                    "parameters",
                    "parametersSubscribe",
                    "services",
                    "connectionGraph",
                    "assets",
                ],
                "include_hidden": False,
                # ★ subscription 큐 깊이 1 — 못 따라가면 그냥 drop. burst 방지.
                # 이미지같이 큰 토픽이 쌓이는 게 stuttering 의 원인이라서.
                "min_qos_depth": 1,
                "max_qos_depth": 1,
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8765",
                              description="WebSocket port."),
        DeclareLaunchArgument("address", default_value="0.0.0.0",
                              description="Listen address. 0.0.0.0 = all interfaces."),
        DeclareLaunchArgument("use_compression", default_value="true",
                              description="permessage-deflate WebSocket 압축."),
        DeclareLaunchArgument("send_buffer_limit", default_value="1000000000",
                              description="송신 버퍼 한계 [bytes]. 100MB. 너무 크면 burst 조장."),
        DeclareLaunchArgument(
            "topic_whitelist",
            default_value="['.*']",
            description="정규식 list. 기본은 전부 허용. 대역폭 줄이려면 명시.",
        ),
        foxglove_node,
    ])
