#!/usr/bin/env python3
"""C-6 통합 검증용 fake topic relay.

state_machine 의 의존 토픽 중 ROS2 ws 에 발행자 없는 토픽들을
/global_waypoints 의 메시지를 alias 로 발행 + 빈 메시지 stub 으로 채움.

발행:
  - /global_waypoints_scaled (WpntArray) — /global_waypoints alias
  - /global_waypoints/overtaking (WpntArray) — /global_waypoints alias
  - /planner/recovery/wpnts (WpntArray) — 빈 (state_machine 가 sub 만 함)
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from f110_msgs.msg import WpntArray


class FakeTopicRelay(Node):
    def __init__(self) -> None:
        super().__init__("fake_topic_relay")

        # /global_waypoints alias 두 개
        self._scaled_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled", 10)
        self._overtaking_pub = self.create_publisher(WpntArray, "/global_waypoints/overtaking", 10)
        self._recovery_pub = self.create_publisher(WpntArray, "/planner/recovery/wpnts", 10)

        self.create_subscription(
            WpntArray, "/global_waypoints", self._on_global_waypoints, 10
        )

        # /planner/recovery/wpnts 는 빈 메시지 1Hz (state_machine 의 sub 가 활성화되도록)
        self._empty_recovery = WpntArray()
        self._empty_recovery.header.frame_id = "map"
        self.create_timer(1.0, self._publish_empty_recovery)

        self.get_logger().info(
            "[FakeTopicRelay] ready: /global_waypoints → /scaled + /overtaking, "
            "/recovery/wpnts (empty 1Hz)"
        )

    def _on_global_waypoints(self, msg: WpntArray) -> None:
        self._scaled_pub.publish(msg)
        self._overtaking_pub.publish(msg)

    def _publish_empty_recovery(self) -> None:
        self._empty_recovery.header.stamp = self.get_clock().now().to_msg()
        self._recovery_pub.publish(self._empty_recovery)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeTopicRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
