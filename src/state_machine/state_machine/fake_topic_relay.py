#!/usr/bin/env python3
"""C-6 / D-1b 통합 검증용 fake topic relay.

state_machine 과 controller 의 의존 토픽 중 ROS2 ws 에 발행자 없는 것들 stub.

발행:
  - /global_waypoints_scaled (WpntArray) — /global_waypoints alias
  - /global_waypoints/overtaking (WpntArray) — /global_waypoints alias
  - /planner/recovery/wpnts (WpntArray) — 빈 (state_machine sub)
  - /car_state/pose (PoseStamped) — /car_state/odom 의 pose alias (controller sub)
  - /scan (LaserScan) — 빈 stub 1Hz (controller mapping_loop 안 써도 옵셔널)
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class FakeTopicRelay(Node):
    def __init__(self) -> None:
        super().__init__("fake_topic_relay")

        # /global_waypoints alias
        self._scaled_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled", 10)
        self._overtaking_pub = self.create_publisher(WpntArray, "/global_waypoints/overtaking", 10)
        self._recovery_pub = self.create_publisher(WpntArray, "/planner/recovery/wpnts", 10)
        # /car_state/pose (controller 의존)
        self._pose_pub = self.create_publisher(PoseStamped, "/car_state/pose", 10)
        # /scan stub (controller mapping_loop 의존, 정상 controller_loop 에선 unused)
        self._scan_pub = self.create_publisher(LaserScan, "/scan", 10)

        self.create_subscription(
            WpntArray, "/global_waypoints", self._on_global_waypoints, 10
        )
        self.create_subscription(
            Odometry, "/car_state/odom", self._on_odom, 10
        )

        self._empty_recovery = WpntArray()
        self._empty_recovery.header.frame_id = "map"
        self._empty_scan = LaserScan()
        self._empty_scan.header.frame_id = "laser"
        self._empty_scan.ranges = [10.0] * 100  # 100 ray, 10m default
        self.create_timer(1.0, self._publish_low_rate_stubs)

        self.get_logger().info(
            "[FakeTopicRelay] ready: /global_waypoints → /scaled + /overtaking, "
            "/odom → /car_state/pose, /recovery/wpnts + /scan (1Hz stubs)"
        )

    def _on_global_waypoints(self, msg: WpntArray) -> None:
        self._scaled_pub.publish(msg)
        self._overtaking_pub.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        """/car_state/odom 의 pose 부분만 PoseStamped 로 alias."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self._pose_pub.publish(pose)

    def _publish_low_rate_stubs(self) -> None:
        now = self.get_clock().now().to_msg()
        self._empty_recovery.header.stamp = now
        self._recovery_pub.publish(self._empty_recovery)
        self._empty_scan.header.stamp = now
        self._scan_pub.publish(self._empty_scan)


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
