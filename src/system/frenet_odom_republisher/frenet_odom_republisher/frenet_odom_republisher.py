#!/usr/bin/env python3
"""
Frenet odom republisher (ROS2 Jazzy port).

/odom (cartesian) 을 받아 /odom_frenet (GB raceline 기준) 와
/odom_frenet_fixed (Smart Static fixed path 기준) 두 토픽으로 변환 발행.

원본: ICRA2026_HJ f110_utils/nodes/frenet_odom_republisher (C++, 240L).
포팅: SH ROS2 Jazzy — Python, frenet_conversion lib import.

미포팅 (의도):
- interactive_markers 의 "Force Full Search" 버튼.
  ROS2 interactive_markers 는 별도 패키지 의존이고, 부수 디버깅 기능이라 보류.
  (필요 시 향후 외부 cli 또는 ros2 service 로 재구현)

토픽 (원본 launch remap 그대로):
  sub:
    - /car_state/odom (Odometry)            ← /odom 으로 listen
    - /global_waypoints (WpntArray)
    - /planner/avoidance/smart_static_otwpnts (OTWpntArray)
    - /trackbounds/markers (MarkerArray, 한 번만 수신)
  pub:
    - /car_state/odom_frenet (Odometry)     ← /odom_frenet 발행
    - /car_state/odom_frenet_fixed (Odometry) ← /odom_frenet_fixed 발행
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray, OTWpntArray

from frenet_conversion.frenet_converter import FrenetConverter
from frenet_odom_republisher.transforms import quaternion_to_yaw


class FrenetOdomRepublisher(Node):
    def __init__(self) -> None:
        super().__init__("frenet_odom_republisher")

        self._converter_gb: Optional[FrenetConverter] = None
        self._converter_fixed: Optional[FrenetConverter] = None
        self._has_global_trajectory = False
        self._has_fixed_path_trajectory = False
        self._has_track_bounds = False

        # Sub
        self.create_subscription(
            WpntArray, "/global_waypoints", self._on_global_traj, 10
        )
        self.create_subscription(
            OTWpntArray,
            "/planner/avoidance/smart_static_otwpnts",
            self._on_fixed_path_traj,
            10,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(
            MarkerArray, "/trackbounds/markers", self._on_trackbounds, 1
        )

        # Pub (원본 토픽명 그대로 — launch 의 remap 으로 /car_state/odom_frenet 으로 노출)
        self._frenet_odom_pub = self.create_publisher(Odometry, "/odom_frenet", 1)
        self._frenet_odom_fixed_pub = self.create_publisher(
            Odometry, "/odom_frenet_fixed", 1
        )

        self.get_logger().info(
            "[FrenetOdomRepublisher] ready, waiting for /global_waypoints + /odom..."
        )

    # ---------- 콜백 (waypoints / track bounds) ----------

    def _on_global_traj(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        z = np.array([w.z_m for w in msg.wpnts])
        try:
            converter = FrenetConverter(x, y, z)
        except (ValueError, IndexError) as e:
            self.get_logger().warn(
                f"[FrenetOdomRepublisher] GB FrenetConverter build failed: {e}"
            )
            return
        # 트랙 바운드가 이미 받아져 있다면 새 converter 에도 적용
        if self._has_track_bounds and self._converter_gb is not None:
            converter.set_track_bounds(
                self._converter_gb.left_bounds, self._converter_gb.right_bounds
            )
        self._converter_gb = converter
        if not self._has_global_trajectory:
            self.get_logger().info(
                f"[FrenetOdomRepublisher] global_waypoints received "
                f"({len(msg.wpnts)} wpnts, raceline_length={converter.raceline_length:.2f}m)"
            )
            self._has_global_trajectory = True

    def _on_fixed_path_traj(self, msg: OTWpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        z = np.array([w.z_m for w in msg.wpnts])
        try:
            converter = FrenetConverter(x, y, z)
        except (ValueError, IndexError) as e:
            self.get_logger().warn(
                f"[FrenetOdomRepublisher] fixed FrenetConverter build failed: {e}"
            )
            return
        if self._has_track_bounds and self._converter_fixed is not None:
            converter.set_track_bounds(
                self._converter_fixed.left_bounds, self._converter_fixed.right_bounds
            )
        self._converter_fixed = converter
        if not self._has_fixed_path_trajectory:
            self.get_logger().info(
                f"[FrenetOdomRepublisher] smart_static_otwpnts received "
                f"({len(msg.wpnts)} wpnts)"
            )
            self._has_fixed_path_trajectory = True

    def _on_trackbounds(self, msg: MarkerArray) -> None:
        """원본은 한 번 받고 unsub. ROS2 도 has_track_bounds flag 로 한 번만 적용."""
        if self._has_track_bounds:
            return
        if self._converter_gb is not None:
            self._converter_gb.set_track_bounds_from_markers(msg.markers)
        if self._converter_fixed is not None:
            self._converter_fixed.set_track_bounds_from_markers(msg.markers)
        self._has_track_bounds = True
        self.get_logger().info(
            f"[FrenetOdomRepublisher] Track bounds received "
            f"({len(msg.markers)} markers); applied to both converters"
        )

    # ---------- 메인 — odom 변환 ----------

    def _on_odom(self, msg: Odometry) -> None:
        # quaternion → yaw
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # GB frame
        if self._converter_gb is not None:
            self._publish_frenet(
                msg, self._converter_gb, yaw, self._frenet_odom_pub
            )

        # Fixed path frame
        if self._converter_fixed is not None:
            self._publish_frenet(
                msg, self._converter_fixed, yaw, self._frenet_odom_fixed_pub
            )

    def _publish_frenet(
        self,
        odom_msg: Odometry,
        converter: FrenetConverter,
        yaw: float,
        publisher,
    ) -> None:
        """원본 cc 의 OdomCallback 한 가지(GB 또는 Fixed) 처리."""
        s, d, vs, vd, idx = converter.get_frenet_odometry(
            x=odom_msg.pose.pose.position.x,
            y=odom_msg.pose.pose.position.y,
            z=odom_msg.pose.pose.position.z,
            yaw=yaw,
            vx_body=odom_msg.twist.twist.linear.x,
            vy_body=odom_msg.twist.twist.linear.y,
        )

        # 원본 그대로: 입력 odom 메시지를 복사하고 position.x/y, twist.linear.x/y 만 갈아끼움.
        # child_frame_id 에 closest_wpt_idx 를 string 으로 abuse.
        out = Odometry()
        out.header = odom_msg.header
        out.child_frame_id = str(idx)
        out.pose = odom_msg.pose
        out.twist = odom_msg.twist
        out.pose.pose.position.x = s
        out.pose.pose.position.y = d
        out.twist.twist.linear.x = vs
        out.twist.twist.linear.y = vd
        publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrenetOdomRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
