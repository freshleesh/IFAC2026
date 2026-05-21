#!/usr/bin/env python3
"""Minimal pure-pursuit controller.

Subscribes:
  /local_waypoints   (f110_msgs/WpntArray)  — state_machine 출력
  /LIVO2/imu_propagate    (nav_msgs/Odometry)    — fast_livo localization

Publishes:
  /vesc/high_level/ackermann_cmd_mux/input/nav_1 (AckermannDriveStamped)

속도는 lookahead waypoint 의 vx_mps 를 그대로 사용 — lat_err / accel limiter /
heading scaler 등 일체의 후처리 없음.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker


def _yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class SimplePP(Node):
    def __init__(self) -> None:
        super().__init__("simple_pp")

        self.declare_parameter("lookahead_distance", 1.2)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("max_steering_rad", 0.4)
        self.declare_parameter("max_speed_mps", 8.0)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter(
            "drive_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
        )

        self.L           = float(self.get_parameter("lookahead_distance").value)
        self.wheelbase   = float(self.get_parameter("wheelbase").value)
        self.max_steer   = float(self.get_parameter("max_steering_rad").value)
        self.v_max       = float(self.get_parameter("max_speed_mps").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        rate             = float(self.get_parameter("control_rate_hz").value)
        drive_topic      = str(self.get_parameter("drive_topic").value)

        self._wpnts: np.ndarray | None = None   # shape (N, 3): x, y, vx
        self._pose: tuple[float, float, float] | None = None  # x, y, yaw

        self.create_subscription(WpntArray, "/local_waypoints", self._wpnts_cb, 10)
        # self.create_subscription(Odometry, "/LIVO2/imu_propagate", self._odom_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self._odom_cb, 10)

        self.drive_pub  = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.target_pub = self.create_publisher(Marker, "/simple_pp/lookahead", 10)

        self.create_timer(1.0 / rate, self._tick)

        self.add_on_set_parameters_callback(self._on_param)

        self.get_logger().info(
            f"[simple_pp] L={self.L} wb={self.wheelbase} v_max={self.v_max} "
            f"scale={self.speed_scale} → {drive_topic}"
        )

    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "lookahead_distance":
                self.L = float(p.value)
            elif p.name == "speed_scale":
                self.speed_scale = float(p.value)
            elif p.name == "max_speed_mps":
                self.v_max = float(p.value)
            elif p.name == "max_steering_rad":
                self.max_steer = float(p.value)
        return SetParametersResult(successful=True)

    def _wpnts_cb(self, msg: WpntArray) -> None:
        if not msg.wpnts:
            return
        self._wpnts = np.array(
            [[w.x_m, w.y_m, w.vx_mps] for w in msg.wpnts], dtype=float
        )

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w))

    def _tick(self) -> None:
        if self._wpnts is None or self._pose is None:
            return

        x, y, yaw = self._pose
        wp = self._wpnts

        # nearest waypoint
        dx = wp[:, 0] - x
        dy = wp[:, 1] - y
        d2 = dx * dx + dy * dy
        idx_near = int(np.argmin(d2))

        # march forward until distance from car >= L
        target_idx = idx_near
        for i in range(idx_near, len(wp)):
            if math.hypot(wp[i, 0] - x, wp[i, 1] - y) >= self.L:
                target_idx = i
                break
        else:
            target_idx = len(wp) - 1

        tx, ty, v_target = wp[target_idx]

        # transform target into vehicle frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        ex = tx - x
        ey = ty - y
        x_local = c * ex - s * ey
        y_local = s * ex + c * ey

        L_sq = x_local * x_local + y_local * y_local
        if L_sq < 1e-6:
            steer = 0.0
        else:
            steer = math.atan2(2.0 * self.wheelbase * y_local, L_sq)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        speed = float(np.clip(v_target * self.speed_scale, 0.0, self.v_max))

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.steering_angle = steer
        cmd.drive.speed = speed
        self.drive_pub.publish(cmd)

        self._publish_target_marker(tx, ty)

    def _publish_target_marker(self, x: float, y: float) -> None:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "simple_pp"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 1.0
        m.color.g = 0.4
        m.color.b = 0.0
        m.color.a = 1.0
        self.target_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimplePP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
