#!/usr/bin/env python3
"""Stanley path-tracking controller.

Subscribes:
  /local_waypoints   (f110_msgs/WpntArray)  — state_machine output
  /car_state/odom    (nav_msgs/Odometry)    — rear-axle pose + speed

Publishes:
  drive_topic               (AckermannDriveStamped)
  /stanley/cte_target       (Marker)   — nearest waypoint
  /stanley/cte              (Float64)  — cross-track error [m]
  /stanley/heading_error    (Float64)  — heading error [rad]
  /stanley/steer            (Float64)  — steering command [rad]

Algorithm:
  1. Front axle position (odom is rear axle):
       x_fa = x + wheelbase * cos(yaw)
       y_fa = y + wheelbase * sin(yaw)
  2. Nearest waypoint to front axle.
  3. Heading error:  psi_e = normalize(psi_path - yaw)
  4. Cross-track error (positive = front axle right of path):
       cross = cos(psi_path)*(y_fa - wy) - sin(psi_path)*(x_fa - wx)
       e_fa  = -cross
  5. Steering:
       delta = psi_e + atan2(k * e_fa, v + k_s)
       delta = clip(delta, -max_steer, max_steer)

  k_s (softening): raises denominator at low speed and adds
  a floor so gain decreases naturally with velocity — suppresses
  high-speed oscillation without separate PID tuning.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker


def _yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _normalize_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Stanley(Node):
    def __init__(self) -> None:
        super().__init__("stanley")

        self.declare_parameter("wheelbase",        0.33)
        self.declare_parameter("k",                2.5)
        self.declare_parameter("k_s",              1.5)
        self.declare_parameter("k_ff",             0.01)
        self.declare_parameter("lookahead_min",    0.02)
        self.declare_parameter("lookahead_k",      0.07)
        self.declare_parameter("max_steering_rad", 0.4)
        self.declare_parameter("max_speed_mps",    8.0)
        self.declare_parameter("speed_scale",      1.0)
        self.declare_parameter("control_rate_hz",  50.0)
        self.declare_parameter("drive_topic",      "/vesc/high_level/ackermann_cmd")

        self.wb             = float(self.get_parameter("wheelbase").value)
        self.k              = float(self.get_parameter("k").value)
        self.k_s            = float(self.get_parameter("k_s").value)
        self.k_ff           = float(self.get_parameter("k_ff").value)
        self.lookahead_min  = float(self.get_parameter("lookahead_min").value)
        self.lookahead_k    = float(self.get_parameter("lookahead_k").value)
        self.max_steer      = float(self.get_parameter("max_steering_rad").value)
        self.v_max          = float(self.get_parameter("max_speed_mps").value)
        self.speed_scale    = float(self.get_parameter("speed_scale").value)
        rate                = float(self.get_parameter("control_rate_hz").value)
        drive_topic         = str(self.get_parameter("drive_topic").value)

        self._wpnts: np.ndarray | None = None
        self._state: tuple[float, float, float, float] | None = None

        self.create_subscription(WpntArray, "/local_waypoints", self._wpnts_cb, 10)
        self.create_subscription(Odometry,  "/car_state/odom",  self._odom_cb,  10)

        self.drive_pub   = self.create_publisher(AckermannDriveStamped, drive_topic,               10)
        self.marker_pub  = self.create_publisher(Marker,                "/stanley/cte_target",     10)
        self.cte_pub     = self.create_publisher(Float64,               "/stanley/cte",            10)
        self.heading_pub = self.create_publisher(Float64,               "/stanley/heading_error",  10)
        self.steer_pub   = self.create_publisher(Float64,               "/stanley/steer",          10)

        self.create_timer(1.0 / rate, self._tick)
        self.add_on_set_parameters_callback(self._on_param)

        self.get_logger().info(
            f"[stanley] wb={self.wb} k={self.k} k_s={self.k_s} k_ff={self.k_ff} "
            f"lookahead=({self.lookahead_min}+{self.lookahead_k}·v)m "
            f"max_steer={self.max_steer} v_max={self.v_max} → {drive_topic}"
        )

    # ------------------------------------------------------------------ #
    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if   p.name == "k":                self.k              = float(p.value)
            elif p.name == "k_s":              self.k_s            = float(p.value)
            elif p.name == "k_ff":             self.k_ff           = float(p.value)
            elif p.name == "lookahead_min":    self.lookahead_min  = float(p.value)
            elif p.name == "lookahead_k":      self.lookahead_k    = float(p.value)
            elif p.name == "max_steering_rad": self.max_steer      = float(p.value)
            elif p.name == "max_speed_mps":    self.v_max          = float(p.value)
            elif p.name == "speed_scale":      self.speed_scale    = float(p.value)
            elif p.name == "wheelbase":        self.wb             = float(p.value)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ #
    def _wpnts_cb(self, msg: WpntArray) -> None:
        if not msg.wpnts:
            return
        self._wpnts = np.array(
            [[w.x_m, w.y_m, w.vx_mps, w.psi_rad] for w in msg.wpnts],
            dtype=float,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear.x
        self._state = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w), v)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self._wpnts is None or self._state is None:
            return

        x, y, yaw, speed = self._state
        wp = self._wpnts

        # 1. Front axle position
        x_fa = x + self.wb * math.cos(yaw)
        y_fa = y + self.wb * math.sin(yaw)

        # 2. Nearest waypoint to front axle
        dx = wp[:, 0] - x_fa
        dy = wp[:, 1] - y_fa
        idx_near = int(np.argmin(dx * dx + dy * dy))

        # 3. Speed-adaptive lookahead:  d = lookahead_min + lookahead_k · |v|
        #    At 3 m/s → 0.5+0.9=1.4m,  at 8 m/s → 0.5+2.4=2.9m
        lookahead = self.lookahead_min + self.lookahead_k * abs(speed)
        N   = len(wp)
        idx = idx_near
        acc = 0.0
        for _ in range(N):
            nxt = (idx + 1) % N
            seg = math.hypot(wp[nxt, 0] - wp[idx, 0], wp[nxt, 1] - wp[idx, 1])
            if acc + seg >= lookahead:
                break
            acc += seg
            idx = nxt
        idx_look = idx

        wx, wy = wp[idx_look, 0], wp[idx_look, 1]
        v_ref  = wp[idx_near, 2]         # speed from nearest (safety)

        # 4. Path heading from geometry (independent of stored psi_rad convention)
        nxt_look  = (idx_look + 1) % N
        psi_path  = math.atan2(wp[nxt_look, 1] - wy, wp[nxt_look, 0] - wx)

        # 5. Heading error
        psi_e = _normalize_angle(psi_path - yaw)

        # 5. Cross-track error at front axle w.r.t. lookahead tangent
        cross = math.cos(psi_path) * (y_fa - wy) - math.sin(psi_path) * (x_fa - wx)
        e_fa  = -cross

        # 6. Path curvature κ = dψ/ds — heading change per arc length (geometry-based)
        nxt2  = (nxt_look + 1) % N
        ds    = math.hypot(wp[nxt_look, 0] - wx, wp[nxt_look, 1] - wy)
        psi2  = math.atan2(wp[nxt2, 1] - wp[nxt_look, 1], wp[nxt2, 0] - wp[nxt_look, 0])
        kappa = _normalize_angle(psi2 - psi_path) / ds if ds > 1e-3 else 0.0

        # 7. Stanley + curvature feedforward
        #    δ = ψ_e + atan2(k·e_fa, |v|+k_s) + k_ff·κ·L
        steer = (psi_e
                 + math.atan2(self.k * e_fa, abs(speed) + self.k_s)
                 + self.k_ff * kappa * self.wb)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # 8. Speed
        ref_speed = float(np.clip(v_ref * self.speed_scale, 0.0, self.v_max))

        cmd = AckermannDriveStamped()
        cmd.header.stamp         = self.get_clock().now().to_msg()
        cmd.header.frame_id      = "base_link"
        cmd.drive.steering_angle = steer
        cmd.drive.speed          = ref_speed
        self.drive_pub.publish(cmd)

        self.cte_pub.publish(Float64(data=e_fa))
        self.heading_pub.publish(Float64(data=psi_e))
        self.steer_pub.publish(Float64(data=steer))
        self._publish_marker(wx, wy)

    # ------------------------------------------------------------------ #
    def _publish_marker(self, x: float, y: float) -> None:
        m = Marker()
        m.header.frame_id    = "map"
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = "stanley"
        m.id                 = 0
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = x
        m.pose.position.y    = y
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 1.0
        self.marker_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Stanley()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
