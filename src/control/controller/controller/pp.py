#!/usr/bin/env python3
"""Pure Pursuit controller (adaptive lookahead + multi-point blend).

Subscribes:
  /local_waypoints   (f110_msgs/WpntArray)  — state_machine output
  /car_state/odom    (nav_msgs/Odometry)    — rear-axle pose + speed

Publishes:
  drive_topic             (AckermannDriveStamped)
  /pp/lookahead           (Marker)  — blended target point

Algorithm:
  1. Adaptive lookahead:  ld = clip(k_v * v / (1 + k_k * |κ|), ld_min, ld_max)
     Two-pass: compute once at nearest kappa, then refine with preview kappa.
     Speed used for ld = max(current_v, 0.75 * target_v_at_ld)  — early corner preview.

  2. Multi-point blend target:
     Sample n_targets waypoints in [ld, ld + k_window*v], average positions.
     Reduces kappa-spike sensitivity and smooths high-speed cornering.

  3. PP steering:  δ = atan(2 * wheelbase * ly / L²)
     + feedforward: k_ff * atan(wheelbase * κ_target)

  4. Speed: mean of n_avg waypoints around ld index, scaled by speed_scale.

  5. Acceleration ramp: speed <= prev_speed + max_accel * dt  (hard braking unlimited).
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


class PurePursuit(Node):
    def __init__(self) -> None:
        super().__init__("pp")

        self.declare_parameter("wheelbase",        0.33)
        self.declare_parameter("max_steering_rad", 0.4)
        self.declare_parameter("max_speed_mps",    8.0)
        self.declare_parameter("speed_scale",      1.0)
        self.declare_parameter("control_rate_hz",  50.0)
        self.declare_parameter("drive_topic",      "/vesc/high_level/ackermann_cmd")
        # adaptive lookahead
        self.declare_parameter("ld_min",           0.4)
        self.declare_parameter("ld_max",           2.5)
        self.declare_parameter("k_v",              0.5)   # ld ∝ v
        self.declare_parameter("k_k",              2.0)   # ld shrinks at corners
        # multi-point blend
        self.declare_parameter("k_window",         0.05)  # window = k_window * v [m]
        self.declare_parameter("n_targets",        3)
        # feedforward
        self.declare_parameter("k_ff",             0.5)   # 0=off, 1=full kinematic FF
        # acceleration ramp
        self.declare_parameter("max_accel",        3.0)   # [m/s²]

        self.wb          = float(self.get_parameter("wheelbase").value)
        self.max_steer   = float(self.get_parameter("max_steering_rad").value)
        self.v_max       = float(self.get_parameter("max_speed_mps").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        rate             = float(self.get_parameter("control_rate_hz").value)
        drive_topic      = str(self.get_parameter("drive_topic").value)
        self.ld_min      = float(self.get_parameter("ld_min").value)
        self.ld_max      = float(self.get_parameter("ld_max").value)
        self.k_v         = float(self.get_parameter("k_v").value)
        self.k_k         = float(self.get_parameter("k_k").value)
        self.k_window    = float(self.get_parameter("k_window").value)
        self.n_targets   = int(self.get_parameter("n_targets").value)
        self.k_ff        = float(self.get_parameter("k_ff").value)
        self.max_accel   = float(self.get_parameter("max_accel").value)
        self._dt         = 1.0 / rate

        self._wpnts: list | None = None
        self._state: tuple[float, float, float, float] | None = None  # x, y, yaw, v
        self._prev_speed = 0.0
        self._log_tick   = 0

        self.create_subscription(WpntArray, "/local_waypoints", self._wpnts_cb, 10)
        self.create_subscription(Odometry,  "/car_state/odom",  self._odom_cb,  10)

        self.drive_pub    = self.create_publisher(AckermannDriveStamped, drive_topic,    10)
        self.lookahead_pub = self.create_publisher(Marker,               "/pp/lookahead", 10)

        self.create_timer(1.0 / rate, self._tick)
        self.add_on_set_parameters_callback(self._on_param)

        self.get_logger().info(
            f"[PP] ld=[{self.ld_min},{self.ld_max}] k_v={self.k_v} k_k={self.k_k} "
            f"k_window={self.k_window}s n_targets={self.n_targets} "
            f"k_ff={self.k_ff} max_accel={self.max_accel} → {drive_topic}"
        )

    # ------------------------------------------------------------------ #
    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if   p.name == "ld_min":           self.ld_min      = float(p.value)
            elif p.name == "ld_max":           self.ld_max      = float(p.value)
            elif p.name == "k_v":              self.k_v         = float(p.value)
            elif p.name == "k_k":              self.k_k         = float(p.value)
            elif p.name == "k_window":         self.k_window    = float(p.value)
            elif p.name == "n_targets":        self.n_targets   = int(p.value)
            elif p.name == "k_ff":             self.k_ff        = float(p.value)
            elif p.name == "max_accel":        self.max_accel   = float(p.value)
            elif p.name == "speed_scale":      self.speed_scale = float(p.value)
            elif p.name == "max_speed_mps":    self.v_max       = float(p.value)
            elif p.name == "max_steering_rad": self.max_steer   = float(p.value)
            elif p.name == "wheelbase":        self.wb          = float(p.value)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ #
    def _wpnts_cb(self, msg: WpntArray) -> None:
        if msg.wpnts:
            self._wpnts = msg.wpnts

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear.x
        self._state = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w), v)

    # ------------------------------------------------------------------ #
    def _adaptive_ld(self, vx: float, kappa: float) -> float:
        ld = self.k_v * vx / (1.0 + self.k_k * abs(kappa))
        return float(np.clip(ld, self.ld_min, self.ld_max))

    def _tick(self) -> None:
        if self._wpnts is None or self._state is None:
            return

        steer, speed = self._compute()

        # acceleration ramp (hard braking always allowed)
        speed = min(speed, self._prev_speed + self.max_accel * self._dt)
        self._prev_speed = speed

        cmd = AckermannDriveStamped()
        cmd.header.stamp         = self.get_clock().now().to_msg()
        cmd.header.frame_id      = "base_link"
        cmd.drive.steering_angle = steer
        cmd.drive.speed          = speed
        self.drive_pub.publish(cmd)

    def _compute(self) -> tuple[float, float]:
        x, y, yaw, vx = self._state
        wpts = self._wpnts
        N    = len(wpts)

        wx = np.array([w.x_m        for w in wpts])
        wy = np.array([w.y_m        for w in wpts])
        s  = np.array([w.s_m        for w in wpts])

        # 1. nearest waypoint
        nearest = int(np.argmin(np.hypot(wx - x, wy - y)))

        s_total  = float(s[-1])
        s_near   = float(s[nearest])
        s_ahead  = (s - s_near) % s_total

        # 2. adaptive lookahead — two-pass
        N_kap = 5
        cur_idx = [(nearest + i) % N for i in range(N_kap)]
        kappa_now = float(np.mean([abs(wpts[i].kappa_radpm) for i in cur_idx]))

        ld_prelim = self._adaptive_ld(vx, kappa_now)
        prev_idx  = int(np.argmin(np.abs(s_ahead - ld_prelim)))
        pre_idx   = [(prev_idx + i) % N for i in range(N_kap)]
        kappa_ahead = float(np.mean([abs(wpts[i].kappa_radpm) for i in pre_idx]))

        vx_preview = float(wpts[prev_idx].vx_mps)
        vx_for_ld  = max(abs(vx), 0.75 * vx_preview)
        ld         = self._adaptive_ld(vx_for_ld, max(kappa_now, kappa_ahead))

        # 3. multi-point blended target
        n_t      = max(1, self.n_targets)
        window_m = self.k_window * abs(vx)
        s_sample = np.linspace(ld, ld + window_m, n_t)
        x_blend  = float(np.mean([wx[int(np.argmin(np.abs(s_ahead - st)))] for st in s_sample]))
        y_blend  = float(np.mean([wy[int(np.argmin(np.abs(s_ahead - st)))] for st in s_sample]))

        target_idx = int(np.argmin(np.abs(s_ahead - ld)))

        # 4. transform blend target to vehicle frame
        cos_y =  math.cos(yaw)
        sin_y =  math.sin(yaw)
        ex    = x_blend - x
        ey    = y_blend - y
        lx    =  ex * cos_y + ey * sin_y
        ly    = -ex * sin_y + ey * cos_y

        self._publish_lookahead(x_blend, y_blend, ld)

        # 5. PP steering + feedforward
        L2 = lx * lx + ly * ly
        if L2 < 1e-6:
            steer = 0.0
        else:
            steer = math.atan2(2.0 * self.wb * ly, L2)
        kappa_target = float(wpts[target_idx].kappa_radpm)
        steer += self.k_ff * math.atan(self.wb * kappa_target)
        steer  = max(-self.max_steer, min(self.max_steer, steer))

        # 6. speed
        N_avg = 3
        spd_idx = [(target_idx + i) % N for i in range(N_avg)]
        speed   = float(np.mean([wpts[i].vx_mps for i in spd_idx]))
        speed   = float(np.clip(speed * self.speed_scale, 0.0, self.v_max))
        if speed <= 0.0:
            speed = 1.5

        self._log_tick += 1
        if self._log_tick >= 50:
            self._log_tick = 0
            self.get_logger().info(
                f"[PP] v={abs(vx):.2f} m/s  ld={ld:.2f} m  win={window_m:.2f} m  "
                f"steer={math.degrees(steer):.1f}°  spd={speed:.2f} m/s"
            )

        return steer, speed

    # ------------------------------------------------------------------ #
    def _publish_lookahead(self, x: float, y: float, ld: float) -> None:
        m = Marker()
        m.header.frame_id    = "map"
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = "pp_lookahead"
        m.id                 = 0
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = x
        m.pose.position.y    = y
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        s = float(np.clip(ld * 0.15, 0.1, 0.4))
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r = 0.1
        m.color.g = 1.0
        m.color.b = 0.2
        m.color.a = 1.0
        self.lookahead_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
