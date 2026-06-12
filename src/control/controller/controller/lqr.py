#!/usr/bin/env python3
"""LQR (Linear Quadratic Regulator) path tracking controller.

Kinematic bicycle model in Frenet frame (small-angle):
  ė_d  = v * e_θ
  ė_θ  = (v/L) * δ  −  v*κ(s)

State   : x = [e_d, e_θ]
  e_d   : lateral error [m]  — signed distance from car to nearest waypoint
           positive = car is to the LEFT of the path
  e_θ   : heading error [rad] — yaw − psi_ref
           positive = car heading to the LEFT

Control : δ = −K(v) · x  +  k_ff · atan(L · κ)
  K(v) is computed offline via DARE for each v in the gain schedule,
  then interpolated at runtime.

Topics:
  sub  /local_waypoints      (f110_msgs/WpntArray)
  sub  /car_state/odom       (nav_msgs/Odometry)
  pub  drive_topic           (ackermann_msgs/AckermannDriveStamped)
  pub  /lqr/nearest_wpnt     (visualization_msgs/Marker)  — 예측 끝점 (주황 구)
  pub  /lqr/predicted_path   (visualization_msgs/Marker)  — rollout 경로 (청록 선 + 초록 구)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker


# ─────────────────────────────────────────────────────────────────────────────
# DARE-based gain solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_lqr_gain(v: float, L: float, dt: float,
                    Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete LQR gain K for the kinematic-bicycle Frenet model at speed v.

    Returns K (1×2) such that δ_fb = −K @ [e_d, e_θ].
    Falls back to zero gain if v is too small for the system to be controllable.
    """
    if v < 1e-3:
        return np.zeros((1, 2))

    # Continuous A, B
    A_c = np.array([[0.0, v],
                    [0.0, 0.0]])
    B_c = np.array([[0.0],
                    [v / L]])

    # Euler discretisation (dt = 0.02 s → error < 0.1 % at 8 m/s)
    A_d = np.eye(2) + A_c * dt
    B_d = B_c * dt

    try:
        P = solve_discrete_are(A_d, B_d, Q, R)
        K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
    except np.linalg.LinAlgError:
        K = np.zeros((1, 2))

    return K  # shape (1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Gain-schedule table
# ─────────────────────────────────────────────────────────────────────────────

class _GainSchedule:
    """Pre-computed LQR gains over a velocity grid, with linear interpolation."""

    def __init__(self, v_min: float, v_max: float, n: int,
                 L: float, dt: float, Q: np.ndarray, R: np.ndarray) -> None:
        self._v_grid = np.linspace(max(v_min, 0.1), v_max, n)
        self._K_grid = np.vstack([
            _solve_lqr_gain(v, L, dt, Q, R) for v in self._v_grid
        ])  # shape (n, 2)

    def get(self, v: float) -> np.ndarray:
        """Return K (shape (1,2)) interpolated at speed v."""
        v_clamped = float(np.clip(v, self._v_grid[0], self._v_grid[-1]))
        K0 = np.interp(v_clamped, self._v_grid, self._K_grid[:, 0])
        K1 = np.interp(v_clamped, self._v_grid, self._K_grid[:, 1])
        return np.array([[K0, K1]])


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LQRController(Node):
    def __init__(self) -> None:
        super().__init__("lqr")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("wheelbase",        0.33)
        self.declare_parameter("max_steering_rad", 0.4)
        self.declare_parameter("max_speed_mps",    8.0)
        self.declare_parameter("speed_scale",      1.0)
        self.declare_parameter("control_rate_hz",  50.0)
        self.declare_parameter("drive_topic",
                               "/vesc/high_level/ackermann_cmd")

        # LQR cost weights
        self.declare_parameter("q_d",     1.0)    # lateral error weight
        self.declare_parameter("q_theta", 1.0)    # heading error weight
        self.declare_parameter("r_delta", 1.0)    # steering input weight

        # Curvature feedforward gain (0 = off, 1 = full kinematic FF)
        self.declare_parameter("k_ff",     1.0)

        # Gain schedule velocity grid
        self.declare_parameter("v_min_schedule", 0.5)
        self.declare_parameter("v_max_schedule", 8.0)
        self.declare_parameter("n_schedule",     100)

        # Adaptive lookahead for heading reference (Preview LQR)
        # e_θ and κ_ff are computed at the lookahead point, not the nearest waypoint.
        # This anticipates corners and suppresses oscillation from localization noise.
        self.declare_parameter("ld_min",  0.4)   # [m] minimum lookahead distance
        self.declare_parameter("ld_max",  2.5)   # [m] maximum lookahead distance
        self.declare_parameter("k_v",     0.5)   # lookahead ∝ speed
        self.declare_parameter("k_k",     2.0)   # lookahead shrinks at corners

        # Iterative rollout: max iterations for fixed-point convergence
        # Each iteration re-rolls with the LQR output from the previous step.
        # 1 = single rollout (original), 2~4 = iterative (recommended: 3)
        self.declare_parameter("n_iter",    3)
        # Convergence threshold [rad] — stop early if |δ_new − δ_old| < tol
        self.declare_parameter("iter_tol",  1e-3)

        # Speed command averaging
        self.declare_parameter("n_avg",     3)

        # Acceleration ramp
        self.declare_parameter("max_accel", 3.0)   # [m/s²]

        # ── Read params ──────────────────────────────────────────────────────
        self.wb          = float(self.get_parameter("wheelbase").value)
        self.max_steer   = float(self.get_parameter("max_steering_rad").value)
        self.v_max       = float(self.get_parameter("max_speed_mps").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        rate             = float(self.get_parameter("control_rate_hz").value)
        drive_topic      = str(self.get_parameter("drive_topic").value)
        self.k_ff      = float(self.get_parameter("k_ff").value)
        self.ld_min    = float(self.get_parameter("ld_min").value)
        self.ld_max    = float(self.get_parameter("ld_max").value)
        self.k_v       = float(self.get_parameter("k_v").value)
        self.k_k       = float(self.get_parameter("k_k").value)
        self.n_iter    = int(self.get_parameter("n_iter").value)
        self.iter_tol  = float(self.get_parameter("iter_tol").value)
        self.n_avg     = int(self.get_parameter("n_avg").value)
        self.max_accel = float(self.get_parameter("max_accel").value)

        q_d     = float(self.get_parameter("q_d").value)
        q_theta = float(self.get_parameter("q_theta").value)
        r_delta = float(self.get_parameter("r_delta").value)
        v_min_s = float(self.get_parameter("v_min_schedule").value)
        v_max_s = float(self.get_parameter("v_max_schedule").value)
        n_s     = int(self.get_parameter("n_schedule").value)

        self._dt = 1.0 / rate

        # ── Build gain-schedule table ────────────────────────────────────────
        Q = np.diag([q_d, q_theta])
        R = np.array([[r_delta]])
        self._gs = _GainSchedule(v_min_s, v_max_s, n_s,
                                 self.wb, self._dt, Q, R)

        # ── State ────────────────────────────────────────────────────────────
        self._wpnts = None
        self._state: Optional[tuple] = None   # (x, y, yaw, v)
        self._prev_speed = 0.0
        self._prev_steer = 0.0   # previous steering command — used as rollout input
        self._log_tick   = 0

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self.create_subscription(WpntArray, "/local_waypoints", self._wpnts_cb, 10)
        self.create_subscription(Odometry,  "/car_state/odom",  self._odom_cb,  10)

        self.drive_pub    = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.marker_pub   = self.create_publisher(Marker, "/lqr/nearest_wpnt",     10)
        self.path_pub     = self.create_publisher(Marker, "/lqr/predicted_path",   10)

        self.create_timer(self._dt, self._tick)
        self.add_on_set_parameters_callback(self._on_param)

        self.get_logger().info(
            f"[LQR] Q=[{q_d}, {q_theta}]  R=[{r_delta}]  k_ff={self.k_ff}  "
            f"ld=[{self.ld_min},{self.ld_max}] k_v={self.k_v} k_k={self.k_k}  "
            f"n_iter={self.n_iter}  iter_tol={self.iter_tol}  "
            f"schedule v=[{v_min_s:.1f},{v_max_s:.1f}] n={n_s}  → {drive_topic}"
        )

    # ── Parameter hot-update ─────────────────────────────────────────────────
    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        rebuild = False
        for p in params:
            if   p.name == "k_ff":            self.k_ff        = float(p.value)
            elif p.name == "speed_scale":     self.speed_scale = float(p.value)
            elif p.name == "max_speed_mps":   self.v_max       = float(p.value)
            elif p.name == "max_steering_rad":self.max_steer   = float(p.value)
            elif p.name == "wheelbase":       self.wb          = float(p.value); rebuild = True
            elif p.name == "max_accel":       self.max_accel   = float(p.value)
            elif p.name == "ld_min":          self.ld_min      = float(p.value)
            elif p.name == "ld_max":          self.ld_max      = float(p.value)
            elif p.name == "k_v":             self.k_v         = float(p.value)
            elif p.name == "k_k":             self.k_k         = float(p.value)
            elif p.name == "n_iter":          self.n_iter      = int(p.value)
            elif p.name == "iter_tol":        self.iter_tol    = float(p.value)
            elif p.name in ("q_d", "q_theta", "r_delta",
                            "v_min_schedule", "v_max_schedule", "n_schedule"):
                rebuild = True
        if rebuild:
            q_d     = float(self.get_parameter("q_d").value)
            q_theta = float(self.get_parameter("q_theta").value)
            r_delta = float(self.get_parameter("r_delta").value)
            v_min_s = float(self.get_parameter("v_min_schedule").value)
            v_max_s = float(self.get_parameter("v_max_schedule").value)
            n_s     = int(self.get_parameter("n_schedule").value)
            Q = np.diag([q_d, q_theta])
            R = np.array([[r_delta]])
            self._gs = _GainSchedule(v_min_s, v_max_s, n_s,
                                     self.wb, self._dt, Q, R)
            self.get_logger().info("[LQR] Gain schedule rebuilt.")
        return __import__("rcl_interfaces.msg", fromlist=["SetParametersResult"]).SetParametersResult(successful=True)

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _wpnts_cb(self, msg: WpntArray) -> None:
        if msg.wpnts:
            self._wpnts = msg.wpnts

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear.x
        self._state = (p.x, p.y, _yaw_from_quat(q.x, q.y, q.z, q.w), v)

    # ── Main timer ───────────────────────────────────────────────────────────
    def _tick(self) -> None:
        if self._wpnts is None or self._state is None:
            return

        steer, speed = self._compute()

        # Acceleration ramp (hard braking always allowed)
        speed = min(speed, self._prev_speed + self.max_accel * self._dt)
        self._prev_speed = speed

        cmd = AckermannDriveStamped()
        cmd.header.stamp         = self.get_clock().now().to_msg()
        cmd.header.frame_id      = "base_link"
        cmd.drive.steering_angle = float(steer)
        cmd.drive.speed          = float(speed)
        self.drive_pub.publish(cmd)

    # ── Adaptive lookahead / prediction horizon ───────────────────────────────
    def _adaptive_ld(self, vx: float, kappa: float) -> float:
        """Lookahead distance = prediction arc length [m].
        T = ld / v = k_v / (1 + k_k·|κ|)  → speed-independent time horizon.
        """
        ld = self.k_v * vx / (1.0 + self.k_k * abs(kappa))
        return float(np.clip(ld, self.ld_min, self.ld_max))

    # ── Kinematic bicycle rollout ─────────────────────────────────────────────
    def _rollout(self, x0: float, y0: float, yaw0: float,
                 v: float, delta: float, N: int) -> tuple[float, float, float]:
        """Simulate kinematic bicycle model N steps forward.

        Uses Euler integration with constant v and delta.
        N steps × dt seconds = total prediction time T.

        Args:
            x0, y0, yaw0 : initial pose
            v            : speed [m/s]  (constant throughout rollout)
            delta        : steering angle [rad]  (constant = previous command)
            N            : number of steps

        Returns:
            (x_pred, y_pred, yaw_pred) — predicted pose after N steps
        """
        x, y, yaw = x0, y0, yaw0
        dt   = self._dt
        # yaw rate is constant when v and delta are constant
        dyaw = v * math.tan(float(np.clip(delta, -self.max_steer, self.max_steer))) / self.wb * dt
        vdt  = v * dt
        for _ in range(N):
            x   += vdt * math.cos(yaw)
            y   += vdt * math.sin(yaw)
            yaw += dyaw
        return x, y, _wrap_angle(yaw)

    # ── Core LQR computation (Rollout LQR) ───────────────────────────────────
    def _compute(self) -> tuple[float, float]:
        """Rollout LQR — predicts where the car will be in T seconds, computes
        LQR error at that predicted position instead of a path lookahead point.

        Why: a path-based lookahead can jump over a corner, causing the car to
        cut corners and hit the wall.  Kinematic rollout cannot jump corners
        because it follows the car's physics.

        Flow:
          1. Nearest waypoint  →  ld (prediction arc length)
          2. Rollout T = ld/v  →  (x_p, y_p, yaw_p) using _prev_steer
          3. Project onto path →  e_d_pred, e_θ_pred
          4. LQR feedback      →  δ_fb = −K(v) · [e_d_pred, e_θ_pred]
          5. Curvature FF      →  δ_ff = k_ff · atan(L · κ_pred)
          6. steer = clip(δ_fb + δ_ff)
        """
        x, y, yaw, vx = self._state
        wpts  = self._wpnts
        N_wpt = len(wpts)
        v_eff = max(abs(vx), 0.5)

        wx = np.array([w.x_m for w in wpts])
        wy = np.array([w.y_m for w in wpts])

        # ── 1. Nearest waypoint → ld ─────────────────────────────────────────
        near_idx   = int(np.argmin(np.hypot(wx - x, wy - y)))
        kappa_near = float(wpts[near_idx].kappa_radpm)
        ld         = self._adaptive_ld(v_eff, kappa_near)

        # ── 2. Rollout 공통 설정 ──────────────────────────────────────────────
        T      = ld / v_eff
        N_roll = max(1, int(round(T / self._dt)))
        K      = self._gs.get(v_eff)

        # ── 3. Iterative rollout (fixed-point iteration) ──────────────────────
        #
        #  목표: rollout 입력 δ 와 LQR 출력 δ 가 일치하는 고정점 탐색.
        #
        #  반복:
        #    δ_{n+1} = clip( -K · [e_d(δ_n), e_θ(δ_n)] + δ_ff(δ_n) )
        #
        #  초기값: _prev_steer — 직전 명령이므로 이미 좋은 근사치.
        #  수렴 조건: |δ_{n+1} − δ_n| < iter_tol  또는  n_iter 회 도달.
        #
        delta          = self._prev_steer
        delta_openloop = self._prev_steer  # 보정 전 open-loop δ — 시각화용
        pred_idx     = near_idx        # fallback (loop 진입 전 초기화)
        e_d_pred     = 0.0
        e_theta_pred = 0.0
        delta_fb     = 0.0
        delta_ff     = 0.0
        iters_done   = 0

        for i in range(self.n_iter):
            # (a) rollout: 현재 delta 추정값으로 T초 시뮬레이션
            x_p, y_p, yaw_p = self._rollout(x, y, yaw, v_eff, delta, N_roll)

            # (b) 예측 위치를 경로에 투영 → 오차 계산
            pred_idx   = int(np.argmin(np.hypot(wx - x_p, wy - y_p)))
            wp_pred    = wpts[pred_idx]
            psi_pred   = float(wp_pred.psi_rad)
            kappa_pred = float(wp_pred.kappa_radpm)

            dxp          = x_p - float(wp_pred.x_m)
            dyp          = y_p - float(wp_pred.y_m)
            e_d_pred     = -math.sin(psi_pred) * dxp + math.cos(psi_pred) * dyp
            e_theta_pred = _wrap_angle(yaw_p - psi_pred)

            # (c) 이 오차로 LQR 명령 계산 → 새 delta 추정값
            delta_fb  = float(-(K @ np.array([e_d_pred, e_theta_pred]))[0])
            delta_ff  = self.k_ff * math.atan(self.wb * kappa_pred)
            delta_new = float(np.clip(delta_fb + delta_ff, -self.max_steer, self.max_steer))

            iters_done = i + 1

            # (d) 수렴 판정 + 포화 조기 종료
            # |δ| ≈ max_steer 이면 이미 최대 조향 — 더 반복해도 발산만 일어남.
            saturated = abs(delta_new) >= self.max_steer * 0.99
            if abs(delta_new - delta) < self.iter_tol or saturated:
                delta = delta_new
                break
            delta = delta_new

        steer = delta
        self._prev_steer = steer   # 다음 스텝 rollout 초기값으로 저장

        # ── 4. Speed from predicted waypoint ──────────────────────────────────
        spd_idxs = [(pred_idx + i) % N_wpt for i in range(self.n_avg)]
        speed    = float(np.mean([wpts[i].vx_mps for i in spd_idxs]))
        speed    = float(np.clip(speed * self.speed_scale, 0.0, self.v_max))
        if speed <= 0.0:
            speed = 1.5

        # ── 5. Visualization + logging ────────────────────────────────────────
        self._publish_nearest_marker(float(wpts[pred_idx].x_m), float(wpts[pred_idx].y_m))
        self._publish_predicted_path(x, y, yaw, v_eff, steer, delta_openloop, N_roll)

        self._log_tick += 1
        if self._log_tick >= 50:
            self._log_tick = 0
            self.get_logger().info(
                f"[LQR] v={abs(vx):.2f} m/s  ld={ld:.2f} m  T={T:.2f} s  "
                f"N={N_roll}  iter={iters_done}/{self.n_iter}  "
                f"e_d={e_d_pred:+.3f} m  e_θ={math.degrees(e_theta_pred):+.1f}°  "
                f"δ_fb={math.degrees(delta_fb):+.1f}°  δ_ff={math.degrees(delta_ff):+.1f}°  "
                f"steer={math.degrees(steer):+.1f}°  spd={speed:.2f} m/s"
            )

        return steer, speed

    # ── Visualization ────────────────────────────────────────────────────────
    def _publish_predicted_path(self, x0: float, y0: float, yaw0: float,
                                v: float, delta_corrected: float,
                                delta_openloop: float, N: int) -> None:
        """두 개의 예측 경로를 발행.

        /lqr/predicted_path:
          id=1 (노란 선) : open-loop  — 보정 전 직전 조향 유지 시 경로
                           포화 여부와 무관하게 항상 앞으로 뻗음
          id=2 (청록 선) : closed-loop — LQR 보정 후 경로
                           포화 시 타이트한 원호, 정상 시 완만한 호
          id=3 (초록 구) : 보정 후 경로의 끝점 (T초 뒤 예측 위치)
        """
        def _rollout_pts(d: float) -> list[tuple[float, float]]:
            pts = [(x0, y0)]
            x, y, yaw = x0, y0, yaw0
            dyaw = v * math.tan(float(np.clip(d, -self.max_steer, self.max_steer))) / self.wb * self._dt
            vdt  = v * self._dt
            for _ in range(N):
                x   += vdt * math.cos(yaw)
                y   += vdt * math.sin(yaw)
                yaw += dyaw
                pts.append((x, y))
            return pts

        pts_open   = _rollout_pts(delta_openloop)
        pts_closed = _rollout_pts(delta_corrected)
        now = self.get_clock().now().to_msg()

        def _make_line(ns, mid, pts, r, g, b, width, z):
            m = Marker()
            m.header.frame_id    = "map"
            m.header.stamp       = now
            m.ns                 = ns
            m.id                 = mid
            m.type               = Marker.LINE_STRIP
            m.action             = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x            = width
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            for px, py in pts:
                p = Point(); p.x = float(px); p.y = float(py); p.z = z
                m.points.append(p)
            return m

        # id=1 : open-loop (노란선) — 현재 조향 유지 시
        self.path_pub.publish(
            _make_line("lqr_pred", 1, pts_open,
                       r=1.0, g=0.85, b=0.0, width=0.06, z=0.08))

        # id=2 : closed-loop (청록선) — LQR 보정 후
        self.path_pub.publish(
            _make_line("lqr_pred", 2, pts_closed,
                       r=0.0, g=0.95, b=1.0, width=0.06, z=0.10))

        # id=3 : closed-loop 끝점 (초록 구)
        tip = Marker()
        tip.header.frame_id    = "map"
        tip.header.stamp       = now
        tip.ns                 = "lqr_pred"
        tip.id                 = 3
        tip.type               = Marker.SPHERE
        tip.action             = Marker.ADD
        tip.pose.position.x    = float(pts_closed[-1][0])
        tip.pose.position.y    = float(pts_closed[-1][1])
        tip.pose.position.z    = 0.15
        tip.pose.orientation.w = 1.0
        tip.scale.x = tip.scale.y = tip.scale.z = 0.25
        tip.color.r = 0.0; tip.color.g = 1.0; tip.color.b = 0.3; tip.color.a = 1.0
        self.path_pub.publish(tip)

    def _publish_nearest_marker(self, wx: float, wy: float) -> None:
        m = Marker()
        m.header.frame_id    = "map"
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = "lqr_nearest"
        m.id                 = 0
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = wx
        m.pose.position.y    = wy
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LQRController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
