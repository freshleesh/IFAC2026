"""Friction-Circle Acceleration-Allocation Controller.

Pipeline (each _loop tick):
  PP geometry → κ_PP (geometric only) → a_lat_req = v²·κ_PP
  feedforward applied to δ_cmd AFTER friction-circle (not before — see Note A)
  velocity profile → v_ref → a_long_des = (v_ref - v) / dt
  friction-circle: lateral priority, remaining budget to ACCELERATION only
  braking budget is independent (a_brake_max, not circle-limited — see Note B)
  back-calculate: δ_cmd = clip(atan(a_lat·L/v²) + δ_ff, ±δ_max)
                  v_cmd = clip(v + a_long·dt, v_min, v_max)
  acceleration ramp: v_cmd ≤ v + max_accel·dt  (prevent front lift — see Note C)

Note A — feedforward not in a_lat_req:
  Including FF in κ_total inflates a_lat_req on the approach straight
  (car is still straight but κ_target is the upcoming hairpin) → exceeds
  a_total_max → a_long_budget=0 → no braking AND steering reduced.
  Fix: compute a_lat_req from geometric κ_PP only; add δ_ff to δ_cmd afterward.

Note B — asymmetric braking budget:
  Braking uses 4-wheel friction; traction circle limit is for drive-wheel
  traction + cornering. Artificially limiting decel to a_long_budget would
  prevent the car from braking hard before corners (same problem PP does NOT have).
  Fix: deceleration budget = a_brake_max (param), independent of a_lat.

Note C — acceleration ramp:
  PP limits speed increase to max_accel·dt per step (VESC/front-lift protection).
  FC without this ramp can accelerate at up to a_total_max = 6 m/s² per step
  at hairpin exit (low a_lat → large budget) → 3× PP acceleration rate.

Responsibility split:
  - velocity profile: anticipatory deceleration before corners (look-ahead)
  - friction circle: 1-step physical safety net / profile-error correction
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

PARAMS = {
    # vehicle physics
    'wheelbase_L':      0.33,
    'delta_max':        0.41,
    'v_min':            0.0,
    'v_max':            7.0,
    'v_min_for_steer':  0.5,
    # friction circle — primary tuning knob (μ·g for traction)
    'a_total_max':      6.0,
    'use_ellipse':      False,
    'a_lat_max':        6.0,
    'a_long_max':       4.0,
    # braking budget — independent of friction circle (4-wheel braking)
    'a_brake_max':      8.0,
    # control timing
    'control_rate_hz':  50.0,
    # speed command horizon [s]: budget × t_cmd_horizon = speed target above vx.
    # Single-step (dt=0.02 s) gives vx+0.1 → VESC PID barely responds.
    # 0.5 s gives vx+2.5 on a straight → VESC accelerates at full budget rate.
    't_cmd_horizon':    0.5,
    # PP adaptive lookahead
    'ld_min':           0.6,
    'ld_max':           1.5,
    'k_v':              0.3,
    'k_k':              2.0,
    'k_window':         0.1,
    'n_targets':        3,
    # speed-adaptive feedforward
    'k_ff':             0.55,
    'k_ff_low':         0.20,
    'v_ff_sat':         4.0,
}

# /fc_debug Float64MultiArray field indices
_D_A_LAT_REQ   = 0   # [m/s²] geometric lateral demand
_D_A_LAT       = 1   # [m/s²] allocated lateral (clipped to friction limit)
_D_A_LONG_BUDG = 2   # [m/s²] longitudinal accel budget (positive side)
_D_A_LONG      = 3   # [m/s²] allocated longitudinal
_D_V_REF       = 4   # [m/s]  velocity profile reference
_D_V_CMD       = 5   # [m/s]  commanded speed
_D_DELTA_DEG   = 6   # [deg]  commanded steering angle
_DEBUG_LEN     = 7


def allocate_acceleration(
        a_lat_req: float,
        a_long_des: float,
        a_total_max: float,
        a_brake_max: float,
        use_ellipse: bool = False,
        a_lat_max: float = 6.0,
        a_long_max: float = 4.0,
) -> tuple[float, float, float]:
    """Friction-circle allocation with asymmetric braking budget.

    Lateral priority: allocate lateral first, remainder to acceleration.
    Braking is NOT limited by friction circle (a_brake_max is independent).

    Returns (a_lat, a_long, a_long_budget_pos).
    """
    if use_ellipse:
        a_lat = float(np.clip(a_lat_req, -a_lat_max, +a_lat_max))
        long_budget_sq = max(0.0, 1.0 - (a_lat / a_lat_max) ** 2) * a_long_max ** 2
        a_long_budget_pos = math.sqrt(long_budget_sq)
    else:
        a_lat = float(np.clip(a_lat_req, -a_total_max, +a_total_max))
        a_long_budget_pos = math.sqrt(max(0.0, a_total_max ** 2 - a_lat ** 2))

    a_long = float(np.clip(a_long_des, -a_brake_max, +a_long_budget_pos))
    return a_lat, a_long, a_long_budget_pos


def accel_to_command(
        a_lat: float, a_long: float, v: float,
        wheelbase: float, v_min_for_steer: float,
        delta_max: float, v_min: float, v_max: float, dt: float,
) -> tuple[float, float]:
    """Inverse kinematic bicycle: (a_lat, a_long) → (δ_geometric, v_cmd).

    Note: feedforward steering is NOT added here — callers add it afterward.
    """
    v_safe = max(v, v_min_for_steer)
    delta_geo = math.atan(a_lat * wheelbase / (v_safe * v_safe))
    delta_geo = float(np.clip(delta_geo, -delta_max, +delta_max))

    v_cmd = v + a_long * dt
    v_cmd = float(np.clip(v_cmd, v_min, v_max))
    return delta_geo, v_cmd


class FrictionCircleController(Node):

    def __init__(self):
        super().__init__('friction_circle_controller')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.wheelbase     = float(p('wheelbase_L'))
        self.delta_max     = float(p('delta_max'))
        self.v_min         = float(p('v_min'))
        self.v_max         = float(p('v_max'))
        self.v_min_steer   = float(p('v_min_for_steer'))
        self.a_total_max   = float(p('a_total_max'))
        self.use_ellipse   = bool(p('use_ellipse'))
        self.a_lat_max     = float(p('a_lat_max'))
        self.a_long_max    = float(p('a_long_max'))
        self.a_brake_max   = float(p('a_brake_max'))
        self._dt           = 1.0 / float(p('control_rate_hz'))
        self.t_cmd_horizon = float(p('t_cmd_horizon'))

        self.ld_min        = float(p('ld_min'))
        self.ld_max        = float(p('ld_max'))
        self.k_v           = float(p('k_v'))
        self.k_k           = float(p('k_k'))
        self.k_window      = float(p('k_window'))
        self.n_targets     = int(p('n_targets'))
        self.k_ff          = float(p('k_ff'))
        self.k_ff_low      = float(p('k_ff_low'))
        self.v_ff_sat      = max(float(p('v_ff_sat')), 0.5)

        self.odom         = None
        self.waypoints    = []
        self._log_tick    = 0

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry,  '/vesc/odom',         self._odom_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints',   self._wp_cb,   10)
        self.drive_pub     = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.debug_pub     = self.create_publisher(Float64MultiArray, '/fc_debug', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/fc/lookahead', 10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info(
            f'[FC] a_total_max={self.a_total_max} m/s²  a_brake_max={self.a_brake_max} m/s²  '
            f't_cmd_horizon={self.t_cmd_horizon} s  use_ellipse={self.use_ellipse}  '
            f'L={self.wheelbase} m  dt={self._dt*1000:.1f} ms'
        )

    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    def _adaptive_lookahead(self, vx: float, kappa: float) -> float:
        ld = self.k_v * vx / (1.0 + self.k_k * abs(kappa))
        return float(np.clip(ld, self.ld_min, self.ld_max))

    def _publish_lookahead(self, x: float, y: float, ld: float) -> None:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = 'fc_lookahead', 0, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.0
        m.pose.orientation.w = 1.0
        s = float(np.clip(ld * 0.15, 0.1, 0.4))
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
        self.lookahead_pub.publish(m)

    def _run_pp(self, vx: float):
        """Pure Pursuit pass — local sliding-window waypoints.

        /local_waypoints is an OPEN, car-centred window (state_machine publishes
        ~80 wpnts ahead at 50 Hz), NOT a closed full-lap array:
          - nearest search is a plain argmin over the whole window (cheap, and
            the window slides with the car so monotonic index tracking would
            break across callbacks anyway)
          - distance-ahead is the cumulative euclidean arc length from the
            nearest point (the global s_m wraps at the lap boundary mid-window,
            so (s - s_nearest) % s_total is NOT usable here)
          - indices clamp at the window end instead of wrapping with % N

        Returns (kappa_pp, steer_ff, v_ref, x_blend, y_blend, ld) or
                (None, ...)  on degenerate geometry.

        kappa_pp  : geometric curvature 2·ly/L² (signed, rad/m).
                    Does NOT include feedforward — feedforward is returned
                    separately so it can be applied to δ_cmd after the
                    friction-circle allocation (see Note A in module docstring).
        steer_ff  : feedforward steering angle [rad] to add to δ_cmd.
        """
        p_x = self.odom.pose.pose.position.x
        p_y = self.odom.pose.pose.position.y
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        wpnts = self.waypoints
        wx = np.array([wp.x_m for wp in wpnts])
        wy = np.array([wp.y_m for wp in wpnts])
        N  = len(wpnts)

        nearest_idx = int(np.argmin(np.hypot(wx - p_x, wy - p_y)))
        last = N - 1

        # Cumulative arc length ahead of the nearest point (open window, no wrap).
        # ahead_s[k] = distance along the raceline from nearest to nearest+k.
        seg = np.hypot(np.diff(wx[nearest_idx:]), np.diff(wy[nearest_idx:]))
        ahead_s = np.concatenate([[0.0], np.cumsum(seg)])

        def idx_at(s_t: float) -> int:
            """Window index closest to s_t metres ahead (clamps at window end)."""
            return nearest_idx + int(np.argmin(np.abs(ahead_s - s_t)))

        N_kappa     = 5
        cur_idx     = [min(nearest_idx + i, last) for i in range(N_kappa)]
        kappa_now   = float(np.mean([abs(wpnts[i].kappa_radpm) for i in cur_idx]))
        ld_prelim   = self._adaptive_lookahead(vx, kappa_now)
        preview_idx = idx_at(ld_prelim)
        pre_idx     = [min(preview_idx + i, last) for i in range(N_kappa)]
        kappa_ahead = float(np.mean([abs(wpnts[i].kappa_radpm) for i in pre_idx]))
        vx_preview  = float(wpnts[preview_idx].vx_mps)
        vx_for_ld   = max(vx, 0.75 * vx_preview)
        ld          = self._adaptive_lookahead(vx_for_ld, max(kappa_now, kappa_ahead))

        n_t      = max(1, self.n_targets)
        s_tgts   = np.linspace(ld, ld + self.k_window * vx, n_t)
        x_blend = y_blend = 0.0
        for s_t in s_tgts:
            i = idx_at(s_t)
            x_blend += wx[i]
            y_blend += wy[i]
        x_blend /= n_t
        y_blend /= n_t

        target_idx = idx_at(ld)

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dtx = x_blend - p_x
        dty = y_blend - p_y
        lx =  dtx * cos_y + dty * sin_y
        ly = -dtx * sin_y + dty * cos_y
        L  = math.hypot(lx, ly)

        if L < 1e-6:
            return None, 0.0, 1.5, x_blend, y_blend, ld

        # Geometric curvature (signed: + = left turn)
        kappa_pp = 2.0 * ly / (L * L)

        # Feedforward: speed-adaptive gain on raceline curvature at target point.
        # Returned separately so it is applied to δ_cmd AFTER friction-circle
        # allocation, not before (avoids inflating a_lat_req — see Note A).
        v_ratio  = min(vx / self.v_ff_sat, 1.0)
        k_ff_eff = self.k_ff_low + (self.k_ff - self.k_ff_low) * v_ratio
        kappa_at_target = float(wpnts[target_idx].kappa_radpm)
        steer_ff = k_ff_eff * math.atan(self.wheelbase * kappa_at_target)

        # Anticipatory velocity: minimum speed in the braking-distance window.
        # vx²/(2·a_brake) is how far ahead the car needs to start decelerating.
        # Using min (not mean-at-target) catches hairpins before they enter
        # the steering lookahead, preventing late braking.
        v_lhd  = vx ** 2 / (2.0 * self.a_brake_max) + self.ld_min
        v_last = int(np.searchsorted(ahead_s, v_lhd, side='right'))
        if v_last > 0:
            v_ref = float(min(
                wpnts[nearest_idx + k].vx_mps for k in range(min(v_last, N - nearest_idx))
            ))
        else:
            v_ref = float(wpnts[target_idx].vx_mps)
        v_ref = max(v_ref, 1.5)

        return kappa_pp, steer_ff, v_ref, x_blend, y_blend, ld

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

        vx = abs(self.odom.twist.twist.linear.x)

        result = self._run_pp(vx)
        kappa_pp, steer_ff, v_ref, tx, ty, ld = result
        self._publish_lookahead(tx, ty, ld)

        if kappa_pp is None:
            self._publish_drive(0.0, 1.5)
            return

        # Low-speed guard: centripetal model a_lat=v²·κ → 0 as v→0.
        # Fall back to pure PP output to maintain steering at low speed.
        if vx < self.v_min_steer:
            steer_pp = math.atan(self.wheelbase * kappa_pp)
            delta_fallback = float(np.clip(steer_pp + steer_ff, -self.delta_max, self.delta_max))
            self._publish_drive(delta_fallback, v_ref)
            self._publish_debug(0.0, 0.0, self.a_total_max, 0.0, v_ref, v_ref,
                                math.degrees(delta_fallback))
            return

        # (2) Required lateral acceleration — geometric κ_PP only.
        # Feedforward is NOT included here (see Note A in module docstring).
        a_lat_req = (vx * vx) * kappa_pp

        # (3) Desired longitudinal acceleration from velocity profile.
        # Profile is a reference/target; friction circle makes the final call.
        a_long_des = (v_ref - vx) / self._dt

        # (4) Friction-circle: lateral allocation only (a_long_budget for debug/accel).
        a_lat, _, a_long_budget_pos = allocate_acceleration(
            a_lat_req, a_long_des, self.a_total_max, self.a_brake_max,
            self.use_ellipse, self.a_lat_max, self.a_long_max,
        )

        # (5) Back-calculate commands.
        v_safe    = max(vx, self.v_min_steer)
        delta_geo = math.atan(a_lat * self.wheelbase / (v_safe * v_safe))
        # Feedforward added after friction-circle so it doesn't inflate a_lat_req.
        delta_cmd = float(np.clip(delta_geo + steer_ff, -self.delta_max, self.delta_max))

        if a_long_des <= 0.0:
            # Braking: command v_ref directly — same as PP, VESC handles decel.
            v_cmd  = float(np.clip(v_ref, self.v_min, self.v_max))
            # Debug: clip to physical brake limit so GG diagram stays in range.
            # (v_cmd - vx)/dt can be hundreds of m/s² when v_ref << vx because
            # we're commanding a distant target, not stepping incrementally.
            a_long = float(np.clip((v_cmd - vx) / self._dt,
                                   -self.a_brake_max, 0.0))
        else:
            # Acceleration: budget-ratio-scaled horizon so VESC gets a meaningful
            # target on straights, but the horizon shrinks automatically in corners.
            #   budget_ratio = 1 (straight)  → t_eff = t_cmd_horizon (aggressive)
            #   budget_ratio ≈ 0 (apex)      → t_eff ≈ dt             (conservative)
            a_long = min(a_long_des, a_long_budget_pos)   # physics-based budget limit
            budget_ratio = a_long_budget_pos / self.a_total_max  # 0..1
            t_eff = max(self._dt, self.t_cmd_horizon * budget_ratio)
            v_cmd = min(v_ref, vx + a_long * t_eff)
            v_cmd = float(np.clip(v_cmd, self.v_min, self.v_max))
            # Keep a_long as the friction-circle-allocated value (NOT (v_cmd-vx)/dt).
            # (v_cmd-vx)/dt = a_long × t_eff/dt = a_long × 50 when t_cmd_horizon=1s
            # → 300 m/s² spikes that make the GG diagram unreadable.

        self._publish_drive(delta_cmd, v_cmd)
        self._publish_debug(a_lat_req, a_lat, a_long_budget_pos, a_long,
                            v_ref, v_cmd, math.degrees(delta_cmd))

        self._log_tick += 1
        if self._log_tick >= 50:
            self._log_tick = 0
            self.get_logger().info(
                f'[FC] v={vx:.2f} ref={v_ref:.2f} cmd={v_cmd:.2f}  '
                f'a_lat={a_lat:.2f}/{a_lat_req:.2f}  '
                f'a_long={a_long:.2f} budg={a_long_budget_pos:.2f}  '
                f'δ={math.degrees(delta_cmd):.1f}° (ff={math.degrees(steer_ff):.1f}°)'
            )

    def _publish_drive(self, steer: float, speed: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _publish_debug(
            self,
            a_lat_req: float, a_lat: float, a_long_budget: float, a_long: float,
            v_ref: float, v_cmd: float, delta_deg: float,
    ) -> None:
        # Field order: [a_lat_req, a_lat, a_long_budget, a_long, v_ref, v_cmd, delta_deg]
        d = Float64MultiArray()
        d.data = [a_lat_req, a_lat, a_long_budget, a_long, v_ref, v_cmd, delta_deg]
        self.debug_pub.publish(d)


def main(args=None):
    rclpy.init(args=args)
    node = FrictionCircleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
