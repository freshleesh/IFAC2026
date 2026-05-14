#!/usr/bin/env python3
"""ROS2 wrapper for the EVO-MPCC core (acados / IPOPT backends).

This is the **skeleton** of the ROS2 node — full feature parity with the
original ROS1 `Nonlinear_MPC_node.py` (56 KB) is reached incrementally.
What's here today:

  - Parameter declaration (mirroring `ddrx_unified_params.yaml`)
  - Subscriptions: odom, pose, goal, clicked_point, external_obstacles
  - Publications: ackermann cmd, MPC trajectory (Path + MPCTrajectory),
                  boundary marker, debug topics
  - 40 Hz control timer that pulls latest state, runs `mpc.solve(...)`,
    publishes outputs
  - boundary_hook adapter that builds MarkerArray from the points the
    core hands back
  - on_set_parameters_callback that pushes live-tunable weights into
    `mpc.q_*_live` attributes (replaces ROS1 dynamic_reconfigure)

What's intentionally still TODO (carried over from the ROS1 node, ported
in follow-up commits):

  - Track data loading: read centerline / boundary / waypoints CSV,
    build CasADi spline interpolants, call `mpc.set_track_data(...)`.
    See `ros1_source/nonlinear_mpc_casadi/scripts/Nonlinear_MPC_node.py`
    (`load_track_data` / `_build_*_lut`).
  - Static-obstacle preload from `~static_obstacles` param.
  - Lap-counter integration (consume LapData).
  - Joy / state machine integration.
  - `/behavior_strategy` direct publish bypass.
  - Visualization: `/center_path`, `/right_path`, `/left_path`,
    `/reference_path` (latched), trajectory marker arrows.

The numerical MPC code (`mpc_core.acados_kinematic.MPC`) is unchanged
from the ROS1 version after Phase 1's ROS-decoupling — the wrapper here
only handles ROS2 IO. Replace `MPC` with `mpc_core.ipopt_kinematic.MPC`
to swap the backend (mirrors the ROS1 `mpc_backend` parameter).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from .track_loader import TrackData, build_track_from_wpnts, find_current_arc_length, load_track

from std_msgs.msg import Bool, ColorRGBA, Float32MultiArray, Float64, Header
from geometry_msgs.msg import (
    Point, PointStamped, PoseArray, PoseStamped, Quaternion, Vector3,
)
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

from osuf1_common.msg import MPCTrajectory, MPCPrediction
from f110_msgs.msg import LapData, WpntArray  # noqa: F401  (used in TODO sections)


# ──────────────────────────────────────────────────────────────────────────
# Backend selection — keep symmetric with ROS1 `mpc_backend` parameter
# ──────────────────────────────────────────────────────────────────────────
def _load_mpc_backend(name: str):
    if name == 'acados':
        from .mpc_core.acados_kinematic import MPC
        return MPC
    if name == 'ipopt':
        from .mpc_core.ipopt_kinematic import MPC
        return MPC
    raise ValueError(f"unknown mpc_backend: {name!r} (expected 'acados' or 'ipopt')")


# ──────────────────────────────────────────────────────────────────────────
# Adapter: rclpy logger → mpc_core logger interface
# ──────────────────────────────────────────────────────────────────────────
class _RclpyLoggerAdapter:
    """Bridge rclpy node logger to the mpc_core logger contract.

    `mpc_core` calls `info(msg, *args)` with %-style format args (matches
    the rospy convention). rclpy's logger expects a pre-formatted string,
    so we format here. Throttled variants emulate `rospy.logwarn_throttle`
    by tracking last-fire time per message template.
    """
    def __init__(self, node_logger):
        self._lg = node_logger
        self._last: dict[str, float] = {}

    @staticmethod
    def _fmt(msg, args):
        try:
            return msg % args if args else msg
        except Exception:
            return msg + " " + " ".join(str(a) for a in args)

    def info(self, msg, *args):    self._lg.info(self._fmt(msg, args))
    def warn(self, msg, *args):    self._lg.warn(self._fmt(msg, args))
    def warning(self, msg, *args): self._lg.warn(self._fmt(msg, args))
    def error(self, msg, *args):   self._lg.error(self._fmt(msg, args))
    def debug(self, msg, *args):   self._lg.debug(self._fmt(msg, args))

    def _throttled(self, fn, period: float, msg, args):
        now = time.monotonic()
        if now - self._last.get(msg, 0.0) >= period:
            self._last[msg] = now
            fn(msg, *args)

    def info_throttle(self, period, msg, *args):
        self._throttled(self.info, period, msg, args)

    def warn_throttle(self, period, msg, *args):
        self._throttled(self.warn, period, msg, args)


# ──────────────────────────────────────────────────────────────────────────
# Live-tunable parameters — names match `mpc_core` `*_live` attributes.
# Updated via `on_set_parameters_callback` (replaces ROS1 dyn_reconfigure).
# ──────────────────────────────────────────────────────────────────────────
LIVE_PARAMS = [
    ('q_cte_live', 8.0),
    ('q_lag_live', 200.0),
    ('q_d_delta_live', 25.0),
    ('R_safe_live', 0.3),
    ('M_slack_live', 2.0e4),
    ('a_lat_safe_live', 6.0),
    ('D_detour_live', 0.15),
    ('D_apex_live', 0.22),
    ('R_car_live', 0.0),
    ('commit_dist_live', 10.0),
    ('cost_spike_thr_live', 500.0),
    ('alpha_steer_live', 0.6),
    # acados-specific scale slots
    ('q_cte_scale_live', 1.0),
    ('q_lag_scale_live', 1.0),
    ('q_psi_scale_live', 1.0),
    ('q_v_scale_live', 1.0),
    ('q_dd_scale_live', 1.0),
    ('q_p_scale_live', 1.0),
    ('q_drate_scale_live', 1.0),
]


class MPCNode(Node):
    CONTROLLER_FREQ = 40.0  # Hz — matches ROS1 default

    def __init__(self):
        super().__init__('mpc_node')
        self._latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('mpc_backend', 'acados')
        self.declare_parameter('odom_topic_name', '/car_state/odom')
        self.declare_parameter('localized_pose_topic_name', '/car_state/pose')
        self.declare_parameter('goal_topic_name', '/move_base_simple/goal')
        self.declare_parameter('cmd_vel_topic_name',
                               '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        # Track / MPC sizing
        self.declare_parameter('track_name', 'f')
        self.declare_parameter('track_dir', '')   # empty → use share/tracks/
        self.declare_parameter('vel_scale', 0.95)
        self.declare_parameter('inflation_factor', 1.2)
        self.declare_parameter('extend_part', 2)
        # Fixed-width corridor (centerline ± half). >0 = enabled, 0 = raw
        # wpnt.d_left/d_right + inflation (기존 동작). 자세한 설명: ddrx_unified_params.yaml.
        self.declare_parameter('mpc_corridor_half_width', 0.0)
        self.declare_parameter('vehicle_L', 0.307)
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('max_speed_p', 6.0)
        self.declare_parameter('mpc_max_steering', 0.4)
        self.declare_parameter('dT', 0.025)
        self.declare_parameter('N_horizon', 18)
        self.declare_parameter('integration_mode', 'Euler')
        self.declare_parameter('params_file', 'BO_params_LTM')
        # Track source: 'centerline' (/centerline_waypoints) or 'raceline'
        # (/global_waypoints, IQP 사전 최적 raceline). raceline 사용 시 mpc
        # 가 racing line 추종 — 코너에서 자연스럽게 apex 통과 (EVO-MPCC /
        # Liniger MPCC 의 표준 패턴).
        self.declare_parameter('track_source', 'centerline')
        # Vehicle dynamics (only `l_wb` consumed in kinematic mode; full set
        # used by acados dynamic Pacejka mode — kept here so swapping
        # use_dynamic doesn't require config changes.)
        self.declare_parameter('vehicle.l_wb', 0.307)
        self.declare_parameter('vehicle.l_f', 0.162)
        self.declare_parameter('vehicle.l_r', 0.145)
        self.declare_parameter('vehicle.m', 3.54)
        for name, default in LIVE_PARAMS:
            self.declare_parameter(name, default)

        backend_name = self.get_parameter('mpc_backend').value
        odom_topic = self.get_parameter('odom_topic_name').value
        pose_topic = self.get_parameter('localized_pose_topic_name').value
        goal_topic = self.get_parameter('goal_topic_name').value
        cmd_topic = self.get_parameter('cmd_vel_topic_name').value

        # ── MPC instance (numerical core) ───────────────────────────
        MPC = _load_mpc_backend(backend_name)
        self._mpc_log = _RclpyLoggerAdapter(self.get_logger())
        self.mpc = MPC(cost_type=None, system_model=None, logger=self._mpc_log)
        self.mpc.boundary_hook = self._on_boundary_points
        self._push_live_params_to_mpc()

        # ── Publications ────────────────────────────────────────────
        self.ackermann_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)
        self.mpc_traj_pub = self.create_publisher(Path, '/mpc_trajectory', 10)
        # MarkerArray twin of /mpc_trajectory — small magenta spheres, one per
        # mpc stage. Matches the visual style of /center_path etc. The Path
        # version is kept for mpc_debug_logger / other consumers.
        self.mpc_traj_markers_pub = self.create_publisher(
            MarkerArray, '/mpc_trajectory/markers', 10)
        self.boundary_pub = self.create_publisher(MarkerArray, '/boundary_marker', 10)
        self.prediction_pub = self.create_publisher(MPCTrajectory, '/mpc/prediction', 1)
        self.cost_pub = self.create_publisher(Float64, '/mpc/cost', 10)
        self.solve_time_pub = self.create_publisher(Float64, '/mpc/solve_time', 10)
        self.feasible_pub = self.create_publisher(Bool, '/mpc/is_feasible', 10)
        self.mpc_debug_pub = self.create_publisher(Float32MultiArray, '/mpc_debug', 10)
        # Latched (TRANSIENT_LOCAL) viz publishers — small-sphere MarkerArray
        # in the style of race-stack's /centerline_waypoints/markers (type=2
        # SPHERE, scale 0.05). Different color per topic so RViz tells them
        # apart at a glance.
        #   /center_path   raw centerline (white)
        #   /right_path    raw right boundary (red)
        #   /left_path     raw left boundary (green)
        #   /reference_path mpc reference = centerline (yellow)
        self.center_path_pub    = self.create_publisher(MarkerArray, '/center_path',    self._latched_qos)
        self.right_path_pub     = self.create_publisher(MarkerArray, '/right_path',     self._latched_qos)
        self.left_path_pub      = self.create_publisher(MarkerArray, '/left_path',      self._latched_qos)
        self.reference_path_pub = self.create_publisher(MarkerArray, '/reference_path', self._latched_qos)

        # ── Subscriptions ───────────────────────────────────────────
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 1)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 1)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 1)
        self.create_subscription(PointStamped, '/clicked_point',
                                 self._clicked_point_cb, 4)
        self.create_subscription(PoseArray, '/external_obstacles',
                                 self._external_obs_cb, 1)

        # ── State cache (most-recent message snapshots) ─────────────
        self._last_odom: Odometry | None = None
        self._last_pose: PoseStamped | None = None
        self._goal: PoseStamped | None = None
        self._obstacles: list[tuple[float, float]] = []
        self._obstacles_stamp: float = 0.0
        self._lap_count: int = 0  # TODO: subscribe to LapData
        self._mpc_ready: bool = False
        self._track: TrackData | None = None

        # ── Live parameter callback (replaces dynamic_reconfigure) ──
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Track ingestion: prefer live race-stack /centerline_waypoints
        # (so mpc shares the sim's coordinate frame — origin / map.yaml
        # mismatches between CSV-baked tracks and race-stack maps would
        # otherwise put the reference path in the wrong world location).
        # CSV fallback fires after `_track_csv_fallback_sec` if no wpnts
        # arrive — preserves standalone-CSV use case (bag replay, dry run).
        self._track_init_done = False
        self._track_csv_fallback_sec = 5.0
        # global_republisher publishes both /centerline_waypoints and
        # /global_waypoints with default QoS (RELIABLE + VOLATILE).
        # track_source param selects which one mpc uses as reference:
        #   centerline → /centerline_waypoints (864 wpnts, raw centerline)
        #   raceline   → /global_waypoints     (766 wpnts, IQP raceline)
        track_source = str(self.get_parameter('track_source').value).strip().lower()
        if track_source not in ('centerline', 'raceline'):
            self.get_logger().warn(
                f"unknown track_source={track_source!r} — falling back to 'centerline'")
            track_source = 'centerline'
        self._track_source = track_source
        ref_topic = ('/global_waypoints' if track_source == 'raceline'
                     else '/centerline_waypoints')
        self.get_logger().info(
            f"mpc reference path: {track_source} ({ref_topic})")
        self.create_subscription(
            WpntArray, ref_topic, self._on_centerline_wpnts, 10)
        self._track_fallback_timer = self.create_timer(
            self._track_csv_fallback_sec, self._track_csv_fallback_cb)

        # ── Control loop ────────────────────────────────────────────
        period = 1.0 / self.CONTROLLER_FREQ
        self.control_timer = self.create_timer(period, self._control_loop_cb)

        self.get_logger().info(
            f"MPC node up — backend={backend_name}, "
            f"rate={self.CONTROLLER_FREQ:.0f} Hz, ready={self._mpc_ready} "
            f"(waiting up to {self._track_csv_fallback_sec:.1f}s for "
            "/centerline_waypoints, then CSV fallback)")

    # ─────────────────────────────────────────────────────────────────
    # Track + MPC setup
    # ─────────────────────────────────────────────────────────────────
    def _share_dir(self) -> str:
        return get_package_share_directory('nonlinear_mpc_acados')

    def _resolve_track_dir(self) -> str:
        explicit = str(self.get_parameter('track_dir').value or '').strip()
        if explicit:
            return explicit
        # Default: share/tracks/ inside this installed package
        return os.path.join(self._share_dir(), 'tracks')

    def _resolve_bo_params(self) -> dict:
        """Load Bayesian-Opt MPC weights (BO_params_*.json) from share/config/mpc/."""
        params_file = self.get_parameter('params_file').value
        json_path = os.path.join(self._share_dir(), 'config', 'mpc', f'{params_file}.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"MPC weights JSON not found: {json_path}")
        with open(json_path) as f:
            return json.load(f)

    def _build_param_dict(self, bo: dict) -> dict:
        """Match `getPreDefinedParas()` from the ROS1 node — keys mpc_core consumes."""
        gp = self.get_parameter
        return {
            'is_jit': False,
            'mpc_v_track':    float(bo.get('q_v', 0.0)),
            'mpc_w_cte':      float(bo['q_cte']),
            'mpc_w_lag':      float(bo['q_lag']),
            'mpc_w_accel':    float(bo['q_dv']),
            'mpc_w_delta_d':  float(bo['q_d_delta']),
            'mpc_w_delta_vp': float(bo['q_dvp']),
            'mpc_vp_project': float(bo['gamma']),
            'N': int(gp('N_horizon').value),
            'dT': float(gp('dT').value),
            'theta_max': float(gp('mpc_max_steering').value),
            'v_max': float(gp('max_speed').value),
            'p_min': 0.0,
            'p_max': float(gp('max_speed_p').value),
            'INTEGRATION_MODE': str(gp('integration_mode').value),
            'L': float(gp('vehicle_L').value),
            'x_min': -200.0, 'x_max': 200.0,
            'y_min': -200.0, 'y_max': 200.0,
            'psi_min': -1000.0, 'psi_max': 1000.0,
            's_min': 0.0, 's_max': 200.0,        # s_max overwritten after track load
            'Vbias_max': 10.0,
            'q_mu': 0.5,
            'M_slack': 100.0,
            'R_safe': 0.5,
            'a_lat_max': 12.0,
            'w_alat': 0.0,
        }

    def _on_centerline_wpnts(self, msg: WpntArray) -> None:
        """Race-stack `/centerline_waypoints` ingestion. Fires once; subsequent
        messages are ignored (the same static track is republished at 0.5Hz).
        Cancels the CSV fallback timer."""
        if self._track_init_done:
            return
        self._track_init_done = True
        if self._track_fallback_timer is not None:
            self._track_fallback_timer.cancel()
            self._track_fallback_timer = None
        try:
            self._initialize_mpc(wpnts=msg.wpnts)
        except Exception as e:
            self.get_logger().error(
                f"MPC init from /centerline_waypoints failed: {e}. Control loop will idle.")
            self._mpc_ready = False

    def _track_csv_fallback_cb(self) -> None:
        """No `/centerline_waypoints` arrived within the grace window — fall
        back to bundled CSV (standalone use, bag replay, IFAC dry-run)."""
        if self._track_init_done:
            return
        self._track_init_done = True
        if self._track_fallback_timer is not None:
            self._track_fallback_timer.cancel()
            self._track_fallback_timer = None
        self.get_logger().warn(
            f"/centerline_waypoints not seen in {self._track_csv_fallback_sec:.1f}s — falling back to CSV")
        try:
            self._initialize_mpc(wpnts=None)
        except Exception as e:
            self.get_logger().error(
                f"MPC CSV-fallback init failed: {e}. Control loop will idle.")
            self._mpc_ready = False

    def _initialize_mpc(self, wpnts=None) -> None:
        vel_scale = float(self.get_parameter('vel_scale').value)
        inflation = float(self.get_parameter('inflation_factor').value)
        extend_part = int(self.get_parameter('extend_part').value)

        if wpnts is not None:
            corridor_half = float(self.get_parameter('mpc_corridor_half_width').value)
            mode_desc = (f"fixed corridor ±{corridor_half:.2f}m"
                         if corridor_half > 1e-3
                         else f"raw d_left/d_right + inflation {inflation}")
            self.get_logger().info(
                f"building track from /centerline_waypoints ({len(wpnts)} wpnts, "
                f"vel_scale={vel_scale}, {mode_desc})")
            self._track = build_track_from_wpnts(
                wpnts, vel_scale=vel_scale,
                inflation_factor=inflation, extend_part=extend_part,
                default_v=float(self.get_parameter('max_speed').value),
                corridor_half_width=corridor_half)
        else:
            track_dir = self._resolve_track_dir()
            track_name = str(self.get_parameter('track_name').value)
            self.get_logger().info(
                f"loading track '{track_name}' from {track_dir} "
                f"(vel_scale={vel_scale}, inflation={inflation})")
            self._track = load_track(track_dir, track_name,
                                     vel_scale=vel_scale,
                                     inflation_factor=inflation,
                                     extend_part=extend_part)
        L_orig = float(self._track.element_arc_lengths_orig[-1])
        L_ext = float(self._track.element_arc_lengths[-1])
        self.get_logger().info(
            f"track loaded: N_orig={len(self._track.element_arc_lengths_orig)}, "
            f"L_orig={L_orig:.2f} m, L_ext={L_ext:.2f} m")

        bo = self._resolve_bo_params()
        param = self._build_param_dict(bo)
        param['s_max'] = L_ext  # match ROS1 (extended)
        is_ot = bool(bo.get('overtaking', False))

        # Vehicle dynamics dict — kinematic mode only consumes `l_wb`
        vheid = {
            'l_wb': float(self.get_parameter('vehicle.l_wb').value),
            'l_f':  float(self.get_parameter('vehicle.l_f').value),
            'l_r':  float(self.get_parameter('vehicle.l_r').value),
            'm':    float(self.get_parameter('vehicle.m').value),
        }

        self.mpc.set_initial_params(param, vheid, is_ot)
        self.mpc.set_track_data(
            self._track.center_lut_x, self._track.center_lut_y,
            self._track.center_lut_dx, self._track.center_lut_dy,
            self._track.right_lut_x, self._track.right_lut_y,
            self._track.left_lut_x,  self._track.left_lut_y,
            self._track.element_arc_lengths,
            float(self._track.element_arc_lengths_orig[-1]),
            self._track.lut_ref_v,
        )
        self.get_logger().info("calling mpc.setup_MPC() — first run codegens acados (~30s)…")
        self.mpc.setup_MPC()
        self._mpc_ready = True
        self.get_logger().info("MPC ready — control loop active")

        # Latched RViz viz: raw (un-inflated) lanes. /boundary_marker (cycle)
        # shows inflated corridor — comparing the two reveals how much
        # safety margin mpc applies vs the actual track walls.
        self._publish_track_viz()

    def _publish_track_viz(self) -> None:
        """One-shot publish of /center_path /right_path /left_path /reference_path
        as small-sphere MarkerArrays. Latched (TRANSIENT_LOCAL) so late RViz
        attaches still receive. Ports ROS1 `preprocess_track_data` viz publish."""
        if self._track is None:
            return
        cl = self._track.raw_center_lane
        rl = self._track.raw_right_lane
        ll = self._track.raw_left_lane
        if cl is None or rl is None or ll is None:
            self.get_logger().warn("track has no raw_*_lane — skipping viz publish")
            return
        #              (r,   g,   b)   ns
        self.center_path_pub.publish(self._np_to_markers(cl, (1.0, 1.0, 1.0), 'center'))
        self.right_path_pub.publish(self._np_to_markers(rl, (1.0, 0.2, 0.2), 'right'))
        self.left_path_pub.publish(self._np_to_markers(ll, (0.2, 1.0, 0.2), 'left'))
        self.reference_path_pub.publish(self._np_to_markers(cl, (1.0, 1.0, 0.2), 'reference'))
        self.get_logger().info(
            f"published latched /center_path /right_path /left_path /reference_path "
            f"({len(cl)} sphere markers each)")

    def _np_to_markers(self, pts, rgb, ns: str) -> MarkerArray:
        """Build a MarkerArray of small spheres (type=2, scale=0.05) — matches
        the visual style of `/centerline_waypoints/markers` so RViz looks
        consistent. One Marker per waypoint with id=i."""
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, p in enumerate(pts):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = ns
            m.id = int(i)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            m.color.a = 1.0
            ma.markers.append(m)
        return ma

    # ─────────────────────────────────────────────────────────────────
    # Live parameter sync (rclpy parameter callback → mpc.* attributes)
    # ─────────────────────────────────────────────────────────────────
    def _push_live_params_to_mpc(self) -> None:
        for name, _default in LIVE_PARAMS:
            val = float(self.get_parameter(name).value)
            setattr(self.mpc, name, val)

    def _on_param_change(self, params: list[Parameter]):
        from rcl_interfaces.msg import SetParametersResult
        live_names = {n for n, _ in LIVE_PARAMS}
        for p in params:
            if p.name in live_names:
                try:
                    setattr(self.mpc, p.name, float(p.value))
                except Exception as e:
                    return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    # ─────────────────────────────────────────────────────────────────
    # Sub callbacks — cache only; heavy work runs on the control timer
    # ─────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._last_odom = msg

    def _pose_cb(self, msg: PoseStamped):
        self._last_pose = msg

    def _goal_cb(self, msg: PoseStamped):
        self._goal = msg

    def _clicked_point_cb(self, msg: PointStamped):
        # Mirrors ROS1: RViz click adds a static obstacle for dev
        self._obstacles.append((msg.point.x, msg.point.y))
        self._obstacles_stamp = time.monotonic()
        self.get_logger().info(
            f"clicked obstacle ({msg.point.x:.2f}, {msg.point.y:.2f}); "
            f"now {len(self._obstacles)} obstacles")

    def _external_obs_cb(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y) for p in msg.poses]
        self._obstacles_stamp = time.monotonic()

    # ─────────────────────────────────────────────────────────────────
    # Boundary visualization hook — called by mpc_core every solve()
    # ─────────────────────────────────────────────────────────────────
    def _on_boundary_points(self, shifted_points: list[tuple[float, float]]) -> None:
        ma = MarkerArray()
        for i, (xx, yy) in enumerate(shifted_points):
            mk = Marker()
            mk.header.frame_id = 'map'
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.scale = Vector3(x=0.10, y=0.10, z=0.10)
            mk.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
            mk.pose.position = Point(x=float(xx), y=float(yy), z=0.0)
            mk.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ma.markers.append(mk)
        self.boundary_pub.publish(ma)

    # ─────────────────────────────────────────────────────────────────
    # Control loop @ 40 Hz
    # ─────────────────────────────────────────────────────────────────
    def _current_state_4(self) -> np.ndarray | None:
        """Assemble [x, y, psi, s] from latest pose/odom. Returns None if not ready."""
        if self._last_pose is None and self._last_odom is None:
            return None
        if self._last_pose is not None:
            p = self._last_pose.pose.position
            q = self._last_pose.pose.orientation
        else:
            p = self._last_odom.pose.pose.position
            q = self._last_odom.pose.pose.orientation
        # Yaw from quaternion (avoid tf_transformations dep for one expr).
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        psi = math.atan2(siny_cosp, cosy_cosp)
        s = 0.0
        if self._track is not None:
            s, _ = find_current_arc_length(self._track, np.array([p.x, p.y]))
        return np.array([p.x, p.y, psi, s], dtype=float)

    def _control_loop_cb(self):
        if not self._mpc_ready:
            return  # waiting on track loader (TODO)
        x0 = self._current_state_4()
        if x0 is None:
            return
        t0 = time.monotonic()
        try:
            con_first, traj, u_seq, opti_value = self.mpc.solve(x0, self._obstacles)
        except Exception as e:
            self.get_logger().error(f"mpc.solve raised: {e}")
            return
        solve_dt = time.monotonic() - t0

        # Output: AckermannDrive
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.drive.speed = float(con_first[0])
        cmd.drive.steering_angle = float(con_first[1])
        self.ackermann_pub.publish(cmd)

        # Diagnostics
        self.cost_pub.publish(Float64(data=float(opti_value)))
        # ROS1 호환: ms 단위 (Nonlinear_MPC_node.py:328). logger [dbg] 출력
        # 도 "ms" 표기라 seconds 그대로 publish 하면 0 으로 보임.
        self.solve_time_pub.publish(Float64(data=float(solve_dt * 1000.0)))
        self.feasible_pub.publish(Bool(data=bool(opti_value < 1e8)))

        # MPC trajectory as nav_msgs/Path
        path = Path()
        path.header.stamp = cmd.header.stamp
        path.header.frame_id = 'map'
        for k in range(traj.shape[0]):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(traj[k, 0]), y=float(traj[k, 1]), z=0.0)
            siny = math.sin(float(traj[k, 2]) * 0.5)
            cosy = math.cos(float(traj[k, 2]) * 0.5)
            ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=siny, w=cosy)
            path.poses.append(ps)
        self.mpc_traj_pub.publish(path)

        # MarkerArray twin — sphere markers in centerline-style. Magenta.
        markers = MarkerArray()
        # DELETEALL first so leftover spheres from previous longer horizon
        # don't linger (defensive; horizon size is fixed but cheap to guard).
        clear = Marker()
        clear.header = path.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for k in range(traj.shape[0]):
            m = Marker()
            m.header = path.header
            m.ns = 'mpc_traj'
            m.id = int(k)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(traj[k, 0])
            m.pose.position.y = float(traj[k, 1])
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.07
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 1.0
            markers.markers.append(m)
        self.mpc_traj_markers_pub.publish(markers)

        # MPC prediction (raw state + inputs per stage) → osuf1_common/MPCTrajectory
        traj_msg = MPCTrajectory()
        traj_msg.header = path.header
        for k in range(traj.shape[0] - 1):
            pred = MPCPrediction()
            pred.state = traj[k].astype(np.float32).tolist()
            pred.inputs = u_seq[k].astype(np.float32).tolist()
            traj_msg.trajectory.append(pred)
        self.prediction_pub.publish(traj_msg)

        # /mpc_debug Float32MultiArray — 16 fields matching ROS1 DBG_FIELDS
        # order (see mpc_debug_logger.DBG_FIELDS). Consumed by the logger
        # node for CSV / event-dump anomaly detection.
        v_actual = 0.0
        if self._last_odom is not None:
            v_actual = float(self._last_odom.twist.twist.linear.x)
        current_s, near_idx = 0.0, 0
        if self._track is not None:
            current_s, near_idx = find_current_arc_length(
                self._track, np.array([x0[0], x0[1]]))
        try:
            ref_v_now = float(self.mpc.ref_v(current_s % float(self._track.element_arc_lengths_orig[-1])))
        except Exception:
            ref_v_now = 0.0
        dbg = Float32MultiArray()
        dbg.data = [
            float(con_first[0]),       # v_cmd
            float(con_first[1]),       # steer_cmd
            v_actual,                  # v_actual
            float(x0[0]),              # car_x
            float(x0[1]),              # car_y
            float(x0[2]),              # car_yaw
            float(current_s),          # current_s
            float(near_idx),           # near_idx
            ref_v_now,                 # ref_v
            float(len(self._obstacles)),  # n_obs_in
            0.0,                       # sel_dmin  (obstacle TODO)
            0.0,                       # sel_x
            0.0,                       # sel_y
            0.0,                       # side_pref
            float(opti_value),         # opti_value
            float(solve_dt * 1000.0),  # solve_ms
        ]
        self.mpc_debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
