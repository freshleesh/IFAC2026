#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 wrapper around upstream TUMRT/sampling_based_3D_local_planning.

Phase 0/1: OBSERVATION MODE ONLY.
  - Subscribes to stack odometry / raceline topics.
  - Calls LocalSamplingPlanner.calc_trajectory() at a fixed rate.
  - Publishes chosen trajectory and debug markers on dedicated topics.
  - Does NOT inject into /global_waypoints_scaled or the control pipeline.

### HJ : 3D sampling planner ROS wrapper (observation mode)
"""

import os
import sys
import time
import yaml
import numpy as np

from std_msgs.msg import String, Float32, Header
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray, Wpnt


# -- Make upstream src importable --------------------------------------------------------------------------------------
# Sampling package owns its own src/ (track3D, sampling_based_planner, ...).
# Shared modules (currently: ggManager.py) live in 3d_gb_optimizer/global_line/src/ and are picked
# up via sys.path fallback — i.e. we removed the local duplicate and resolve the shared one there.
# ### HJ : share ggManager from 3d_gb_optimizer to reduce duplication
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPSTREAM_SRC = os.path.join(_PKG_DIR, 'src')
_SHARED_SRC   = os.path.abspath(os.path.join(_PKG_DIR, '..', '3d_gb_optimizer', 'global_line', 'src'))

# Local src first (for track3D, sampling_based_planner, …), then the shared dir (for ggManager).
for _p in (_UPSTREAM_SRC, _SHARED_SRC):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

from track3D import Track3D                        # noqa: E402  (local src)
from ggManager import GGManager                    # noqa: E402  (shared from 3d_gb_optimizer)
from sampling_based_planner import LocalSamplingPlanner  # noqa: E402  (local src)


class SamplingPlannerNode:

    def __init__(self):
        rospy.init_node('sampling_planner_node', anonymous=False)

        # -- Parameters --------------------------------------------------------------------------------------------
        # Paths
        self.track_csv_path      = self._get_param_or_default('~track_csv_path')          # *_3d_smoothed.csv
        self.gg_dir_path         = self._get_param_or_default('~gg_dir_path')             # .../gg_diagrams/<veh>/velocity_frame/
        self.vehicle_params_path = self._get_param_or_default('~vehicle_params_path')     # params_*.yml
        self.raceline_csv_path   = self._get_param_or_default('~raceline_csv_path', '')   # optional, for init raceline

        # Rates / IO
        self.rate_hz             = self._get_param_or_default('~rate_hz', 10.0)
        self.frame_id            = self._get_param_or_default('~frame_id', 'map')

        # Sampling planner params (see upstream LocalSamplingPlanner.calc_trajectory)
        self.horizon             = self._get_param_or_default('~horizon', 3.0)
        self.num_samples         = self._get_param_or_default('~num_samples', 30)
        self.n_samples           = self._get_param_or_default('~n_samples', 11)
        self.v_samples           = self._get_param_or_default('~v_samples', 5)
        self.safety_distance     = self._get_param_or_default('~safety_distance', 0.05)
        self.gg_abs_margin       = self._get_param_or_default('~gg_abs_margin', 0.0)
        self.gg_margin_rel       = self._get_param_or_default('~gg_margin_rel', 0.0)
        self.friction_check_2d   = self._get_param_or_default('~friction_check_2d', False)
        self.relative_generation = self._get_param_or_default('~relative_generation', True)
        self.s_dot_min           = self._get_param_or_default('~s_dot_min', 1.0)
        self.kappa_thr           = self._get_param_or_default('~kappa_thr', 0.1)

        # Cost weights
        self.w_raceline   = self._get_param_or_default('~cost/raceline_weight', 0.1)
        self.w_velocity   = self._get_param_or_default('~cost/velocity_weight', 100.0)
        self.w_prediction = self._get_param_or_default('~cost/prediction_weight', 5000.0)

        # MPPI-style weighted blending (optional).
        self.mppi_enable           = self._get_param_or_default('~mppi/enable',          False)
        self.mppi_temperature_rel  = self._get_param_or_default('~mppi/temperature_rel', 0.25)
        self.mppi_temporal_weight  = self._get_param_or_default('~mppi/temporal_weight', 0.0)
        self._prev_blended_s       = None
        self._prev_blended_n       = None

        # -- Load vehicle params ----------------------------------------------------------------------------------
        with open(self.vehicle_params_path, 'r') as f:
            vehicle_yml = yaml.safe_load(f)
        # upstream expects {'vehicle_params': {...}, 'tire_params': {...}, ...}
        self.params = dict(vehicle_yml)
        self.get_logger().info('[sampling] Loaded vehicle params from %s', self.vehicle_params_path)

        # -- Instantiate upstream core ---------------------------------------------------------------------------
        self.get_logger().info('[sampling] Loading Track3D from %s', self.track_csv_path)
        self.track = Track3D(path=self.track_csv_path)

        self.get_logger().info('[sampling] Loading gg-diagrams from %s', self.gg_dir_path)
        self.gg = GGManager(gg_path=self.gg_dir_path, gg_margin=self.gg_margin_rel)

        self.planner = LocalSamplingPlanner(
            params=self.params,
            track_handler=self.track,
            gg_handler=self.gg,
        )
        self.get_logger().info('[sampling] LocalSamplingPlanner ready.')

        # ### HJ : log Track3D sanity — is the centerline polyline what we expect?
        self.get_logger().info(
            '[sampling][track3d] N=%d  s=[%.3f .. %.3f]  (delta_s mean=%.4f, min=%.4f, max=%.4f)',
            len(self.track.s), float(self.track.s[0]), float(self.track.s[-1]),
            float(np.mean(np.diff(self.track.s))),
            float(np.min(np.diff(self.track.s))),
            float(np.max(np.diff(self.track.s))),
        )
        self.get_logger().info(
            '[sampling][track3d] xyz bbox  x=[%.2f, %.2f]  y=[%.2f, %.2f]  z=[%.2f, %.2f]',
            float(np.min(self.track.x)), float(np.max(self.track.x)),
            float(np.min(self.track.y)), float(np.max(self.track.y)),
            float(np.min(self.track.z)), float(np.max(self.track.z)),
        )

        # -- Reference raceline (as dict) ------------------------------------------------------------------------
        self.raceline_dict = self._load_raceline_dict()
        # ### HJ : raceline sanity dump
        _rs = self.raceline_dict['s']; _rt = self.raceline_dict['t']; _rv = self.raceline_dict['V']
        self.get_logger().info(
            '[sampling][raceline] N=%d  s=[%.3f .. %.3f]  t=[%.3f .. %.3f]  V=[%.2f .. %.2f]',
            len(_rs), float(_rs[0]), float(_rs[-1]),
            float(_rt[0]), float(_rt[-1]), float(np.min(_rv)), float(np.max(_rv)),
        )

        # -- State caches ----------------------------------------------------------------------------------------
        # Stack publishes frenet in RACELINE frame (/car_state/odom_frenet). Track3D/solver expect
        # CENTERLINE frame. We reproject cartesian (x,y,z) onto Track3D's own polyline at each tick.
        # ### HJ : follow local_raceline_mux_node_HJ._cart_to_cl_frenet_exact pattern
        self._cur_x = None
        self._cur_y = None
        self._cur_z = None
        self._cur_vs = None         # linear.x magnitude from /car_state/odom

        # Debug: previous-tick projection to detect unphysical jumps.
        self._prev_s_cent = None
        self._prev_xyz    = None
        self._prev_stamp  = None
        self.debug_jump_threshold_m = self._get_param_or_default('~debug_jump_threshold_m', 2.0)

        # -- Subscribers -----------------------------------------------------------------------------------------
        self.create_subscription(Odometry, '/car_state/odom', self._cb_odom, 10)

        # -- Publishers ------------------------------------------------------------------------------------------
        self.pub_wpnts         = self.create_publisher(WpntArray, '~best_trajectory', 1)
        self.pub_best_sample   = self.create_publisher(Path, '~best_sample', 1)
        # ### HJ : best sample = SPHERE per point (speed-colored) + CYLINDER (speed-height)
        self.pub_best_markers  = self.create_publisher(MarkerArray, '~best_sample/markers', 10)
        self.pub_vel_markers   = self.create_publisher(MarkerArray, '~best_sample/vel_markers', 10)
        # ### HJ : every valid sampled candidate as thin gray LINE_STRIPs so the
        #         "fan of samples" is visible in RViz. Invalid (friction / bounds)
        #         candidates are suppressed.
        self.pub_candidates    = self.create_publisher(MarkerArray, '~candidates', 10)
        self.pub_status        = rospy.Publisher('~status',                  String,      queue_size=1, latch=True)
        self.pub_timing        = self.create_publisher(Float32, '~timing_ms', 1)

        self._publish_status('INIT_OK')
        self.get_logger().info('[sampling] Node ready — observation mode.')

    # =============================================================================================================
    # Helpers
    # =============================================================================================================
    def _load_raceline_dict(self):
        """Build the minimal raceline dict that LocalSamplingPlanner expects.

        Upstream uses keys at least: s, s_dot, n (and derivatives). If a CSV is provided we load it;
        otherwise we synthesize an "along track centerline at constant v" raceline so the planner can run.
        """
        if self.raceline_csv_path and os.path.exists(self.raceline_csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(self.raceline_csv_path, comment='#')
                self.get_logger().info('[sampling] Loaded raceline CSV: %d rows, cols=%s',
                              len(df), list(df.columns))

                def _pick(candidates, fallback=None):
                    low = {c.lower(): c for c in df.columns}
                    for k in candidates:
                        if k.lower() in low:
                            return df[low[k.lower()]].to_numpy()
                    return fallback

                s = _pick(['s_opt', 's_m', 's'])
                v = _pick(['v_opt', 'vx_mps', 'v', 'vs'])
                if s is None or v is None:
                    raise KeyError(
                        "required s/v columns not found in CSV; got {}".format(list(df.columns))
                    )
                n   = _pick(['n_opt', 'd_m', 'n'],      fallback=np.zeros_like(s))
                chi = _pick(['chi_opt', 'chi'],          fallback=np.zeros_like(s))
                ax  = _pick(['ax_opt', 'ax_mps2', 'ax'], fallback=np.zeros_like(s))
                ay  = _pick(['ay_opt', 'ay_mps2', 'ay'], fallback=np.zeros_like(s))

                # ### HJ : critical raceline s sanity for upstream's `np.interp(..., period=L)` calls.
                # ----------------------------------------------------------------------------
                # Upstream uses `period=track_handler.s[-1]` everywhere it interpolates raceline
                # data. numpy's periodic-interp implementation does:
                #   1) xp_wrap = xp % period
                #   2) sort xp_wrap, REORDER fp accordingly
                #   3) extend with wrap-around boundary points
                #     xp_ext = [xp_sorted[-1]-period, ...xp_sorted..., xp_sorted[0]+period]
                #     fp_ext = [fp_sorted[-1],         ...fp_sorted..., fp_sorted[0]]
                #   4) linear interp on the extended (sorted, monotonic) arrays
                # Two failure modes follow from a tiny mismatch between raceline['s'][-1] and L:
                #
                #  (a) `s[-1] >= L`  → step (1) aliases the last sample to s ≈ 0 of the next
                #      period, sorting it next to xp_sorted[0]. fp gets reordered, dragging the
                #      lap-end t value (~T_lap) into fp_sorted[1]. Any query at s ≈ 0..s_2nd
                #      then interpolates between `T_lap` and `~0` → returns t ≈ 2.5 s for
                #      s=0.17 m. Bad t indexes the wrong slice in postprocess_raceline →
                #      s_post first step jumps several meters → polynomial boundary blows up
                #      → every candidate shoots 10–30 m in 1 s.
                #
                #  (b) `s[-1] < L by more than ε`  → after sort, xp_ext gap between the real
                #      last sample s[-1] and the wrapped-front sample (xp_sorted[0] + L) covers
                #      a region [s[-1], L). For any query in that gap, np.interp linearly
                #      blends fp_sorted[-1] (= T_lap-ish) and fp_sorted[0] (= 0) — same kind
                #      of huge-vs-tiny interpolation. Same downstream blow-up.
                #
                # Symmetric remedy: PIN the last raceline sample's s to exactly `L - eps`. That
                # makes the periodic-interp wrap region degenerate (width = eps) so neither
                # failure mode can fire, regardless of CSV precision. Closed-loop CSVs already
                # duplicate the first point as the last, so adjusting s by tens of micrometres
                # introduces no physical error (we keep the original t/V/n/chi/ax/ay).
                # Pin s[-1] to strictly less than L (the period).  numpy's
                # `np.interp(..., period=L)` does `xp % period`, so any sample at s == L (or
                # >= L) wraps to 0 and pollutes the lap-start neighbourhood. Likewise any
                # sample appreciably below L leaves a wrap-extension gap that interpolates
                # between lap-end-time and lap-start-time. Both yield the "trajectory jumps
                # halfway across the map" symptom at lap boundary.
                L_track = float(self.track.s[-1])
                EPS_S   = 1e-6
                target_last = L_track - EPS_S
                if len(s) >= 2 and s[-1] != target_last:
                    self.get_logger().warning(
                        '[sampling] pinning raceline s[-1]: %.9f → %.9f  (track L=%.9f) '
                        'to keep upstream periodic np.interp valid.',
                        float(s[-1]), target_last, L_track,
                    )
                    s = s.copy()
                    s[-1] = target_last

                # Derive time t(s) by integrating ds / v (upstream postprocess_raceline indexes by t).
                v_safe = np.clip(v, 1e-3, None)
                ds = np.diff(s, prepend=s[0])
                t  = np.cumsum(ds / v_safe)
                s_ddot = np.gradient(v, s)
                z = np.zeros_like(s)

                # ### HJ : single-lap raceline — NO 2-lap extension.
                # We tried a 2-lap concat to cover wrap queries but it interacted badly with
                # upstream's polynomial target selection (candidate s_end pulled far forward
                # → 1 s horizon producing 30 m s-spans at V≤5.7 m/s near lap end).
                # Reverting to single lap; upstream uses np.interp(..., period=L) and np.mod
                # for wrap in most places, so normal lap-interior behaviour is correct.
                # Lap-end horizon slice may be shorter than configured, tolerated for now.
                self.get_logger().info('[sampling][raceline] single-lap  N=%d  s=[%.3f..%.3f]  t=[%.3f..%.3f]',
                              len(s), float(s[0]), float(s[-1]), float(t[0]), float(t[-1]))

                return {
                    't':       t,
                    's':       s,
                    's_dot':   v,
                    's_ddot':  s_ddot,
                    'n':       n,
                    'n_dot':   z,
                    'n_ddot':  z,
                    'V':       v,
                    'chi':     chi,
                    'ax':      ax,
                    'ay':      ay,
                }
            except Exception as e:
                self.get_logger().warning('[sampling] Failed to parse raceline CSV (%s), using centerline fallback.', e)

        self.get_logger().warning('[sampling] No raceline CSV — using centerline + constant 5 m/s fallback.')
        L = float(self.track.s[-1])
        s = np.linspace(0.0, L, 500)
        v = np.full_like(s, 5.0)
        t = s / 5.0
        z = np.zeros_like(s)
        return {
            't': t,
            's': s, 's_dot': v, 's_ddot': z,
            'n': z, 'n_dot': z, 'n_ddot': z,
            'V': v, 'chi': z, 'ax': z, 'ay': z,
        }

    def _publish_status(self, msg):
        self.pub_status.publish(String(data=msg))

    # =============================================================================================================
    # Callbacks
    # =============================================================================================================
    def _cb_odom(self, msg: Odometry):
        # World cartesian pose + body-frame longitudinal speed (twist.linear.x ≈ |v_long|).
        self._cur_x  = float(msg.pose.pose.position.x)
        self._cur_y  = float(msg.pose.pose.position.y)
        self._cur_z  = float(msg.pose.pose.position.z)
        self._cur_vs = float(msg.twist.twist.linear.x)
        # ### HJ : quick sanity once per second; confirms odom arrives and is finite.
        rospy.loginfo_throttle(
            2.0,
            '[sampling][odom] xyz=(%.3f, %.3f, %.3f)  v=%.3f  frame=%s',
            self._cur_x, self._cur_y, self._cur_z, self._cur_vs,
            msg.header.frame_id,
        )

    # =============================================================================================================
    # Frenet conversion (cartesian → Track3D centerline s, n)
    # =============================================================================================================
    def _cart_to_cl_frenet_exact(self, x, y, z, debug=False):
        """(x,y,z) → (s_cent, n_cent) on Track3D's own polyline.

        Mirrors local_raceline_mux_node_HJ._cart_to_cl_frenet_exact. Guarantees s/n use the SAME
        spline that sampling_based_planner's Track3D uses for theta/normals/bounds — avoids the
        ~cm-level drift we would get from a separate FrenetConverter instance.
        """
        xs    = np.asarray(self.track.x)
        ys    = np.asarray(self.track.y)
        zs    = np.asarray(self.track.z)
        s_arr = np.asarray(self.track.s)
        L = float(s_arr[-1])
        N = len(xs)

        d2 = (xs - x) ** 2 + (ys - y) ** 2 + (zs - z) ** 2
        i = int(np.argmin(d2))
        d2_i = float(d2[i])

        best_s, best_d2, best_pair = None, np.inf, None
        for ja, jb in (((i - 1) % N, i), (i, (i + 1) % N)):
            xa, ya, za = xs[ja], ys[ja], zs[ja]
            xb, yb, zb = xs[jb], ys[jb], zs[jb]
            dxab = xb - xa; dyab = yb - ya; dzab = zb - za
            len2 = dxab*dxab + dyab*dyab + dzab*dzab
            if len2 < 1e-12:
                continue
            t_seg = ((x - xa)*dxab + (y - ya)*dyab + (z - za)*dzab) / len2
            t_seg = max(0.0, min(1.0, t_seg))
            fx = xa + t_seg*dxab; fy = ya + t_seg*dyab; fz = za + t_seg*dzab
            dseg2 = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
            if dseg2 < best_d2:
                best_d2 = dseg2
                sa = s_arr[ja]; sb = s_arr[jb]
                if sb < sa:                # loop wrap
                    sb = sb + L
                best_s = sa + t_seg * (sb - sa)
                if best_s >= L:
                    best_s -= L
                best_pair = (int(ja), int(jb), float(t_seg))

        theta = float(np.interp(best_s, s_arr, self.track.theta))
        mu    = float(np.interp(best_s, s_arr, self.track.mu))
        phi   = float(np.interp(best_s, s_arr, self.track.phi))
        normal = Track3D.get_normal_vector_numpy(theta, mu, phi)
        xc = float(np.interp(best_s, s_arr, xs))
        yc = float(np.interp(best_s, s_arr, ys))
        zc = float(np.interp(best_s, s_arr, zs))
        n  = float(np.dot(np.array([x - xc, y - yc, z - zc]), normal))

        if debug:
            # Top-3 nearest indices to see ambiguity (loop self-intersection, close neighbors)
            top3 = np.argsort(d2)[:3]
            top3_info = ', '.join(
                '[i={} s={:.3f} d={:.3f}]'.format(int(k), float(s_arr[k]), float(np.sqrt(d2[k])))
                for k in top3
            )
            ja, jb, t_seg = best_pair if best_pair is not None else (-1, -1, -1)
            self.get_logger().info(
                '[sampling][proj] car=(%.3f, %.3f, %.3f)  nearest_i=%d (s=%.3f d=%.3f) top3=%s | '
                'best_seg=(ja=%d s=%.3f, jb=%d s=%.3f, t=%.3f)  s_cent=%.3f n=%.4f foot=(%.3f,%.3f,%.3f)',
                x, y, z, i, float(s_arr[i]), np.sqrt(d2_i), top3_info,
                ja, float(s_arr[ja] if ja >= 0 else -1),
                jb, float(s_arr[jb] if jb >= 0 else -1),
                t_seg, float(best_s), n, xc, yc, zc,
            )

        return float(best_s), n

    # =============================================================================================================
    # Main loop
    # =============================================================================================================
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self._cur_x is None:
                self._publish_status('WAITING_ODOM')
                rate.sleep()
                continue

            # RACELINE-frenet (from /car_state/odom_frenet) would mis-index Track3D.
            # Always reproject from cartesian onto Track3D's centerline spline.
            x_cur, y_cur, z_cur = self._cur_x, self._cur_y, self._cur_z
            v_cur = self._cur_vs or 0.0
            now   = self.get_clock().now().to_msg().to_sec()

            # ### HJ : detect unphysical jumps in s_cent or cartesian between ticks.
            #           First call runs with debug=True to log the projection internals.
            force_debug = self._prev_s_cent is None
            s_cent, n_cent = self._cart_to_cl_frenet_exact(
                x_cur, y_cur, z_cur, debug=force_debug)
            L = float(self.track.s[-1])

            if self._prev_s_cent is not None:
                # Shortest-signed arc-length delta accounting for wrap at L.
                ds = s_cent - self._prev_s_cent
                if ds >  L / 2: ds -= L
                if ds < -L / 2: ds += L
                dt = max(now - (self._prev_stamp or now), 1e-3)
                dx, dy, dz = x_cur - self._prev_xyz[0], y_cur - self._prev_xyz[1], z_cur - self._prev_xyz[2]
                cart_jump = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                s_jump    = abs(float(ds))

                if s_jump > self.debug_jump_threshold_m:
                    # Re-run projection with verbose logging so we can see the ambiguity.
                    self.get_logger().warning(
                        '[sampling][JUMP] |ds|=%.3fm > %.2f in dt=%.3fs  '
                        '(s: %.3f -> %.3f)  cart_jump=%.3fm (x:%.3f->%.3f y:%.3f->%.3f z:%.3f->%.3f)  '
                        'v=%.2f  expected_s_rate ≈ %.2f m/s',
                        s_jump, self.debug_jump_threshold_m, dt,
                        self._prev_s_cent, s_cent, cart_jump,
                        self._prev_xyz[0], x_cur, self._prev_xyz[1], y_cur, self._prev_xyz[2], z_cur,
                        v_cur, (s_jump / dt),
                    )
                    # One-shot verbose projection re-dump
                    self._cart_to_cl_frenet_exact(x_cur, y_cur, z_cur, debug=True)
                else:
                    rospy.loginfo_throttle(
                        1.0,
                        '[sampling][frenet] s=%.3f n=%.4f  ds=%+.3f m  dt=%.3f s  '
                        'cart_jump=%.3f m  v=%.2f m/s',
                        s_cent, n_cent, float(ds), dt, cart_jump, v_cur,
                    )

            self._prev_s_cent = s_cent
            self._prev_xyz    = (x_cur, y_cur, z_cur)
            self._prev_stamp  = now

            # ### HJ : no latency compensation — plan from the TRUE car position.
            # The trajectory starting ~V/rate_hz behind the car in RViz is normal
            # for any planner (processing + publish delay). At 30 Hz / 5 m/s this
            # is ~17 cm — barely visible. Compensation by predicting s forward
            # was tried but adds error on corners (assumes straight-line motion)
            # and doesn't account for RViz render lag. Controller lookahead
            # handles this naturally when the planner is actually wired in.

            state = {
                's':        s_cent,
                'n':        n_cent,
                's_dot':    max(self.s_dot_min, v_cur),
                's_ddot':   0.0,
                'n_dot':    0.0,
                'n_ddot':   0.0,
            }

            # No obstacle predictions yet (Phase 3). `prediction` is dict-of-dicts keyed by
            # opponent id, each value carrying 't','s','n' arrays. Empty dict = no opponents.
            prediction = {}

            t0 = time.time()
            try:
                self.planner.calc_trajectory(
                    state=state,
                    prediction=prediction,
                    raceline=self.raceline_dict,
                    relative_generation=self.relative_generation,
                    n_samples=self.n_samples,
                    v_samples=self.v_samples,
                    horizon=self.horizon,
                    num_samples=self.num_samples,
                    safety_distance=self.safety_distance,
                    gg_abs_margin=self.gg_abs_margin,
                    friction_check_2d=self.friction_check_2d,
                    s_dot_min=self.s_dot_min,
                    kappa_thr=self.kappa_thr,
                    raceline_cost_weight=self.w_raceline,
                    velocity_cost_weight=self.w_velocity,
                    prediction_cost_weight=self.w_prediction,
                )
                dt_ms = (time.time() - t0) * 1000.0
                self.pub_timing.publish(Float32(data=dt_ms))

                traj = self.planner.trajectory
                if traj and 'x' in traj:
                    # ### HJ : log what the chosen trajectory actually spans
                    try:
                        ts  = np.asarray(traj['s'])
                        Vs  = np.asarray(traj['V'])
                        xs_ = np.asarray(traj['x']); ys_ = np.asarray(traj['y']); zs_ = np.asarray(traj['z'])
                        # Shortest-arc span, handling one possible L-wrap
                        ds_span = float(ts[-1] - ts[0])
                        if ds_span < -L/2: ds_span += L
                        if ds_span >  L/2: ds_span -= L
                        cart_len = float(np.sum(np.sqrt(np.diff(xs_)**2 + np.diff(ys_)**2 + np.diff(zs_)**2)))
                        rospy.loginfo_throttle(
                            1.0,
                            '[sampling][traj] npts=%d  s:[%.3f..%.3f] (span=%.3f)  V:[%.2f..%.2f]  '
                            'cart_len=%.3f m  dt_solve=%.1f ms',
                            len(ts), float(ts[0]), float(ts[-1]), ds_span,
                            float(np.min(Vs)), float(np.max(Vs)), cart_len, dt_ms,
                        )
                    except Exception as _e:
                        rospy.logwarn_throttle(2.0, '[sampling][traj] log failed: %s', _e)
                    # ### HJ : optional MPPI-style weighted blending.
                    # When enabled, replace the argmin-picked traj with a weighted
                    # average over valid candidates. Keeps argmin intact by default.
                    if self.mppi_enable:
                        blended = self._mppi_blend()
                        if blended is not None:
                            traj = blended
                    self._publish_trajectory(traj)
                    self._publish_candidates(traj.get('optimal_idx', -1))
                    self._publish_status('OK')
                else:
                    self._publish_status('NO_FEASIBLE')

            except Exception as e:
                rospy.logerr_throttle(2.0, '[sampling] calc_trajectory failed: %s', e)
                self._publish_status('EXCEPTION:' + type(e).__name__)

            rate.sleep()

    # =============================================================================================================
    # Publishing
    # =============================================================================================================
    def _publish_trajectory(self, traj: dict):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        # ### HJ : upstream does `s = np.mod(s, L)` (line 582 of sampling_based_planner.py)
        # so the output s can jump backward at the lap boundary (e.g. 89.5 → 0.1).
        # That makes WpntArray consumers/RViz see non-monotonic `s_m` and the cartesian
        # (x,y,z) can also come from slightly-inconsistent interpolator endpoints when
        # the CSV is not perfectly closed. We:
        #   1) unwrap `traj['s']` in time-order so `s_m` is monotonically increasing,
        #   2) recompute (x,y,z) via Track3D.sn2cartesian on `s_unwrapped % L` so the
        #      cartesian samples use the SAME lap-origin for every point and stay
        #      continuous across the wrap.
        L        = float(self.track.s[-1])
        s_raw    = np.asarray(traj['s'], dtype=np.float64)
        n_arr    = np.asarray(traj['n'], dtype=np.float64)
        ds       = np.diff(s_raw)
        # ### HJ : bidirectional wrap detection (see _publish_candidates for rationale).
        wrap_adj = np.cumsum(
            np.where(ds < -L / 2.0,  L, 0.0) +
            np.where(ds >  L / 2.0, -L, 0.0)
        )
        s_unwr   = s_raw.copy()
        s_unwr[1:] += wrap_adj

        # ### HJ : TRUNCATE on first non-monotonic step instead of clamping with cummax.
        # Earlier band-aid (np.maximum.accumulate) pinned tail samples that went backward
        # to the previous max s. That stacked multiple samples at the same physical
        # location → "points clumping/mushing at the lap end" visual artefact reported by
        # user. Now that the periodic-interp root-cause is fixed (load_raceline_dict pin),
        # any residual backward step indicates a polynomial overshoot in a single tick;
        # the cleanest response is to TRUNCATE the trajectory at that point and publish
        # only the strictly-monotonic prefix. Visualization stays clean, no fake stacked
        # waypoints sent to consumers.
        # Threshold tuned: only truncate on PHYSICALLY MEANINGFUL backward steps. The
        # quintic polynomial can produce sub-cm noise at the horizon end which is
        # invisible in RViz and harmless to publish; truncating on it would chop the
        # trajectory in half. A backward step > 0.20 m means a real overshoot worth
        # cutting. (For RC car, 1 sample step at 5 m/s × 0.033 s ≈ 16 cm forward.)
        BACKWARD_THR = -0.20
        ds_unwr = np.diff(s_unwr)
        bad = np.where(ds_unwr < BACKWARD_THR)[0]
        if len(bad) > 0:
            cut = int(bad[0]) + 1   # keep up to and including the last forward sample
            rospy.loginfo_throttle(
                1.0,
                '[sampling][trunc] truncating at idx %d (of %d): s=%.3f → %.3f (ds=%.4f, thr=%.2f)',
                cut, len(s_unwr), float(s_unwr[cut - 1]), float(s_unwr[cut]),
                float(s_unwr[cut] - s_unwr[cut - 1]), BACKWARD_THR,
            )
            s_unwr = s_unwr[:cut]
            n_arr  = n_arr[:cut]

        # ### HJ : safety clamp before sn2cartesian.
        # `s_unwr % L` can land at exactly 0 or very close to L due to float precision,
        # where the underlying CasADi LUT interpolators may behave oddly at the boundary.
        # Clamp into the strictly-interior [eps, L-eps].
        s_for_cart = np.mod(s_unwr, L)
        s_for_cart = np.clip(s_for_cart, 1e-6, L - 1e-6)

        # Recompute cartesian from the unwrapped (mod-L, clamped) s for consistency.
        try:
            xyz = self.track.sn2cartesian(s=s_for_cart, n=n_arr)
            xs  = np.asarray(xyz[:, 0], dtype=np.float64)
            ys  = np.asarray(xyz[:, 1], dtype=np.float64)
            zs  = np.asarray(xyz[:, 2], dtype=np.float64)
        except Exception as _e:
            rospy.logwarn_throttle(2.0, '[sampling] sn2cartesian recomputation failed: %s', _e)
            n_pts = len(s_unwr)
            xs = np.asarray(traj['x'][:n_pts], dtype=np.float64)
            ys = np.asarray(traj['y'][:n_pts], dtype=np.float64)
            zs = np.asarray(traj['z'][:n_pts], dtype=np.float64)

        # Debug: if we detected any wrap, log how much we adjusted
        n_wraps = int(np.sum(ds < -L / 2.0))
        if n_wraps > 0:
            rospy.loginfo_throttle(
                1.0,
                '[sampling][wrap] unwrapped %d backward jumps  raw s=[%.3f..%.3f]  unwrapped=[%.3f..%.3f]',
                n_wraps, float(s_raw.min()), float(s_raw.max()),
                float(s_unwr.min()), float(s_unwr.max()),
            )

        # ### HJ : cart-gap sanity — TRUNCATE rather than stack-snap on big jumps.
        # Periodic-interp bug is fixed but defensive truncation kept for any residual
        # numerical edge case. Drops the offending suffix instead of stacking duplicate
        # points at one location (which caused the "clumping" the user reported).
        if len(xs) >= 2:
            cart_d = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
            gap_thr = 3.0   # meters
            big_gaps = np.where(cart_d > gap_thr)[0]
            if len(big_gaps) > 0:
                k = int(big_gaps[0])
                rospy.logwarn_throttle(
                    1.0,
                    '[sampling][JUMP] cart gap %.2f m at idx %d→%d  '
                    's_unwr[k]=%.3f s_unwr[k+1]=%.3f  xyz[k]=(%.3f,%.3f,%.3f)  '
                    'xyz[k+1]=(%.3f,%.3f,%.3f)  → truncating at idx %d',
                    float(cart_d[k]), k, k + 1,
                    float(s_unwr[k]), float(s_unwr[k + 1]),
                    float(xs[k]), float(ys[k]), float(zs[k]),
                    float(xs[k + 1]), float(ys[k + 1]), float(zs[k + 1]),
                    k + 1,
                )
                xs     = xs[:k + 1]
                ys     = ys[:k + 1]
                zs     = zs[:k + 1]
                s_unwr = s_unwr[:k + 1]
                n_arr  = n_arr[:k + 1]

        # WpntArray
        wp_arr = WpntArray()
        wp_arr.header = header
        for i in range(len(xs)):
            w = Wpnt()
            w.id           = i
            w.s_m          = float(s_unwr[i])         # monotonic across wrap
            w.d_m          = float(n_arr[i])
            w.x_m          = float(xs[i])
            w.y_m          = float(ys[i])
            if hasattr(w, 'z_m'):
                w.z_m = float(zs[i])
            w.vx_mps       = float(traj['V'][i])
            w.ax_mps2      = float(traj['ax'][i])
            w.kappa_radpm  = float(traj['kappa'][i])
            wp_arr.wpnts.append(w)
        self.pub_wpnts.publish(wp_arr)

        # Path (3D-friendly)
        path = Path()
        path.header = header
        for i in range(len(xs)):
            ps = PoseStamped()
            ps.header = header
            ps.pose.position.x = float(xs[i])
            ps.pose.position.y = float(ys[i])
            ps.pose.position.z = float(zs[i])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_best_sample.publish(path)

        # Expose for downstream marker building
        traj_x = xs; traj_y = ys; traj_z = zs

        # -- Marker styles (mirror 3d_state_machine_node local_waypoints) --------------------------
        # ### HJ : SPHERE per point + CYLINDER vel marker, speed-colored (red→green gradient)
        # Use only as many V samples as we have surviving cartesian samples after truncation.
        n_pts = len(traj_x)
        vx_vals = [float(v) for v in traj['V'][:n_pts]]
        vx_min = min(vx_vals) if vx_vals else 0.0
        vx_max = max(vx_vals) if vx_vals else 1.0

        def _vel_color(vx):
            t = (vx - vx_min) / (vx_max - vx_min) if vx_max > vx_min else 0.5
            return (max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5))),
                    max(0.0, min(1.0, 2.0 * t)),
                    0.0)

        # SPHERE markers — each point, color by speed
        loc_markers = MarkerArray()
        clr = Marker()
        clr.header = header
        clr.action = Marker.DELETEALL
        loc_markers.markers.append(clr)
        # ### HJ : use the wrap-consistent (traj_x, traj_y, traj_z) computed above
        for i in range(len(traj_x)):
            mk = Marker()
            mk.header = header
            mk.type = Marker.SPHERE
            mk.id = i + 1                 # id 0 reserved for DELETEALL
            mk.scale.x = 0.15
            mk.scale.y = 0.15
            mk.scale.z = 0.15
            mk.color.a = 1.0
            mk.color.r, mk.color.g, mk.color.b = _vel_color(vx_vals[i])
            mk.pose.position.x = float(traj_x[i])
            mk.pose.position.y = float(traj_y[i])
            mk.pose.position.z = float(traj_z[i])
            mk.pose.orientation.w = 1.0
            loc_markers.markers.append(mk)
        self.pub_best_markers.publish(loc_markers)

        # CYLINDER vel markers — height = V * 0.1317 (same scale as state machine)
        VEL_SCALE = 0.1317
        vel_markers = MarkerArray()
        clr2 = Marker()
        clr2.header = header
        clr2.action = Marker.DELETEALL
        vel_markers.markers.append(clr2)
        for i in range(len(traj_x)):
            mk = Marker()
            mk.header = header
            mk.type = Marker.CYLINDER
            mk.id = i + 1
            mk.scale.x = 0.1
            mk.scale.y = 0.1
            height = max(vx_vals[i] * VEL_SCALE, 0.02)
            mk.scale.z = height
            mk.color.a = 0.7
            mk.color.r, mk.color.g, mk.color.b = _vel_color(vx_vals[i])
            mk.pose.position.x = float(traj_x[i])
            mk.pose.position.y = float(traj_y[i])
            mk.pose.position.z = float(traj_z[i]) + height * 0.5
            mk.pose.orientation.w = 1.0
            vel_markers.markers.append(mk)
        self.pub_vel_markers.publish(vel_markers)

    # =============================================================================================================
    # Candidate-samples publishing (gray fan)
    # =============================================================================================================
    def _publish_candidates(self, optimal_idx: int):
        """Publish all sampled candidates as LINE_STRIPs.

        - Valid non-best : gray, thin (0.03 m), alpha 0.35
        - Invalid        : dark gray, very thin (0.015 m), alpha 0.15
        - Best (optimal) : RED, thick (0.07 m), alpha 1.0 — drawn last so it
                           is always on top.
        """
        cands = getattr(self.planner, 'candidates', None)
        if cands is None:
            return
        s_all     = np.asarray(cands['s'])
        n_all     = np.asarray(cands['n'])
        valid_all = np.asarray(cands['valid'], dtype=bool)
        N = s_all.shape[0]

        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        ma = MarkerArray()
        clr = Marker()
        clr.header = header
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        L = float(self.track.s[-1])
        drawn = 0
        best_marker = None

        # ### HJ : precompute numpy arrays for fast cartesian approximation.
        # Bypass CasADi sn2cartesian (1650 calls → slow) — use pure np.interp
        # on Track3D's raw polyline arrays. For visualization accuracy this is
        # sufficient; the normal-vector offset for n is applied via the same
        # numpy-level theta lookup.
        _s_arr = np.asarray(self.track.s)
        _x_arr = np.asarray(self.track.x)
        _y_arr = np.asarray(self.track.y)
        _z_arr = np.asarray(self.track.z)
        _th_arr = np.asarray(self.track.theta)

        for i in range(N):
            is_best  = (i == optimal_idx)
            is_valid = bool(valid_all[i])

            # ### HJ : skip invalid candidates entirely (diagnostic).
            # Invalid ones include polynomial overshoots + constraint violations
            # which can produce visually rough/zigzag lines. If these are the
            # source of the "lightning" the user sees, we'll know.
            if not is_valid and not is_best:
                continue

            s_row = s_all[i]
            n_row = n_all[i]
            ds = np.diff(s_row)
            wrap_adj = np.cumsum(
                np.where(ds < -L / 2.0,  L, 0.0) +
                np.where(ds >  L / 2.0, -L, 0.0)
            )
            s_unwr = s_row.copy().astype(np.float64)
            s_unwr[1:] += wrap_adj
            s_mod = np.clip(s_unwr % L, 1e-6, L - 1e-6)

            # ### HJ : lightning-bolt diagnostic — if consecutive cartesian points
            # end up far apart AFTER unwrap (shouldn't happen on closed loop),
            # dump the s/n values so we can see exactly what upstream emitted.
            # Only logged for first candidate near lap boundary, once a second.
            if (i == optimal_idx) and (s_row.min() < 5.0 or s_row.max() > L - 5.0):
                # find where wrap happens (either direction)
                big_neg = np.where(ds < -L / 2.0)[0]
                big_pos = np.where(ds > L / 2.0)[0]
                rospy.loginfo_throttle(
                    2.0,
                    '[sampling][cand-debug] i=%d  s_row min/max=[%.3f, %.3f]  '
                    'big_neg_ds@=%s (vals=%s)  big_pos_ds@=%s (vals=%s)  '
                    's_unwr min/max=[%.3f, %.3f]  s_mod min/max=[%.3f, %.3f]',
                    i, float(s_row.min()), float(s_row.max()),
                    big_neg.tolist(),
                    np.array2string(ds[big_neg], precision=2) if len(big_neg) else '[]',
                    big_pos.tolist(),
                    np.array2string(ds[big_pos], precision=2) if len(big_pos) else '[]',
                    float(s_unwr.min()), float(s_unwr.max()),
                    float(s_mod.min()), float(s_mod.max()),
                )

            # Fast numpy-only cartesian: centerline pos + 2D normal × n
            xc = np.interp(s_mod, _s_arr, _x_arr)
            yc = np.interp(s_mod, _s_arr, _y_arr)
            zc = np.interp(s_mod, _s_arr, _z_arr)
            th = np.interp(s_mod, _s_arr, _th_arr)
            # 2D normal approximation: rotate heading 90° (sufficient for viz)
            xs_ = xc - np.sin(th) * n_row
            ys_ = yc + np.cos(th) * n_row
            zs_ = zc

            mk = Marker()
            mk.header = header
            mk.ns     = 'candidates'
            mk.id     = drawn + 1
            mk.type   = Marker.LINE_STRIP
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0

            if is_best:
                mk.scale.x = 0.07          # thick
                mk.color.r = 1.0; mk.color.g = 0.1; mk.color.b = 0.1
                mk.color.a = 1.0
            elif is_valid:
                mk.scale.x = 0.03          # thin but visible
                mk.color.r = 0.1; mk.color.g = 0.1; mk.color.b = 0.1   # near-black
                mk.color.a = 0.45
            else:
                mk.scale.x = 0.015         # thin
                mk.color.r = 0.6; mk.color.g = 0.6; mk.color.b = 0.6   # light gray
                mk.color.a = 0.25

            for k in range(len(xs_)):
                p = Point()
                p.x = float(xs_[k]); p.y = float(ys_[k]); p.z = float(zs_[k])
                mk.points.append(p)

            if is_best:
                best_marker = mk   # draw last (on top)
            else:
                ma.markers.append(mk)
            drawn += 1

        # best on top — appended last so RViz renders it above the others
        if best_marker is not None:
            ma.markers.append(best_marker)

        self.pub_candidates.publish(ma)

    # =============================================================================================================
    # MPPI-style weighted blending
    # =============================================================================================================
    def _mppi_blend(self):
        """Replace the argmin-picked trajectory with a cost-weighted average of
        all VALID candidates. Returns a dict with the same fields as
        self.planner.trajectory, or None if blending is not applicable.

        Weights: w_i = exp(-(cost_i - cost_min) / T), T = temperature_rel × cost_range.
        Invalid candidates get 0 weight (never contribute).
        Optional temporal term penalizes candidates that deviate from the
        previous tick's blended trajectory — smooths across ticks further.

        Averaging quintic polynomials preserves the quintic property
        (C² continuous), so the blended trajectory stays smooth physically.
        """
        cands = getattr(self.planner, 'candidates', None)
        cost_arr = getattr(self.planner, 'cost_array', None)
        if cands is None or cost_arr is None:
            return None

        valid = np.asarray(cands['valid'], dtype=bool)
        if valid.sum() == 0:
            return None

        # Base cost + optional temporal regularization
        cost = np.asarray(cost_arr, dtype=np.float64).copy()
        if self.mppi_temporal_weight > 0.0 and self._prev_blended_s is not None:
            s_arr = np.asarray(cands['s'])
            n_arr = np.asarray(cands['n'])
            # Align length (prev and current may differ by 0 when horizon fixed)
            m = min(self._prev_blended_s.shape[0], s_arr.shape[1])
            ds = s_arr[:, :m] - self._prev_blended_s[:m]
            dn = n_arr[:, :m] - self._prev_blended_n[:m]
            tempo = np.sum(ds * ds + dn * dn, axis=1)
            cost = cost + self.mppi_temporal_weight * tempo

        valid_costs = cost[valid]
        c_min = float(valid_costs.min())
        c_max = float(valid_costs.max())
        c_range = max(c_max - c_min, 1e-6)
        T = max(self.mppi_temperature_rel * c_range, 1e-6)

        # Softmax-style weights (shifted for numerical stability)
        w = np.exp(-(valid_costs - c_min) / T)
        w /= w.sum()

        # ### HJ : unwrap each candidate's s BEFORE averaging.
        # Near the lap boundary, candidates wrap at different indices (some at
        # idx k, some at idx k+1). Averaging raw (mod-L) s values would mix
        # lap-end and lap-start numbers → nonsensical intermediate s values
        # (the "번개 모양" zigzag). Unwrap each candidate to a monotonic
        # arc-length, THEN weighted-average, THEN mod back for cartesian lookup.
        L = float(self.track.s[-1])
        s_valid = np.asarray(cands['s'])[valid].astype(np.float64).copy()
        # Per-candidate bidirectional wrap correction
        ds_valid = np.diff(s_valid, axis=1)
        wrap_adj = np.cumsum(
            np.where(ds_valid < -L / 2.0,  L, 0.0) +
            np.where(ds_valid >  L / 2.0, -L, 0.0),
            axis=1,
        )
        s_valid[:, 1:] += wrap_adj
        # Anchor each unwrapped trajectory to [0, 2L) so the weighted average
        # stays bounded. Use the candidate-average start s as reference.
        anchor = float(np.mean(s_valid[:, 0]))
        # If a candidate's first sample is far from anchor (wrap side flip),
        # shift it by ±L to align.
        for k in range(s_valid.shape[0]):
            while s_valid[k, 0] - anchor > L / 2.0:
                s_valid[k] -= L
            while s_valid[k, 0] - anchor < -L / 2.0:
                s_valid[k] += L

        # Weighted average of each field (only over valid candidates)
        blended = {'traj_cnt': self.planner.traj_cnt,
                   'optimal_idx': int(np.arange(valid.size)[valid][int(np.argmax(w))])}
        blended['s'] = np.sum(w[:, None] * s_valid, axis=0)
        for key in ('n', 'V', 'chi', 'ax', 'ay', 'kappa', 't'):
            arr = np.asarray(cands[key])
            if arr.ndim == 1:
                continue
            blended[key] = np.sum(w[:, None] * arr[valid], axis=0)

        # sn2cartesian for blended (s, n)
        try:
            s_for_cart = np.clip(np.mod(blended['s'], L), 1e-6, L - 1e-6)
            xyz = self.track.sn2cartesian(s=s_for_cart, n=blended['n'])
            blended['x'] = np.asarray(xyz[:, 0], dtype=np.float64)
            blended['y'] = np.asarray(xyz[:, 1], dtype=np.float64)
            blended['z'] = np.asarray(xyz[:, 2], dtype=np.float64)
        except Exception as _e:
            rospy.logwarn_throttle(2.0, '[sampling][mppi] cartesian failed: %s', _e)
            return None

        # Cache for next-tick temporal smoothing
        self._prev_blended_s = np.asarray(blended['s'], dtype=np.float64).copy()
        self._prev_blended_n = np.asarray(blended['n'], dtype=np.float64).copy()

        rospy.loginfo_throttle(
            1.0,
            '[sampling][mppi] valid=%d/%d  T=%.3f  cost=[%.3f..%.3f]  '
            'top-3 weights=%s',
            int(valid.sum()), int(valid.size), T, c_min, c_max,
            np.array2string(np.sort(w)[-3:][::-1], precision=3),
        )
        return blended


def main():
    node = SamplingPlannerNode()
    node.spin()


if __name__ == '__main__':
    main()
