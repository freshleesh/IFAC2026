#!/usr/bin/env python3
### HJ : RACELINE-BASED variant of the local raceline MPC node.
###
###      Sibling file (untouched, CENTERLINE-based): local_raceline_mux_node_HJ.py
###      Paired Track3D variant:                     track3D_raceline_based.py
###
###      Goal of this fork
###      -----------------
###      Shift the solver's internal frenet frame from the TRACK CENTERLINE to
###      the GLOBAL RACING LINE. In the centerline-based version the cost
###      1/s_dot = (1-n·Ω_z) / (V cos chi) makes corner-cutting emerge because
###      Ω_z is tight and `n·Ω_z` term pays for moving inside. Once the frame
###      is the raceline, Ω_z is already the minimum-time curvature and n=0
###      means "on the raceline". The solver's optimum coincides with the
###      global raceline in the nominal case, and MPC only deviates when it
###      has a reason to (obstacles, bound violation, GGV shift).
###
###      What changes vs centerline-based:
###        * Track3D_raceline_based builds its spline + interpolators from the
###          global_waypoints.json raceline wpnts (x/y/z, psi, kappa, d_left,
###          d_right, mu) rather than smoothed.csv.
###        * s is raceline arc-length (length ≈ 85.8m, not 89.9m centerline).
###        * w_tr_right/left are raceline-centric distances to walls (already
###          present in JSON as d_right / d_left).
###        * theta/Ω_z are raceline tangent/curvature (psi_rad / kappa_radpm).
###
###      What stays the same:
###        * point_mass_model.py — frame-agnostic; takes whatever Track3D
###          provides.
###        * local_racing_line_planner.py — same solver, same cost shape.
###        * _cart_to_cl_frenet_exact math (renamed to _cart_to_rl_frenet_exact
###          conceptually); still a brute-force+segment projection on the
###          SAME spline Track3D uses.
###
###      Status: scaffolding only — file is currently a byte-identical copy
###      of the centerline version; raceline refactor is WIP.

import os
import sys
import yaml
import numpy as np
import time
import tf.transformations

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from f110_msgs.msg import WpntArray, Wpnt
### HJ : RViz markers for test-only visualization
from visualization_msgs.msg import Marker, MarkerArray

# Add paths for solver imports
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)  # for track3D_lite
_solver_src = os.path.join(_dir, '..', 'src')
sys.path.append(os.path.abspath(_solver_src))

### HJ : raceline-based variant imports Track3D from track3D_raceline_based
#        (still scaffold — to be refactored to load raceline JSON). The
#        centerline version keeps using track3D_local; IY's pipeline keeps
#        using track3D_lite. Three Track3D variants live side-by-side.
from track3D_raceline_based import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model

### HJ : centerline frenet converter — (x,y,z) → (s_cent, n_cent)
from frenet_conversion.frenet_converter import FrenetConverter


class LocalRacelineMux:
    """
    ### IY : Local racing line mux node
    - PASSTHROUGH: global waypoints 그대로 전달
    - ACTIVE: solver로 local racing line 계산 후 global waypoints에 덮어쓰기
    """

    PASSTHROUGH = "PASSTHROUGH"
    ACTIVE = "ACTIVE"

    def __init__(self):
        ### HJ : raceline-based variant registers under a DIFFERENT node
        #        name so it can run alongside both IY's original and the
        #        centerline-based HJ node without collision.
        rospy.init_node('local_raceline_mux_hj_raceline', anonymous=False)

        # ── Parameters ──
        self.d_threshold = self._get_param_or_default('~d_threshold', 0.3)
        self.v_threshold = self._get_param_or_default('~v_threshold', 1.5)
        self.convergence_d_threshold = self._get_param_or_default('~convergence_d_threshold', 0.1)
        self.convergence_v_threshold = self._get_param_or_default('~convergence_v_threshold', 0.5)
        ### HJ : horizon 10m → 7.5m. 5m was overly short for tracking
        #        anticipation; 7.5m balances smooth visual + enough corner
        #        lookahead. Reduces stage-end kink while keeping turn-in
        #        planning useful.
        self.optimization_horizon = self._get_param_or_default('~optimization_horizon', 7.5)
        self.N_steps = self._get_param_or_default('~N_steps', 30)
        self.gg_mode = self._get_param_or_default('~gg_mode', 'diamond')
        self.safety_distance = self._get_param_or_default('~safety_distance', 0.05)
        self.is_sim = self._get_param_or_default('/sim', False)
        self.map_name = self._get_param_or_default('/map', '')
        ### IY : filenames as params (map name and file prefix may differ)
        ### HJ : default the smoothed-track filename from /map so switching
        #        maps (gazebo_wall_2 → gazebo_wall_2_iy) doesn't need a
        #        manual param override. Convention: "<map>_3d_smoothed.csv".
        _default_smoothed = '{}_3d_smoothed.csv'.format(self.map_name) \
            if self.map_name else 'gazebo_wall_2_3d_smoothed.csv'
        self.smoothed_track_file = self._get_param_or_default(
            '~smoothed_track_file', _default_smoothed)

        # ── Paths ──
        pkg_path = os.path.abspath(os.path.join(_dir, '..'))  # global_line/
        map_path = os.path.abspath(os.path.join(
            _dir, '..', '..', '..', '..', 'stack_master', 'maps', self.map_name))

        self.smoothed_track_path = os.path.join(map_path, self.smoothed_track_file)
        self.vehicle_params_path = os.path.join(
            pkg_path, 'data', 'vehicle_params', 'params_rc_car_10th.yml')
        self.gg_diagram_path = os.path.join(
            pkg_path, 'data', 'gg_diagrams', 'rc_car_10th', 'velocity_frame')

        # ── State variables ──
        self.mode = self.PASSTHROUGH
        self.prev_solution = None
        ### HJ : TEST version no longer needs gb_wpnts — we don't overwrite
        ###      /global_waypoints_scaled. Keep removed to prevent accidental use.
        self.cur_s = 0.0   ### raceline frenet s (from /car_state/odom_frenet)
        self.cur_d = 0.0   ### raceline frenet n (from /car_state/odom_frenet)
        self.cur_vs = 0.0
        self.cur_yaw = 0.0
        self.cur_imu_ax = 0.0
        self.cur_imu_ay = 0.0
        ### HJ : cartesian pose (from /car_state/odom) — used for converting
        ###      to centerline frenet so the MPC (which is centerline-based)
        ###      sees consistent coordinates.
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = 0.0
        self.have_odom = False   # set True after first /car_state/odom message
        ### HJ : end
        ### HJ : sim fallback — accelerations from /car_state/odom twist
        #       numerical differentiation + low-pass filter. Used in sim mode
        #       (no IMU). Frame note: nav_msgs/Odometry twist is in child_frame
        #       (typically base_link) so linear.x/y are body-frame vx/vy →
        #       their derivatives are body-frame ax/ay, which is what the MPC
        #       state expects.
        self.cur_odom_ax = 0.0
        self.cur_odom_ay = 0.0
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_odom_stamp = None
        ### HJ : heavier LPF — sim's twist.linear.x/y occasionally jumps
        #   (integrator step boundaries, physics reset), producing >30 m/s²
        #   spurious ax/ay after raw differentiation. alpha=0.1 attenuates
        #   those spikes at the cost of ~100ms lag, which is acceptable
        #   since the MPC horizon is ~10s.
        self._lpf_alpha = 0.1
        ### HJ : end
        self.solver_initialized = False

        # ── Load vehicle params ──
        with open(self.vehicle_params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        # ── Load global raceline reference (for deviation check) ──
        self._load_global_raceline_ref()

        # ── Initialize solver (Track3D, GGManager, model, planner) ──
        self._init_solver()

        ### HJ : centerline_ref (s, n in centerline frame for every raceline
        #        point) is only available after Track3D is ready, so we
        #        compute it AFTER _init_solver. If the JSON already carried
        #        a valid centerline_ref it was loaded above; otherwise we
        #        derive it on the fly using Track3D's own spline via
        #        _cart_to_cl_frenet_exact — same conversion the MPC input
        #        path uses, so the cold-start guess and the solver's
        #        coordinate system stay numerically identical.
        self._ensure_centerline_ref()
        ### HJ : end

        ### HJ : TEST-ONLY outputs — do NOT overwrite /global_waypoints_scaled.
        #       We publish to a dedicated topic so the car keeps following the
        #       original vel_planner output, and we can visually compare the
        #       solver's plan against actual driving in RViz.
        #
        #   /3d_optimized_local_waypoints_raceline              WpntArray  — solver output
        #   /3d_optimized_local_waypoints_raceline/markers      MarkerArray (sphere, speed-color)
        #   /3d_optimized_local_waypoints_raceline/vel_markers  MarkerArray (cylinder height = V)
        #   /local_raceline/status                     String      (PASSTHROUGH/ACTIVE)
        self.wpnt_pub = rospy.Publisher(
            '/3d_optimized_local_waypoints_raceline', WpntArray, queue_size=1)
        self.loc_markers_pub = rospy.Publisher(
            '/3d_optimized_local_waypoints_raceline/markers', MarkerArray, queue_size=10)
        self.vel_markers_pub = rospy.Publisher(
            '/3d_optimized_local_waypoints_raceline/vel_markers', MarkerArray, queue_size=10)
        self.status_pub = rospy.Publisher(
            '/3d_optimized_local_waypoints_raceline/status', String, queue_size=1)
        ### HJ : end

        ### HJ : Subscribers — TEST version reads ONLY the car state.
        #       IY's original also subscribed to /global_waypoints_scaled_raw
        #       to copy-and-overwrite as the published output. We don't
        #       republish /global_waypoints_scaled at all, so the upstream
        #       waypoints topic is not needed here.
        rospy.Subscriber(
            '/car_state/odom_frenet', Odometry, self._frenet_cb)
        rospy.Subscriber(
            '/car_state/odom', Odometry, self._odom_cb)
        ### HJ : end

        if not self.is_sim:
            self.create_subscription(Imu, '/ekf/imu/data', self._imu_cb, 10)

        # ── Timer (10Hz) ──
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

        self.get_logger().info("[LocalRacelineMux] Initialized. Mode: PASSTHROUGH, sim=%s, map=%s",
                      self.is_sim, self.map_name)

    # ═══════════════════════════════════════════════════════
    # Initialization helpers
    # ═══════════════════════════════════════════════════════

    def _load_global_raceline_ref(self):
        """### IY : global_waypoints.json에서 deviation check + solver 초기 guess용 배열 구축"""
        import json
        ### HJ : cache the JSON path so _init_solver can hand it to
        #        Track3D_raceline_based as the source of track-spline data.
        self.global_waypoints_json_path = os.path.join(
            os.path.abspath(os.path.join(
                _dir, '..', '..', '..', '..', 'stack_master', 'maps', self.map_name)),
            'global_waypoints.json')
        with open(self.global_waypoints_json_path, 'r') as f:
            data = json.load(f)
        wpnts = data['global_traj_wpnts_iqp']['wpnts']
        self.ref_s = np.array([w['s_m'] for w in wpnts])
        self.ref_v = np.array([w['vx_mps'] for w in wpnts])
        self.ref_n = np.array([w['d_m'] for w in wpnts])

        ### HJ : cartesian of the RACING LINE — used to build the raceline
        #       FrenetConverter (for output step). All waypoints consumed
        #       downstream are raceline-indexed, so we need to express the MPC
        #       output in raceline frenet before writing back.
        self.ref_x = np.array([w['x_m'] for w in wpnts])
        self.ref_y = np.array([w['y_m'] for w in wpnts])
        self.ref_z = np.array([w.get('z_m', 0.0) for w in wpnts])
        ### HJ : end

        ### HJ : always defer — we compute centerline_ref ourselves in
        #        _ensure_centerline_ref() by projecting each raceline wpnt
        #        onto Track3D's own spline. This is numerically identical
        #        to the frame the MPC uses (both read smoothed.csv via the
        #        same spline), so the cold-start initial guess lines up
        #        exactly with the solver's coordinate system.
        #
        #        JSON-stored centerline_ref was written by an upstream tool
        #        with its own spline; trusting it risks a subtle offset
        #        between guess coords and solver coords. Recomputing costs
        #        ~2ms once at startup — negligible.
        self.ref_s_center = None   # filled in _ensure_centerline_ref
        self.ref_n_center = None
        self._centerline_ref_source = 'deferred'
        if 'centerline_ref' in data:
            self.get_logger().info("[LocalRacelineMux] JSON has centerline_ref, but "
                          "ignoring it — recomputing from Track3D spline for "
                          "coordinate-system consistency.")

        self.get_logger().info("[LocalRacelineMux] Loaded raceline ref from JSON: %d pts, s=[%.1f, %.1f]",
                      len(self.ref_s), self.ref_s[0], self.ref_s[-1])

    def _init_solver(self):
        """sim_local_racing_line.py 패턴대로 solver 초기화"""
        self.get_logger().info("[LocalRacelineMux] Initializing solver (acados compile may take 10-30s)...")
        t0 = time.time()

        ### HJ : raceline-framed Track3D. Primary data source = global_waypoints
        #        JSON (raceline xyz, psi, kappa, d_left/right, mu). Optional
        #        secondary = smoothed centerline CSV for 3D banking (mu, phi,
        #        Ω_x, Ω_y interpolated to each raceline point).
        #        After this, self.track_handler.s is RACELINE arc length,
        #        and n=0 means "on the raceline" in the solver's frame.
        self.track_handler = Track3D(
            raceline_json_path=self.global_waypoints_json_path,
            smoothed_csv_path=self.smoothed_track_path,
        )
        self.track_length = float(self.track_handler.s[-1])
        self.get_logger().info("[LocalRacelineMux] Raceline-framed Track3D ready: "
                      "N=%d, L=%.3fm  (n=0 ≡ on-raceline)",
                      len(self.track_handler.x), self.track_length)

        ### HJ : raceline FrenetConverter — converts (x, y, z) → (s_race, n_race).
        ###      Used at the OUTPUT step: MPC gives cartesian per sample; we
        ###      need to map each sample to the raceline s_m grid so the
        ###      published wpnts stay raceline-indexed (consistent with
        ###      vel_planner / controller consumers).
        self.frenet_race = FrenetConverter(
            waypoints_x=self.ref_x,
            waypoints_y=self.ref_y,
            waypoints_z=self.ref_z,
        )
        self.race_length = float(self.frenet_race.raceline_length)
        self.get_logger().info("[LocalRacelineMux] Raceline FrenetConverter ready: "
                      "N=%d, L=%.3fm (centerline L=%.3fm)",
                      len(self.ref_x), self.race_length, self.track_length)
        ### HJ : end

        # GGManager
        self.gg_handler = GGManager(
            gg_path=self.gg_diagram_path,
            gg_margin=0.0
        )

        # Point mass ODE model
        self.model = export_point_mass_ode_model(
            vehicle_params=self.params['vehicle_params'],
            track_handler=self.track_handler,
            gg_handler=self.gg_handler,
            optimization_horizon=self.optimization_horizon,
            gg_mode=self.gg_mode
        )

        ### HJ : revert IY's SQP_RTI/3-iter override — the original
        #        sim_local_racing_line.py uses the class defaults
        #        (nlp_solver_type='SQP', sqp_max_iter=20). SQP_RTI is
        #        designed for high-rate (100Hz+) real-time re-linearization
        #        with always-warm prev_solution; in our 10Hz cold-start-prone
        #        setup it under-iterates and the horizon output shows
        #        dynamics-infeasible kinks (dn≈1m at stage 0→1). Full SQP
        #        with the default 20-iter cap converges per call, at the
        #        cost of a longer but still comfortably <100ms solve time.
        #        w_slack_n=1e3 stays (strong track-bound penalty is useful).
        self.planner = LocalRacinglinePlanner(
            params=self.params,
            track_handler=self.track_handler,
            gg_handler=self.gg_handler,
            model=self.model,
            optimization_horizon=self.optimization_horizon,
            gg_mode=self.gg_mode,
            N_steps=self.N_steps,
            w_slack_n=1e3,
        )

        self.solver_initialized = True
        self.get_logger().info("[LocalRacelineMux] Solver initialized in %.1fs", time.time() - t0)

    # ═══════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════

    ### HJ : _gb_wpnts_cb removed — /global_waypoints_scaled_raw is no
    ###      longer subscribed in the test-only build.

    def _frenet_cb(self, msg):
        """Frenet odometry: s, d, vs"""
        self.cur_s = msg.pose.pose.position.x
        self.cur_d = msg.pose.pose.position.y
        self.cur_vs = msg.twist.twist.linear.x

    def _odom_cb(self, msg):
        """Map-frame odometry: yaw for chi calculation + cartesian for centerline conversion"""
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cur_yaw = yaw
        ### HJ : save cartesian pose for centerline-frenet conversion
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        self.cur_z = msg.pose.pose.position.z
        self.have_odom = True

        ### HJ : sim-mode ax/ay estimation via twist differentiation + LPF.
        #       Always compute (cheap), but only used when IMU is unavailable.
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        stamp_sec = msg.header.stamp.to_sec()
        if self._prev_odom_stamp is not None:
            dt = stamp_sec - self._prev_odom_stamp
            if dt > 1e-3:  # avoid division by near-zero
                raw_ax = (vx - self._prev_vx) / dt
                raw_ay = (vy - self._prev_vy) / dt
                a = self._lpf_alpha
                self.cur_odom_ax = a * raw_ax + (1.0 - a) * self.cur_odom_ax
                self.cur_odom_ay = a * raw_ay + (1.0 - a) * self.cur_odom_ay
        self._prev_vx = vx
        self._prev_vy = vy
        self._prev_odom_stamp = stamp_sec
        ### HJ : end

    def _imu_cb(self, msg):
        """IMU acceleration (real car only)"""
        self.cur_imu_ax = msg.linear_acceleration.x
        self.cur_imu_ay = msg.linear_acceleration.y

    # ═══════════════════════════════════════════════════════
    # Main timer loop (10Hz)
    # ═══════════════════════════════════════════════════════

    def _timer_cb(self, event):
        ### HJ : guard on car state instead of upstream wpnts (which we no longer read)
        if not self.solver_initialized or not self.have_odom:
            return

        ### HJ : Deviation check — all in RACELINE frenet (consistent with
        #       /car_state/odom_frenet + ref_s/ref_n which both come from
        #       the raceline). Use race_length, not track_length (centerline).
        s = self.cur_s % self.race_length
        n_ref = np.interp(s, self.ref_s, self.ref_n, period=self.race_length)
        v_ref = np.interp(s, self.ref_s, self.ref_v, period=self.race_length)
        d_err = abs(self.cur_d - n_ref)
        v_err = abs(self.cur_vs - v_ref)
        ### HJ : end

        # ── Mode transitions ──
        if self.mode == self.PASSTHROUGH:
            if d_err > self.d_threshold or v_err > self.v_threshold:
                self.mode = self.ACTIVE
                self.get_logger().warning("[LocalRacelineMux] -> ACTIVE (d_err=%.3f, v_err=%.3f)",
                              d_err, v_err)

        if self.mode == self.ACTIVE:
            self._run_active_mode(s, d_err, v_err)
        else:
            ### HJ : PASSTHROUGH — TEST-only build. Do NOT republish
            #       /global_waypoints_scaled; the car stays on the existing
            #       vel_planner output. We clear any stale test markers.
            self._publish_local_wpnts_and_markers(None)

        # ── Status publish ──
        self.status_pub.publish(String(data=self.mode))

    def _run_active_mode(self, s, d_err, v_err):
        """Solver 호출 + WpntArray 변환 + 수렴 체크"""
        ### HJ : convert car pose (raceline frenet + cartesian) → centerline frenet
        #       ----------------------------------------------------------------
        #       The MPC solver and Track3D are indexed by CENTERLINE s. The car
        #       state we receive is in two forms:
        #         - self.cur_s, self.cur_d  : RACELINE frenet (/car_state/odom_frenet)
        #         - self.cur_x/y/z, cur_yaw : cartesian (/car_state/odom)
        #       IY's original code fed (cur_s, cur_d) straight into the MPC, which
        #       silently mis-indexed theta_interpolator and made chi/n wrong.
        #       Here we go cartesian → centerline via the FrenetConverter.
        if not self.have_odom:
            rospy.logwarn_throttle(2.0,
                "[LocalRacelineMux] ACTIVE but no /car_state/odom yet — skipping")
            self._publish_local_wpnts_and_markers(None)
            return
        ### HJ : use Track3D-native exact conversion (not FrenetConverter).
        #        Guarantees s, n live on the SAME spline as the solver's
        #        theta/Omega_z/w_tr_* interpolators — no more chi/bound
        #        mismatch from spline drift.
        s_cent, n_cent = self._cart_to_cl_frenet_exact(
            self.cur_x, self.cur_y, self.cur_z)

        # chi in the centerline frame (ONLY reliable after s→s_cent conversion)
        theta_track = float(self.track_handler.theta_interpolator(s_cent))
        chi = self._normalize_angle(self.cur_yaw - theta_track)
        ### HJ : end

        # ── ax, ay ──
        ax, ay = self._get_accelerations()

        ### HJ : sanitize ax/ay before feeding x0.
        #   In sim, odom twist differentiation occasionally spikes to
        #   ±30+ m/s² (>3g) which is physically impossible for the RC car
        #   and corrupts the SQP initial state. Clamp to a generous
        #   envelope that still covers any realistic friction event,
        #   and zero out NaN/Inf.
        AX_CLAMP = 8.0
        AY_CLAMP = 10.0
        if not np.isfinite(ax):
            ax = 0.0
        if not np.isfinite(ay):
            ay = 0.0
        ax = float(np.clip(ax, -AX_CLAMP, AX_CLAMP))
        ay = float(np.clip(ay, -AY_CLAMP, AY_CLAMP))
        ### HJ : end

        # ── Solver 호출 ──
        V = max(self.cur_vs, 0.5)  # minimum velocity for solver stability

        ### HJ : ===== DEBUG BLOCK 1 — inputs to solver =====
        #   Verify the frenet conversion round-trip (cart → cl → cart) has
        #   near-zero error, and flag any NaN/Inf/out-of-range inputs.
        self._debug_inputs(s_cent, n_cent, V, chi, ax, ay, theta_track)
        ### HJ : end

        ### IY : cold start 시 global raceline 기반 fake prev_solution 생성
        ### prev_solution이 없으면 solver가 현재 상태를 상수로 복사해서 initial guess로 씀
        ### → 트랙 밖이면 infeasible guess → SQP_RTI 수렴 실패
        ### global raceline의 centerline 기준 (s, n, V)를 initial guess로 제공
        prev_sol = self.prev_solution
        if prev_sol is None:
            ### HJ : use s_cent (centerline) for guess building, not raceline s
            prev_sol = self._build_global_raceline_guess(s_cent)
            self.get_logger().info("[LocalRacelineMux] Cold start: using global raceline as initial guess")

        t0 = time.time()
        try:
            ### HJ : feed CENTERLINE frenet (s_cent, n_cent, chi_cent) into MPC.
            #       previously passed raceline frenet (self.cur_s, self.cur_d)
            #       which silently mismatched Track3D's centerline indexing.
            raceline = self.planner.calc_raceline(
                s=s_cent,
                V=V,
                n=n_cent,
                chi=chi,
                ax=ax,
                ay=ay,
                safety_distance=self.safety_distance,
                prev_solution=prev_sol,
                V_max=self.params['vehicle_params']['v_max']
            )
            dt_ms = (time.time() - t0) * 1000

            ### HJ : guard against storing a poisoned warm start.
            #   Once the solver emits a large-slack / NaN solution, feeding it
            #   back as the next initial guess permanently strands SQP_RTI
            #   (we observed eps_n=60+ persisting for 10+ iterations).
            #   Reject it — next call will rebuild from global raceline.
            has_nan = any(
                isinstance(v, np.ndarray) and not np.all(np.isfinite(v))
                for v in raceline.values())
            eps_n_val = float(np.max(raceline.get('epsilon_n', np.zeros(1))))
            eps_ax_val = float(np.max(raceline.get('epsilon_a_x', np.zeros(1))))
            eps_ay_val = float(np.max(raceline.get('epsilon_a_y', np.zeros(1))))
            EPS_N_REJECT = 1.0     # 1 m track-bound slack is already extreme
            EPS_GG_REJECT = 5.0    # diamond slack units
            poisoned = (has_nan or eps_n_val > EPS_N_REJECT
                        or eps_ax_val > EPS_GG_REJECT
                        or eps_ay_val > EPS_GG_REJECT)
            if poisoned:
                self.prev_solution = None
                rospy.logwarn_throttle(1.0,
                    "[LocalRacelineMux] Rejecting poisoned warm start "
                    "(NaN=%s eps_n=%.2f eps_ax=%.2f eps_ay=%.2f)",
                    has_nan, eps_n_val, eps_ax_val, eps_ay_val)
            else:
                self.prev_solution = raceline
            ### HJ : end

            ### HJ : ===== DEBUG BLOCK 2 — solver stats + output health =====
            self._debug_outputs(raceline, s_cent, n_cent, V, chi, ax, ay,
                                d_err, v_err, dt_ms, prev_sol)
            self._debug_horizon_smoothness(raceline)
            ### HJ : end
        except Exception as e:
            import traceback
            self.get_logger().error("[LocalRacelineMux] Solver failed: %s\n%s", str(e), traceback.format_exc())
            self._publish_local_wpnts_and_markers(None)
            return

        ### HJ : convergence check in RACELINE frenet — mirror the coordinate
        #       system of ref_s / ref_n (which come from the json raceline).
        #       solver output's 's' and 'n' are in CENTERLINE frame, so convert
        #       via the cartesian x,y,z and the raceline FrenetConverter.
        tail_start = max(0, len(raceline['n']) - 10)
        tail_x = np.asarray(raceline['x'][tail_start:], dtype=np.float64)
        tail_y = np.asarray(raceline['y'][tail_start:], dtype=np.float64)
        tail_z = np.asarray(raceline['z'][tail_start:], dtype=np.float64)
        tail_s_guess = (np.asarray(raceline['s'][tail_start:]) % self.track_length) \
            * (self.race_length / self.track_length)
        fr_tail = self.frenet_race.get_frenet_3d(tail_x, tail_y, tail_z, s=tail_s_guess)
        tail_s_race = np.asarray(fr_tail[0]).flatten() % self.race_length
        tail_n_race = np.asarray(fr_tail[1]).flatten()
        tail_V = np.asarray(raceline['V'][tail_start:])

        tail_n_ref = np.interp(tail_s_race, self.ref_s, self.ref_n, period=self.race_length)
        tail_v_ref = np.interp(tail_s_race, self.ref_s, self.ref_v, period=self.race_length)

        max_n_err = np.max(np.abs(tail_n_race - tail_n_ref))
        max_v_err = np.max(np.abs(tail_V - tail_v_ref))
        ### HJ : end

        if (max_n_err < self.convergence_d_threshold and
                max_v_err < self.convergence_v_threshold and
                d_err < self.convergence_d_threshold and
                v_err < self.convergence_v_threshold):
            self.mode = self.PASSTHROUGH
            self.prev_solution = None
            self.get_logger().warning("[LocalRacelineMux] -> PASSTHROUGH (converged)")
            ### HJ : TEST mode — clear test markers when converged; do NOT
            #       touch /global_waypoints_scaled (car keeps following the
            #       original vel_planner output).
            self._publish_local_wpnts_and_markers(None)
            return

        ### HJ : publish SOLVER HORIZON only as a dedicated test topic.
        #       Shape (color / scale / cylinder vel markers) mirrors
        #       3d_state_machine_node's local_waypoints style so RViz configs
        #       with "/local_waypoints/*" subscribers can be reused by just
        #       changing the topic prefix.
        self._publish_local_wpnts_and_markers(raceline)

    # ═══════════════════════════════════════════════════════
    # Test-mode publish (WpntArray + markers, solver horizon only)
    # ═══════════════════════════════════════════════════════

    def _publish_local_wpnts_and_markers(self, raceline):
        """### HJ : publish solver horizon to /3d_optimized_local_waypoints_raceline{,/markers,/vel_markers}.
        #       Style mirrors 3d_state_machine_node's local_waypoints (SPHERE
        #       speed-color, CYLINDER vel height). Does NOT touch
        #       /global_waypoints_scaled — car keeps normal driving.
        #
        #       If raceline is None (PASSTHROUGH/converged), publish DELETEALL
        #       to clear any stale markers.
        """
        stamp = self.get_clock().now().to_msg()

        if raceline is None:
            # clear markers
            clear = MarkerArray()
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.action = Marker.DELETEALL
            clear.markers.append(m)
            self.loc_markers_pub.publish(clear)
            self.vel_markers_pub.publish(clear)
            # empty wpnt array
            empty = WpntArray()
            empty.header.stamp = stamp
            empty.header.frame_id = "map"
            self.wpnt_pub.publish(empty)
            return

        # --- Build WpntArray (solver samples directly, same as sim output) ---
        xs = np.asarray(raceline['x'], dtype=np.float64)
        ys = np.asarray(raceline['y'], dtype=np.float64)
        zs = np.asarray(raceline['z'], dtype=np.float64)
        N = len(xs)

        # raceline frenet (for d_m and s_m)
        s_guess_race = (np.asarray(raceline['s']) % self.track_length) \
            * (self.race_length / self.track_length)
        fr_rl = self.frenet_race.get_frenet_3d(xs, ys, zs, s=s_guess_race)
        s_race = np.asarray(fr_rl[0]).flatten()
        n_race = np.asarray(fr_rl[1]).flatten()

        ### HJ : Track3D-native exact conversion for output publishing, same
        #        reasoning as the input path — keeps s aligned with the spline
        #        the solver used. Loop of N≈30 is <1ms.
        s_cent, _ = self._cart_to_cl_frenet_exact_batch(xs, ys, zs)

        V_arr   = np.asarray(raceline['V'])
        ax_arr  = np.asarray(raceline['ax'])
        ay_arr  = np.asarray(raceline['ay'])
        chi_arr = np.asarray(raceline['chi'])

        wpnts_out = []
        for i in range(N):
            w = Wpnt()
            w.id = i
            w.s_m = float(s_race[i] % self.race_length)
            w.d_m = float(n_race[i])
            w.x_m = float(xs[i])
            w.y_m = float(ys[i])
            w.z_m = float(zs[i])
            theta_i = float(self.track_handler.theta_interpolator(float(s_cent[i])))
            w.psi_rad = self._normalize_angle(theta_i + float(chi_arr[i]))
            w.kappa_radpm = float(ay_arr[i]) / max(float(V_arr[i]) ** 2, 0.01)
            w.vx_mps = float(V_arr[i])
            w.ax_mps2 = float(ax_arr[i])
            # track bounds at this centerline s
            w_tr_right = float(np.interp(
                float(s_cent[i]), self.track_handler.s, self.track_handler.w_tr_right,
                period=self.track_length))
            w_tr_left = float(np.interp(
                float(s_cent[i]), self.track_handler.s, self.track_handler.w_tr_left,
                period=self.track_length))
            w.d_right = abs(w_tr_right) - float(n_race[i])
            w.d_left = w_tr_left - float(n_race[i])
            w.mu_rad = float(self.track_handler.mu_interpolator(float(s_cent[i])))
            wpnts_out.append(w)

        out = WpntArray()
        out.header.stamp = stamp
        out.header.frame_id = "map"
        out.wpnts = wpnts_out
        self.wpnt_pub.publish(out)

        # --- Marker styles (mirror 3d_state_machine_node local_waypoints) ---
        vx_vals = [w.vx_mps for w in wpnts_out]
        vx_min = min(vx_vals) if vx_vals else 0.0
        vx_max = max(vx_vals) if vx_vals else 1.0
        def _vel_color(vx):
            t = (vx - vx_min) / (vx_max - vx_min) if vx_max > vx_min else 0.5
            return (max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5))),
                    max(0.0, min(1.0, 2.0 * t)),
                    0.0)

        # SPHERE markers — each point colored by speed
        loc_markers = MarkerArray()
        # clear old first
        clr = Marker()
        clr.header.frame_id = "map"
        clr.header.stamp = stamp
        clr.action = Marker.DELETEALL
        loc_markers.markers.append(clr)
        for i, w in enumerate(wpnts_out):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = stamp
            mrk.type = Marker.SPHERE
            mrk.scale.x = 0.15
            mrk.scale.y = 0.15
            mrk.scale.z = 0.15
            mrk.color.a = 1.0
            mrk.color.r, mrk.color.g, mrk.color.b = _vel_color(w.vx_mps)
            mrk.id = i + 1  # id 0 reserved for DELETEALL
            mrk.pose.position.x = w.x_m
            mrk.pose.position.y = w.y_m
            mrk.pose.position.z = w.z_m
            mrk.pose.orientation.w = 1
            loc_markers.markers.append(mrk)
        self.loc_markers_pub.publish(loc_markers)

        # CYLINDER vel markers — height = speed * VEL_SCALE
        VEL_SCALE = 0.1317
        vel_markers = MarkerArray()
        clr2 = Marker()
        clr2.header.frame_id = "map"
        clr2.header.stamp = stamp
        clr2.action = Marker.DELETEALL
        vel_markers.markers.append(clr2)
        for i, w in enumerate(wpnts_out):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = stamp
            mrk.type = Marker.CYLINDER
            mrk.id = i + 1
            mrk.scale.x = 0.1
            mrk.scale.y = 0.1
            height = max(w.vx_mps * VEL_SCALE, 0.02)
            mrk.scale.z = height
            mrk.color.a = 0.7
            mrk.color.r, mrk.color.g, mrk.color.b = _vel_color(w.vx_mps)
            mrk.pose.position.x = w.x_m
            mrk.pose.position.y = w.y_m
            mrk.pose.position.z = w.z_m + height * 0.5
            mrk.pose.orientation.w = 1
            vel_markers.markers.append(mrk)
        self.vel_markers_pub.publish(vel_markers)


    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════

    def _build_global_raceline_guess(self, cur_s):
        """### IY : global raceline 기반 fake prev_solution 생성 (cold start용)
        solver의 __gen_raceline이 prev_solution으로부터 initial guess를 보간함.
        global raceline의 (s_center, n_center, V)를 horizon 구간만큼 구성.
        """
        horizon = self.optimization_horizon
        N = self.N_steps
        s_array = np.linspace(cur_s, cur_s + horizon, N)
        s_mod = s_array % self.track_length

        # centerline 기준 s, n, V를 global raceline에서 보간
        # ref_s_center는 centerline s 기준, 등간격이 아닐 수 있으므로 period 보간
        V_array = np.interp(s_mod, self.ref_s_center, self.ref_v, period=self.track_length)
        n_array = np.interp(s_mod, self.ref_s_center, self.ref_n_center, period=self.track_length)

        return {
            's': s_array,
            'V': V_array,
            'n': n_array,
            'chi': np.zeros(N),
            'ax': np.zeros(N),
            'ay': V_array ** 2 * np.interp(
                s_mod, self.track_handler.s, self.track_handler.Omega_z,
                period=self.track_length),
            'jx': np.zeros(N),
            'jy': np.zeros(N),
        }

    # ═══════════════════════════════════════════════════════
    ### HJ : Exact cartesian → centerline-frenet using Track3D's OWN spline.
    #        Avoids FrenetConverter's Newton-on-different-spline drift.
    # ═══════════════════════════════════════════════════════

    def _cart_to_cl_frenet_exact(self, x, y, z):
        """Exact (x,y,z) → (s_cent, n_cent) consistent with the MPC solver.

        Why not FrenetConverter:
          - FrenetConverter fits its own internal spline and runs Newton.
          - Track3D fits a DIFFERENT (linear) spline — which the solver uses
            for theta, Omega_z, w_tr_*.
          - Feeding FrenetConverter's s into Track3D's interpolators returns
            a theta for a slightly different arc-length, so chi and track
            bounds are off. Near high-curvature / loop-boundary spots, Newton
            can also converge to the wrong branch (bad initial guess), giving
            |n|>>track_width — immediately infeasible for the MPC.

        Algorithm (O(N), N≈1800 is fine at 10Hz):
          1. Global brute-force nearest discrete waypoint.
          2. Analytical segment projection onto the two adjacent polyline
             segments → exact arc-length foot-of-perpendicular.
          3. n = signed distance along Track3D's 3D normal at that s.

        Uses ONLY np.interp on Track3D's own (x, y, z, theta, mu, phi) arrays
        — exactly the linear interpolation the CasADi interpolants do inside
        the solver — so s and theta lookups line up to numerical precision.
        """
        xs = np.asarray(self.track_handler.x)
        ys = np.asarray(self.track_handler.y)
        zs = np.asarray(self.track_handler.z)
        s_arr = np.asarray(self.track_handler.s)
        L = self.track_length
        N = len(xs)

        # 1. Global nearest discrete waypoint
        d2 = (xs - x) ** 2 + (ys - y) ** 2 + (zs - z) ** 2
        i = int(np.argmin(d2))

        # 2. Exact projection on the two adjacent segments
        best_s, best_d2 = None, np.inf
        for ja, jb in (((i - 1) % N, i), (i, (i + 1) % N)):
            xa, ya, za = xs[ja], ys[ja], zs[ja]
            xb, yb, zb = xs[jb], ys[jb], zs[jb]
            dxab = xb - xa
            dyab = yb - ya
            dzab = zb - za
            len2 = dxab * dxab + dyab * dyab + dzab * dzab
            if len2 < 1e-12:
                continue
            t = ((x - xa) * dxab + (y - ya) * dyab + (z - za) * dzab) / len2
            t = max(0.0, min(1.0, t))
            fx = xa + t * dxab
            fy = ya + t * dyab
            fz = za + t * dzab
            dseg2 = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
            if dseg2 < best_d2:
                best_d2 = dseg2
                sa = s_arr[ja]
                sb = s_arr[jb]
                if sb < sa:  # loop wrap
                    sb = sb + L
                best_s = sa + t * (sb - sa)
                if best_s >= L:
                    best_s -= L

        # 3. n = signed distance along the Track3D 3D normal at best_s
        theta = float(np.interp(best_s, s_arr, self.track_handler.theta))
        mu = float(np.interp(best_s, s_arr, self.track_handler.mu))
        phi = float(np.interp(best_s, s_arr, self.track_handler.phi))
        normal = Track3D.get_normal_vector_numpy(theta, mu, phi)
        xc = float(np.interp(best_s, s_arr, xs))
        yc = float(np.interp(best_s, s_arr, ys))
        zc = float(np.interp(best_s, s_arr, zs))
        disp = np.array([x - xc, y - yc, z - zc])
        n = float(np.dot(disp, normal))
        return float(best_s), n

    def _cart_to_cl_frenet_exact_batch(self, xs_car, ys_car, zs_car):
        """Vectorized version for publishing the solver horizon (N≈30)."""
        out_s = np.empty(len(xs_car), dtype=np.float64)
        out_n = np.empty(len(xs_car), dtype=np.float64)
        for i in range(len(xs_car)):
            out_s[i], out_n[i] = self._cart_to_cl_frenet_exact(
                float(xs_car[i]), float(ys_car[i]), float(zs_car[i]))
        return out_s, out_n

    def _ensure_centerline_ref(self):
        """### HJ : trivial in the raceline-framed variant.

        In the raceline frame, n=0 IS the raceline itself. The cold-start
        initial guess _build_global_raceline_guess(s_cur) wants (s_ref, n_ref)
        arrays describing the raceline as seen from the solver's frame — that
        is literally (s, 0) for every point. So we just set:
          ref_s_center = Track3D's own s grid (already raceline arc-length)
          ref_n_center = zeros
        No nearest-point projection needed; the solver's state variables s
        and n already live in the raceline frame.
        """
        self.ref_s_center = np.asarray(self.track_handler.s, dtype=np.float64)
        self.ref_n_center = np.zeros_like(self.ref_s_center)
        self._centerline_ref_source = 'raceline-trivial'
        self.get_logger().info(
            "[LocalRacelineMux] raceline-frame: ref_s_center = track_handler.s "
            "(N=%d, L=%.2fm), ref_n_center = 0 everywhere.",
            len(self.ref_s_center), float(self.ref_s_center[-1]))

    # ═══════════════════════════════════════════════════════
    ### HJ : DEBUG helpers — verbose logging for Frenet consistency,
    #        solver I/O, NaN/bound violations. Intended for diagnosis:
    #        once the node runs clean, the calls can be commented out.
    # ═══════════════════════════════════════════════════════

    def _debug_inputs(self, s_cent, n_cent, V, chi, ax, ay, theta_track):
        """Round-trip check + NaN/range screening on MPC inputs. Throttled."""
        # 1. Frenet round-trip: (s_cent, n_cent) → cart → (s_cent', n_cent')
        try:
            xyz_back = self.track_handler.sn2cartesian(
                np.array([s_cent]), np.array([n_cent]))
            x_back = float(xyz_back[0, 0])
            y_back = float(xyz_back[0, 1])
            z_back = float(xyz_back[0, 2])
            cart_err = float(np.sqrt(
                (x_back - self.cur_x) ** 2 +
                (y_back - self.cur_y) ** 2 +
                (z_back - self.cur_z) ** 2))
        except Exception as e:
            cart_err = float('nan')
            x_back = y_back = z_back = float('nan')
            rospy.logwarn_throttle(2.0, "[DBG] sn2cartesian failed: %s", e)

        # 2. Input sanity
        bad_flags = []
        for name, val in [('s_cent', s_cent), ('n_cent', n_cent), ('V', V),
                          ('chi', chi), ('ax', ax), ('ay', ay),
                          ('theta', theta_track)]:
            if not np.isfinite(val):
                bad_flags.append(f"{name}=NaN/Inf")
        if abs(ay) > 15.0:
            bad_flags.append(f"|ay|={ay:.1f} (>15 suspicious)")
        if abs(ax) > 15.0:
            bad_flags.append(f"|ax|={ax:.1f} (>15 suspicious)")
        if abs(chi) > np.pi / 2:
            bad_flags.append(f"|chi|={chi:.2f} (>pi/2)")

        # 3. Track bound check at current s_cent
        try:
            w_right = float(np.interp(
                s_cent % self.track_length, self.track_handler.s,
                self.track_handler.w_tr_right, period=self.track_length))
            w_left = float(np.interp(
                s_cent % self.track_length, self.track_handler.s,
                self.track_handler.w_tr_left, period=self.track_length))
            # n_cent should satisfy w_right < n_cent < w_left (IY convention)
            # w_right is typically NEGATIVE (right side of track)
            n_margin_right = n_cent - w_right
            n_margin_left = w_left - n_cent
            if n_margin_right < 0 or n_margin_left < 0:
                bad_flags.append(
                    f"n_cent={n_cent:.3f} OUTSIDE bounds"
                    f" [w_r={w_right:.3f}, w_l={w_left:.3f}]")
        except Exception as e:
            w_left = w_right = float('nan')
            rospy.logwarn_throttle(2.0, "[DBG] track bound interp failed: %s", e)

        rospy.loginfo_throttle(1.0,
            "[DBG-IN] cart=(%.3f,%.3f,%.3f) yaw=%.3f | "
            "cl(s=%.2f n=%.3f) round-trip err=%.3fm | "
            "track_bnd=[%.3f,%.3f] | theta=%.3f chi=%.3f | "
            "V=%.2f ax=%.2f ay=%.2f%s",
            self.cur_x, self.cur_y, self.cur_z, self.cur_yaw,
            s_cent, n_cent, cart_err,
            w_right, w_left, theta_track, chi,
            V, ax, ay,
            (" | BAD: " + "; ".join(bad_flags)) if bad_flags else "")

    def _debug_horizon_smoothness(self, raceline):
        """Detect whether the output path 'kinks' by measuring:
          1. Per-stage spatial step ds_cart (Euclidean distance between
             consecutive horizon samples). Sudden large deltas = output
             spatial jump.
          2. Per-stage lateral delta dn (|n[i+1] - n[i]|). Spikes indicate
             the solver snapped laterally between stages.
          3. Per-stage arc-length delta ds. If uniform s-sampling is
             expected, spikes mean the solver re-parametrized oddly.
          4. Same-stage drift vs previous solve: if stage i at t and t-dt
             correspond to nearly same s, their cartesian distance should
             be ~(V * dt). Excess = index-tracking drift.
        """
        xs = np.asarray(raceline['x'])
        ys = np.asarray(raceline['y'])
        zs = np.asarray(raceline['z'])
        ss = np.asarray(raceline['s'])
        ns = np.asarray(raceline['n'])
        Vs = np.asarray(raceline['V'])

        if len(xs) < 2:
            return

        dxy = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
        dn = np.diff(ns)
        ### HJ : ds un-wrapped across the centerline loop seam (L≈89.87m),
        #        otherwise stage-29 at s≈0 right after stage-28 at s≈89.8
        #        shows ds=-89.5 and is mis-flagged as a "kink". This is a
        #        coordinate-system wrap, not a real jump; xyz is continuous.
        L = float(self.track_length)
        ds_raw = np.diff(ss)
        ds = ds_raw.copy()
        ds[ds < -L / 2.0] += L
        ds[ds > L / 2.0] -= L

        # expected: ds should be nearly constant (horizon uniformly spaced in s)
        ds_nominal = float(self.optimization_horizon) / max(self.N_steps - 1, 1)
        ds_jitter = np.abs(ds - ds_nominal)

        # expected cartesian step ≈ ds_nominal (since ds is ~V*dt along track
        # and cartesian step ≈ ds when n doesn't change much)
        # Detect stages where dxy/ds ratio deviates a lot (sharp jump or stop)
        ratio = dxy / np.maximum(np.abs(ds), 1e-6)

        # worst-spike stages
        i_worst_cart = int(np.argmax(dxy))
        i_worst_n = int(np.argmax(np.abs(dn)))
        i_worst_ds = int(np.argmax(ds_jitter))

        # also check index-tracking vs previous stored horizon
        drift_str = ""
        if hasattr(self, '_prev_horizon') and self._prev_horizon is not None:
            pxs, pys = self._prev_horizon
            if len(pxs) == len(xs):
                dxys_idx = np.sqrt((xs - pxs) ** 2 + (ys - pys) ** 2)
                idx_drift_max = int(np.argmax(dxys_idx))
                drift_str = (" | idx-drift max=%.3fm @stage%d" %
                             (float(dxys_idx.max()), idx_drift_max))
        self._prev_horizon = (xs.copy(), ys.copy())

        rospy.loginfo_throttle(0.5,
            "[DBG-HORIZON] dxy max=%.3f@%d (stage %d→%d) | "
            "|dn| max=%.3f@%d | ds jitter max=%.3f@%d (nominal=%.3f) | "
            "dxy/|ds| range=[%.2f, %.2f]%s",
            float(dxy.max()), i_worst_cart, i_worst_cart, i_worst_cart + 1,
            float(np.abs(dn).max()), i_worst_n,
            float(ds_jitter.max()), i_worst_ds, ds_nominal,
            float(ratio.min()), float(ratio.max()),
            drift_str)

        # additional: print full stage table when a jump is large
        if float(dxy.max()) > 3.0 * ds_nominal or float(np.abs(dn).max()) > 0.5:
            self.get_logger().warning(
                "[DBG-HORIZON] SHARP KINK — stage-by-stage dump:\n"
                "  i   s       n      V     x       y      dxy   dn    ds")
            for i in range(len(xs)):
                dxy_i = float(dxy[i - 1]) if i > 0 else 0.0
                dn_i = float(dn[i - 1]) if i > 0 else 0.0
                ds_i = float(ds[i - 1]) if i > 0 else 0.0
                self.get_logger().warning(
                    "  %2d  %6.2f  %+5.3f  %4.2f  %+6.2f %+6.2f  %5.3f %+5.3f %+5.3f",
                    i, float(ss[i]), float(ns[i]), float(Vs[i]),
                    float(xs[i]), float(ys[i]), dxy_i, dn_i, ds_i)

    def _debug_outputs(self, raceline, s_cent, n_cent, V, chi, ax, ay,
                       d_err, v_err, dt_ms, prev_sol):
        """Solver output health: status, slacks, range, NaN."""
        # solver status (0=success, 1=NAN, 2=MAXITER, 3=MINSTEP, 4=QP_FAIL)
        # acados has both `get_status()` and `get_stats('status')`; try both.
        status = -1
        try:
            status = int(self.planner.solver.get_status())
        except Exception:
            try:
                status = int(self.planner.solver.get_stats('status'))
            except Exception:
                pass

        # NaN check across all horizon arrays
        nan_fields = [k for k, v in raceline.items()
                      if isinstance(v, np.ndarray) and not np.all(np.isfinite(v))]

        # slacks
        eps_n = raceline.get('epsilon_n', np.zeros(1))
        eps_ax = raceline.get('epsilon_a_x', np.zeros(1))
        eps_ay = raceline.get('epsilon_a_y', np.zeros(1))
        eps_axy = raceline.get('epsilon_a_xy', np.zeros(1))

        # ranges over horizon
        def _rng(a):
            a = np.asarray(a)
            finite = a[np.isfinite(a)]
            if len(finite) == 0:
                return (float('nan'), float('nan'))
            return (float(finite.min()), float(finite.max()))

        V_rng = _rng(raceline['V'])
        n_rng = _rng(raceline['n'])
        chi_rng = _rng(raceline['chi'])
        ax_rng = _rng(raceline['ax'])
        ay_rng = _rng(raceline['ay'])

        # prev_sol source info (cold start vs warm)
        ps_tag = "warm" if (self.prev_solution is not None and
                            prev_sol is self.prev_solution) else "cold"

        rospy.loginfo_throttle(1.0,
            "[DBG-OUT] status=%d %s | %.1fms | d_err=%.3f v_err=%.3f | "
            "INPUT: s=%.2f n=%.3f V=%.2f chi=%.2f ax=%.2f ay=%.2f | "
            "OUT V[%.2f,%.2f] n[%.3f,%.3f] chi[%.2f,%.2f] "
            "ax[%.2f,%.2f] ay[%.2f,%.2f] | "
            "eps: n_max=%.3f ax_max=%.3f ay_max=%.3f axy_max=%.3f%s",
            status, ps_tag, dt_ms, d_err, v_err,
            s_cent, n_cent, V, chi, ax, ay,
            V_rng[0], V_rng[1], n_rng[0], n_rng[1], chi_rng[0], chi_rng[1],
            ax_rng[0], ax_rng[1], ay_rng[0], ay_rng[1],
            float(np.max(eps_n)), float(np.max(eps_ax)),
            float(np.max(eps_ay)), float(np.max(eps_axy)),
            (" | NaN in: " + ",".join(nan_fields)) if nan_fields else "")

        # status==-1 is our probe-failed sentinel (API couldn't fetch it),
        # not a real solver failure. Warn only on concrete nonzero codes.
        if status > 0:
            rospy.logwarn_throttle(2.0,
                "[DBG-OUT] SOLVER UNHAPPY status=%d "
                "(1=NAN,2=MAXITER,3=MINSTEP,4=QP_FAIL)", status)

    # ═══════════════════════════════════════════════════════
    # Original helpers
    # ═══════════════════════════════════════════════════════

    def _get_accelerations(self):
        """ax, ay 반환.
           실차: /ekf/imu/data (직접 측정, 노이즈 적음)
           sim:  /car_state/odom twist 미분 + LPF (IMU 없으므로)
           fallback: prev_solution 의 ax[0], ay[0] 또는 0
        """
        ### HJ : sim 에서는 odom 미분 값 사용 (IMU 안 읽음)
        if self.is_sim:
            if self._prev_odom_stamp is not None:
                return self.cur_odom_ax, self.cur_odom_ay
            # 아직 첫 odom 안 온 경우
            if self.prev_solution is not None:
                return float(self.prev_solution['ax'][0]), float(self.prev_solution['ay'][0])
            return 0.0, 0.0
        ### HJ : end
        # 실차
        return self.cur_imu_ax, self.cur_imu_ay

    @staticmethod
    def _normalize_angle(angle):
        """Wrap angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


if __name__ == '__main__':
    try:
        node = LocalRacelineMux()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
