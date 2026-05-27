#!/usr/bin/env python3
### IY : Event-triggered local racing line mux node
### Normally pass-through global waypoints, compute local racing line when vehicle deviates

import os
import sys
import yaml
import numpy as np
import time
import tf_transformations

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from f110_msgs.msg import WpntArray, Wpnt

# Add paths for solver imports
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)  # for track3D_lite
_solver_src = os.path.join(_dir, '..', 'src')
sys.path.append(os.path.abspath(_solver_src))

### IY : use track3D_lite (pandas-free) instead of track3D
from track3D_lite import Track3D
from ggManager import GGManager
from local_racing_line_planner import LocalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model


class LocalRacelineMux:
    """
    ### IY : Local racing line mux node
    - PASSTHROUGH: global waypoints 그대로 전달
    - ACTIVE: solver로 local racing line 계산 후 global waypoints에 덮어쓰기
    """

    PASSTHROUGH = "PASSTHROUGH"
    ACTIVE = "ACTIVE"

    def __init__(self):
        rospy.init_node('local_raceline_mux', anonymous=False)

        # ── Parameters ──
        self.d_threshold = self._get_param_or_default('~d_threshold', 0.3)
        self.v_threshold = self._get_param_or_default('~v_threshold', 1.5)
        self.convergence_d_threshold = self._get_param_or_default('~convergence_d_threshold', 0.1)
        self.convergence_v_threshold = self._get_param_or_default('~convergence_v_threshold', 0.5)
        self.optimization_horizon = self._get_param_or_default('~optimization_horizon', 10.0)
        self.N_steps = self._get_param_or_default('~N_steps', 30)
        self.gg_mode = self._get_param_or_default('~gg_mode', 'diamond')
        self.safety_distance = self._get_param_or_default('~safety_distance', 0.05)
        self.is_sim = self._get_param_or_default('/sim', False)
        self.map_name = self._get_param_or_default('/map', '')
        ### IY : filenames as params (map name and file prefix may differ)
        self.smoothed_track_file = self._get_param_or_default('~smoothed_track_file', 'gazebo_wall_2_3d_smoothed.csv')

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
        self.gb_wpnts = None  # latest WpntArray from vel_scaler
        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0
        self.cur_yaw = 0.0
        self.cur_imu_ax = 0.0
        self.cur_imu_ay = 0.0
        self.solver_initialized = False

        # ── Load vehicle params ──
        with open(self.vehicle_params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        # ── Load global raceline reference (for deviation check) ──
        self._load_global_raceline_ref()

        # ── Initialize solver (Track3D, GGManager, model, planner) ──
        self._init_solver()

        # ── Publishers ──
        self.wpnt_pub = rospy.Publisher(
            '/global_waypoints_scaled', WpntArray, queue_size=1)
        ### IY : online_waypoints — solver가 계산한 local racing line (RViz 시각화용)
        self.online_wpnt_pub = rospy.Publisher(
            '/online_waypoints', WpntArray, queue_size=1)
        self.status_pub = rospy.Publisher(
            '/local_raceline/status', String, queue_size=1)

        # ── Subscribers ──
        rospy.Subscriber(
            '/global_waypoints_scaled_raw', WpntArray, self._gb_wpnts_cb)
        rospy.Subscriber(
            '/car_state/odom_frenet', Odometry, self._frenet_cb)
        rospy.Subscriber(
            '/car_state/odom', Odometry, self._odom_cb)

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
        json_path = os.path.join(
            os.path.abspath(os.path.join(
                _dir, '..', '..', '..', '..', 'stack_master', 'maps', self.map_name)),
            'global_waypoints.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        wpnts = data['global_traj_wpnts_iqp']['wpnts']
        self.ref_s = np.array([w['s_m'] for w in wpnts])
        self.ref_v = np.array([w['vx_mps'] for w in wpnts])
        self.ref_n = np.array([w['d_m'] for w in wpnts])

        ### IY : centerline 기준 s, n 로드 (solver cold start 초기 guess용)
        ### export_global_waypoints.py에서 별도 섹션 'centerline_ref'로 저장됨
        if 'centerline_ref' in data:
            self.ref_s_center = np.array(data['centerline_ref']['s_center_m'])
            self.ref_n_center = np.array(data['centerline_ref']['n_center_m'])
            self.get_logger().info("[LocalRacelineMux] Loaded centerline ref: n_center=[%.3f, %.3f]",
                          self.ref_n_center.min(), self.ref_n_center.max())
        else:
            ### IY : 이전 JSON 호환 — centerline_ref 없으면 fallback
            self.ref_s_center = self.ref_s
            self.ref_n_center = np.zeros_like(self.ref_s)
            self.get_logger().warning("[LocalRacelineMux] No centerline_ref in JSON, using fallback (n=0)")

        self.get_logger().info("[LocalRacelineMux] Loaded raceline ref from JSON: %d pts, s=[%.1f, %.1f]",
                      len(self.ref_s), self.ref_s[0], self.ref_s[-1])

    def _init_solver(self):
        """sim_local_racing_line.py 패턴대로 solver 초기화"""
        self.get_logger().info("[LocalRacelineMux] Initializing solver (acados compile may take 10-30s)...")
        t0 = time.time()

        # Track3D
        self.track_handler = Track3D(path=self.smoothed_track_path)
        self.track_length = self.track_handler.s[-1]

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

        ### IY : solver 파라미터 강화
        ### - SQP_RTI 1 iter → 3 iter (수렴 개선, 여전히 real-time 가능)
        ### - w_slack_n 1.0 → 1e3 (track bound 위반 페널티 대폭 강화)
        self.planner = LocalRacinglinePlanner(
            params=self.params,
            track_handler=self.track_handler,
            gg_handler=self.gg_handler,
            model=self.model,
            nlp_solver_type='SQP_RTI',
            optimization_horizon=self.optimization_horizon,
            gg_mode=self.gg_mode,
            N_steps=self.N_steps,
            sqp_max_iter=3,
            qp_max_iter=50,
            w_slack_n=1e3,
        )

        self.solver_initialized = True
        self.get_logger().info("[LocalRacelineMux] Solver initialized in %.1fs", time.time() - t0)

    # ═══════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════

    def _gb_wpnts_cb(self, msg):
        """Global waypoints (from vel_scaler, remapped to _raw)"""
        self.gb_wpnts = msg

    def _frenet_cb(self, msg):
        """Frenet odometry: s, d, vs"""
        self.cur_s = msg.pose.pose.position.x
        self.cur_d = msg.pose.pose.position.y
        self.cur_vs = msg.twist.twist.linear.x

    def _odom_cb(self, msg):
        """Map-frame odometry: yaw for chi calculation"""
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cur_yaw = yaw

    def _imu_cb(self, msg):
        """IMU acceleration (real car only)"""
        self.cur_imu_ax = msg.linear_acceleration.x
        self.cur_imu_ay = msg.linear_acceleration.y

    # ═══════════════════════════════════════════════════════
    # Main timer loop (10Hz)
    # ═══════════════════════════════════════════════════════

    def _timer_cb(self, event):
        if self.gb_wpnts is None or not self.solver_initialized:
            return

        # ── Deviation check ──
        s = self.cur_s % self.track_length
        n_ref = np.interp(s, self.ref_s, self.ref_n, period=self.track_length)
        v_ref = np.interp(s, self.ref_s, self.ref_v, period=self.track_length)
        d_err = abs(self.cur_d - n_ref)
        v_err = abs(self.cur_vs - v_ref)

        # ── Mode transitions ──
        if self.mode == self.PASSTHROUGH:
            if d_err > self.d_threshold or v_err > self.v_threshold:
                self.mode = self.ACTIVE
                self.get_logger().warning("[LocalRacelineMux] -> ACTIVE (d_err=%.3f, v_err=%.3f)",
                              d_err, v_err)

        if self.mode == self.ACTIVE:
            self._run_active_mode(s, d_err, v_err)
        else:
            # Pass-through: publish global waypoints unchanged
            self.wpnt_pub.publish(self.gb_wpnts)

        # ── Status publish ──
        self.status_pub.publish(String(data=self.mode))

    def _run_active_mode(self, s, d_err, v_err):
        """Solver 호출 + WpntArray 변환 + 수렴 체크"""
        # ── chi 계산 ──
        theta_track = float(self.track_handler.theta_interpolator(s))
        chi = self._normalize_angle(self.cur_yaw - theta_track)

        # ── ax, ay ──
        ax, ay = self._get_accelerations()

        # ── Solver 호출 ──
        V = max(self.cur_vs, 0.5)  # minimum velocity for solver stability

        ### IY : cold start 시 global raceline 기반 fake prev_solution 생성
        ### prev_solution이 없으면 solver가 현재 상태를 상수로 복사해서 initial guess로 씀
        ### → 트랙 밖이면 infeasible guess → SQP_RTI 수렴 실패
        ### global raceline의 centerline 기준 (s, n, V)를 initial guess로 제공
        prev_sol = self.prev_solution
        if prev_sol is None:
            prev_sol = self._build_global_raceline_guess(self.cur_s)
            self.get_logger().info("[LocalRacelineMux] Cold start: using global raceline as initial guess")

        t0 = time.time()
        try:
            raceline = self.planner.calc_raceline(
                s=self.cur_s,
                V=V,
                n=self.cur_d,
                chi=chi,
                ax=ax,
                ay=ay,
                safety_distance=self.safety_distance,
                prev_solution=prev_sol,
                V_max=self.params['vehicle_params']['v_max']
            )
            self.prev_solution = raceline
            dt_ms = (time.time() - t0) * 1000
            ### IY : epsilon_n (track bound slack) 로그 추가 — 0이면 정상, 클수록 트랙 이탈
            eps_n = raceline.get('epsilon_n', np.zeros(1))
            eps_n_max = float(np.max(eps_n))
            n_min, n_max = float(np.min(raceline['n'])), float(np.max(raceline['n']))
            rospy.loginfo_throttle(2.0,
                "[LocalRacelineMux] Solver: %.1fms, d_err=%.3f, v_err=%.3f | "
                "input: s=%.2f V=%.2f n=%.3f chi=%.3f ax=%.2f ay=%.2f | "
                "output: V=[%.1f,%.1f] n=[%.3f,%.3f] | "
                "eps_n_max=%.4f (0=OK)",
                dt_ms, d_err, v_err,
                self.cur_s, V, self.cur_d, chi, ax, ay,
                raceline['V'][0], raceline['V'][-1],
                n_min, n_max,
                eps_n_max)
        except Exception as e:
            import traceback
            self.get_logger().error("[LocalRacelineMux] Solver failed: %s\n%s", str(e), traceback.format_exc())
            self.wpnt_pub.publish(self.gb_wpnts)
            return

        # ── 수렴 체크 (solver 끝부분 + 현재 위치 모두 global에 근접) ──
        tail_start = max(0, len(raceline['n']) - 10)
        tail_s = raceline['s'][tail_start:] % self.track_length
        tail_n = raceline['n'][tail_start:]
        tail_V = raceline['V'][tail_start:]
        tail_n_ref = np.interp(tail_s, self.ref_s, self.ref_n, period=self.track_length)
        tail_v_ref = np.interp(tail_s, self.ref_s, self.ref_v, period=self.track_length)

        max_n_err = np.max(np.abs(tail_n - tail_n_ref))
        max_v_err = np.max(np.abs(tail_V - tail_v_ref))

        if (max_n_err < self.convergence_d_threshold and
                max_v_err < self.convergence_v_threshold and
                d_err < self.convergence_d_threshold and
                v_err < self.convergence_v_threshold):
            self.mode = self.PASSTHROUGH
            self.prev_solution = None
            self.get_logger().warning("[LocalRacelineMux] -> PASSTHROUGH (converged)")
            self.wpnt_pub.publish(self.gb_wpnts)
            return

        # ── Solver 출력 → WpntArray 변환 ──
        modified = self._raceline_to_wpntarray(raceline)
        self.wpnt_pub.publish(modified)

        ### IY : online_waypoints — solver 구간만 별도 publish (RViz용)
        online = self._raceline_to_online_wpntarray(raceline)
        self.online_wpnt_pub.publish(online)

    # ═══════════════════════════════════════════════════════
    # Solver output → WpntArray conversion
    # ═══════════════════════════════════════════════════════

    def _raceline_to_wpntarray(self, raceline):
        """
        ### IY : solver 출력을 global waypoint 그리드에 보간하여 WpntArray 생성
        global WpntArray를 복사 → solver horizon 구간만 덮어쓰기
        """
        wpnts = list(self.gb_wpnts.wpnts)
        num_wpnts = len(wpnts)
        if num_wpnts < 2:
            return self.gb_wpnts

        wpnt_dist = wpnts[1].s_m - wpnts[0].s_m

        # ### IY : Solver 출력 s를 unwrap (연속 증가하도록)
        # np.unwrap period arg는 numpy>=1.21 — 수동 처리로 호환성 확보
        solver_s = raceline['s'].copy()
        for i in range(1, len(solver_s)):
            diff = solver_s[i] - solver_s[i - 1]
            if diff < -self.track_length / 2.0:
                solver_s[i:] += self.track_length
            elif diff > self.track_length / 2.0:
                solver_s[i:] -= self.track_length

        # Global waypoint 그리드 중 solver 구간에 해당하는 인덱스
        s_start = solver_s[0]
        s_end = solver_s[-1]
        idx_start = int(round((s_start % self.track_length) / wpnt_dist)) % num_wpnts

        n_cover = int(round((s_end - s_start) / wpnt_dist))
        n_cover = min(n_cover, num_wpnts)  # 트랙 한 바퀴 초과 방지

        for i in range(n_cover):
            idx = (idx_start + i) % num_wpnts
            s_target = s_start + i * wpnt_dist  # unwrapped s

            # Solver 출력에서 보간
            V_interp = float(np.interp(s_target, solver_s, raceline['V']))
            n_interp = float(np.interp(s_target, solver_s, raceline['n']))
            chi_interp = float(np.interp(s_target, solver_s, raceline['chi']))
            ax_interp = float(np.interp(s_target, solver_s, raceline['ax']))
            ay_interp = float(np.interp(s_target, solver_s, raceline['ay']))
            x_interp = float(np.interp(s_target, solver_s, raceline['x']))
            y_interp = float(np.interp(s_target, solver_s, raceline['y']))
            z_interp = float(np.interp(s_target, solver_s, raceline['z']))

            s_mod = s_target % self.track_length
            theta_track = float(self.track_handler.theta_interpolator(s_mod))

            w = Wpnt()
            w.id = idx
            w.s_m = wpnts[idx].s_m
            w.d_m = n_interp
            w.x_m = x_interp
            w.y_m = y_interp
            w.z_m = z_interp
            w.psi_rad = self._normalize_angle(theta_track + chi_interp)
            w.kappa_radpm = ay_interp / max(V_interp ** 2, 0.01)
            w.vx_mps = V_interp
            w.ax_mps2 = ax_interp

            # Track bounds adjusted by lateral offset
            w_tr_right = float(np.interp(
                s_mod, self.track_handler.s, self.track_handler.w_tr_right,
                period=self.track_length))
            w_tr_left = float(np.interp(
                s_mod, self.track_handler.s, self.track_handler.w_tr_left,
                period=self.track_length))
            w.d_right = abs(w_tr_right) - n_interp
            w.d_left = w_tr_left - n_interp

            # mu from track
            w.mu_rad = float(self.track_handler.mu_interpolator(s_mod))

            wpnts[idx] = w

        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.wpnts = wpnts
        return out

    def _raceline_to_online_wpntarray(self, raceline):
        """### IY : solver 출력 포인트를 그대로 WpntArray로 변환 (RViz 시각화용)"""
        wpnts = []
        cartesian = self.track_handler.sn2cartesian(
            raceline['s'] % self.track_length, raceline['n'])
        for i in range(len(raceline['s'])):
            w = Wpnt()
            w.id = i
            w.s_m = float(raceline['s'][i] % self.track_length)
            w.d_m = float(raceline['n'][i])
            w.x_m = float(cartesian[i, 0])
            w.y_m = float(cartesian[i, 1])
            w.z_m = float(cartesian[i, 2])
            s_mod = float(raceline['s'][i] % self.track_length)
            theta_track = float(self.track_handler.theta_interpolator(s_mod))
            w.psi_rad = self._normalize_angle(theta_track + float(raceline['chi'][i]))
            w.vx_mps = float(raceline['V'][i])
            w.ax_mps2 = float(raceline['ax'][i])
            w.kappa_radpm = float(raceline['ay'][i]) / max(float(raceline['V'][i]) ** 2, 0.01)
            wpnts.append(w)
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.wpnts = wpnts
        return out

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

    def _get_accelerations(self):
        """ax, ay 반환. sim이면 prev_solution/0, 실차면 IMU"""
        if not self.is_sim:
            return self.cur_imu_ax, self.cur_imu_ay

        # Sim: use previous solution if available
        if self.prev_solution is not None:
            return float(self.prev_solution['ax'][0]), float(self.prev_solution['ay'][0])
        return 0.0, 0.0

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
