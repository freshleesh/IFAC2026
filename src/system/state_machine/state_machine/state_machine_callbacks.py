"""Callback mixin — state_machine 노드의 ROS 콜백 24개 모음.

이 mixin이 사용하는 외부 의존:
    - 본체의 헬퍼/유틸 메서드: `update_velocity`, `ot_splinification`, `_find_nearest_ot_s`
    - 본체의 인스턴스 attribute (광범위): cur_*, gb_*, smart_*, sectors_params 등

정적 sector obstacle filtering 관련 상수 (ENABLE_STATIC_SECTOR_FILTERING / HORIZON_FOR_TTL)
는 obstacle_perception_cb 의 일부라 이 모듈에 함께 둔다. 메인의 `_check_getting_closer`
wrapper 도 같은 정책을 공유하기 위해 여기서 import 한다.
"""
from typing import Any
import numpy as np


from f110_msgs.msg import OTWpntArray, WpntArray
from nav_msgs.msg import Odometry

from state_machine.states_types import StateType


# 정적 sector obstacle filtering 정책 — obstacle_perception_cb + 메인의 _check_getting_closer 공유
ENABLE_STATIC_SECTOR_FILTERING = False
HORIZON_FOR_TTL = 3.0


class CallbackMixin:
    """state_machine 노드의 모든 ROS Subscriber 콜백 모음."""

    def save_start_traj_cb(self, msg):
        if len(self.cur_start_wpnts_candidate.wpnts) != 0:
            self.update_velocity(
                self.cur_start_wpnts_candidate, self.cur_start_wpnts.vel_planner_safety_factor
            )
            self.cur_start_wpnts.initialize_traj(self.cur_start_wpnts_candidate)
            self.cur_state = StateType.START

    def vesc_state_cb(self, data):
        """vesc state callback, reads the voltage"""
        self.cur_volt = data.state.voltage_input

    def frenet_planner_cb(self, data: WpntArray):
        """frenet planner waypoints"""
        self.frenet_wpnts = data

    def recovery_wpnts_cb(self, data: WpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor)
        self.recovery_wpnts = data

    def avoidance_cb(self, data: OTWpntArray):
        """splini waypoints"""
        # ROS2: update_velocity throw 시에도 self.avoidance_wpnts 보장 — 순서 변경
        self.avoidance_wpnts = data
        if len(data.wpnts) != 0:
            try:
                self.update_velocity(data, self.cur_avoidance_wpnts.vel_planner_safety_factor)
            except Exception as _e:
                self.get_logger().warn(f"[avoidance_cb] update_velocity raised: {_e}")

    def static_avoidance_cb(self, data: OTWpntArray):
        """static splini waypoints"""
        if not hasattr(self, "_static_cb_count"):
            self._static_cb_count = 0
        self._static_cb_count += 1
        if self._static_cb_count % 50 == 1:
            self.get_logger().info(f"[static_avoidance_cb] #{self._static_cb_count} len={len(data.wpnts)}")
        self.static_avoidance_wpnts = data
        if len(data.wpnts) != 0:
            try:
                self.update_velocity(data, self.cur_static_avoidance_wpnts.vel_planner_safety_factor)
            except Exception as _e:
                self.get_logger().warn(f"[static_avoidance_cb] update_velocity raised: {_e}")

    def smart_static_avoidance_cb(self, data: OTWpntArray):
        """Smart static avoidance waypoints from GB optimizer fixed path.

        Only update if timestamp is newer than current — prevents Smart node's old messages
        from overwriting global_velocity_planner's updates.
        """
        if (self.smart_static_wpnts is None or
            data.header.stamp > self.smart_static_wpnts.header.stamp):
            self.smart_static_wpnts = data
            self.cur_smart_static_avoidance_wpnts.initialize_traj(data)
        else:
            return  # 이미 최신이면 무시

        # 첫 메시지 받으면 track_length / 평균 wpnt 간격 초기화
        # OTWpntArray는 s_m 을 직접 갖고 있어 FrenetConverter 불필요.
        if len(data.wpnts) > 0 and self.smart_track_length is None:
            self.smart_track_length = data.wpnts[-1].s_m

            if len(data.wpnts) >= 2:
                s_diffs = [
                    data.wpnts[i].s_m - data.wpnts[i - 1].s_m
                    for i in range(1, len(data.wpnts))
                ]
                self.smart_wpnt_dist = np.mean(s_diffs)
            else:
                self.smart_wpnt_dist = self.smart_track_length

            self.get_logger().info(
                f"[{self.name}] Smart Static path initialized: "
                f"track_length={self.smart_track_length:.2f}m, "
                f"wpnt_dist={self.smart_wpnt_dist:.3f}m, "
                f"num_wpnts={len(data.wpnts)}"
            )

    def smart_static_active_cb(self, data):
        """Flag from spliner: is smart static mode currently active?"""
        self.smart_static_active = data.data

    def start_wpnts_cb(self, data: OTWpntArray):
        """start trajectory candidate (실제 활성화는 save_start_traj_cb 에서)"""
        if len(data.wpnts) != 0:
            self.cur_start_wpnts_candidate = data

    def overtake_cb(self, data):
        """Pre-computed 오버테이킹 라인 (graph_based 등 비-spliner planner 용)."""
        self.overtake_wpnts = data.wpnts
        self.num_ot_points = len(self.overtake_wpnts)

        # 새 spline 들어왔을 때만 OT spline 재계산
        if self.recompute_ot_spline and self.num_ot_points != 0:
            self.ot_splinification()
            self.recompute_ot_spline = False

    def glb_wpnts_cb(self, data: WpntArray):
        """Global waypoints (velocity scaler 출력)."""
        data.wpnts = data.wpnts[:-1]  # 마지막 wpnt 제외 (첫 wpnt 와 동일)
        self.gb_wpnts = data
        self.num_glb_wpnts = len(data.wpnts)

        self.n_loc_wpnts = min(self.n_loc_wpnts, int(self.num_glb_wpnts / 2))

        self.max_s = data.wpnts[-1].s_m
        self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m
        self.waypoints_dist = self.wpnt_dist
        self.gb_max_idx = data.wpnts[-1].id
        if self.ot_planner == "graph_based":
            self.gb_wpnts_arr = np.array([
                [w.s_m, w.d_m, w.x_m, w.y_m, w.d_right, w.d_left, w.psi_rad,
                 w.kappa_radpm, w.vx_mps, w.ax_mps2] for w in data.wpnts
            ])

    def glb_wpnts_og_cb(self, data):
        """OG global waypoints (100% 속도) — max_speed 1회만 측정."""
        if self.max_speed == -1:
            self.max_speed = max([wpnt.vx_mps for wpnt in data.wpnts])

    def graphbased_wpts_cb(self, data):
        arr = np.asarray(data.data)
        self.graph_based_wpts = arr.reshape(data.layout.dim[0].size, data.layout.dim[1].size)
        self.graph_based_action = data.layout.dim[0].label

    def scan_cb(self, data):
        """sim 라이다 visibility 필터용 — /scan 캐시. sim 모드에서만 sub됨."""
        self._latest_scan_ranges = data.ranges
        self._latest_scan_angle_min = data.angle_min
        self._latest_scan_angle_inc = data.angle_increment
        self._latest_scan_range_max = data.range_max

    def _obstacle_visible_to_lidar(self, obs) -> bool:
        """sim 모드: 장애물이 차의 라이다 시야에 있는지 (벽 너머가 아닌지) 검사.

        obstacle 의 cartesian (x_m, y_m) 을 차 기준으로 변환 → 해당 각도의 라이다 ray
        끝점 거리와 비교. 장애물이 ray 끝점보다 가까우면 (or 비슷하면) → 차에서 보임.
        벽 너머에 있으면 ray 가 wall 까지만 닿아서 (장애물 거리 > ray 거리) → reject.

        scan 못 받았거나 current_position 없으면 통과 (안전 default — 필터 비활성과 같음).
        """
        import math
        if (self._latest_scan_ranges is None or
                not getattr(self, "current_position", None) or
                len(self.current_position) < 3):
            return True

        car_x, car_y, car_yaw = self.current_position[0], self.current_position[1], self.current_position[2]
        dx = obs.x_m - car_x
        dy = obs.y_m - car_y
        dist = math.sqrt(dx * dx + dy * dy)

        # 라이다 사거리 밖 → reject
        if self._latest_scan_range_max and dist > self._latest_scan_range_max:
            return False

        angle_world = math.atan2(dy, dx)
        angle_rel = angle_world - car_yaw
        # normalize to [-pi, pi]
        while angle_rel > math.pi:
            angle_rel -= 2 * math.pi
        while angle_rel < -math.pi:
            angle_rel += 2 * math.pi

        idx = int(round((angle_rel - self._latest_scan_angle_min) / self._latest_scan_angle_inc))
        n = len(self._latest_scan_ranges)
        if idx < 0 or idx >= n:
            # FOV 밖 → reject (라이다가 못 봄)
            return False

        # 주변 3개 ray 중 최대 거리 — angle 정확도 / noise 마진
        lo = max(0, idx - 1)
        hi = min(n, idx + 2)
        ranges_around = [r for r in self._latest_scan_ranges[lo:hi] if r > 0.0 and not math.isinf(r)]
        if not ranges_around:
            return False
        scan_dist = max(ranges_around)

        # 장애물이 ray 끝(=벽)까지보다 가깝거나 비슷하면 → 시야 안. margin 0.3m.
        return dist <= scan_dist + 0.3

    def obstacle_perception_cb(self, data):
        """장애물 감지 메시지 처리. 정적 sector 장애물에 대한 stricter filtering 포함.

        정적 sector 안의 정적 장애물은 너무 일찍 TRAILING 이 걸리지 않도록 작은 horizon
        (HORIZON_FOR_TTL) 만 사용. ENABLE_STATIC_SECTOR_FILTERING 으로 on/off.

        sim 모드: 라이다 visibility 추가 검사 — 클릭으로 들어온 장애물이 벽 너머이면 무시.
        """
        if self.timetrials_only:
            return

        self.obstacles_perception = data.obstacles[:]
        self.obstacles = data.obstacles

        obstacles_in_interest = []
        for obs in data.obstacles:
            # sim 모드: 라이다 시야 검사 — 벽 너머 클릭으로 들어온 가짜 장애물 무시
            if self.sim and not self._obstacle_visible_to_lidar(obs):
                continue
            gap = (obs.s_start - self.cur_s) % self.track_length
            is_static_in_static_sector = obs.in_static_obs_sector and obs.is_static
            if is_static_in_static_sector and ENABLE_STATIC_SECTOR_FILTERING:
                horizon = HORIZON_FOR_TTL
            else:
                horizon = self.interest_horizon_m
            if gap < horizon:
                obstacles_in_interest.append(obs)
        self.obstacles_in_interest = obstacles_in_interest

    def ego_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.ego_prediction = data.predictions
        else:
            self.ego_prediction = []

    def obstacle_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.obstacles_prediction_id = data.id
            self.obstacles_prediction = data.predictions
        else:
            self.obstacles_prediction = []

    def frenet_pose_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        if self.num_ot_points != 0:
            self.cur_id_ot = int(self._find_nearest_ot_s())

    def odom_cb(self, data):
        """/car_state/odom — Cartesian (x, y, theta) 위치.

        ROS2 포팅: tf_transformations.euler_from_quaternion → 직접 atan2 계산.
        """
        import math
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        q = data.pose.pose.orientation
        # yaw (z 축 회전, ZYX 오일러) — euler_from_quaternion 의 [2] 와 동일
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.current_position = [x, y, theta]

    def dyn_param_cb(self, params: Any):
        """rqt 에서 state_machine 파라미터 변경 시 갱신."""
        self.lateral_width_gb_m = self._get_param_or_default("dyn_statemachine/lateral_width_gb_m", 0.75)
        self.lateral_width_ot_m = self._get_param_or_default("dyn_statemachine/lateral_width_ot_m", 0.3)
        if self.ot_planner == "spliner":
            self.splini_ttl = self._get_param_or_default("dyn_statemachine/splini_ttl")
        else:
            self.splini_ttl = self._get_param_or_default("dyn_statemachine/pred_splini_ttl")
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
        self.splini_hyst_timer_sec = self._get_param_or_default("dyn_statemachine/splini_hyst_timer_sec", 0.75)
        self.emergency_break_horizon = self._get_param_or_default("dyn_statemachine/emergency_break_horizon", 1.1)
        self.ftg_speed_mps = self._get_param_or_default("dyn_statemachine/ftg_speed_mps", 1.0)
        self.ftg_timer_sec = self._get_param_or_default("dyn_statemachine/ftg_timer_sec", 3.0)

        self.overtaking_ttl_sec = self._get_param_or_default("dyn_statemachine/overtaking_ttl_sec", 3.0)
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        self.ftg_disabled = not self._get_param_or_default("dyn_statemachine/ftg_active", False)
        self.force_gbtrack_state = self._get_param_or_default("dyn_statemachine/force_GBTRACK", False)
        self.use_force_trailing = self._get_param_or_default("dyn_statemachine/use_force_trailing", False)

        if self.force_gbtrack_state:
            self.get_logger().warning(f"[{self.name}] GBTRACK state force activated!!!")

        self.get_logger().debug(
            "[{}] Received new parameters for state machine: lateral_width_gb_m: {}, "
            "lateral_width_ot_m: {}, splini_ttl: {}, splini_hyst_timer_sec: {}, ftg_speed_mps: {}, "
            "ftg_timer_sec: {}, GBTRACK_force: {}".format(
                self.name,
                self.lateral_width_gb_m,
                self.lateral_width_ot_m,
                self.splini_ttl,
                self.splini_hyst_timer_sec,
                self.ftg_speed_mps,
                self.ftg_timer_sec,
                self.force_gbtrack_state,
            )
        )

    def sector_dyn_param_cb(self, params: Any):
        """rqt 에서 ftg-only sector 변경 시 갱신."""
        self.only_ftg_zones = []
        for i in range(self.n_sectors):
            self.sectors_params[f"Sector{i}"]["only_FTG"] = params.bools[2 * i + 1].value
            if self.sectors_params[f"Sector{i}"]["only_FTG"]:
                self.only_ftg_zones.append([
                    self.sectors_params[f"Sector{i}"]["start"],
                    self.sectors_params[f"Sector{i}"]["end"],
                ])

    def ot_dyn_param_cb(self, params: Any):
        """rqt 에서 overtaking sector 변경 시 갱신."""
        self.overtake_zones = []
        try:
            for i in range(self.n_ot_sectors):
                self.ot_sectors_params[f"Overtaking_sector{i}"]["ot_flag"] = params.bools[i + 1].value
                if self.ot_sectors_params[f"Overtaking_sector{i}"]["ot_flag"]:
                    self.overtake_zones.append([
                        self.ot_sectors_params[f"Overtaking_sector{i}"]["start"],
                        self.ot_sectors_params[f"Overtaking_sector{i}"]["end"] + 1,
                    ])
        except IndexError as e:
            raise IndexError(
                f"[State Machine] Error in overtaking sector numbers. \n"
                f"Try switching map with the script in stack_master/scripts and re-source in every terminal. \n"
                f"Error thrown: {e}"
            )

        self.ot_begin_margin = params.doubles[2].value
        self.get_logger().warning(
            f"[{self.name}] Using OT beginning {self.ot_begin_margin}[m] "
            f"from param: {params.doubles[2].name}"
        )
        # 기존 OT 가 있으면 다음 overtake_cb 에서 spline 재계산
        self.recompute_ot_spline = True

    def merger_cb(self, data):
        self.merger = data.data

    def force_trailing_cb(self, data):
        if self.use_force_trailing:
            self.force_trailing = data.data
        else:
            self.force_trailing = False

    def fail_trailing_cb(self, data):
        self.fail_trailing = data.data
