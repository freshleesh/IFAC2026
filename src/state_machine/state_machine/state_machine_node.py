#!/usr/bin/env python3
import logging
import time
from typing import Any, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from f110_msgs.msg import WpntArray, Wpnt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

# trajectory_planning_helpers 와 vel_planner_25d 는 ROS1 환경 (docker container) 에만 있다.
# ROS2 ws 검증 환경에서는 conditional import — 사용 시 명시 ImportError.
try:
    import trajectory_planning_helpers as tph
except ImportError:
    tph = None
try:
    from vel_planner_25d.vel_planner import calc_vel_profile
except ImportError:
    calc_vel_profile = None

_logger = logging.getLogger(__name__)

# Pure-function path checker (단계적으로 분리 중인 collision check 로직)
from state_machine.path_checker import (
    EgoFrenetState,
    FrenetCheckParams,
    GettingCloserParams,
    OnSplineParams,
    OvertakingModeChecks,
    StaticOvertakingChecks,
    TrajFreshnessParams,
    check_free_frenet,
    is_getting_closer,
    is_in_overtaking_zone,
    is_on_spline,
    is_traj_msg_fresh,
    should_engage_overtaking,
    should_engage_static_overtaking,
)

# Visualization / init / callback 메서드 묶음 (mixin)
from state_machine.state_machine_visualization import VisualizationMixin
from state_machine.state_machine_init import InitMixin
from state_machine.state_machine_callbacks import (
    CallbackMixin,
    ENABLE_STATIC_SECTOR_FILTERING,
    HORIZON_FOR_TTL,
)

# ===== HJ ADDED: Debug logging helper for state_machine_node =====
DEBUG_LOGGING_ENABLED = False  # Set to False to disable all debug logging
_debug_log_cache = {}

# NOTE: ENABLE_STATIC_SECTOR_FILTERING / HORIZON_FOR_TTL 은 state_machine_callbacks 로 이동.
# 메인은 callback 모듈에서 import 한다 (위 import 블록 참조).

MAX_VEL_RIGHT_BEFORE_STATIC_OT = 5.0

# ===== HJ ADDED: Global flag for state transition debug logging =====
DEBUG_STATE_TRANSITION = False  # Set to True to see state transition warnings
# ===== HJ ADDED END =====


# NOTE: STATE_COLORS / _STATE_COLOR_DEFAULT 는 state_machine_visualization 으로 이동.

def debug_log_on_change(tag, **kwargs):
    """이전 호출과 kwargs 값이 다를 때만 로그.

    DEBUG_LOGGING_ENABLED 가 False 면 no-op. 캐시는 모듈 전역 dict (`_debug_log_cache`).
    item 할당만 하므로 `global` 선언 필요 없음.
    """
    if not DEBUG_LOGGING_ENABLED:
        return

    cache_key = tag
    prev_values = _debug_log_cache.get(cache_key, None)
    if prev_values != kwargs:
        msg_parts = [f"{k}={v}" for k, v in kwargs.items()]
        # 모듈-레벨 함수라 self 없음 — 표준 logging 사용
        _logger.warning(f"[DEBUG {tag}] " + ", ".join(msg_parts))
        _debug_log_cache[cache_key] = kwargs.copy()


from state_machine.states_types import StateType
from state_machine.waypoint_data import WaypointData


class StateMachine(Node, InitMixin, VisualizationMixin, CallbackMixin):
    """F1TENTH 행동 FSM. ROS2 Jazzy 포팅 — Node 상속 + 3 mixin.

    원본 ROS1 (HJ): 3d_state_machine_node.py.

    State transitions and state behaviors → state_transitions.py, states.py.
    """

    def __init__(self, name: str) -> None:
        # 1) ROS2 Node 초기화 (rospy.init_node 대체).
        #    - allow_undeclared_parameters=True: rospy 의 has_param/get_param 호환
        #    - automatically_declare_parameters_from_overrides=True: launch 에서 -p 로
        #      준 모든 파라미터를 자동 declare (rospy.get_param 호환)
        Node.__init__(
            self, name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name

        # 2) rosparam / vehicle dyn / vel_planner / state attr 셋업 (mixin helper 들)
        self._load_rosparams()
        self._load_vehicle_dynamics()
        self._load_vel_planner_params()
        self._init_state_attributes()

        # 3) ROS2 IO 셋업
        self._setup_ros_subscribers()
        self._setup_ros_publishers()

        # 4) Smart Static helper — publisher/state attr 모두 만들어진 뒤
        from state_machine.state_helper_for_smart import SmartStaticChecker
        self.smart_helper = SmartStaticChecker(self)
        self.get_logger().info(
            f"[{self.name}] Smart Static helper initialized for Fixed Frenet transitions"
        )

        # 5) ROS2 native parameter callback — 원본의 dynamic_reconfigure 서버 대체.
        #    dyn_statemachine/* 변경 시 self.* 동기화. ros2 param set / rqt_reconfigure 호환.
        self.add_on_set_parameters_callback(self._on_dyn_param_change)

        # 6) 메인 loop — ROS2 timer 패턴 (rospy.Rate 대체)
        self._loop_period_sec = 1.0 / self.rate_hz
        self.create_timer(self._loop_period_sec, self.loop)
        self.get_logger().info(
            f"[{self.name}] StateMachine ready, loop @ {self.rate_hz} Hz"
        )

    # =========================================================================
    # ROS2 native parameter callback (dynamic_reconfigure 대체) — C-5
    # =========================================================================

    def _on_dyn_param_change(self, params):
        """ros2 param set / rqt_reconfigure 로 파라미터 변경 시 호출.

        - dyn_statemachine/* prefix 만 처리 (self.* 동기화)
        - 다른 prefix (dyn_sector_tuner/*, global_velplanner_3d/*) 는 무시:
          sector tuner / vel_planner_25d 패키지 미포팅이라 호출 자체 안 일어남.
        - 받은 params 의 value 직접 사용 (callback 은 set 전 hook — get_parameter
          하면 옛 값. 받은 params 가 truth).
        """
        from rcl_interfaces.msg import SetParametersResult

        for p in params:
            n = p.name
            v = p.value
            if n == "dyn_statemachine/lateral_width_gb_m":
                self.lateral_width_gb_m = float(v)
            elif n == "dyn_statemachine/lateral_width_ot_m":
                self.lateral_width_ot_m = float(v)
            elif n == "dyn_statemachine/splini_ttl" and self.ot_planner == "spliner":
                self.splini_ttl = float(v)
                self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
            elif n == "dyn_statemachine/pred_splini_ttl" and self.ot_planner != "spliner":
                self.splini_ttl = float(v)
                self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
            elif n == "dyn_statemachine/splini_hyst_timer_sec":
                self.splini_hyst_timer_sec = float(v)
            elif n == "dyn_statemachine/emergency_break_horizon":
                self.emergency_break_horizon = float(v)
            elif n == "dyn_statemachine/ftg_speed_mps":
                self.ftg_speed_mps = float(v)
            elif n == "dyn_statemachine/ftg_timer_sec":
                self.ftg_timer_sec = float(v)
            elif n == "dyn_statemachine/overtaking_ttl_sec":
                self.overtaking_ttl_sec = float(v)
                self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)
            elif n == "dyn_statemachine/ftg_active":
                self.ftg_disabled = not bool(v)
            elif n == "dyn_statemachine/force_GBTRACK":
                self.force_gbtrack_state = bool(v)
                if self.force_gbtrack_state:
                    self.get_logger().warning(
                        f"[{self.name}] GBTRACK state force activated!!!"
                    )
            elif n == "dyn_statemachine/use_force_trailing":
                self.use_force_trailing = bool(v)
        return SetParametersResult(successful=True)

    # =========================================================================
    # Properties — Smart helper / GB 좌표계 분기를 한 곳에 모아 패턴 반복 제거.
    # SmartStaticChecker 는 부모(StateMachine)에 self.parent 를 추가로 가지므로
    # `is_smart_helper` 가 두 클래스에서 다른 값을 반환한다.
    # =========================================================================

    @property
    def is_smart_helper(self) -> bool:
        """SmartStaticChecker 인스턴스인지 (parent attribute 보유 여부)."""
        return hasattr(self, 'parent')

    @property
    def role_tag(self) -> str:
        """디버그 로그용 'HELPER'(SmartStaticChecker) / 'PARENT'(StateMachine) 태그."""
        return "HELPER" if self.is_smart_helper else "PARENT"

    @property
    def gb_cur_s(self) -> float:
        """GB Frenet cur_s. SmartStaticChecker 일 때는 parent 의 GB cur_s 사용
        (self.cur_s 는 Fixed Frenet 으로 override 되어 있으므로)."""
        return self.parent.cur_s if self.is_smart_helper else self.cur_s

    @property
    def gb_waypoints_dist(self) -> float:
        """GB raceline 의 waypoint 간격. SmartStaticChecker 일 때는 parent 의 것."""
        return self.parent.waypoints_dist if self.is_smart_helper else self.waypoints_dist

    @property
    def active_helper(self):
        """현재 활성 모드의 데이터 source.

        Smart 모드(`smart_static_active=True`)면 `smart_helper` (Fixed Frenet 기반).
        그 외엔 `self` (GB Frenet 기반).

        주의: `smart_helper.cur_gb_wpnts.closest_target` 와 `self.gb_closest_target` 는
        다른 attribute 라 모든 분기를 통합하지는 못한다 (cur_*_wpnts.closest_target / closest_gap
        만 같은 인터페이스를 공유).
        """
        return self.smart_helper if self.smart_static_active else self

    @property
    def current_mode_tag(self) -> str:
        """디버그 로그용 현재 모드 — 'SMART' / 'GB'."""
        return "SMART" if self.smart_static_active else "GB"

    # NOTE: __init__ 헬퍼 6개 (_load_rosparams, _load_vehicle_dynamics,
    # _load_vel_planner_params, _init_state_attributes, _setup_ros_subscribers,
    # _setup_ros_publishers) 는 InitMixin 으로 이동.

    def on_shutdown(self):
        self.get_logger().info(f"[{self.name}] Shutting down state machine")

    # NOTE: 24개 ROS Subscriber callback 은 CallbackMixin 으로 이동.
    # ENABLE_STATIC_SECTOR_FILTERING / HORIZON_FOR_TTL 도 함께 이동.

    ######################################
    # ATTRIBUTES/CONDITIONS CALCULATIONS #
    ######################################
    """ For consistency, all conditions should be calculated in this section, and should all have the same signature:
    def _check_condition(self) -> bool:
    ...
    """

    # ===== ORIGINAL FUNCTION (before HJ modification) =====
    # def _check_only_ftg_zone(self) -> bool:
    #     ftg_only = False
    #     # check if the car is in a ftg only zone, but only if there is an only ftg zone
    #     if len(self.only_ftg_zones) != 0:
    #         for sector in self.only_ftg_zones:
    #             if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
    #                 ftg_only = True
    #                 # self.get_logger().warning(f"[{self.name}] IN FTG ONLY ZONE")
    #                 break  # cannot be in two ftg zones
    #     return ftg_only
    # ===== ORIGINAL FUNCTION END =====

    # ===== HJ MODIFIED: Always use GB Frenet coordinates for zone checks =====
    def _check_only_ftg_zone(self) -> bool:
        """Check if in FTG-only zone using GB raceline coordinates

        Zones are defined using GB raceline waypoint indices.
        When called from SmartStaticChecker, uses parent's GB cur_s instead of Fixed cur_s.
        """
        ftg_only = False
        # zones 는 GB raceline waypoint 인덱스 기준 — gb_cur_s/gb_waypoints_dist property 사용
        if len(self.only_ftg_zones) != 0:
            for sector in self.only_ftg_zones:
                if sector[0] <= self.gb_cur_s / self.gb_waypoints_dist <= sector[1]:
                    ftg_only = True
                    break  # cannot be in two ftg zones
        return ftg_only
    # ===== HJ MODIFIED END =====

    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        if threshold_m is None:
            return np.abs(self.cur_d) < self.gb_ego_width_m  # [m]
        else:
            return np.abs(self.cur_d) < threshold_m  # [m]

    def _get_adaptive_close_threshold(self) -> float:
        """Speed-adaptive close-to-raceline threshold.
        - At launch / struggling (v_cur << v_target): tight → recovery triggers easily
        - Normal driving (v_cur ≈ v_target): loose → recovery rarely triggers
        """
        tight = self._get_param_or_default("state_machine/close_threshold_tight", 0.2)
        loose = self._get_param_or_default("state_machine/close_threshold_loose", 0.5)
        speed_ratio_thres = self._get_param_or_default("state_machine/close_speed_ratio_thres", 0.5)
        try:
            cur_s_idx = int(self.cur_s / self.waypoints_dist) % self.num_glb_wpnts
            v_target = self.cur_gb_wpnts.list[cur_s_idx].vx_mps
            if v_target < 0.05:
                return loose
            if self.cur_vs < v_target * speed_ratio_thres:
                return tight
            return loose
        except Exception:
            return loose

    def _check_close_to_raceline_heading(self, threshold_deg=None) -> bool:

        cloest_wpnt_idx = int(self.cur_s / self.waypoints_dist)%self.num_glb_wpnts
        cloest_wpnt_psi = self.cur_gb_wpnts.list[cloest_wpnt_idx].psi_rad
        if threshold_deg is None:
            return np.abs(self.current_position[2] - cloest_wpnt_psi) < np.deg2rad(20)
        else:
            return np.abs(self.cur_d) < np.deg2rad(threshold_deg)

    # ===== ORIGINAL FUNCTION (before HJ modification) =====
    # def _check_ot_sector(self) -> bool:
    #     # self.ot_section_check_pub.publish(True)
    #     # return True
    #
    #     for sector in self.overtake_zones:
    #         if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
    #             # self.get_logger().info(f"[{self.name}] In overtaking sector!")
    #             self.ot_section_check_pub.publish(True)
    #             return True
    #     self.ot_section_check_pub.publish(False)
    #
    #     return False
    # ===== ORIGINAL FUNCTION END =====

    # ===== HJ MODIFIED: Always use GB Frenet coordinates for zone checks =====
    def _check_ot_sector(self) -> bool:
        """OT zone 진입 여부.

        결정 로직(zone matching)은 path_checker.is_in_overtaking_zone 으로 위임.
        zone 은 GB raceline 기준이라 gb_cur_s/gb_waypoints_dist property 사용.
        """
        in_zone = is_in_overtaking_zone(
            s_m=self.gb_cur_s,
            waypoints_dist=self.gb_waypoints_dist,
            zones=self.overtake_zones,
        )
        from std_msgs.msg import Bool as _Bool
        self.ot_section_check_pub.publish(_Bool(data=bool(in_zone)))
        return in_zone
    # ===== HJ MODIFIED END =====

    # ===== HJ COMMENTED: Original version without distance check =====
    # def _check_getting_closer(self, threshold_m=3.0) -> bool:
    #     obs = None
    #     # return True
    #     if (
    #         len(self.obstacles_in_interest) != 0
    #         and self.cur_vs - self.obstacles_in_interest[0].vs > -0.5
    #     ):
    #         return True
    #     else:
    #         return False
    # ===== HJ COMMENTED END =====

    def _check_getting_closer(self, threshold_m=7.0) -> bool:
        """관심 장애물(첫 번째)이 자차에 가까워지고 있는지 판정.

        결정 로직은 path_checker.is_getting_closer 로 위임. 정적 sector 필터링 동작은
        전역 ENABLE_STATIC_SECTOR_FILTERING / HORIZON_FOR_TTL 그대로 유지한다.

        Args:
            threshold_m: 인터페이스 보존용 (현재 미사용, 호출자들이 명시적으로 넘기던 값).
        """
        first_obstacle = self.obstacles_in_interest[0] if self.obstacles_in_interest else None
        return is_getting_closer(
            cur_s=self.cur_s,
            cur_vs=self.cur_vs,
            first_obstacle=first_obstacle,
            params=GettingCloserParams(
                horizon_for_ttl=HORIZON_FOR_TTL,
                static_sector_filtering=ENABLE_STATIC_SECTOR_FILTERING,
                track_length=self.track_length,
            ),
        )


    def _check_enemy_in_front(self) -> bool:
        # If we are in time trial only mode -> return free overtake i.e. GB_FREE True
        horizon = self.gb_horizon_m  # Horizon in front of cur_s [m]
        for obs in self.obstacles:
            # if not obs.is_static:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < horizon:
                return True
        return False


    # ===== HJ ADDED: Helper to get base state and trajectory =====
    def _get_base_state_and_trajectory(self) -> Tuple[StateType, StateType]:
        """Returns (SMART_STATIC, SMART_STATIC) if active, otherwise (GB_TRACK, GB_TRACK)"""
        if self.smart_static_active and len(self.cur_smart_static_avoidance_wpnts.list) > 0:
            return StateType.SMART_STATIC, StateType.SMART_STATIC
        return StateType.GB_TRACK, StateType.GB_TRACK
    # ===== HJ ADDED END =====

    ##################################################################
    def _check_latest_wpnts(self, src_wpnts, wpnts_data: WaypointData):
        """수신된 trajectory 가 fresh 한지 + on_spline 한지 종합 판정.

        timestamp/freshness 결정 로직은 path_checker.is_traj_msg_fresh 로 위임.
        initialize_traj (wpnts_data 내부 array/list/is_init 갱신)와 _check_on_spline 호출은
        wrapper에 남는다 (side effect / 다른 sub-check 합성).
        """
        is_smart_static = (wpnts_data.name == 'static_avoidance_planner')

        is_fresh = is_traj_msg_fresh(
            src_msg=src_wpnts,
            now_sec=(self.get_clock().now().nanoseconds * 1e-9),
            params=TrajFreshnessParams(
                is_smart_static=is_smart_static,
                latest_threshold_sec=wpnts_data.latest_threshold,
            ),
        )

        if not is_fresh:
            # Smart Static에서 stamp 미초기화 케이스만 별도 디버그 로그 (원본 동작 유지)
            if is_smart_static and src_wpnts is not None and len(src_wpnts.wpnts) > 0:
                self.get_logger().warning("[_check_latest_wpnts] Smart Static waypoints timestamp not initialized")
            return False

        # Side effect: wpnts_data 내부 array/list/is_init 갱신
        wpnts_data.initialize_traj(src_wpnts)
        on_spline = self._check_on_spline(wpnts_data)

        if not on_spline and is_smart_static:
            self.get_logger().warning("[_check_latest_wpnts] Smart Static _check_on_spline FAILED! "
                "(timestamp check was OK)")

        return on_spline


    def _check_ftg(self) -> bool:
        # If we have been standing still for 3 seconds inside TRAILING -> FTG
        threshold = self.ftg_timer_sec * self.rate_hz
        if self.ftg_disabled:
            return False
        else:
            if (self.cur_state == StateType.TRAILING or self.cur_state == StateType.ATTACK) and self.cur_vs < self.ftg_speed_mps:
                self.ftg_counter += 1
                self.get_logger().warning(f"[{self.name}] FTG counter: {self.ftg_counter}/{threshold}")
            else:
                self.ftg_counter = 0

            if self.ftg_counter > threshold:
                return True
            else:
                return False

    # def _check_emergency_break(self) -> bool:
    #     emergency_break = False
    #     if self.ot_planner == "predictive_spliner":
    #         if not self.timetrials_only:
    #             obstacles = self.obstacles_perception.copy()
    #             if obstacles != []:
    #                 horizon = self.emergency_break_horizon # Horizon in front of cur_s [m]

    #                 for obs in obstacles:
    #                     # Only use opponent for emergency break
    #                     # Wrapping madness to check if infront
    #                     dist_to_obj = (obs.s_start - self.cur_s) % self.max_s
    #                     # Check if opponent is closer than emegerncy
    #                     if dist_to_obj < horizon:
                    
    #                         # Get estimated d from local waypoints
    #                         local_wpnt_idx = np.argmin(
    #                             np.array([abs(avoid_s.s_m - obs.s_center) for avoid_s in self.local_wpnts.wpnts])
    #                         )
    #                         ot_d = self.local_wpnts.wpnts[local_wpnt_idx].d_m
    #                         ot_obs_dist = ot_d - obs.d_center
    #                         if abs(ot_obs_dist) < self.emergency_break_d:
    #                             emergency_break = True
    #                             self.get_logger().warning("[State Machine] emergency break")
    #         else:
    #             emergency_break = False
    #         return emergency_break
    
    def _check_on_spline(self, wpnt_data) -> bool:
        """자차가 wpnt spline 위에 있는지 판정.

        결정 로직은 path_checker.is_on_spline 으로 위임. 디버그 로깅은 wrapper에 남는다.
        """
        result = is_on_spline(
            cur_s=self.cur_s,
            current_position_xy=np.asarray(self.current_position[:2]),
            waypoints=wpnt_data,
            params=OnSplineParams(
                track_length=self.track_length,
                front_horizon_thres_m=wpnt_data.on_spline_front_horizon_thres_m,
                min_dist_thres_m=wpnt_data.on_spline_min_dist_thres_m,
            ),
        )

        # ===== HJ ADDED: Debug logging for failed checks =====
        if not result and wpnt_data.is_init:
            gap = (wpnt_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = np.min(
                np.linalg.norm(wpnt_data.array[:, 0:2] - self.current_position[:2], axis=1)
            )
            self.get_logger().warning(f"[DEBUG {self.role_tag} _check_on_spline FAIL] planner={wpnt_data.name}, "
                f"gap={gap:.2f}m (need>{wpnt_data.on_spline_front_horizon_thres_m:.2f}): "
                f"{gap > wpnt_data.on_spline_front_horizon_thres_m}, "
                f"min_dist={min_dist:.3f}m (need<{wpnt_data.on_spline_min_dist_thres_m:.3f}): "
                f"{min_dist < wpnt_data.on_spline_min_dist_thres_m}")
        # ===== HJ ADDED END =====

        return result
    
    def _check_free_frenet(self, wpnts_data) -> bool:
        """path_checker.check_free_frenet 순수 함수로 위임.

        side effect (wpnts_data.closest_target / closest_gap 갱신) 만 wrapper에서 처리하여
        외부 호출자의 동작은 변하지 않게 유지한다.
        """
        result = check_free_frenet(
            ego=EgoFrenetState(s=self.cur_s, vs=self.cur_vs),
            waypoints=wpnts_data,
            obstacles=self.cur_obstacles_in_interest,
            obstacle_predictions=self.obstacles_prediction,
            obstacle_prediction_id=self.obstacles_prediction_id,
            params=FrenetCheckParams(
                max_s=self.max_s,
                veh_length=self.pars["veh_params"]["length"],
                ego_width=self.gb_ego_width_m,
            ),
        )
        wpnts_data.closest_target = result.closest_obstacle
        wpnts_data.closest_gap = result.closest_gap
        return result.is_free

    def _check_free_cartesian(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = None
        # Slightly different for spliner
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m
        
        obstacles = self.cur_obstacles_in_interest
        if wpnts_data.is_init:
            for obs in obstacles:
                obs_s = obs.s_center
                # 자차 기준 wrap-around gap
                gap = (obs_s - self.cur_s) % self.max_s

                if gap < max_horizon or min_horizon < (gap - self.max_s):
                    dists = np.linalg.norm(wpnts_data.array[:, 0:2] - np.array([obs.x_m, obs.y_m]), axis=1)
                    min_dist = np.min(dists)
                    free_dist = min_dist - obs.size / 2 - self.gb_ego_width_m / 2
                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)

                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                        self.get_logger().info(f"[{self.name}] RECOVERY_FREE False, obs dist to recovery lane: {min_dist} m")
        else:
            is_free = True
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free
        
    def _check_availability(self, wpnts, wpnts_data) -> bool:
        # ===== HJ MODIFIED: Check if wpnts is None (not published yet) =====
        if wpnts is None or wpnts_data.stamp is None:
            return False
        # ===== HJ MODIFIED END =====

        # self.get_logger().warning((self.get_clock().now().to_msg() - wpnts_data.stamp).to_sec())
        if (self.get_clock().now().nanoseconds * 1e-9 - (wpnts_data.stamp.sec + wpnts_data.stamp.nanosec * 1e-9)) > wpnts_data.killing_timer_sec:
            wpnts_data.is_init = False
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True
            else:
                return False

        if (self.get_clock().now().nanoseconds * 1e-9 - (wpnts_data.stamp.sec + wpnts_data.stamp.nanosec * 1e-9)) > wpnts_data.hyst_timer_sec:
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True


        if not self._check_on_spline(wpnts_data):
            if self._check_latest_wpnts(wpnts, wpnts_data):
                return True
            else:
                return False
            
        return True

        # else:
    
    def _check_sustainability(self, src_wpnts, wpnts_data) -> bool:
        if (
            self._check_availability(src_wpnts, wpnts_data)
            # self._check_on_spline()
            and self._check_free_frenet(wpnts_data)
            # and self.last_valid_avoidance_wpnts is not None
        ):
            return True

        return False
    
    def _check_overtaking_mode(self) -> bool:
        """동적 OT 모드 진입 여부 결정.

        결정 로직은 path_checker.should_engage_overtaking 으로 위임. 진입 시 wrapper에서
        명시적으로 self.static_overtaking_mode = False 세팅.
        """
        checks = OvertakingModeChecks(
            in_ot_sector=self._check_ot_sector(),
            is_getting_closer=self._check_getting_closer(threshold_m=10.0),
            wpnts_are_latest=self._check_latest_wpnts(self.avoidance_wpnts, self.cur_avoidance_wpnts),
            path_is_free=self._check_free_frenet(self.cur_avoidance_wpnts),
        )

        # 디버그 로그
        wpnts_info = "None"
        if self.avoidance_wpnts is not None:
            wpnts_info = f"exists(len={len(self.avoidance_wpnts.wpnts)})"
        debug_log_on_change(
            f"{self.role_tag}_check_OT",
            ot_sector=checks.in_ot_sector,
            closer=checks.is_getting_closer,
            latest=checks.wpnts_are_latest,
            free=checks.path_is_free,
            wpnts_avail=self.avoidance_wpnts is not None,
            wpnts=wpnts_info,
            num_obs=len(self.obstacles_in_interest)
        )

        if should_engage_overtaking(checks):
            self.static_overtaking_mode = False
            return True
        return False
        
    def _check_static_overtaking_mode(self) -> bool:
        """정적 OT 모드 진입 여부 결정.

        결정 로직은 path_checker.should_engage_static_overtaking 으로 위임. 진입 시 wrapper에서
        명시적으로 self.static_overtaking_mode = True 세팅.
        """
        checks = StaticOvertakingChecks(
            velocity_safe=self.cur_vs < MAX_VEL_RIGHT_BEFORE_STATIC_OT,
            is_getting_closer=self._check_getting_closer(threshold_m=7.0),
            wpnts_are_latest=self._check_latest_wpnts(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts),
            path_is_free=self._check_free_frenet(self.cur_static_avoidance_wpnts),
        )

        # 디버그 로그
        wpnts_info = "None"
        if self.static_avoidance_wpnts is not None:
            wpnts_info = f"exists(len={len(self.static_avoidance_wpnts.wpnts)})"
        debug_log_on_change(
            f"{self.role_tag}_check_static_OT",
            vs=round(self.cur_vs, 2),
            vs_ok=checks.velocity_safe,
            closer=checks.is_getting_closer,
            latest=checks.wpnts_are_latest,
            free=checks.path_is_free,
            wpnts_avail=self.static_avoidance_wpnts is not None,
            wpnts=wpnts_info,
            num_obs=len(self.obstacles_in_interest)
        )

        if should_engage_static_overtaking(checks):
            self.static_overtaking_mode = True
            return True
        return False

    def _check_overtaking_mode_sustainability(self) -> bool:
        """현재 OT 모드(static/dynamic)의 경로가 여전히 사용 가능한지 판정."""
        if self.static_overtaking_mode:
            wpnts_msg = self.static_avoidance_wpnts
            wpnts_data = self.cur_static_avoidance_wpnts
        else:
            wpnts_msg = self.avoidance_wpnts
            wpnts_data = self.cur_avoidance_wpnts

        if not self._check_availability(wpnts_msg, wpnts_data):
            return False
        if not self.static_overtaking_mode:
            self.get_logger().warning("AVAILABLE")  # 원본의 dynamic 분기 로그 보존
        return self._check_free_frenet(wpnts_data)

    # def _check_on_merger(self) -> bool:
    #     if self.merger is not None:
    #         if self.merger[0] < self.merger[1]:
    #             if self.cur_s > self.merger[0] and self.cur_s < self.merger[1]:
    #                 return True
    #         elif self.merger[0] > self.merger[1]:
    #             if self.cur_s > self.merger[0] or self.cur_s < self.merger[1]:
    #                 return True
    #         else:
    #             return False
    #     return False
        
    # def _check_force_trailing(self) -> bool:
    #     return self.force_trailing

    # def _check_fail_trailing(self) -> bool:
    #     return self.fail_trailing

    ################
    # HELPER FUNCS #
    ################
    def _apply_vel_planner_params(self, params: dict):
        """
        Apply vel planner params from yaml or rqt dynamic_reconfigure.
        Updates ggv, ax_max tables, and 3D-specific params in-place.
        Same logic as global_velocity_planner_3d._apply_params_to_ggv.
        """
        self._slope_correction = params.get('slope_correction', self._slope_correction)
        self._slope_brake_margin = params.get('slope_brake_margin', self._slope_brake_margin)
        self._slope_brake_vmax = params.get('slope_brake_vmax', self._slope_brake_vmax)
        self._grip_scale_exp = params.get('grip_scale_exp', self._grip_scale_exp)

        # ggv / ax_max table overrides
        if 'a_x_max' in params and params['a_x_max'] is not None:
            self.ggv[:, 1] = params['a_x_max']
        if 'a_y_max' in params and params['a_y_max'] is not None:
            self.ggv[:, 2] = params['a_y_max']
        if 'ax_max_motor' in params and params['ax_max_motor'] is not None:
            self.ax_max_machines[:, 1] = params['ax_max_motor']
        if 'ax_max_brake' in params and params['ax_max_brake'] is not None:
            self.b_ax_max_machines[:, 1] = params['ax_max_brake']
        if 'v_max' in params and params['v_max'] is not None:
            self.pars["veh_params"]["v_max"] = params['v_max']
        if 'dyn_model_exp' in params and params['dyn_model_exp'] is not None:
            self.pars["vel_calc_opts"]["dyn_model_exp"] = params['dyn_model_exp']

    def _vel_planner_3d_param_cb(self, msg: Any):  # was: dynamic_reconfigure.msg.Config
        """
        Subscribe to /global_velplanner_3d/parameter_updates (dynamic_reconfigure).
        When user edits rqt vel_planner → this fires → state machine tables updated in real-time.
        """
        params = {}
        for p in msg.doubles:
            params[p.name] = p.value
        for p in msg.bools:
            params[p.name] = p.value

        # Skip save/load triggers
        if params.get('save_config') or params.get('save_csv') or params.get('load_yaml'):
            return

        self._apply_vel_planner_params(params)
        self.get_logger().info(f"[StateMachine3D] vel_planner rqt sync: grip_scale_exp={self._grip_scale_exp:.2f}, "
            f"a_y_max={self.ggv[0,2]:.1f}, v_max={self.pars['veh_params']['v_max']:.1f}")

    def _build_track_3d_params(self, wpnts):
        """
        Build slope and track_3d_params from planner waypoints' mu_rad/s_m/kappa.
        Ported from global_velocity_planner_3d.py:286-334.
        """
        n = len(wpnts)
        mu = np.array([wp.mu_rad for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        s = np.array([wp.s_m for wp in wpnts])

        # Flat track → skip 3D params (pure 2D behavior)
        if np.all(np.abs(mu) < 1e-8):
            return mu, None

        slope = mu

        # Unwrap s across track wraparound (e.g., 84, 85, 0.5, 1.5 → 84, 85, 86.3, 87.3)
        # max_s from converter (raceline_length)
        max_s = self.gb_wpnts.wpnts[-1].s_m + (
            self.gb_wpnts.wpnts[-1].s_m - self.gb_wpnts.wpnts[-2].s_m)
        ds = np.diff(s)
        # If a step is largely negative (>= half track), it's a wraparound: add max_s
        ds = np.where(ds < -max_s / 2, ds + max_s, ds)
        ds = np.where(ds > max_s / 2, ds - max_s, ds)
        ds = np.maximum(ds, 1e-6)  # avoid zero-length intervals
        s_for_grad = np.concatenate([[0], np.cumsum(ds)])

        dmu_ds = np.gradient(slope, s_for_grad)

        phi = np.zeros(n)  # no banking

        # Angular rates via Euler→body Jacobian
        omega_x = -np.sin(mu) * kappa
        omega_y = np.cos(phi) * dmu_ds + np.cos(mu) * np.sin(phi) * kappa
        omega_z = -np.sin(phi) * dmu_ds + np.cos(mu) * np.cos(phi) * kappa
        track_3d_params = {
            'mu': mu,
            'phi': phi,
            'omega_x': omega_x,
            'omega_y': omega_y,
            'omega_z': omega_z,
            'd_omega_x': np.gradient(omega_x, s_for_grad),
            'd_omega_y': np.gradient(omega_y, s_for_grad),
            'd_omega_z': np.gradient(omega_z, s_for_grad),
            'dmu_ds': dmu_ds,
            'h': self._h_cog,
            'slope_correction': self._slope_correction,
            'slope_brake_margin': self._slope_brake_margin,
            'slope_brake_vmax': self._slope_brake_vmax,
        }
        return slope, track_3d_params

    def update_velocity(self, wpnts_msg, safety_factor=1.0):
        """3D: uses vel_planner_25d with slope, track_3d_params, grip_scale_exp.
        Passes s_global for correct friction sector matching.

        ROS2 포팅: vel_planner_25d / tph 미설치 시 noop (smoke 검증용).
        진짜 운영 시 calc_vel_profile + tph.calc_ax_profile 필수.
        """
        if calc_vel_profile is None or tph is None:
            return  # smoke mode — vel_planner 비활성
        wpnts = wpnts_msg.wpnts
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = np.array([
            np.linalg.norm([
                wpnts[i+1].x_m - wpnts[i].x_m,
                wpnts[i+1].y_m - wpnts[i].y_m
            ])
            for i in range(len(wpnts)-1)
        ])

        glb_start_idx = int(wpnts_msg.wpnts[-1].s_m / self.wpnt_dist)
        v_end = self.gb_wpnts.wpnts[glb_start_idx % len(self.gb_wpnts.wpnts)].vx_mps

        ax_max_machines_sf = self.ax_max_machines.copy()
        b_ax_max_machines_sf = self.b_ax_max_machines.copy()
        ax_max_machines_sf[:, 1] *= safety_factor
        b_ax_max_machines_sf[:, 1] *= safety_factor

        # 3D: build slope and track_3d_params from waypoint mu_rad
        slope, track_3d_params = self._build_track_3d_params(wpnts)

        # Global s from waypoints for friction sector matching
        s_global = np.array([wp.s_m for wp in wpnts])

        vx_profile = calc_vel_profile(
            ax_max_machines=ax_max_machines_sf,
            kappa=kappa,
            el_lengths=el_lengths,
            closed=False,
            drag_coeff=self.pars["veh_params"]["dragcoeff"],
            m_veh=self.pars["veh_params"]["mass"],
            b_ax_max_machines=b_ax_max_machines_sf,
            ggv=self.ggv,
            v_max=self.pars["veh_params"]["v_max"],
            filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
            v_start=self.cur_vs,
            v_end=v_end,
            slope=slope,
            track_3d_params=track_3d_params,
            grip_scale_exp=self._grip_scale_exp,
            s_global=s_global,
        )

        for i in range(len(vx_profile)):
            wpnts_msg.wpnts[i].vx_mps = vx_profile[i]

        ax_profile = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile,
                                                            el_lengths=el_lengths,
                                                            eq_length_output=False)

        for i in range(len(ax_profile)):
            wpnts_msg.wpnts[i].ax_mps2 = ax_profile[i]
        wpnts[len(ax_profile)].ax_mps2 = ax_profile[-1]


    def mincurv_splinification(self):
        coords = np.empty((len(self.cur_gb_wpnts.list), 4))
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.vx_mps

        self.mincurv_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.mincurv_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.mincurv_spline_v = Spline(coords[:, 0], coords[:, 3])
        self.get_logger().info(f"[{self.name}] Splinified Min Curve")

    def ot_splinification(self):
        coords = np.empty((len(self.overtake_wpnts), 5))
        for i, wpnt in enumerate(self.overtake_wpnts):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.d_m
            coords[i, 4] = wpnt.vx_mps

        # Sort s_m to start splining at 0
        coords = coords[coords[:, 0].argsort()]
        self.ot_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.ot_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.ot_spline_d = Spline(coords[:, 0], coords[:, 3])
        self.ot_spline_v = Spline(coords[:, 0], coords[:, 4])
        self.get_logger().info(f"[{self.name}] Splinified Overtaking Curve")

    def _find_nearest_ot_s(self) -> float:
        half_search_dim = 5

        # create indices
        idxs = [
            i % self.num_ot_points for i in range(self.cur_id_ot - half_search_dim, self.cur_id_ot + half_search_dim)
        ]
        ses = np.array([self.overtake_wpnts[i].s_m for i in idxs])

        dists = np.abs(self.cur_s - ses)
        chose_id = np.argmin(dists)
        s_ot = idxs[chose_id]
        s_ot %= self.num_ot_points

        return s_ot

    def get_splini_wpts(self) -> WpntArray:
        """Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        # splini_glob = self.cur_gb_wpnts.list.copy()

        # Handle wrapping
        wpnts = None
        if self.static_overtaking_mode:
            wpnts = self.cur_static_avoidance_wpnts
        else:
            wpnts = self.cur_avoidance_wpnts

        # ===== HJ ADDED: Safety check - fallback if spliner fails =====
        if wpnts is None or not hasattr(wpnts, 'array') or wpnts.array is None or len(wpnts.array) == 0:
            self.get_logger().warning("[state_machine] Spliner waypoints invalid, using fallback")
            # Fallback based on smart_static_active flag
            if self.smart_static_active and self.cur_smart_static_avoidance_wpnts.is_init:
                # Use Fixed path with Fixed Frenet cur_s
                self.get_logger().warning("[state_machine] Fallback: Using Fixed path")
                cur_s_fixed = self.smart_helper.cur_s
                s = int(cur_s_fixed / self.smart_wpnt_dist + 0.5)
                num_smart = len(self.cur_smart_static_avoidance_wpnts.list)
                return [self.cur_smart_static_avoidance_wpnts.list[(s + i) % num_smart] for i in range(self.n_loc_wpnts)]
            else:
                # Use GB path with GB Frenet cur_s
                self.get_logger().warning("[state_machine] Fallback: Using GB path")
                s = int(self.cur_s / self.waypoints_dist + 0.5)
                return [self.cur_gb_wpnts.list[(s + i) % self.num_glb_wpnts] for i in range(self.n_loc_wpnts)]
        # ===== HJ ADDED END =====

        diff = np.linalg.norm(wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
        min_idx = np.argmin(diff)
        avoidance_wpnts = wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

        if len(avoidance_wpnts) < self.n_loc_wpnts:
            # Use helper function to get extra waypoints (Smart or GB based on flag)
            last_s_m = wpnts.list[-1].s_m
            num_needed = self.n_loc_wpnts - len(avoidance_wpnts)
            extra_wpnts = self.get_extra_waypoints(last_s_m, num_needed)
            avoidance_wpnts.extend(extra_wpnts)
        # self.get_logger().warning(f"WORK WELL {self.last_valid_avoidance_wpnts.wpnts[-1].s_m}")
        return avoidance_wpnts
        
    def get_recovery_wpts(self) -> WpntArray:
        """Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        # splini_glob = self.cur_gb_wpnts.list.copy()

        # Handle wrapping
        if self.cur_recovery_wpnts.is_init:
            
            diff = np.linalg.norm(self.cur_recovery_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            wpnts = self.cur_recovery_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

            if len(wpnts) < self.n_loc_wpnts:
                # Use helper function to get extra waypoints (Smart or GB based on flag)
                last_s_m = self.cur_recovery_wpnts.list[-1].s_m
                num_needed = self.n_loc_wpnts - len(wpnts)
                extra_wpnts = self.get_extra_waypoints(last_s_m, num_needed)
                wpnts.extend(extra_wpnts)
            # self.get_logger().warning(f"WORK WELL {self.last_valid_avoidance_wpnts.wpnts[-1].s_m}")
            return wpnts

    # ===== HJ ADDED: Helper function for waypoint shortage =====
    def get_extra_waypoints(self, last_s_m: float, num_needed: int) -> List[Wpnt]:
        """
        Get extra waypoints to fill shortage.

        If smart_static_active: wrap within Smart Static path (using s_m matching)
        Otherwise: fill with GB waypoints (original behavior)

        Args:
            last_s_m: GB Frenet s coordinate of last waypoint in current list
            num_needed: Number of extra waypoints needed

        Returns:
            List of extra waypoints
        """
        if self.smart_static_active and self.cur_smart_static_avoidance_wpnts.is_init:
            # ===== HJ FIXED: Use index-based approach (same as original code) to prevent reverse wrapping =====
            # Convert s_m to index, then +1 to get NEXT waypoint
            # This ensures we always move forward, even at track boundaries (modulo handles wrap-around)
            start_idx = int(last_s_m / self.smart_wpnt_dist) + 1
            num_smart = len(self.cur_smart_static_avoidance_wpnts.list)

            # Extract waypoints with modulo wrap-around
            extra = [self.cur_smart_static_avoidance_wpnts.list[(start_idx + i) % num_smart]
                     for i in range(num_needed)]

            self.get_logger().debug(f"[{self.name}] Shortage filled with Smart waypoints: start_idx={start_idx}, num={num_needed}")
            # ===== HJ FIXED END =====
        else:
            # GB mode: original behavior
            gb_start_idx = int(last_s_m / self.wpnt_dist) + 1
            extra = [self.cur_gb_wpnts.list[(gb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                     for i in range(num_needed)]

            self.get_logger().debug(f"[{self.name}] Shortage filled with GB waypoints: start_idx={gb_start_idx}, num={num_needed}")

        return extra
    # ===== HJ ADDED END =====

    # ===== HJ ADDED: Get smart static avoidance waypoints =====
    def get_smart_static_wpts(self) -> WpntArray:
        """Obtain waypoints from smart static avoidance fixed path.

        Finds closest waypoint by s_m value (with closed-loop wrap-around), then uses modulo for extraction.
        Uses Fixed Frenet cur_s from smart_helper for accurate positioning.
        """
        if self.cur_smart_static_avoidance_wpnts.is_init:
            # Fixed Frenet cur_s (smart_helper) — wpnt.s_m 와 같은 좌표계
            cur_s_fixed = self.smart_helper.cur_s

            # ===== HJ FIXED: Use GlobalTracking-style rounding for closest waypoint =====
            # Round to nearest waypoint index (same approach as GlobalTracking)
            min_idx = int(cur_s_fixed / self.smart_wpnt_dist + 0.5)
            num_smart_wpnts = len(self.cur_smart_static_avoidance_wpnts.list)
            # ===== HJ FIXED END =====

            # Use modulo to wrap around - pure Smart waypoints, no GB fallback!
            wpnts = [self.cur_smart_static_avoidance_wpnts.list[(min_idx + i) % num_smart_wpnts]
                     for i in range(self.n_loc_wpnts)]

            return wpnts

    # ===== HJ ADDED END =====

    def get_start_wpts(self) -> WpntArray:
        """Obtain the waypoints by fusing those obtained by spliner with the
        global ones.
        """
        # Handle wrapping
        if self.cur_start_wpnts.is_init:
            diff = np.linalg.norm(self.cur_start_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            start_wpnts = self.cur_start_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

            if len(start_wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_start_wpnts.list[-1].s_m / self.wpnt_dist) + 1
                extra_wpnts = [self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)] 
                            for i in range(self.n_loc_wpnts - len(start_wpnts))]

                start_wpnts.extend(extra_wpnts)
            # self.get_logger().warning(f"WORK WELL {self.last_valid_avoidance_wpnts.wpnts[-1].s_m}")
            return start_wpnts

        else:
            self.get_logger().warning(f"[{self.name}] No valid avoidance waypoints, passing global waypoints")
            pass

        # return splini_glob

    # NOTE: visualization 메서드 (_pub_local_wpnts / visualize_state /
    # _compute_visualization_anchor / _publish_state_text_marker /
    # publish_not_ready_marker / _speed_to_color) 는 VisualizationMixin 으로 이동.

    def update_waypoints(self):
        if not self.cur_gb_wpnts.is_init:
            self.cur_gb_wpnts.initialize_traj(self.gb_wpnts)
        else:
            self.cur_gb_wpnts.list = self.gb_wpnts.wpnts

        self.cur_obstacles_in_interest = self.obstacles_in_interest

        # ===== HJ ADDED: Update smart_helper synchronously =====
        # Update smart_helper's obstacles_in_interest at same time as parent
        # This ensures perfect synchronization between GB and Smart modes
        if self.smart_static_active and self.smart_helper is not None:
            self.smart_helper.update()
        # ===== HJ ADDED END =====

        return

        
    def get_overtaking_target(self):
        """현재 모드(Smart/GB)에 맞는 closest_target 반환 — active_helper property 로 분기 통일."""
        helper = self.active_helper
        if helper.cur_gb_wpnts.closest_target is not None:
            return [helper.cur_gb_wpnts.closest_target]
        if helper.cur_recovery_wpnts.closest_target is not None:
            return [helper.cur_recovery_wpnts.closest_target]
        return []



    def get_traling_target(self):
        if self.local_wpnts_src == StateType.GB_TRACK and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.OVERTAKE and self.ot_closest_target is not None:
            return [self.ot_closest_target]
        else:
            return []
        
    def get_farthest_target(self, local_wpnts_src):
        """현재 모드의 데이터 source(active_helper)에서 base wpnts 의 closest_target 으로 시작해
        avoidance / static_avoidance / start 의 더 먼 closest_gap 으로 갱신한다.

        Smart 모드에서 base 는 SMART_STATIC, GB 모드에서는 GB_TRACK. RECOVERY 는 양 모드 공통.
        """
        helper = self.active_helper
        base_state = StateType.SMART_STATIC if self.smart_static_active else StateType.GB_TRACK

        # base waypoint 결정 — local_wpnts_src 에 따라 base / recovery 갈래
        if local_wpnts_src == base_state and helper.cur_gb_wpnts.closest_target is not None:
            base_wpnts = helper.cur_gb_wpnts
        elif local_wpnts_src == StateType.RECOVERY and helper.cur_recovery_wpnts.closest_target is not None:
            base_wpnts = helper.cur_recovery_wpnts
        else:
            return [], local_wpnts_src

        closest_target = base_wpnts.closest_target
        closest_gap = base_wpnts.closest_gap

        # 더 먼 (큰 gap) closest_target 으로 순차 갱신
        # 주의: GB_TRACK base 일 때만 첫 비교가 `<=` (>= 가 아니라). 원본 동작 보존.
        first_op_le = (local_wpnts_src == base_state)

        avoidance = helper.cur_avoidance_wpnts
        if avoidance.closest_target is not None:
            if (first_op_le and closest_gap <= avoidance.closest_gap) or \
               (not first_op_le and closest_gap < avoidance.closest_gap):
                closest_target = avoidance.closest_target
                closest_gap = avoidance.closest_gap
                local_wpnts_src = StateType.OVERTAKE

        static_avoidance = helper.cur_static_avoidance_wpnts
        if static_avoidance.closest_target is not None and closest_gap < static_avoidance.closest_gap:
            closest_target = static_avoidance.closest_target
            closest_gap = static_avoidance.closest_gap
            local_wpnts_src = StateType.OVERTAKE

        start = helper.cur_start_wpnts
        if start.closest_target is not None and closest_gap < start.closest_gap:
            closest_target = start.closest_target
            closest_gap = start.closest_gap
            local_wpnts_src = StateType.START

        return [closest_target], local_wpnts_src

    
    def check_ot_cloest_target(self):
        # ===== HJ MODIFIED: Use appropriate Frenet coordinate system based on mode =====
        if self.smart_static_active:
            # Smart mode: use smart_helper's Fixed Frenet based calculations
            smart_helper = self.smart_helper
            if smart_helper.cur_gb_wpnts.closest_target is not None and smart_helper.ot_closest_target is not None and self.local_wpnts_src == StateType.SMART_STATIC:
                if smart_helper.ot_closest_gap > smart_helper.cur_gb_wpnts.closest_gap:
                    self.local_wpnts_src = StateType.OVERTAKE
            elif smart_helper.cur_recovery_wpnts.closest_target is not None and smart_helper.ot_closest_target is not None and self.local_wpnts_src == StateType.RECOVERY:
                if smart_helper.ot_closest_gap > smart_helper.cur_recovery_wpnts.closest_gap:
                    self.local_wpnts_src = StateType.OVERTAKE
        else:
            # GB mode: use self's GB Frenet based calculations
            if self.gb_closest_target is not None and self.ot_closest_target is not None and self.local_wpnts_src == StateType.GB_TRACK:
                if self.ot_closest_gap > self.gb_closest_gap:
                    self.local_wpnts_src = StateType.OVERTAKE
            elif self.cur_recovery_wpnts.closest_target is not None and self.ot_closest_target is not None and self.local_wpnts_src == StateType.RECOVERY:
                if self.ot_closest_gap > self.cur_recovery_wpnts.closest_gap:
                    self.local_wpnts_src = StateType.OVERTAKE
        # ===== HJ MODIFIED END =====       

    #############
    # MAIN LOOP HELPERS
    #############

    def _reset_check_result_caches(self):
        """매 loop iteration 시작 시 이전 결과 캐시 초기화."""
        self.gb_closest_target = None
        self.ot_closest_target = None
        self.cur_gb_wpnts.closest_target = None
        self.cur_recovery_wpnts.closest_target = None
        self.cur_avoidance_wpnts.closest_target = None
        self.cur_static_avoidance_wpnts.closest_target = None
        self.cur_start_wpnts.closest_target = None

    def _check_low_voltage_warning(self):
        """배터리 전압이 임계 이하면 경고 마커 발행."""
        if self.cur_volt < self.volt_threshold:
            self.get_logger().error(f"[{self.name}] VOLTS TOO LOW, STOP THE CAR")
            self.publish_not_ready_marker()

    def _decide_next_state(self):
        """다음 상태 결정. 우선순위: force_gbtrack > ftg_only_zone > 정상 transition."""
        if self.force_gbtrack_state:
            return StateType.GB_TRACK, StateType.GB_TRACK
        if self._check_only_ftg_zone():
            self.get_logger().warning(f"[{self.name}] FTGONLY sector !!!")
            return StateType.FTGONLY, StateType.FTGONLY
        return self.state_transitions[self.cur_state](self)

    def _log_state_change_if_debug(self, prev_state, prev_wpnts_src):
        """state 또는 wpnts_src 가 변경됐으면 디버그 로그 (DEBUG_STATE_TRANSITION 활성 시)."""
        if not DEBUG_STATE_TRANSITION:
            return
        if prev_state != self.cur_state or prev_wpnts_src != self.local_wpnts_src:
            self.get_logger().warning(
                f"[STATE CHANGE {self.current_mode_tag}] {prev_state.name} → {self.cur_state.name} | "
                f"wpnts: {prev_wpnts_src.name} → {self.local_wpnts_src.name}"
            )

    def _publish_behavior_strategy(self, local_wpnts):
        """BehaviorStrategy 메시지 채우고 발행."""
        self.behavior_strategy.header.stamp = self.get_clock().now().to_msg()
        self.behavior_strategy.local_wpnts = local_wpnts
        self.behavior_strategy.state = self.cur_state.value
        self.behavior_strategy.need_vel_planner = False
        self.behavior_strategy_pub.publish(self.behavior_strategy)

    # NOTE: _publish_target_marker 도 VisualizationMixin 으로 이동.

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        """ROS 노드가 고정 주기로 호출하는 state machine 메인 루프."""
        if self.measuring:
            start = time.perf_counter()

        self.update_waypoints()
        self._reset_check_result_caches()
        self._check_low_voltage_warning()

        # 다음 상태 결정 + 디버그 로그
        prev_state, prev_wpnts_src = self.cur_state, self.local_wpnts_src
        self.cur_state, self.local_wpnts_src = self._decide_next_state()
        self._log_state_change_if_debug(prev_state, prev_wpnts_src)

        # TRAILING 시 trailing target 계산 (그 외엔 비움)
        if self.cur_state == StateType.TRAILING:
            self.check_ot_cloest_target()
            self.behavior_strategy.trailing_targets, self.local_wpnts_src = (
                self.get_farthest_target(self.local_wpnts_src)
            )
        else:
            self.behavior_strategy.trailing_targets = []
        self.behavior_strategy.overtaking_targets = self.get_overtaking_target()

        # 현재 상태에 맞는 local waypoints 생성
        local_wpnts = self.states[self.local_wpnts_src](self)
        if self.cur_state == StateType.LOSTLINE:
            self.cur_state = StateType.GB_TRACK

        # behavior + state + 시각화 발행 (Round 2: 모든 strict 타입 fix 후 재활성)
        from std_msgs.msg import String as _String
        self._publish_behavior_strategy(local_wpnts)
        self.state_pub.publish(_String(data=self.cur_state.value))
        self.visualize_state(state=self.cur_state.value)
        self._pub_local_wpnts(local_wpnts)

        # TRAILING/ATTACK 이외에서는 FTG 카운터 초기화
        if self.cur_state != StateType.TRAILING and self.cur_state != StateType.ATTACK:
            self.ftg_counter = 0

        # target 시각화 (overtaking=blue, trailing=green)
        self._publish_target_marker(
            self.overtaking_marker_pub, self.behavior_strategy.overtaking_targets, color_b=1.0
        )
        self._publish_target_marker(
            self.trailing_marker_pub, self.behavior_strategy.trailing_targets, color_g=1.0
        )

        if self.measuring:
            from std_msgs.msg import Float32 as _Float32
            self.latency_pub.publish(_Float32(data=float(1 / (time.perf_counter() - start))))

def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = StateMachine("state_machine")
    except (ImportError, RuntimeError, FileNotFoundError) as e:
        # tph / vel_planner_25d / stack_master config 미설치 등 startup 실패
        print(f"[state_machine] startup error: {e}")
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # rospy.on_shutdown 대체
        try:
            node.on_shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()