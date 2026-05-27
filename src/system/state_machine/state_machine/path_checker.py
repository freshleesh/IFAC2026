"""Pure-function path validation for the state machine.

이 모듈은 state machine의 collision check 로직을 ROS 의존성에서 분리한 순수 함수 모음이다.
모든 함수는 명시적 인자만 받으며, 입력 객체를 변경하지 않는다 (logger 출력 제외).
호출자(wrapper)는 결과 dataclass를 받아 필요한 side effect를 처리한다.

ROS2 포팅: 원본 ROS1 의 rospy.loginfo/logwarn → 표준 Python logging 모듈로 대체.
ROS2 launch 환경에서도 정상 동작 (rclpy 가 standard logging 으로 통합).
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


_logger = logging.getLogger(__name__)


# =============================================================================
# 입출력 dataclass
# =============================================================================

@dataclass(frozen=True)
class EgoFrenetState:
    """자차의 Frenet 상태 (체크 함수가 의존하는 최소 필드만)."""
    s: float    # 진행 거리 [m]
    vs: float   # 종방향 속도 [m/s]


@dataclass(frozen=True)
class FrenetCheckParams:
    """충돌 체크에 필요한 차량/트랙 파라미터."""
    max_s: float                    # 트랙 한 바퀴 길이 [m]
    veh_length: float               # 자차 길이 [m]
    ego_width: float                # 자차 폭 [m]
    safety_factor_sec: float = 0.5  # 안전 시간 마진 [s] (현재 미사용, 향후 확장용)


@dataclass
class FreeFrenetResult:
    """check_free_frenet 결과.

    - is_free=True 이면 경로에 충돌 없음
    - is_free=False 이면 closest_obstacle / closest_gap 에 가장 가까운 충돌 정보가 담김
    """
    is_free: bool
    closest_obstacle: Optional[Any] = None
    closest_gap: float = 2.0


@dataclass(frozen=True)
class OvertakingModeChecks:
    """동적 OT 모드 진입 조건들 (모두 충족 시 진입)."""
    in_ot_sector: bool
    is_getting_closer: bool
    wpnts_are_latest: bool
    path_is_free: bool


@dataclass(frozen=True)
class StaticOvertakingChecks:
    """정적 OT 모드 진입 조건들 (모두 충족 시 진입)."""
    velocity_safe: bool
    is_getting_closer: bool
    wpnts_are_latest: bool
    path_is_free: bool


@dataclass(frozen=True)
class GettingCloserParams:
    horizon_for_ttl: float
    static_sector_filtering: bool
    track_length: float


@dataclass(frozen=True)
class OnSplineParams:
    track_length: float
    front_horizon_thres_m: float
    min_dist_thres_m: float


@dataclass(frozen=True)
class TrajFreshnessParams:
    is_smart_static: bool
    latest_threshold_sec: float


# =============================================================================
# Pure functions
# =============================================================================

def check_free_frenet(
    ego: EgoFrenetState,
    waypoints: Any,
    obstacles: List[Any],
    obstacle_predictions: List[Any],
    obstacle_prediction_id: int,
    params: FrenetCheckParams,
) -> FreeFrenetResult:
    """경로(waypoints) 위에 충돌 가능한 장애물이 있는지 Frenet 좌표계에서 체크.

    `3d_state_machine_node.StateMachine._check_free_frenet`에서 추출한 순수 함수.
    원본과 동일한 로직을 유지하며, side effect(`wpnts_data.closest_target/closest_gap` 수정)만
    제거하여 결과 dataclass로 반환한다.

    waypoints 객체에서 읽는 필드 (duck typing):
        - is_init, is_closed, is_gb_track_wpnts, is_ot_wpnts: bool
        - max_horizon, lateral_width_m, free_scaling_reference_distance_m: float
        - array: np.ndarray, columns = [x_m, y_m, s_m, d_m]
        - list: List[Wpnt-like], 각 원소는 .d_m 을 가짐

    obstacle 객체에서 읽는 필드:
        - s_center, d_center: float
        - is_static: bool
        - size: float
        - vs: float
        - id: int

    obstacle_predictions 원소에서 읽는 필드:
        - pred_s, pred_d: float
    """
    if not waypoints.is_init:
        return FreeFrenetResult(is_free=True)

    is_free = True
    closest_obs: Optional[Any] = None
    min_gap: float = 2.0

    max_horizon = waypoints.max_horizon
    is_gb_track_wpnts = waypoints.is_gb_track_wpnts
    is_ot_wpnts = waypoints.is_ot_wpnts
    free_scaling_reference_distance_m = waypoints.free_scaling_reference_distance_m
    lateral_width_m = waypoints.lateral_width_m

    max_gap = (waypoints.array[-1, 2] - ego.s) % params.max_s

    for obs in obstacles:
        obs_s = obs.s_center
        gap = (obs_s - ego.s) % params.max_s
        relative_vs = ego.vs - obs.vs
        clip_vs = max(relative_vs, 0.5)
        ttc = (gap - params.veh_length) / clip_vs
        tt0 = (gap + 0.3 * params.veh_length) / clip_vs

        if obs.is_static:
            if not waypoints.is_closed and gap > max_gap:
                is_free = False
                if closest_obs is None or min_gap > gap:
                    closest_obs = obs
                    min_gap = gap

            elif gap < max_horizon:
                ot_d = 0
                if not is_gb_track_wpnts:
                    avoid_wpnt_idx = np.argmin(abs(waypoints.array[:, 2] - obs_s))
                    ot_d = waypoints.list[avoid_wpnt_idx].d_m
                min_dist = abs(ot_d - obs.d_center)

                free_dist = min_dist - obs.size / 2 - params.ego_width / 2

                scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                if free_dist < lateral_width_m * scaling_factor:
                    is_free = False
                    _logger.info(
                        "[State Machine] FREE False, obs dist to ot lane: %s m", free_dist
                    )
                    if closest_obs is None or min_gap > gap:
                        closest_obs = obs
                        min_gap = gap
        else:
            if len(obstacle_predictions) != 0 and obstacle_prediction_id == obs.id:
                start_idx = 0
                end_idx = len(obstacle_predictions)

                if is_ot_wpnts:
                    if ttc > 0:
                        start_idx = min(int(ttc / 0.05), len(obstacle_predictions))
                    if tt0 > 0:
                        end_idx = min(int(tt0 / 0.05), len(obstacle_predictions))

                for obs_pred in obstacle_predictions[start_idx:end_idx]:
                    wpnt_idx = np.argmin(abs(waypoints.array[:, 2] - obs_pred.pred_s))
                    wpnt_d = waypoints.list[wpnt_idx].d_m
                    min_dist = abs(wpnt_d - obs_pred.pred_d)
                    free_dist = min_dist - obs.size / 2 - params.ego_width / 2
                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                    if is_ot_wpnts:
                        _logger.warning(
                            "free_dist: %s, lateral_width_m: %s, scaling_factor: %s, "
                            "obs.size: %s, wpnt_d: %s, obs_pred.pred_d: %s",
                            free_dist, lateral_width_m, scaling_factor,
                            obs.size, wpnt_d, obs_pred.pred_d,
                        )
                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
            else:
                if not waypoints.is_closed and gap > max_gap:
                    is_free = False
                    if closest_obs is None or min_gap > gap:
                        closest_obs = obs
                        min_gap = gap
                elif gap < max_horizon:
                    ot_d = 0
                    if not is_gb_track_wpnts:
                        avoid_wpnt_idx = np.argmin(abs(waypoints.array[:, 2] - obs.s_center))
                        ot_d = waypoints.list[avoid_wpnt_idx].d_m
                    min_dist = abs(ot_d - obs.d_center)

                    free_dist = min_dist - obs.size / 2 - params.ego_width / 2

                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap

    return FreeFrenetResult(
        is_free=is_free,
        closest_obstacle=closest_obs,
        closest_gap=min_gap,
    )


def should_engage_overtaking(checks: OvertakingModeChecks) -> bool:
    """동적 OT 진입 결정. 4개 조건 전부 참이어야 진입."""
    return (
        checks.in_ot_sector
        and checks.is_getting_closer
        and checks.wpnts_are_latest
        and checks.path_is_free
    )


def should_engage_static_overtaking(checks: StaticOvertakingChecks) -> bool:
    """정적 OT 진입 결정. 4개 조건 전부 참이어야 진입."""
    return (
        checks.velocity_safe
        and checks.is_getting_closer
        and checks.wpnts_are_latest
        and checks.path_is_free
    )


def is_in_overtaking_zone(
    s_m: float,
    waypoints_dist: float,
    zones: List[Tuple[float, float]],
) -> bool:
    """현재 진행거리(s_m)가 OT 허용 zone 안에 있는지 확인."""
    idx = s_m / waypoints_dist
    return any(start <= idx <= end for start, end in zones)


def is_getting_closer(
    cur_s: float,
    cur_vs: float,
    first_obstacle: Optional[Any],
    params: GettingCloserParams,
) -> bool:
    """관심 장애물(첫 번째)이 자차에 가까워지고 있는지 판정."""
    if first_obstacle is None:
        return False

    obs = first_obstacle
    is_static_in_static_sector = obs.in_static_obs_sector and obs.is_static

    if is_static_in_static_sector and params.static_sector_filtering:
        distance = (obs.s_start - cur_s) % params.track_length
        if distance > params.horizon_for_ttl:
            return False

    return cur_vs - obs.vs > -0.5


def is_on_spline(
    cur_s: float,
    current_position_xy: np.ndarray,
    waypoints: Any,
    params: OnSplineParams,
) -> bool:
    """자차가 waypoint spline 위에 있는지 (시작점 충분 + 가까이 붙어있는지) 판정."""
    if not waypoints.is_init:
        return False
    gap = (waypoints.list[-1].s_m - cur_s) % params.track_length
    min_dist = np.min(np.linalg.norm(waypoints.array[:, 0:2] - current_position_xy, axis=1))
    return bool(gap > params.front_horizon_thres_m and min_dist < params.min_dist_thres_m)


def is_traj_msg_fresh(
    src_msg: Optional[Any],
    now_sec: float,
    params: TrajFreshnessParams,
) -> bool:
    """들어온 trajectory 메시지가 fresh 한지 (사용 가능한지) 판정.

    src_msg 객체에서 읽는 필드:
        - wpnts: list (비어있는지 확인)
        - header.stamp.{sec, nanosec}: builtin_interfaces/Time 필드 (ROS2)

    Smart Static 플래너는 stamp 초기화만 됐으면 영구 유효 (sec=0 & nanosec=0 이면 미초기화).
    그 외 플래너는 (now - stamp) <= latest_threshold_sec 이어야 fresh.
    """
    if src_msg is None or len(src_msg.wpnts) == 0:
        return False

    stamp = src_msg.header.stamp  # builtin_interfaces/Time
    stamp_sec = stamp.sec + stamp.nanosec * 1e-9

    if params.is_smart_static:
        return stamp.sec != 0 or stamp.nanosec != 0

    time_diff = now_sec - stamp_sec
    return time_diff <= params.latest_threshold_sec
