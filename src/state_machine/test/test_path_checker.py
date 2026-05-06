"""Scenario tests for path_checker pure functions."""
import itertools

import pytest

from fake_msgs import FakeObstacle, FakeWaypointData
from state_machine.path_checker import (
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
from state_machine.waypoint_data import WaypointData


def test_no_obstacles_returns_free(ego_at_10m, straight_track_wpnts, default_params):
    """장애물이 하나도 없으면 경로는 자유로 반환되어야 한다."""
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=straight_track_wpnts,
        obstacles=[],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is True
    assert result.closest_obstacle is None
    assert result.closest_gap == 2.0  # default 값 유지


def test_static_obstacle_in_front_blocks_path(
    ego_at_10m, straight_track_wpnts, default_params
):
    """자차 5m 앞 정적 장애물 (lateral 0) 이면 충돌로 판정되어야 한다.

    free_dist = abs(0 - 0) - 0.3/2 - 0.3/2 = -0.3 < lateral_width(0.4)*scaling -> 충돌.
    """
    obstacle = FakeObstacle(
        s_center=15.0, d_center=0.0, is_static=True, size=0.3, vs=0.0, id=1
    )
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=straight_track_wpnts,
        obstacles=[obstacle],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is False
    assert result.closest_obstacle is obstacle
    assert result.closest_gap == 5.0  # 15 - 10


def test_static_obstacle_behind_is_ignored(
    ego_at_10m, straight_track_wpnts, default_params
):
    """자차 뒤에 있는 정적 장애물은 무시되어야 한다.

    gap = (5 - 10) % 200 = 195. closed track + gap > max_horizon(30) 이므로 체크 스킵.
    """
    obstacle = FakeObstacle(s_center=5.0, d_center=0.0, is_static=True)
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=straight_track_wpnts,
        obstacles=[obstacle],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is True
    assert result.closest_obstacle is None


def test_static_obstacle_far_lateral_does_not_block(
    ego_at_10m, straight_track_wpnts, default_params
):
    """lateral 안전거리 밖 정적 장애물은 충돌로 보지 않는다.

    GB 트랙 위 d=2.0 인 장애물.
    free_dist = abs(0 - 2.0) - 0.3/2 - 0.3/2 = 1.7
    lateral_width * scaling = 0.4 * (5/5)=0.4
    1.7 > 0.4 이므로 자유.
    """
    obstacle = FakeObstacle(s_center=15.0, d_center=2.0, is_static=True, size=0.3)
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=straight_track_wpnts,
        obstacles=[obstacle],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is True
    assert result.closest_obstacle is None


def test_uninitialized_waypoints_return_free(ego_at_10m, default_params):
    """is_init=False 인 wpnts 데이터는 항상 자유로 반환되어야 한다."""
    waypoints = FakeWaypointData(is_init=False)
    obstacle = FakeObstacle(s_center=15.0, d_center=0.0, is_static=True)
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=waypoints,
        obstacles=[obstacle],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is True
    assert result.closest_obstacle is None


def test_closest_obstacle_is_the_nearest_one(
    ego_at_10m, straight_track_wpnts, default_params
):
    """충돌 장애물이 여러 개면 가장 가까운 것이 closest_obstacle 로 반환되어야 한다."""
    far_obs = FakeObstacle(s_center=20.0, d_center=0.0, is_static=True, id=1)
    near_obs = FakeObstacle(s_center=13.0, d_center=0.0, is_static=True, id=2)
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=straight_track_wpnts,
        obstacles=[far_obs, near_obs],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is False
    assert result.closest_obstacle is near_obs
    assert result.closest_gap == 3.0  # 13 - 10


# ===========================================================================
# should_engage_overtaking
# ===========================================================================

def test_overtaking_engages_only_when_all_conditions_pass():
    """4개 조건 전부 참이어야 동적 OT 진입."""
    checks = OvertakingModeChecks(
        in_ot_sector=True,
        is_getting_closer=True,
        wpnts_are_latest=True,
        path_is_free=True,
    )
    assert should_engage_overtaking(checks) is True


@pytest.mark.parametrize("missing_field", [
    "in_ot_sector",
    "is_getting_closer",
    "wpnts_are_latest",
    "path_is_free",
])
def test_overtaking_blocked_if_any_condition_fails(missing_field):
    """4개 조건 중 하나라도 거짓이면 OT 진입 거부."""
    fields = {
        "in_ot_sector": True,
        "is_getting_closer": True,
        "wpnts_are_latest": True,
        "path_is_free": True,
    }
    fields[missing_field] = False
    assert should_engage_overtaking(OvertakingModeChecks(**fields)) is False


def test_overtaking_truth_table():
    """4개 boolean 조합 16가지 중 (True,True,True,True) 만 진입."""
    for combo in itertools.product([False, True], repeat=4):
        checks = OvertakingModeChecks(*combo)
        expected = all(combo)
        assert should_engage_overtaking(checks) is expected, f"failed for {combo}"


# ===========================================================================
# should_engage_static_overtaking
# ===========================================================================

def test_static_overtaking_engages_only_when_all_conditions_pass():
    """4개 조건 전부 참이어야 정적 OT 진입."""
    checks = StaticOvertakingChecks(
        velocity_safe=True,
        is_getting_closer=True,
        wpnts_are_latest=True,
        path_is_free=True,
    )
    assert should_engage_static_overtaking(checks) is True


@pytest.mark.parametrize("missing_field", [
    "velocity_safe",
    "is_getting_closer",
    "wpnts_are_latest",
    "path_is_free",
])
def test_static_overtaking_blocked_if_any_condition_fails(missing_field):
    """4개 조건 중 하나라도 거짓이면 정적 OT 진입 거부."""
    fields = {
        "velocity_safe": True,
        "is_getting_closer": True,
        "wpnts_are_latest": True,
        "path_is_free": True,
    }
    fields[missing_field] = False
    assert should_engage_static_overtaking(StaticOvertakingChecks(**fields)) is False


# ===========================================================================
# is_in_overtaking_zone
# ===========================================================================

def test_in_overtaking_zone_when_inside():
    """진행거리가 zone 범위 안이면 True."""
    # zone (50, 150) — waypoints_dist=0.1 기준 인덱스
    # cur_s=10.0 -> idx 100 -> 50 <= 100 <= 150 -> True
    assert is_in_overtaking_zone(s_m=10.0, waypoints_dist=0.1, zones=[(50, 150)]) is True


def test_in_overtaking_zone_when_outside():
    """진행거리가 어느 zone에도 속하지 않으면 False."""
    # cur_s=20.0 -> idx 200, zone (50, 150) 밖
    assert is_in_overtaking_zone(s_m=20.0, waypoints_dist=0.1, zones=[(50, 150)]) is False


def test_in_overtaking_zone_with_no_zones():
    """zone 리스트가 비어있으면 항상 False."""
    assert is_in_overtaking_zone(s_m=10.0, waypoints_dist=0.1, zones=[]) is False


def test_in_overtaking_zone_with_multiple_zones():
    """여러 zone 중 하나만 매칭되어도 True."""
    zones = [(0, 50), (200, 300), (500, 600)]
    # cur_s=25.0 -> idx 250 -> (200, 300) 매칭
    assert is_in_overtaking_zone(s_m=25.0, waypoints_dist=0.1, zones=zones) is True
    # cur_s=40.0 -> idx 400 -> 어느 zone에도 안 속함
    assert is_in_overtaking_zone(s_m=40.0, waypoints_dist=0.1, zones=zones) is False


# ===========================================================================
# is_getting_closer
# ===========================================================================

@pytest.fixture
def default_closer_params():
    """필터링 비활성화 default — 원본 ENABLE_STATIC_SECTOR_FILTERING=False."""
    return GettingCloserParams(
        horizon_for_ttl=3.0,
        static_sector_filtering=False,
        track_length=200.0,
    )


def test_getting_closer_with_no_obstacle(default_closer_params):
    """장애물 없으면 False."""
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=None,
        params=default_closer_params,
    ) is False


def test_getting_closer_when_ego_faster(default_closer_params):
    """자차가 장애물보다 빠르면 다가오는 중 → True."""
    obs = FakeObstacle(s_start=15.0, vs=2.0, is_static=False)
    # cur_vs(5) - obs.vs(2) = 3 > -0.5 -> True
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=obs,
        params=default_closer_params,
    ) is True


def test_getting_closer_when_obstacle_faster(default_closer_params):
    """장애물이 자차보다 충분히 빠르면 멀어지는 중 → False."""
    obs = FakeObstacle(s_start=15.0, vs=10.0, is_static=False)
    # cur_vs(5) - obs.vs(10) = -5 > -0.5 -> False
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=obs,
        params=default_closer_params,
    ) is False


def test_getting_closer_static_sector_filtering_blocks_far_obstacle():
    """필터링 활성 + 정적 sector 장애물이 멀면 False (TTL 거리 초과)."""
    params = GettingCloserParams(
        horizon_for_ttl=3.0,
        static_sector_filtering=True,
        track_length=200.0,
    )
    obs = FakeObstacle(
        s_start=20.0,           # cur_s=10에서 10m 앞 (>3m)
        vs=0.0,
        is_static=True,
        in_static_obs_sector=True,
    )
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=obs,
        params=params,
    ) is False


def test_getting_closer_static_sector_filtering_passes_near_obstacle():
    """필터링 활성 + 정적 sector 장애물이 가까우면 정상 판정 (속도 기준)."""
    params = GettingCloserParams(
        horizon_for_ttl=3.0,
        static_sector_filtering=True,
        track_length=200.0,
    )
    obs = FakeObstacle(
        s_start=12.0,           # cur_s=10에서 2m 앞 (<3m)
        vs=0.0,
        is_static=True,
        in_static_obs_sector=True,
    )
    # 거리 통과 후 속도 체크: 5 - 0 = 5 > -0.5 -> True
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=obs,
        params=params,
    ) is True


def test_getting_closer_dynamic_obstacle_skips_distance_filter():
    """동적 장애물은 거리 필터를 건너뛰고 속도만 본다."""
    params = GettingCloserParams(
        horizon_for_ttl=3.0,
        static_sector_filtering=True,
        track_length=200.0,
    )
    obs = FakeObstacle(
        s_start=50.0,           # 멀리 있어도
        vs=0.0,
        is_static=False,        # 동적이라 필터 무시
        in_static_obs_sector=False,
    )
    assert is_getting_closer(
        cur_s=10.0, cur_vs=5.0,
        first_obstacle=obs,
        params=params,
    ) is True


# ===========================================================================
# Real WaypointData via for_test classmethod
#   ROS 의존성 없이 진짜 WaypointData 인스턴스 사용 — FakeWaypointData를 점진적으로 대체.
# ===========================================================================

def test_for_test_factory_creates_real_instance_without_ros():
    """WaypointData.for_test() 가 ROS 없이 인스턴스를 만들고 default 파라미터를 갖는다."""
    wd = WaypointData.for_test()

    # 식별자
    assert wd.name == "test_planner"
    assert wd.node_name == "/dyn_planners_statemachine/test_planner"
    # 데이터 빈 상태
    assert wd.list == []
    assert wd.array is None
    assert wd.is_init is False
    # 트랙 메타 default
    assert wd.is_closed is True
    assert wd.is_gb_track_wpnts is False
    assert wd.is_ot_wpnts is False
    # 결과 캐시 default
    assert wd.closest_target is None
    assert wd.closest_gap is None
    # 동적 파라미터 default 존재
    assert wd.min_horizon == 1.0
    assert wd.max_horizon == 30.0
    assert wd.lateral_width_m == 0.4
    # ROS 핸들 없음
    assert wd.dyn_sub is None


def test_for_test_param_overrides_take_effect():
    """for_test의 **kwargs로 default 덮어쓰기 가능."""
    wd = WaypointData.for_test(
        planner_name="dynamic_avoidance_planner",
        is_closed=False,
        max_horizon=20.0,
        lateral_width_m=0.6,
    )
    assert wd.name == "dynamic_avoidance_planner"
    assert wd.is_closed is False
    assert wd.max_horizon == 20.0
    assert wd.lateral_width_m == 0.6
    # 덮어쓰지 않은 필드는 default
    assert wd.min_horizon == 1.0


def test_real_waypoint_data_works_with_check_free_frenet(ego_at_10m, default_params):
    """진짜 WaypointData (for_test) 객체가 check_free_frenet에 그대로 들어간다.

    FakeWaypointData 없이 동일 시나리오 검증 — 미래에 모든 테스트를 진짜 클래스로 옮길 수 있음.
    """
    import numpy as np

    # 직선 트랙 데이터 직접 채워넣음
    n = 50
    array = np.array([[float(i), 0.0, float(i), 0.0] for i in range(n)])

    # FakeWpnt와 같은 인터페이스 (.d_m 만 있으면 됨)
    class _Wpnt:
        def __init__(self, d_m):
            self.d_m = d_m
    wpnt_list = [_Wpnt(d_m=0.0) for _ in range(n)]

    wd = WaypointData.for_test(
        planner_name="global_tracking",
        is_closed=True,
        is_gb_track_wpnts=True,
        max_horizon=30.0,
        lateral_width_m=0.4,
        free_scaling_reference_distance_m=5.0,
    )
    wd.list = wpnt_list
    wd.array = array
    wd.is_init = True

    # 5m 앞 정적 장애물
    obstacle = FakeObstacle(s_center=15.0, d_center=0.0, is_static=True)
    result = check_free_frenet(
        ego=ego_at_10m,
        waypoints=wd,
        obstacles=[obstacle],
        obstacle_predictions=[],
        obstacle_prediction_id=-1,
        params=default_params,
    )
    assert result.is_free is False
    assert result.closest_obstacle is obstacle


# ===========================================================================
# is_on_spline
# ===========================================================================

import numpy as np


def _make_straight_spline(length_m=50, ds=1.0):
    """xy=직선, s=거리, d=0 인 wpnt 데이터를 WaypointData.for_test에 넣는다."""
    n = int(length_m / ds)
    array = np.array([[i * ds, 0.0, i * ds, 0.0] for i in range(n)])

    class _Wpnt:
        def __init__(self, s_m, d_m=0.0):
            self.s_m = s_m
            self.d_m = d_m

    wpnt_list = [_Wpnt(s_m=i * ds) for i in range(n)]
    wd = WaypointData.for_test(planner_name="test", is_closed=True)
    wd.list = wpnt_list
    wd.array = array
    wd.is_init = True
    return wd


@pytest.fixture
def on_spline_params():
    return OnSplineParams(
        track_length=200.0,
        front_horizon_thres_m=5.0,
        min_dist_thres_m=0.5,
    )


def test_on_spline_uninitialized_returns_false(on_spline_params):
    """is_init=False 면 무조건 False."""
    wd = WaypointData.for_test()
    # is_init default False
    assert is_on_spline(
        cur_s=10.0,
        current_position_xy=np.array([10.0, 0.0]),
        waypoints=wd,
        params=on_spline_params,
    ) is False


def test_on_spline_passes_when_close_and_far_from_end(on_spline_params):
    """자차가 spline에 가깝고 (dist 작음) 끝점까지 충분히 남았으면 True."""
    wd = _make_straight_spline(length_m=50)
    # cur_s=10, last_s=49 -> gap=39 > 5 (front_horizon)
    # current_position=(10, 0) -> 가장 가까운 wpnt와 dist=0 < 0.5 (min_dist)
    assert is_on_spline(
        cur_s=10.0,
        current_position_xy=np.array([10.0, 0.0]),
        waypoints=wd,
        params=on_spline_params,
    ) is True


def test_on_spline_fails_when_too_close_to_end(on_spline_params):
    """spline 끝점에 너무 가까우면 False (gap < front_horizon_thres_m)."""
    wd = _make_straight_spline(length_m=50)
    # cur_s=47, last_s=49 -> gap=2 < 5 -> 실패
    assert is_on_spline(
        cur_s=47.0,
        current_position_xy=np.array([47.0, 0.0]),
        waypoints=wd,
        params=on_spline_params,
    ) is False


def test_on_spline_fails_when_too_far_from_path(on_spline_params):
    """자차가 spline에서 너무 떨어져 있으면 False (min_dist > min_dist_thres_m)."""
    wd = _make_straight_spline(length_m=50)
    # cur_s=10, current_position=(10, 1.0) -> 가장 가까운 wpnt와 dist=1.0 > 0.5 -> 실패
    assert is_on_spline(
        cur_s=10.0,
        current_position_xy=np.array([10.0, 1.0]),
        waypoints=wd,
        params=on_spline_params,
    ) is False


# ===========================================================================
# is_traj_msg_fresh
# ===========================================================================

class _FakeStamp:
    """builtin_interfaces/Time 흉내 — sec / nanosec 필드 (ROS2 포팅 후)."""
    def __init__(self, sec: float):
        # sec 인자를 ROS2 native 의 sec (int) + nanosec (uint32) 로 분해
        self.sec = int(sec)
        self.nanosec = int((sec - int(sec)) * 1e9)


class _FakeHeader:
    def __init__(self, stamp_sec: float):
        self.stamp = _FakeStamp(stamp_sec)


class _FakeTrajMsg:
    """f110_msgs/WpntArray (또는 OTWpntArray) 흉내."""
    def __init__(self, n_wpnts: int, stamp_sec: float):
        self.wpnts = [object()] * n_wpnts
        self.header = _FakeHeader(stamp_sec)


def test_traj_fresh_returns_false_when_msg_is_none():
    """메시지가 None이면 fresh False."""
    params = TrajFreshnessParams(is_smart_static=False, latest_threshold_sec=1.0)
    assert is_traj_msg_fresh(src_msg=None, now_sec=100.0, params=params) is False


def test_traj_fresh_returns_false_when_no_wpnts():
    """wpnts 리스트가 비어있으면 fresh False."""
    params = TrajFreshnessParams(is_smart_static=False, latest_threshold_sec=1.0)
    msg = _FakeTrajMsg(n_wpnts=0, stamp_sec=99.5)
    assert is_traj_msg_fresh(src_msg=msg, now_sec=100.0, params=params) is False


def test_traj_fresh_normal_planner_within_threshold():
    """일반 플래너: stamp 차이 < threshold 면 fresh."""
    params = TrajFreshnessParams(is_smart_static=False, latest_threshold_sec=1.0)
    msg = _FakeTrajMsg(n_wpnts=10, stamp_sec=99.5)  # 0.5s old
    assert is_traj_msg_fresh(src_msg=msg, now_sec=100.0, params=params) is True


def test_traj_fresh_normal_planner_over_threshold():
    """일반 플래너: stamp 차이 > threshold 면 stale."""
    params = TrajFreshnessParams(is_smart_static=False, latest_threshold_sec=1.0)
    msg = _FakeTrajMsg(n_wpnts=10, stamp_sec=98.0)  # 2.0s old
    assert is_traj_msg_fresh(src_msg=msg, now_sec=100.0, params=params) is False


def test_traj_fresh_smart_static_with_nonzero_stamp():
    """Smart Static: stamp가 0이 아니면 영원히 fresh (시간 지나도)."""
    params = TrajFreshnessParams(is_smart_static=True, latest_threshold_sec=1.0)
    msg = _FakeTrajMsg(n_wpnts=10, stamp_sec=10.0)  # 90s old
    assert is_traj_msg_fresh(src_msg=msg, now_sec=100.0, params=params) is True


def test_traj_fresh_smart_static_with_zero_stamp():
    """Smart Static: stamp 가 0이면 (초기화 안 됨) fresh False."""
    params = TrajFreshnessParams(is_smart_static=True, latest_threshold_sec=1.0)
    msg = _FakeTrajMsg(n_wpnts=10, stamp_sec=0.0)
    assert is_traj_msg_fresh(src_msg=msg, now_sec=100.0, params=params) is False
