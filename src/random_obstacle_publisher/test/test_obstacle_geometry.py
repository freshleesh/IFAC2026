"""obstacle_geometry 의 순수 함수 단위 테스트 — ROS 의존 없음."""
import math

import pytest
from numpy.random import default_rng

from random_obstacle_publisher.obstacle_geometry import (
    ObstacleSpec,
    WaypointSpec,
    build_sector_obstacles,
    generate_random_obstacle,
    get_closest_point_on_traj,
    select_obstacles_in_lookahead,
)


# ---------- 픽스처 ----------

@pytest.fixture
def linear_track():
    """0~10 m 트랙. d_left=1.0, d_right=1.0 균등 폭."""
    return [
        WaypointSpec(id=i, s_m=float(i), d_left=1.0, d_right=1.0)
        for i in range(11)  # s = 0,1,...,10
    ]


@pytest.fixture
def asymmetric_track():
    """s 에 따라 좌우 폭이 달라지는 트랙."""
    return [
        WaypointSpec(id=0, s_m=0.0, d_left=2.0, d_right=0.5),
        WaypointSpec(id=1, s_m=5.0, d_left=0.5, d_right=2.0),
        WaypointSpec(id=2, s_m=10.0, d_left=1.5, d_right=1.5),
    ]


# ---------- get_closest_point_on_traj ----------

def test_closest_at_exact_match(linear_track):
    assert get_closest_point_on_traj(0.0, linear_track) == 0
    assert get_closest_point_on_traj(5.0, linear_track) == 5


def test_closest_at_in_between(linear_track):
    """원본 알고리즘은 ≤ 비교가 아니라 < 비교 — 동률일 때 먼저 만난 것 유지."""
    # s=3.4 → 3 이 더 가까움 (|3.4-3|² < |3.4-4|²)
    assert get_closest_point_on_traj(3.4, linear_track) == 3
    # s=3.6 → 4 가 더 가까움
    assert get_closest_point_on_traj(3.6, linear_track) == 4


def test_closest_at_outside_range(linear_track):
    # 음수 s → 0 이 가장 가까움
    assert get_closest_point_on_traj(-2.0, linear_track) == 0
    # 트랙 끝보다 큰 s → 마지막 wpnt
    assert get_closest_point_on_traj(15.0, linear_track) == 10


# ---------- generate_random_obstacle ----------

def test_generate_obstacle_within_sector(linear_track):
    """obstacle.s_start 가 [s_start, s_end] 안에 있어야 함."""
    gen = default_rng(42)
    ob = generate_random_obstacle(
        obs_id=3, s_start=2.0, s_end=4.0, gen=gen,
        obstacle_length=0.3, obstacle_width=0.2,
        obstacle_max_d_from_traj=1.0, gb_wpnts=linear_track,
    )
    assert ob.id == 3
    assert 2.0 <= ob.s_start <= 4.0
    assert ob.s_end == pytest.approx(ob.s_start + 0.3)
    assert not ob.is_actually_a_gap


def test_generate_obstacle_d_width(linear_track):
    """d_left - d_right 는 obstacle_width 와 정확히 일치."""
    gen = default_rng(7)
    ob = generate_random_obstacle(
        obs_id=0, s_start=1.0, s_end=2.0, gen=gen,
        obstacle_length=0.3, obstacle_width=0.4,
        obstacle_max_d_from_traj=1.0, gb_wpnts=linear_track,
    )
    assert ob.d_left - ob.d_right == pytest.approx(0.4)


def test_generate_obstacle_respects_max_d(asymmetric_track):
    """obstacle_max_d_from_traj 가 트랙 폭보다 작으면 obstacle 이 그 안으로 clamp."""
    gen = default_rng(1)
    # asymmetric s=0 wpnt: d_left=2.0, d_right=0.5 → max_d=0.3 으로 강제 좁히면
    # track_right = -min(0.5, 0.3) = -0.3
    # track_left  = min(2.0 - 0.2, 0.3) = 0.3
    ob = generate_random_obstacle(
        obs_id=0, s_start=0.0, s_end=0.5, gen=gen,
        obstacle_length=0.3, obstacle_width=0.2,
        obstacle_max_d_from_traj=0.3, gb_wpnts=asymmetric_track,
    )
    assert -0.3 <= ob.d_right <= 0.3
    assert -0.3 + 0.2 <= ob.d_left <= 0.3 + 0.2


def test_generate_obstacle_deterministic_with_seed(linear_track):
    """같은 seed 로 두 번 호출하면 결과 동일."""
    gen1 = default_rng(123)
    gen2 = default_rng(123)
    ob1 = generate_random_obstacle(
        obs_id=0, s_start=1.0, s_end=3.0, gen=gen1,
        obstacle_length=0.3, obstacle_width=0.2,
        obstacle_max_d_from_traj=1.0, gb_wpnts=linear_track,
    )
    ob2 = generate_random_obstacle(
        obs_id=0, s_start=1.0, s_end=3.0, gen=gen2,
        obstacle_length=0.3, obstacle_width=0.2,
        obstacle_max_d_from_traj=1.0, gb_wpnts=linear_track,
    )
    assert ob1.s_start == ob2.s_start
    assert ob1.d_right == ob2.d_right


# ---------- build_sector_obstacles ----------

def test_build_n_sectors_count(linear_track):
    gen = default_rng(0)
    obs = build_sector_obstacles(
        n_sectors=4, gb_wpnts=linear_track, gen=gen,
        obstacle_length=0.3, obstacle_width=0.2, obstacle_max_d_from_traj=1.0,
    )
    assert len(obs) == 4


def test_build_sector_ids_sequential(linear_track):
    gen = default_rng(0)
    obs = build_sector_obstacles(
        n_sectors=5, gb_wpnts=linear_track, gen=gen,
        obstacle_length=0.3, obstacle_width=0.2, obstacle_max_d_from_traj=1.0,
    )
    assert [o.id for o in obs] == [0, 1, 2, 3, 4]


def test_build_sector_obstacles_within_track(linear_track):
    """모든 obstacle 이 트랙 (0, final_s) 안에 들어가야 함."""
    gen = default_rng(0)
    final_s = linear_track[-1].s_m
    obs = build_sector_obstacles(
        n_sectors=8, gb_wpnts=linear_track, gen=gen,
        obstacle_length=0.3, obstacle_width=0.2, obstacle_max_d_from_traj=1.0,
    )
    for ob in obs:
        # 각 sector spacing = 10/8 = 1.25, margin = max(0.5, 0.3) = 0.5
        # 각 sector 의 s_end = sec * 1.25 + 1.25 - 0.5 = sec * 1.25 + 0.75
        # obstacle.s_start ≤ sec*1.25 + 0.75, s_end = s_start + 0.3 → 최대 sec*1.25 + 1.05
        # 마지막 sector(7) 의 최대값 = 9.8 < 10 (final_s)
        assert ob.s_start >= 0.0
        assert ob.s_end < final_s + 0.3  # 다소 여유


# ---------- select_obstacles_in_lookahead ----------

def test_lookahead_selects_close_only():
    """current_s=2.0, lookahead=3.0, final_s=20.0 → s_start ∈ (2.0, 5.0+1) 만."""
    obs = [
        ObstacleSpec(id=0, s_start=1.0, s_end=1.3, d_left=0, d_right=0),  # 너무 뒤
        ObstacleSpec(id=1, s_start=3.5, s_end=3.8, d_left=0, d_right=0),  # 안 (5.0-3.5=1.5)
        ObstacleSpec(id=2, s_start=4.5, s_end=4.8, d_left=0, d_right=0),  # 안 (5.0-4.5=0.5)
        ObstacleSpec(id=3, s_start=10.0, s_end=10.3, d_left=0, d_right=0),  # 너무 멀어
    ]
    selected = select_obstacles_in_lookahead(
        obs, current_s=2.0, lookahead_distance=3.0, final_s=20.0
    )
    ids = [o.id for o in selected]
    # 원본의 dist = fmod(2 + 3, 20) - s_start = 5 - s_start
    # 0 < dist < lookahead+1 = 4 인 것: id=1 (dist=1.5), id=2 (dist=0.5)
    assert 1 in ids
    assert 2 in ids
    assert 0 not in ids
    assert 3 not in ids


def test_lookahead_empty_when_all_passed():
    """모든 obstacle 이 current 보다 뒤에 있으면 아무것도 선택 안 됨 (wrap 처리 X)."""
    obs = [ObstacleSpec(id=0, s_start=15.0, s_end=15.3, d_left=0, d_right=0)]
    selected = select_obstacles_in_lookahead(
        obs, current_s=18.0, lookahead_distance=3.0, final_s=20.0
    )
    # dist = fmod(18+3, 20) - 15 = 1 - 15 = -14 → 음수 → 선택 안 됨
    assert selected == []
