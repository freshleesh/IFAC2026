"""
FrenetConverter 단위 테스트 — ROS 의존 없음.

핵심 회귀:
- 직선 트랙: get_cartesian(s,0) 가 정확히 (s,0) 위치
- get_frenet(get_cartesian) 의 round-trip
- 3D slope 트랙: build_raceline 의 s 가 평면 거리가 아니라 3D 거리
- d 음/양 부호: 좌측이 양 (perp = (-tangent_y, tangent_x))
- get_approx_s_3d_with_idx 가 (s, idx) 둘 다 반환
"""
import math

import numpy as np
import pytest

from frenet_conversion.frenet_converter import FrenetConverter


# ---------- 픽스처 ----------

@pytest.fixture
def straight_track():
    """x 축 따라 평지 100 wpnts (s=0..99)."""
    x = np.linspace(0.0, 99.0, 100)
    y = np.zeros(100)
    z = np.zeros(100)
    return FrenetConverter(x, y, z)


@pytest.fixture
def slope_track():
    """20% 슬로프 (z = 0.2 x). s 는 평면 거리가 아니라 3D 거리."""
    x = np.linspace(0.0, 10.0, 21)
    y = np.zeros(21)
    z = 0.2 * x
    return FrenetConverter(x, y, z)


@pytest.fixture
def curved_track():
    """반지름 10 m 원호 (1 사분면 90 도)."""
    n = 91
    theta = np.linspace(0.0, np.pi / 2.0, n)
    x = 10.0 * np.cos(theta)
    y = 10.0 * np.sin(theta)
    z = np.zeros(n)
    return FrenetConverter(x, y, z)


# ---------- build_raceline ----------

def test_build_raceline_straight(straight_track):
    assert straight_track.raceline_length == pytest.approx(99.0, rel=1e-9)
    # 균등 spacing → median 도 1.0
    assert straight_track.waypoints_distance_m == pytest.approx(1.0, rel=1e-9)


def test_build_raceline_3d_arc_length(slope_track):
    """3D 거리: dx=0.5, dz=0.1 → 한 wpnt 당 sqrt(0.5²+0.1²) ≈ 0.5099."""
    expected_segment = math.sqrt(0.5 ** 2 + 0.1 ** 2)
    assert slope_track.waypoints_distance_m == pytest.approx(expected_segment, rel=1e-9)
    assert slope_track.raceline_length == pytest.approx(20 * expected_segment, rel=1e-9)


def test_build_raceline_psi_mu(slope_track):
    """20% 슬로프 → mu = arctan(0.2) 균일."""
    expected_mu = math.atan(0.2)
    # 끝점은 spline 미분 정확도 떨어짐 → 중간만
    assert slope_track.waypoints_mu[10] == pytest.approx(expected_mu, abs=1e-3)


# ---------- get_cartesian ----------

def test_get_cartesian_zero_d_straight(straight_track):
    xy = straight_track.get_cartesian(np.array([5.0]), np.array([0.0]))
    assert xy[0][0] == pytest.approx(5.0, abs=1e-6)
    assert xy[1][0] == pytest.approx(0.0, abs=1e-6)


def test_get_cartesian_d_offset_left_is_positive_y(straight_track):
    """좌측이 양 d (perp = (-tangent_y, tangent_x), tangent=(1,0) → perp=(0,1))."""
    xy = straight_track.get_cartesian(np.array([5.0]), np.array([1.0]))
    assert xy[0][0] == pytest.approx(5.0, abs=1e-6)
    assert xy[1][0] == pytest.approx(1.0, abs=1e-6)


def test_get_cartesian_3d_z_from_spline(slope_track):
    """20% 슬로프에서 s = 0.5099 → x≈0.5, z≈0.1."""
    s = slope_track.waypoints_s[1]  # 첫 segment 끝
    xyz = slope_track.get_cartesian_3d(np.array([s]), np.array([0.0]))
    assert xyz[0][0] == pytest.approx(0.5, abs=1e-6)
    assert xyz[2][0] == pytest.approx(0.1, abs=1e-6)


# ---------- get_frenet round-trip ----------

def test_frenet_round_trip_straight_zero_d(straight_track):
    """(s, 0) → cartesian → frenet 다시. round-trip 오차 < 1mm."""
    s_input = np.array([3.7, 50.5, 90.0])
    d_input = np.array([0.0, 0.0, 0.0])
    xy = straight_track.get_cartesian(s_input, d_input)
    sd = straight_track.get_frenet(xy[0], xy[1])
    assert sd[0] == pytest.approx(s_input, abs=1e-3)
    assert sd[1] == pytest.approx(d_input, abs=1e-3)


def test_frenet_round_trip_straight_nonzero_d(straight_track):
    s_input = np.array([20.0, 60.0])
    d_input = np.array([0.5, -0.3])
    xy = straight_track.get_cartesian(s_input, d_input)
    sd = straight_track.get_frenet(xy[0], xy[1])
    assert sd[0] == pytest.approx(s_input, abs=1e-3)
    assert sd[1] == pytest.approx(d_input, abs=1e-3)


def test_frenet_3d_round_trip(slope_track):
    """3D 슬로프에서 round-trip — get_frenet_3d 로 z 까지 사용."""
    s_input = np.array([2.0, 5.0])
    d_input = np.array([0.0, 0.0])
    xyz = slope_track.get_cartesian_3d(s_input, d_input)
    sd = slope_track.get_frenet_3d(xyz[0], xyz[1], xyz[2])
    assert sd[0] == pytest.approx(s_input, abs=1e-2)
    assert sd[1] == pytest.approx(d_input, abs=1e-2)


# ---------- 헬퍼 ----------

def test_get_approx_s_2d(straight_track):
    s = straight_track.get_approx_s(np.array([3.4, 9.0]), np.array([0.0, 0.0]))
    # waypoint id 3 (s=3.0) 이 가장 가까움 (3.4 → 3 보다 4 까지 0.6, 3 까지 0.4)
    # waypoint id 9 (s=9.0)
    assert s[0] == pytest.approx(3.0)
    assert s[1] == pytest.approx(9.0)


def test_get_approx_s_3d_with_idx(straight_track):
    s, idx = straight_track.get_approx_s_3d_with_idx(
        np.array([3.4]), np.array([0.0]), np.array([0.0])
    )
    assert idx[0] == 3
    assert s[0] == pytest.approx(3.0)


# ---------- 트랙 boundary ----------

def test_set_track_bounds_disabled_by_default(straight_track):
    assert straight_track.has_track_bounds is False
    # 호출 시 False 반환 (단축 평가)
    assert straight_track._is_line_crossing_boundary(0, 0, 1, 0, 0) is False


def test_set_track_bounds_enables_check(straight_track):
    """좌측 (y=+1) / 우측 (y=-1) 평행 벽 + 트랙을 가로지르는 선분."""
    left = [[i, 1.0, 0.0] for i in range(10)]
    right = [[i, -1.0, 0.0] for i in range(10)]
    straight_track.set_track_bounds(left, right)
    assert straight_track.has_track_bounds is True

    # (0, -2) → (0, 2) — 양쪽 벽 모두 가로지름
    assert straight_track._is_line_crossing_boundary(0.5, -2.0, 0.5, 2.0, 0.0) is True
    # (0, 0) → (5, 0) — 트랙 내부, 가로지르지 않음
    assert straight_track._is_line_crossing_boundary(0, 0, 5, 0, 0) is False


# ---------- 곡선 트랙 (회귀) ----------

def test_curved_track_round_trip(curved_track):
    """원호 트랙에서 (s, 0) → cartesian → frenet round-trip."""
    s_input = np.array([5.0, 10.0])  # 두 임의 s
    d_input = np.array([0.0, 0.0])
    xy = curved_track.get_cartesian(s_input, d_input)
    sd = curved_track.get_frenet(xy[0], xy[1])
    assert sd[0] == pytest.approx(s_input, abs=1e-2)
    assert sd[1] == pytest.approx(d_input, abs=1e-2)


# ---------- 헤딩 오차 ----------

def test_e_psi_zero_when_aligned(straight_track):
    """직선 트랙 위에서 yaw=0 → e_psi=0."""
    e_psi = straight_track.get_e_psi(5.0, 0.0, 0.0)
    assert e_psi == pytest.approx(0.0, abs=1e-3)


def test_e_psi_pi_over_2_when_perpendicular(straight_track):
    """직선 트랙 위에서 yaw=π/2 → e_psi=π/2."""
    e_psi = straight_track.get_e_psi(5.0, 0.0, np.pi / 2.0)
    assert e_psi == pytest.approx(np.pi / 2.0, abs=1e-3)
