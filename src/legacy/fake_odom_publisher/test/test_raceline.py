"""
fake_odom_publisher.raceline 의 순수 함수 단위 테스트.
ROS 의존 없이 pytest 만으로 검증.
"""
import math

import pytest

from fake_odom_publisher.raceline import (
    Pose3D,
    Waypoint,
    find_segment_index,
    interpolate_pose,
    waypoints_from_dicts,
    yaw_to_quaternion,
)


# ---------- 픽스처 ----------

@pytest.fixture
def straight_track():
    """일직선 평지 트랙: x 축으로 0 → 1 → 2 → 3 m, vx=2 m/s."""
    return [
        Waypoint(s_m=0.0, x_m=0.0, y_m=0.0, z_m=0.0, psi_rad=0.0, vx_mps=2.0),
        Waypoint(s_m=1.0, x_m=1.0, y_m=0.0, z_m=0.0, psi_rad=0.0, vx_mps=2.0),
        Waypoint(s_m=2.0, x_m=2.0, y_m=0.0, z_m=0.0, psi_rad=0.0, vx_mps=2.0),
        Waypoint(s_m=3.0, x_m=3.0, y_m=0.0, z_m=0.0, psi_rad=0.0, vx_mps=2.0),
    ]


@pytest.fixture
def slope_track():
    """20% 슬로프: x = s, z = 0.2 * s (s_3d ≠ s)."""
    return [
        Waypoint(s_m=0.0, x_m=0.0, y_m=0.0, z_m=0.0, psi_rad=0.0, vx_mps=10.0),
        Waypoint(s_m=1.0, x_m=1.0, y_m=0.0, z_m=0.2, psi_rad=0.0, vx_mps=10.0),
    ]


# ---------- find_segment_index ----------

def test_find_segment_index_zero(straight_track):
    assert find_segment_index(straight_track, 0.0) == 0


def test_find_segment_index_mid(straight_track):
    # s=0.5 는 [0,1] segment
    assert find_segment_index(straight_track, 0.5) == 0
    # s=1.5 는 [1,2] segment
    assert find_segment_index(straight_track, 1.5) == 1
    # s=2.7 는 [2,3] segment
    assert find_segment_index(straight_track, 2.7) == 2


def test_find_segment_index_at_or_past_end(straight_track):
    # 마지막 wpnt 이상이면 마지막 인덱스 반환 (run loop 가 wrap-around 처리)
    assert find_segment_index(straight_track, 3.0) == len(straight_track) - 1
    assert find_segment_index(straight_track, 100.0) == len(straight_track) - 1


# ---------- interpolate_pose ----------

def test_interpolate_at_first_wpnt(straight_track):
    pose = interpolate_pose(straight_track, 0.0, speed_scale=1.0, total_s=3.0)
    assert pose.x == pytest.approx(0.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.z == pytest.approx(0.0)
    assert pose.vx == pytest.approx(2.0)
    assert pose.vz == pytest.approx(0.0)  # 평지


def test_interpolate_midpoint_linear(straight_track):
    pose = interpolate_pose(straight_track, 0.5, speed_scale=1.0, total_s=3.0)
    assert pose.x == pytest.approx(0.5)


def test_interpolate_speed_scale(straight_track):
    pose = interpolate_pose(straight_track, 0.0, speed_scale=2.5, total_s=3.0)
    assert pose.vx == pytest.approx(5.0)


def test_interpolate_slope_vz(slope_track):
    """20% 슬로프, vx=10 → vz = 10 * 0.2 / sqrt(1+0.04) ≈ 1.961."""
    pose = interpolate_pose(slope_track, 0.0, speed_scale=1.0, total_s=1.0)
    expected_vz = 10.0 * 0.2 / math.sqrt(1.0 + 0.04)
    assert pose.vz == pytest.approx(expected_vz, rel=1e-6)


def test_interpolate_clamps_t(straight_track):
    """s_current 가 segment 의 정확히 끝에 있어도 t 가 [0,1] 안."""
    pose = interpolate_pose(straight_track, 1.0, speed_scale=1.0, total_s=3.0)
    # s=1.0 은 wpnt[1] 위치 → x=1.0
    assert pose.x == pytest.approx(1.0)


# ---------- yaw_to_quaternion ----------

def test_yaw_to_quaternion_zero():
    qx, qy, qz, qw = yaw_to_quaternion(0.0)
    assert (qx, qy, qz, qw) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_yaw_to_quaternion_half_pi():
    qx, qy, qz, qw = yaw_to_quaternion(math.pi / 2.0)
    assert qx == pytest.approx(0.0)
    assert qy == pytest.approx(0.0)
    assert qz == pytest.approx(math.sin(math.pi / 4.0))
    assert qw == pytest.approx(math.cos(math.pi / 4.0))


def test_yaw_to_quaternion_norm():
    """단위 quaternion 보장: |q| = 1."""
    for psi in [-math.pi, -1.0, 0.0, 0.7, math.pi]:
        qx, qy, qz, qw = yaw_to_quaternion(psi)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        assert norm == pytest.approx(1.0, rel=1e-9)


# ---------- waypoints_from_dicts ----------

def test_waypoints_from_dicts_minimal():
    dicts = [
        {"s_m": 0.0, "x_m": 1.0, "y_m": 2.0, "z_m": 0.3, "psi_rad": 0.5, "vx_mps": 4.0},
        {"s_m": 1.0, "x_m": 2.0, "y_m": 2.0, "z_m": 0.3, "psi_rad": 0.5, "vx_mps": 4.0},
    ]
    wpnts = waypoints_from_dicts(dicts)
    assert len(wpnts) == 2
    assert wpnts[0].x_m == 1.0
    assert wpnts[1].vx_mps == 4.0


def test_waypoints_from_dicts_ignores_extra_keys():
    """global_waypoints.json 에 d_left, kappa_radpm 등 추가 키가 있어도 동작."""
    dicts = [{
        "s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
        "psi_rad": 0.0, "vx_mps": 1.0, "kappa_radpm": 0.1, "d_left": 1.5,
    }]
    wpnts = waypoints_from_dicts(dicts)
    assert wpnts[0].vx_mps == 1.0
