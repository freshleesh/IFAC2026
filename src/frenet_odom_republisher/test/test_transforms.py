"""quaternion_to_yaw 단위 테스트 — ROS 의존 없음."""
import math

import pytest

from frenet_odom_republisher.transforms import quaternion_to_yaw


def test_identity_quaternion_yaw_zero():
    """단위 quaternion (0,0,0,1) → yaw=0."""
    yaw = quaternion_to_yaw(0.0, 0.0, 0.0, 1.0)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_yaw_pi_over_2():
    """yaw=π/2 quaternion (0, 0, sin(π/4), cos(π/4)) → yaw=π/2."""
    qz = math.sin(math.pi / 4.0)
    qw = math.cos(math.pi / 4.0)
    yaw = quaternion_to_yaw(0.0, 0.0, qz, qw)
    assert yaw == pytest.approx(math.pi / 2.0, abs=1e-9)


def test_yaw_negative_pi_over_2():
    """yaw=-π/2 → quaternion (0, 0, -sin(π/4), cos(π/4))."""
    qz = -math.sin(math.pi / 4.0)
    qw = math.cos(math.pi / 4.0)
    yaw = quaternion_to_yaw(0.0, 0.0, qz, qw)
    assert yaw == pytest.approx(-math.pi / 2.0, abs=1e-9)


def test_yaw_pi():
    """yaw=π → quaternion (0, 0, ±1, 0). atan2 가 +π 또는 -π 를 반환."""
    yaw = quaternion_to_yaw(0.0, 0.0, 1.0, 0.0)
    assert abs(yaw) == pytest.approx(math.pi, abs=1e-9)


def test_yaw_for_arbitrary():
    """yaw=0.7 rad."""
    psi = 0.7
    qz = math.sin(psi / 2.0)
    qw = math.cos(psi / 2.0)
    yaw = quaternion_to_yaw(0.0, 0.0, qz, qw)
    assert yaw == pytest.approx(psi, abs=1e-9)


def test_round_trip_with_yaw_to_quaternion():
    """B-1 fake_odom_publisher 의 yaw_to_quaternion 와 round-trip 확인 (가능하면)."""
    # 직접 계산 (외부 의존 회피)
    for psi in [-2.5, -1.0, 0.0, 0.3, 1.5, 2.9]:
        qx, qy, qz, qw = 0.0, 0.0, math.sin(psi / 2.0), math.cos(psi / 2.0)
        yaw = quaternion_to_yaw(qx, qy, qz, qw)
        # ±π 경계 처리: yaw 와 psi 가 2π 차이일 수 있음
        diff = (yaw - psi + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-9
