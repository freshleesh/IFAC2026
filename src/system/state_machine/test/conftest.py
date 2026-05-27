"""Shared pytest fixtures for state_machine tests (ROS2 port).

원본 ROS1 conftest 는 sys.modules 로 rospy / dynamic_reconfigure mock 했으나,
ROS2 포팅에서는 path_checker / state_transitions 의 rospy → logging 변경으로
mock 불필요. f110_msgs 만 ROS2 install 에서 import.

state_machine 패키지는 ament_python — 정상 설치 후 `from state_machine.path_checker
import ...` 로 직접 import 가능. sys.path 조작 불필요.
"""
import os
import sys

import pytest

# fake_msgs.py import 위해 test 디렉터리만 path 에 추가
_TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)


# ---------------------------------------------------------------------------
# 공용 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_track_wpnts():
    """0~50m 직선 트랙 (closed=True 가정)."""
    from fake_msgs import make_straight_track
    return make_straight_track(length_m=50, ds=1.0)


@pytest.fixture
def default_params():
    """일반적인 f1tenth 차량 파라미터."""
    from state_machine.path_checker import FrenetCheckParams
    return FrenetCheckParams(
        max_s=200.0,
        veh_length=0.5,
        ego_width=0.3,
    )


@pytest.fixture
def ego_at_10m():
    """진행거리 10m, 속도 5 m/s 인 자차."""
    from state_machine.path_checker import EgoFrenetState
    return EgoFrenetState(s=10.0, vs=5.0)
