"""Minimal fake message classes for unit-testing pure path-check functions.

실제 ROS 메시지 (f110_msgs/Wpnt, Obstacle, PredictionStep, WaypointData) 는 ROS 환경 없이는
인스턴스화가 어렵다. 테스트에서는 path_checker가 실제로 읽는 필드만 갖춘 dataclass로 대체한다.
필드 추가가 필요해지면 여기서만 수정하면 된다.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FakeWpnt:
    """f110_msgs/Wpnt 흉내 — path_checker가 .d_m 만 사용."""
    x_m: float = 0.0
    y_m: float = 0.0
    s_m: float = 0.0
    d_m: float = 0.0
    z_m: float = 0.0


@dataclass
class FakeObstacle:
    """f110_msgs/Obstacle 흉내 — path_checker가 읽는 필드만."""
    s_center: float = 0.0
    d_center: float = 0.0
    is_static: bool = True
    size: float = 0.3                  # 장애물 직경 [m]
    vs: float = 0.0                    # 종방향 속도 [m/s]
    id: int = 0
    s_start: float = 0.0               # 장애물 진입 s [m] (is_getting_closer용)
    in_static_obs_sector: bool = False # 정적 sector 안 장애물 여부


@dataclass
class FakePrediction:
    """f110_msgs/PredictionStep 흉내."""
    pred_s: float = 0.0
    pred_d: float = 0.0


@dataclass
class FakeWaypointData:
    """state_machine.WaypointData 흉내.

    실제 클래스는 생성자에서 ROS Subscriber를 만들기 때문에 테스트에서 사용 불가.
    pure function이 의존하는 필드만 dataclass로 노출한다.
    """
    list: List[FakeWpnt] = field(default_factory=list)
    array: Optional[np.ndarray] = None
    is_init: bool = True
    is_closed: bool = True
    is_gb_track_wpnts: bool = True
    is_ot_wpnts: bool = False
    min_horizon: float = 1.0
    max_horizon: float = 30.0
    lateral_width_m: float = 0.4
    free_scaling_reference_distance_m: float = 5.0
    closest_target: Optional[FakeObstacle] = None
    closest_gap: Optional[float] = None


def make_straight_track(length_m: int = 50, ds: float = 1.0) -> FakeWaypointData:
    """직선 트랙 (d=0) wpnt 데이터 생성. column = [x, y, s, d]."""
    n = int(length_m / ds)
    wpnts = [FakeWpnt(x_m=i * ds, y_m=0.0, s_m=i * ds, d_m=0.0) for i in range(n)]
    array = np.array([[w.x_m, w.y_m, w.s_m, w.d_m] for w in wpnts])
    return FakeWaypointData(list=wpnts, array=array)
