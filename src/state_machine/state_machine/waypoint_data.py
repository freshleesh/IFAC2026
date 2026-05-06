"""WaypointData — state_machine 노드들이 공유하는 planner-별 wpnt 컨테이너.

원본 ROS1: state_machine/src/waypoint_data.py.

ROS2 포팅 (C-2):
- `for_test` classmethod 는 그대로 (ROS 의존 없음).
- `__init__` 의 ROS Subscriber + get_param 호출은 stub 처리 — C-3 에서 메인 노드와
  통합 시 ROS2 native parameter (declare_parameter + add_on_set_parameters_callback)
  로 다시 짠다. C-2 는 단위 테스트 통과만 목적.
- `dynamic_reconfigure.msg.Config` import 제거 (ROS2 에 없음, ROS2 native 가
  별도 msg 안 받음).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class WaypointData:
    """플래너 한 개의 wpnt 데이터 + 동적 파라미터 + 결과 캐시 컨테이너.

    필드 묶음 (역사적 이유로 한 클래스에 모여있음):
        - wpnt 데이터: list, array, stamp, is_init
        - 트랙 메타: is_closed, is_gb_track_wpnts, is_ot_wpnts
        - 외부 함수가 적어두는 결과 캐시: closest_target, closest_gap
        - 동적 파라미터: min_horizon, max_horizon, lateral_width_m, ...
        - MPC 출처 표식: from_mpc

    생성자는 ROS 노드와 연결 (declare_parameter) 에 의존하므로, 테스트 환경에서는
    `for_test` 사용. C-2 시점에서는 __init__ 이 stub 라 실제 노드와 미통합.
    """

    def __init__(self, planner_name: str, is_closed: bool, node: Any | None = None):
        """
        Args:
            planner_name: planner 식별자 (예: 'global_traj', 'dynamic_avoidance_planner')
            is_closed: 트랙이 closed loop 인지
            node: ROS2 rclpy.Node 인스턴스 (옵셔널). 주어지면 declare_parameter
                  + add_on_set_parameters_callback 로 동적 파라미터 갱신 활성.
                  None 이면 default 값으로 채우고 ROS 와 미통합 (C-3 에서 보강).
        """
        self.name = planner_name
        self.node_name = "/dyn_planners_statemachine/" + self.name
        self.list = []
        self.array = None
        self.stamp = None
        self.is_init = False
        self.is_gb_track_wpnts = False
        self.is_ot_wpnts = False
        self.closest_target = None
        self.closest_gap = None
        self.is_closed = is_closed
        self.vel_planner_safety_factor = 1.0
        # ### HJ : Phase X (refactored) — provenance tag. When the last
        # initialize_traj came from the unified MPC topic, set this True
        # so get_splini_wpts / get_recovery_wpts can choose NOT to pad
        # with GB waypoints (user directive 2026-04-24 "mpc 출력 그대로").
        self.from_mpc = False
        self._node = node

        # 동적 파라미터 default — 실 ROS2 wiring 은 attach_to_node 에서 갱신.
        self._set_default_params()
        if node is not None:
            self.attach_to_node(node)

    def _set_default_params(self) -> None:
        self.min_horizon = 1.0
        self.max_horizon = 30.0
        self.lateral_width_m = 0.4
        self.free_scaling_reference_distance_m = 5.0
        self.latest_threshold = 1.0
        self.on_spline_front_horizon_thres_m = 5.0
        self.on_spline_min_dist_thres_m = 0.5
        self.hyst_timer_sec = 1.0
        self.killing_timer_sec = 3.0

    def attach_to_node(self, node: Any) -> None:
        """C-3 에서 구현: 노드의 declare_parameter / add_on_set_parameters_callback
        으로 동적 파라미터 wiring. C-2 단위 테스트에서는 호출되지 않는다."""
        # TODO C-3: ROS2 native parameter callback 으로 self.{min_horizon, max_horizon, ...} 갱신
        pass

    def initialize_traj(self, wpnt) -> None:
        """들어온 trajectory 메시지를 list / array / stamp 로 변환 + is_init=True."""
        if len(wpnt.wpnts) != 0:
            self.stamp = wpnt.header.stamp
            self.list = wpnt.wpnts
            self.array = np.array(
                [[w.x_m, w.y_m, w.s_m, w.d_m] for w in wpnt.wpnts]
            )
            self.is_init = True

    # =========================================================================
    # Test-only entry point
    # =========================================================================

    @classmethod
    def for_test(cls, planner_name="test_planner", is_closed=True, **param_overrides):
        """ROS 의존성 없이 인스턴스 생성. pytest용.

        __init__ 우회로 ROS 호출을 피한다. 모든 동적 파라미터는 합리적 default 로
        채우고, **param_overrides 로 개별 덮어쓰기 가능.

        예:
            wd = WaypointData.for_test(
                planner_name="dynamic_avoidance_planner",
                max_horizon=20.0,
                lateral_width_m=0.5,
            )
        """
        obj = cls.__new__(cls)  # __init__ 우회 (ROS 호출 없음)
        obj.name = planner_name
        obj.node_name = "/dyn_planners_statemachine/" + planner_name
        obj.list = []
        obj.array = None
        obj.stamp = None
        obj.is_init = False
        obj.is_gb_track_wpnts = False
        obj.is_ot_wpnts = False
        obj.is_closed = is_closed
        obj.closest_target = None
        obj.closest_gap = None
        obj.vel_planner_safety_factor = 1.0
        obj.from_mpc = False
        obj._node = None
        # ROS subscription 핸들 — ROS2 에서는 의미 없는 legacy 필드지만 테스트 호환 위해 유지
        obj.dyn_sub = None
        # 동적 파라미터 default
        obj.min_horizon = 1.0
        obj.max_horizon = 30.0
        obj.lateral_width_m = 0.4
        obj.free_scaling_reference_distance_m = 5.0
        obj.latest_threshold = 1.0
        obj.on_spline_front_horizon_thres_m = 5.0
        obj.on_spline_min_dist_thres_m = 0.5
        obj.hyst_timer_sec = 1.0
        obj.killing_timer_sec = 3.0
        for key, val in param_overrides.items():
            setattr(obj, key, val)
        return obj
