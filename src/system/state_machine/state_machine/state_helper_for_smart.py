#!/usr/bin/env python3
"""SmartStaticChecker — Fixed Frenet 좌표계 기반 state checker.

상속 + composition 혼합 패턴:
- StateMachine 을 상속해 모든 check 함수 (`_check_close_to_raceline`, `_check_free_frenet`,
  `_check_overtaking_mode_sustainability` 등) 를 그대로 사용.
- self.cur_s / cur_d / obstacles / cur_gb_wpnts 등 일부 attribute 만 Fixed Frenet
  데이터로 override (생성 시 + `update()` 매 iteration 에서).
- 그 외 attribute 는 모두 `__getattr__` 으로 parent 에서 동적으로 fallback.

Usage:
    checker = SmartStaticChecker(state_machine)
    checker.update()
    close = checker._check_close_to_raceline()

ROS2 포팅 (C-3): rospy.Subscriber → parent 노드의 create_subscription 위임.
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from nav_msgs.msg import Odometry

if TYPE_CHECKING:
    from state_machine.state_machine_node import StateMachine

# parent 가 만들어진 후 base class 도 import 가능. 단 import 자체가 lazy 이므로
# 클래스 정의 시점엔 parent 모듈 로딩 안 됨 (TYPE_CHECKING).

_logger = logging.getLogger(__name__)


# 베이스는 메인 노드의 StateMachine. 다중 상속 회피 위해 동적 import 후 클래스 합성도
# 가능하지만 원본 패턴 (직접 상속) 유지. C-4 에서 메인 노드 정의 후에만 import 가능.
def _get_state_machine_base():
    """state_machine_node 의 StateMachine class 를 lazy import — circular 회피."""
    from state_machine.state_machine_node import StateMachine
    return StateMachine


class SmartStaticChecker:
    """StateMachine 의 check 함수를 Fixed Frenet 좌표계로 재사용.

    원본은 StateMachine 직접 상속. ROS2 포팅에서는 lazy 상속 패턴 유지 어렵고
    runtime 에 클래스 합성하는 게 부담 → __getattr__ 만으로 충분히 동작 (모든
    StateMachine 메서드 / attribute 가 fallback).
    """

    def __init__(self, parent_state_machine: "StateMachine"):
        """parent (메인 StateMachine) 를 참조로 들고 Fixed Frenet override 만 set.

        주의: 부모 `StateMachine.__init__` 호출 안 함 (ROS 노드 중복 init 방지).
        모든 attribute / method 는 `__getattr__` 으로 fallback.
        """
        self.__dict__["parent"] = parent_state_machine

        # ── Fixed Frenet 전용 override ──
        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0
        self.cur_vd = 0.0

        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []

        self.static_overtaking_mode = False

        self.cur_gb_wpnts = parent_state_machine.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = 0
        self.waypoints_dist = 0.0

        # Fixed Frenet odom subscription — parent 노드에 위임
        # (rospy.Subscriber → parent.create_subscription)
        parent_state_machine.create_subscription(
            Odometry,
            "/car_state/odom_frenet_fixed",
            self._odom_fixed_cb,
            10,
        )
        _logger.info("[SmartStaticChecker] Initialized with Fixed Frenet odom subscription")

    def __getattr__(self, name):
        """self / 클래스에 없는 attribute 는 parent 에서 자동 fallback.

        - cur_s / cur_d 등 helper 가 직접 set 한 attribute 는 helper 의 것 사용
        - StateMachine 의 메서드들은 parent attr 로 fallback
        - 'parent' 자체 접근은 __dict__ hit 라 fallback 트리거 안 됨 (무한재귀 X)
        """
        if name == "parent":
            raise AttributeError("SmartStaticChecker has no 'parent' yet")
        parent = object.__getattribute__(self, "parent")
        return getattr(parent, name)

    def _odom_fixed_cb(self, data):
        """Fixed Frenet odom callback — helper 자체의 cur_s/d/vs/vd override."""
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        self.cur_vd = data.twist.twist.linear.y

    def update(self):
        """매 iteration parent.update_waypoints() 가 동기적으로 호출.

        (1) Smart Static path 메타데이터 갱신
        (2) parent 의 obstacles 를 Fixed Frenet 좌표로 변환
        (3) interest_horizon_m 안의 obstacles 만 추려 obstacles_in_interest 에 저장
        """
        if len(self.parent.cur_smart_static_avoidance_wpnts.list) == 0:
            self.num_glb_wpnts = 0
            self.obstacles = []
            self.obstacles_in_interest = []
            self.cur_obstacles_in_interest = []
            return

        self.cur_gb_wpnts = self.parent.cur_smart_static_avoidance_wpnts
        self.num_glb_wpnts = len(self.cur_gb_wpnts.list)
        self.track_length = self.cur_gb_wpnts.list[-1].s_m
        self.waypoints_dist = self.track_length / self.num_glb_wpnts
        self.max_s = self.track_length

        self.obstacles = []
        for obs in self.parent.obstacles:
            obs_copy = copy.copy(obs)
            obs_copy.s_start = obs.s_start_fixed
            obs_copy.s_end = obs.s_end_fixed
            obs_copy.s_center = obs.s_center_fixed
            obs_copy.d_center = obs.d_center_fixed
            obs_copy.d_right = obs.d_right_fixed
            obs_copy.d_left = obs.d_left_fixed
            obs_copy.vs = obs.vs_fixed
            obs_copy.vd = obs.vd_fixed
            obs_copy.s_var = obs.s_var_fixed
            obs_copy.d_var = obs.d_var_fixed
            obs_copy.vs_var = obs.vs_var_fixed
            obs_copy.vd_var = obs.vd_var_fixed
            self.obstacles.append(obs_copy)

        self._update_obstacles_in_interest()

    def _update_obstacles_in_interest(self):
        """Fixed Frenet 기반 obstacles_in_interest 필터링."""
        obstacles_in_interest = []
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < self.interest_horizon_m:
                obstacles_in_interest.append(obs)
        self.obstacles_in_interest = obstacles_in_interest
        self.cur_obstacles_in_interest = obstacles_in_interest
