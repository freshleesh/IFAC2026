"""각 state 별 local waypoint 생성 함수 모음.

원본 ROS1 그대로 (rospy 의존 없음, f110_msgs.msg.Wpnt 만 import).
TYPE_CHECKING 으로 StateMachine forward reference — circular import 회피.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List
from f110_msgs.msg import Wpnt

if TYPE_CHECKING:
    # 메인 노드 타입은 C-4 에서 정의. 일단 forward reference 만.
    from state_machine.state_machine_node import StateMachine


def GlobalTracking(state_machine: "StateMachine") -> List[Wpnt]:
    s = int(state_machine.cur_s / state_machine.waypoints_dist + 0.5)
    return [
        state_machine.cur_gb_wpnts.list[(s + i) % state_machine.num_glb_wpnts]
        for i in range(state_machine.n_loc_wpnts)
    ]


def Overtaking(state_machine: "StateMachine") -> List[Wpnt]:
    """Overtaking waypoint 생성. Priority:
    1. 정적 장애물 회피 (`static_overtaking_mode`) — ot_planner 무관
    2. spliner / predictive_spliner 동적 회피
    3. 사전 계산된 overtake_wpnts (다른 planner fallback)
    """
    if state_machine.static_overtaking_mode:
        return state_machine.get_splini_wpts()
    if state_machine.ot_planner in ("spliner", "predictive_spliner"):
        return state_machine.get_splini_wpts()
    s = state_machine.cur_id_ot
    return [
        state_machine.overtake_wpnts[(s + i) % state_machine.num_ot_points]
        for i in range(state_machine.n_loc_wpnts)
    ]


def RECOVERY(state_machine: "StateMachine"):
    return state_machine.get_recovery_wpts()


def START(state_machine: "StateMachine"):
    return state_machine.get_start_wpts()


def FTGOnly(state_machine: "StateMachine"):
    """FTG-only state — 제어 입력은 control 노드가 직접 생성, wpnts 없음."""
    return []


def SmartStatic(state_machine: "StateMachine") -> List[Wpnt]:
    """Smart Static — GB optimizer fixed path 사용."""
    return state_machine.get_smart_static_wpts()
