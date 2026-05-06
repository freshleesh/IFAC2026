"""state_transitions.py 의 transition 함수들 단위 테스트.

목적: B-2 (GB/Smart 이중 closed loop 통합) 작업의 회귀 안전망.
각 transition 함수가 입력 (state_machine attribute + sub-check 결과) 에 따라
어떤 (cur_state, local_wpnts_src) 튜플을 반환하는지 분기별로 검증.

mock 전략:
  - state_machine 객체는 MagicMock — attribute 자유롭게 세팅
  - sub-check 메서드 (`_check_close_to_raceline` 등) 는 mock 의 return_value 사용
  - smart_helper 도 MagicMock — Fixed Frenet 기반 분기 검증
"""
from unittest.mock import MagicMock

import pytest

from state_machine.states_types import StateType
import state_machine.state_transitions as st


# ============================================================================
# Fixture: 기본 state_machine mock — 분기 안 타도 동작하는 default
# ============================================================================

@pytest.fixture
def sm():
    """Default state_machine mock — GB 모드, 장애물 없음, 모든 check True."""
    m = MagicMock()

    # Mode flag
    m.smart_static_active = False

    # Obstacle list
    m.cur_obstacles_in_interest = []

    # Overtaking TTL
    m.overtaking_ttl_count = 0
    m.overtaking_ttl_count_threshold = 100

    # waypoint slots — 그냥 MagicMock (속성 접근 자유)
    # smart_static_wpnts 등은 MagicMock 으로 자동

    # Default sub-check returns
    m._check_close_to_raceline.return_value = True
    m._check_close_to_raceline_heading.return_value = True
    m._check_ftg.return_value = False
    m._check_overtaking_mode.return_value = False
    m._check_static_overtaking_mode.return_value = False
    m._check_overtaking_mode_sustainability.return_value = True
    m._check_enemy_in_front.return_value = True
    m._check_sustainability.return_value = True
    m._check_latest_wpnts.return_value = True
    m._check_on_spline.return_value = True
    m._check_free_frenet.return_value = True
    m._check_free_cartesian.return_value = True
    m._get_adaptive_close_threshold.return_value = 0.3

    # smart_helper — 같은 default
    m.smart_helper.smart_static_active = True
    m.smart_helper.cur_obstacles_in_interest = []
    m.smart_helper._check_close_to_raceline.return_value = True
    m.smart_helper._check_close_to_raceline_heading.return_value = True
    m.smart_helper._check_overtaking_mode.return_value = False
    m.smart_helper._check_static_overtaking_mode.return_value = False
    m.smart_helper._check_overtaking_mode_sustainability.return_value = True
    m.smart_helper._check_enemy_in_front.return_value = True
    m.smart_helper._check_sustainability.return_value = True
    m.smart_helper._check_latest_wpnts.return_value = True
    m.smart_helper._check_on_spline.return_value = True
    m.smart_helper._check_free_frenet.return_value = True

    # waypoint list 길이 (NonObstacleTransition_SmartMode 의 debug log 용)
    m.cur_smart_static_avoidance_wpnts.list = []

    return m


# ============================================================================
# NonObstacleTransition_GBMode (3 분기)
# ============================================================================

class TestNonObstacleTransitionGBMode:
    def test_close_to_gb_returns_gb_track(self, sm):
        """close_to_gb=True 면 즉시 GB_TRACK."""
        result = st.NonObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)

    def test_not_close_recovery_available_returns_recovery(self, sm):
        """close=False + recovery 가용 → RECOVERY."""
        sm._check_latest_wpnts.return_value = True
        sm._check_on_spline.return_value = True
        result = st.NonObstacleTransition_GBMode(sm, close_to_gb=False)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)

    def test_not_close_recovery_unavailable_returns_lostline(self, sm):
        """close=False + recovery 불가 → LOSTLINE (wpnts_src=GB_TRACK)."""
        sm._check_latest_wpnts.return_value = False
        result = st.NonObstacleTransition_GBMode(sm, close_to_gb=False)
        assert result == (StateType.LOSTLINE, StateType.GB_TRACK)


# ============================================================================
# ObstacleTransition_GBMode (5+ 분기)
# ============================================================================

class TestObstacleTransitionGBMode:
    def test_close_and_path_free_returns_gb_track(self, sm):
        sm._check_free_frenet.return_value = True
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)

    def test_not_close_recovery_available_returns_recovery(self, sm):
        """close=False, recovery 가용+free → RECOVERY."""
        sm._check_free_frenet.return_value = True
        sm._check_latest_wpnts.return_value = True
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=False)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)

    def test_static_overtaking_triggers_overtake(self, sm):
        """close + path blocked + static_ot OK → OVERTAKE."""
        sm._check_free_frenet.return_value = False  # gb path 막힘
        sm._check_static_overtaking_mode.return_value = True
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.OVERTAKE, StateType.OVERTAKE)

    def test_dynamic_overtaking_triggers_overtake_when_smart_off(self, sm):
        """smart_static_active=False + dynamic_ot OK → OVERTAKE."""
        sm._check_free_frenet.return_value = False
        sm._check_static_overtaking_mode.return_value = False
        sm._check_overtaking_mode.return_value = True
        sm.smart_static_active = False
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.OVERTAKE, StateType.OVERTAKE)

    def test_dynamic_overtaking_skipped_when_smart_active(self, sm):
        """smart_static_active=True 면 dynamic OT 무시 → TRAILING (close 면 GB_TRACK src)."""
        sm._check_free_frenet.return_value = False
        sm._check_static_overtaking_mode.return_value = False
        sm._check_overtaking_mode.return_value = True  # 그래도 무시되어야
        sm.smart_static_active = True
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.TRAILING, StateType.GB_TRACK)

    def test_trailing_close_returns_gb_track(self, sm):
        """No OT but close → TRAILING + GB_TRACK src."""
        sm._check_free_frenet.return_value = False
        sm._check_static_overtaking_mode.return_value = False
        sm._check_overtaking_mode.return_value = False
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=True)
        assert result == (StateType.TRAILING, StateType.GB_TRACK)

    def test_trailing_not_close_recovery_returns_recovery(self, sm):
        """No OT, not close, recovery 가용 → RECOVERY (Priority 우선)."""
        sm._check_free_frenet.return_value = True   # recovery 도 free
        sm._check_static_overtaking_mode.return_value = False
        sm._check_overtaking_mode.return_value = False
        sm._check_latest_wpnts.return_value = True
        result = st.ObstacleTransition_GBMode(sm, close_to_gb=False)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)


# ============================================================================
# NonObstacleTransition_SmartMode (4 분기)
# ============================================================================

class TestNonObstacleTransitionSmartMode:
    def test_smart_valid_and_close_returns_smart_static(self, sm):
        sm.smart_helper._check_latest_wpnts.return_value = True
        result = st.NonObstacleTransition_SmartMode(sm, close_to_smart=True)
        assert result == (StateType.SMART_STATIC, StateType.SMART_STATIC)

    def test_smart_valid_not_close_returns_smart_static(self, sm):
        """Priority 2: smart valid 만 — 여전히 SMART_STATIC."""
        sm.smart_helper._check_latest_wpnts.return_value = True
        result = st.NonObstacleTransition_SmartMode(sm, close_to_smart=False)
        assert result == (StateType.SMART_STATIC, StateType.SMART_STATIC)

    def test_smart_invalid_recovery_available_returns_recovery(self, sm):
        # 첫 _check_latest_wpnts (smart wpnts) → False
        # 두 번째 _check_latest_wpnts (recovery wpnts) → True
        # _check_on_spline (recovery) → True
        sm.smart_helper._check_latest_wpnts.side_effect = [False, True]
        sm.smart_helper._check_on_spline.return_value = True
        result = st.NonObstacleTransition_SmartMode(sm, close_to_smart=False)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)

    def test_smart_invalid_recovery_unavailable_returns_lostline(self, sm):
        sm.smart_helper._check_latest_wpnts.return_value = False
        result = st.NonObstacleTransition_SmartMode(sm, close_to_smart=False)
        assert result == (StateType.LOSTLINE, StateType.SMART_STATIC)


# ============================================================================
# ObstacleTransition_SmartMode (5+ 분기)
# ============================================================================

class TestObstacleTransitionSmartMode:
    def test_valid_close_free_returns_smart_static(self, sm):
        sm.smart_helper._check_latest_wpnts.return_value = True
        sm.smart_helper._check_free_frenet.return_value = True
        sm.smart_helper.cur_obstacles_in_interest = ["obs"]  # for log
        result = st.ObstacleTransition_SmartMode(sm, close_to_smart=True)
        assert result == (StateType.SMART_STATIC, StateType.SMART_STATIC)

    def test_not_close_recovery_available_returns_recovery(self, sm):
        sm.smart_helper._check_latest_wpnts.return_value = True
        sm.smart_helper._check_free_frenet.return_value = True
        sm.smart_helper.cur_obstacles_in_interest = ["obs"]
        result = st.ObstacleTransition_SmartMode(sm, close_to_smart=False)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)

    def test_static_overtaking_triggers_overtake(self, sm):
        # smart path blocked + static_ot 가능
        sm.smart_helper._check_latest_wpnts.return_value = True
        sm.smart_helper._check_free_frenet.return_value = False  # path 막힘
        sm.smart_helper._check_static_overtaking_mode.return_value = True
        sm.smart_helper.cur_obstacles_in_interest = ["obs"]
        result = st.ObstacleTransition_SmartMode(sm, close_to_smart=True)
        assert result == (StateType.OVERTAKE, StateType.OVERTAKE)

    def test_dynamic_overtaking_disabled_in_smart_mode(self, sm):
        """Smart 모드에서는 dynamic OT 무시 — TRAILING + SMART_STATIC."""
        sm.smart_helper._check_latest_wpnts.return_value = True
        sm.smart_helper._check_free_frenet.return_value = False
        sm.smart_helper._check_static_overtaking_mode.return_value = False
        sm.smart_helper._check_overtaking_mode.return_value = True  # 무시되어야
        sm.smart_helper.cur_obstacles_in_interest = ["obs"]
        result = st.ObstacleTransition_SmartMode(sm, close_to_smart=True)
        assert result == (StateType.TRAILING, StateType.SMART_STATIC)

    def test_smart_invalid_recovery_returns_trailing_recovery(self, sm):
        # smart wpnts invalid + recovery 가용
        sm.smart_helper._check_latest_wpnts.side_effect = [False, True]  # smart, recovery
        sm.smart_helper._check_free_frenet.return_value = False
        sm.smart_helper._check_static_overtaking_mode.return_value = False
        sm.smart_helper.cur_obstacles_in_interest = ["obs"]
        result = st.ObstacleTransition_SmartMode(sm, close_to_smart=False)
        # Priority 2 recovery — 그러나 free_frenet 도 False 라 못 감
        # → Priority 4 fallback: TRAILING + RECOVERY
        assert result == (StateType.TRAILING, StateType.RECOVERY)


# ============================================================================
# StartTransition (2 분기)
# ============================================================================

class TestStartTransition:
    def test_start_free_and_on_spline_returns_start(self, sm):
        sm._check_free_cartesian.return_value = True
        sm._check_on_spline.return_value = True
        result = st.StartTransition(sm)
        assert result == (StateType.START, StateType.START)

    def test_start_blocked_returns_global_tracking(self, sm):
        """blocked → GlobalTrackingTransition 으로 위임."""
        sm._check_free_cartesian.return_value = False
        sm.cur_obstacles_in_interest = []
        sm.smart_static_active = False
        result = st.StartTransition(sm)
        # close_to_gb=True default, 장애물 없음 → NonObstacleTransition_GBMode → GB_TRACK
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)


# ============================================================================
# OvertakingTransition (TTL 카운터 + sustainability + enemy)
# ============================================================================

class TestOvertakingTransition:
    def test_sustainable_enemy_resets_ttl_returns_overtake(self, sm):
        sm._check_overtaking_mode_sustainability.return_value = True
        sm._check_enemy_in_front.return_value = True
        sm.overtaking_ttl_count = 50
        result = st.OvertakingTransition(sm)
        assert result == (StateType.OVERTAKE, StateType.OVERTAKE)
        assert sm.overtaking_ttl_count == 0  # reset

    def test_sustainable_no_enemy_within_ttl_continues(self, sm):
        sm._check_overtaking_mode_sustainability.return_value = True
        sm._check_enemy_in_front.return_value = False
        sm.overtaking_ttl_count = 50
        sm.overtaking_ttl_count_threshold = 100
        result = st.OvertakingTransition(sm)
        assert result == (StateType.OVERTAKE, StateType.OVERTAKE)
        assert sm.overtaking_ttl_count == 51  # increment

    def test_ttl_exhausted_returns_to_global_tracking(self, sm):
        sm._check_overtaking_mode_sustainability.return_value = True
        sm._check_enemy_in_front.return_value = False
        sm.overtaking_ttl_count = 100
        sm.overtaking_ttl_count_threshold = 100  # 같음 → 더 이상 진행 불가
        sm.smart_static_active = False
        sm.cur_obstacles_in_interest = []
        result = st.OvertakingTransition(sm)
        assert sm.overtaking_ttl_count == 0  # reset
        # GlobalTrackingTransition 호출 — close_to_gb default True, no obs → GB_TRACK
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)

    def test_unsustainable_returns_to_global_tracking(self, sm):
        sm._check_overtaking_mode_sustainability.return_value = False
        sm.overtaking_ttl_count = 50
        sm.smart_static_active = False
        sm.cur_obstacles_in_interest = []
        result = st.OvertakingTransition(sm)
        assert sm.overtaking_ttl_count == 0
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)


# ============================================================================
# RecoveryTransition (sustainability + close)
# ============================================================================

class TestRecoveryTransition:
    def test_sustainable_not_close_continues_recovery(self, sm):
        sm._check_sustainability.return_value = True
        sm._check_close_to_raceline.return_value = False  # not close → continue recovery
        result = st.RecoveryTransition(sm)
        assert result == (StateType.RECOVERY, StateType.RECOVERY)

    def test_close_to_gb_returns_global_tracking(self, sm):
        """close 면 recovery 종료 → GB mode 진입."""
        sm._check_sustainability.return_value = True
        sm._check_close_to_raceline.return_value = True
        sm.smart_static_active = False
        sm.cur_obstacles_in_interest = []
        result = st.RecoveryTransition(sm)
        # GlobalTracking → no obs + close → GB_TRACK
        assert result == (StateType.GB_TRACK, StateType.GB_TRACK)
