from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

from state_machine.states_types import StateType
import logging
_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from state_machine.state_machine_node import StateMachine

close_threshold_smart = 0.5
# close_threshold_gb replaced with state_machine._get_adaptive_close_threshold() (speed-adaptive)


# ============================================================================
# Mode Strategy — GB / Smart 이중 closed loop 통합용 컨텍스트.
# 각 모드의 차이점을 ModeContext 하나에 묶고, NonObstacle/ObstacleTransition
# 통합 함수가 이를 받아 동일 로직으로 처리한다.
# ============================================================================

@dataclass(frozen=True)
class ModeContext:
    """Mode-specific data sources for unified transition functions.

    Fields:
        helper                    : decision 의 기준이 되는 객체 (state_machine 또는 smart_helper)
        base_state                : 해당 모드의 기본 state (GB_TRACK 또는 SMART_STATIC)
        base_wpnts_data           : base wpnts 컨테이너 (cur_gb_wpnts 또는 cur_smart_static_avoidance_wpnts)
        base_wpnts_msg            : ROS 메시지 (Smart 만 — GB raceline 은 항상 valid 라 None)
        allow_dynamic_overtaking  : dynamic OT 허용 여부 (Smart 모드에선 항상 False)
        close_threshold           : close_to_raceline 체크용 임계 (모드별로 다름)
    """
    helper: Any
    base_state: StateType
    base_wpnts_data: Any
    base_wpnts_msg: Optional[Any]
    allow_dynamic_overtaking: bool
    close_threshold: float


def _make_gb_context(state_machine) -> ModeContext:
    """GB 모드 context. helper=state_machine 자신, base=GB_TRACK/cur_gb_wpnts.

    GB raceline 은 항상 valid 라 base_wpnts_msg=None (validity 체크 생략).
    Dynamic OT 는 smart_static_active=False 일 때만 허용.
    """
    return ModeContext(
        helper=state_machine,
        base_state=StateType.GB_TRACK,
        base_wpnts_data=state_machine.cur_gb_wpnts,
        base_wpnts_msg=None,
        allow_dynamic_overtaking=not state_machine.smart_static_active,
        close_threshold=state_machine._get_adaptive_close_threshold(),
    )


def _make_smart_context(state_machine) -> ModeContext:
    """Smart 모드 context. helper=smart_helper, base=SMART_STATIC/cur_smart_static_avoidance_wpnts.

    Smart static path 는 동적으로 publish 되므로 base_wpnts_msg=smart_static_wpnts (validity 체크 필요).
    Dynamic OT 는 Smart 모드에선 항상 비활성 (정책).
    """
    return ModeContext(
        helper=state_machine.smart_helper,
        base_state=StateType.SMART_STATIC,
        base_wpnts_data=state_machine.cur_smart_static_avoidance_wpnts,
        base_wpnts_msg=state_machine.smart_static_wpnts,
        allow_dynamic_overtaking=False,
        close_threshold=close_threshold_smart,
    )


def _is_base_path_valid(ctx: ModeContext) -> bool:
    """base path 자체가 valid 한지.

    GB: raceline 은 항상 valid (base_wpnts_msg is None).
    Smart: smart_static_wpnts 가 fresh + on_spline 한지 helper 로 체크.
    """
    if ctx.base_wpnts_msg is None:
        return True
    return ctx.helper._check_latest_wpnts(ctx.base_wpnts_msg, ctx.base_wpnts_data)

# ===== HJ ADDED: Debug logging helper - only logs when values change =====
_debug_log_cache = {}
DEBUG_LOGGING_ENABLED = False  # Set to False to disable all debug logging

def debug_log_on_change(tag, **kwargs):
    """이전 호출과 kwargs 값이 다를 때만 로그.

    DEBUG_LOGGING_ENABLED 가 False 면 no-op. 캐시는 모듈 전역 dict (`_debug_log_cache`).
    item 할당만 하므로 `global` 선언 필요 없음.
    """
    if not DEBUG_LOGGING_ENABLED:
        return

    cache_key = tag
    prev_values = _debug_log_cache.get(cache_key, None)

    # Check if any value changed
    if prev_values != kwargs:
        # Build log message
        msg_parts = [f"{k}={v}" for k, v in kwargs.items()]
        _logger.warning(f"[DEBUG {tag}] " + ", ".join(msg_parts))

        # Update cache
        _debug_log_cache[cache_key] = kwargs.copy()
# ===== HJ ADDED END =====

"""
Transitions should loosely follow the following template (basically a match-case)

if (logic sum of bools obtained by methods of state_machine):
    return StateType.<DESIRED STATE>
elif (e.g. state_machine.obstacles are near):
    return StateType.<ANOTHER DESIRED STATE>
...

NOTE: ideally put the most common cases on top of the match-case

NOTE 2: notice that, when implementing new states, if an attribute/condition in the
    StateMachine is not available, your IDE will tell you, but only if you have a smart
    enough IDE. So use vscode, pycharm, fleet or whatever has specific python syntax highlights.

NOTE 3: transistions must not have side effects on the state machine!
    i.e. any attribute of the state machine should not be modified in the transitions.
"""

# ===== HJ MODIFIED: Complete mode separation - each mode has its own closed loop =====
def GlobalTrackingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.GB_TRACK`

    Routes to completely separate Smart or GB mode transitions.
    """
    smart_active = state_machine.smart_static_active

    # Complete mode switching - call separate function sets
    if smart_active:
        smart_helper = state_machine.smart_helper
        close_to_smart = smart_helper._check_close_to_raceline(close_threshold_smart) * smart_helper._check_close_to_raceline_heading(20)
        num_obs = len(smart_helper.cur_obstacles_in_interest)

        debug_log_on_change("GlobalTracking_SMART",
                           close=close_to_smart,
                           num_obs=num_obs,
                           cur_s=round(smart_helper.cur_s, 2),
                           cur_d=round(smart_helper.cur_d, 3))

        if num_obs == 0:
            return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
        else:
            return ObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        close_to_gb = state_machine._check_close_to_raceline(state_machine._get_adaptive_close_threshold()) * state_machine._check_close_to_raceline_heading(20)
        # _logger.warning(f"[GlobalTracking] GB MODE: close_to_gb={close_to_gb}, num_obs={len(state_machine.cur_obstacles_in_interest)}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_GBMode(state_machine, close_to_gb)
        else:
            return ObstacleTransition_GBMode(state_machine, close_to_gb)


def RecoveryTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.RECOVERY`

    Recovery operates within the mode's closed loop.
    """
    smart_active = state_machine.smart_static_active

    # Use appropriate Frenet coordinate system based on mode
    # In Smart mode, recovery waypoints are also Fixed Frenet based
    if smart_active:
        smart_helper = state_machine.smart_helper
        recovery_sustainability = smart_helper._check_sustainability(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        close_to_smart = smart_helper._check_close_to_raceline(close_threshold_smart) * smart_helper._check_close_to_raceline_heading(20)

        debug_log_on_change("Recovery_SMART",
                           sustainable=recovery_sustainability,
                           close=close_to_smart,
                           continuing=recovery_sustainability and not close_to_smart)

        if recovery_sustainability and not close_to_smart:
            return StateType.RECOVERY, StateType.RECOVERY
        # Recovery ended - return to Smart mode closed loop
        return SmartStaticTransition(state_machine)
    else:
        recovery_sustainability = state_machine._check_sustainability(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        close_to_gb = state_machine._check_close_to_raceline(state_machine._get_adaptive_close_threshold()) * state_machine._check_close_to_raceline_heading(20)
        # _logger.warning(f"[Recovery] GB MODE: close_to_gb={close_to_gb}, sustainable={recovery_sustainability}")

        if recovery_sustainability and not close_to_gb:
            return StateType.RECOVERY, StateType.RECOVERY
        # Recovery ended - return to GB mode closed loop
        return GlobalTrackingTransition(state_machine)


def TrailingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.TRAILING`"""
    smart_active = state_machine.smart_static_active

    if smart_active:
        smart_helper = state_machine.smart_helper
        close_to_smart = smart_helper._check_close_to_raceline(close_threshold_smart) * smart_helper._check_close_to_raceline_heading(20)
        num_obs = len(smart_helper.cur_obstacles_in_interest)
        ftg_check = state_machine._check_ftg()

        debug_log_on_change("Trailing_SMART",
                           close=close_to_smart,
                           num_obs=num_obs,
                           ftg=ftg_check)

        if num_obs == 0:
            return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
        else:
            if ftg_check:
                return StateType.FTGONLY, StateType.FTGONLY
            return ObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        close_to_gb = state_machine._check_close_to_raceline(state_machine._get_adaptive_close_threshold()) * state_machine._check_close_to_raceline_heading(20)
        # _logger.warning(f"[Trailing] GB MODE: close_to_gb={close_to_gb}")

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_GBMode(state_machine, close_to_gb)
        else:
            if state_machine._check_ftg():
                return StateType.FTGONLY, StateType.FTGONLY
            return ObstacleTransition_GBMode(state_machine, close_to_gb)


def OvertakingTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.OVERTAKE`"""
    smart_active = state_machine.smart_static_active

    # Use appropriate Frenet coordinate system based on mode
    # In Smart mode, overtaking waypoints are also Fixed Frenet based
    if smart_active:
        smart_helper = state_machine.smart_helper
        ot_sustainability = smart_helper._check_overtaking_mode_sustainability()
        enemy_in_front = smart_helper._check_enemy_in_front()
    else:
        ot_sustainability = state_machine._check_overtaking_mode_sustainability()
        enemy_in_front = state_machine._check_enemy_in_front()

    debug_log_on_change("Overtaking",
                       mode="SMART" if smart_active else "GB",
                       sustainable=ot_sustainability,
                       enemy=enemy_in_front,
                       ttl_count=state_machine.overtaking_ttl_count,
                       continuing=ot_sustainability and (enemy_in_front or state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold))

    if ot_sustainability and enemy_in_front:
        state_machine.overtaking_ttl_count = 0
        return StateType.OVERTAKE, StateType.OVERTAKE
    if ot_sustainability and state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold:
        state_machine.overtaking_ttl_count += 1
        return StateType.OVERTAKE, StateType.OVERTAKE
    state_machine.overtaking_ttl_count = 0

    # Overtaking ended - return to appropriate mode's closed loop
    if smart_active:
        return SmartStaticTransition(state_machine)
    else:
        return GlobalTrackingTransition(state_machine)


def StartTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.START`"""
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)

    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine)


def FTGOnlyTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.FTGONLY`"""

    smart_active = state_machine.smart_static_active

    # Use appropriate Frenet coordinate system based on mode
    if smart_active:
        smart_helper = state_machine.smart_helper
        close_to_raceline = smart_helper._check_close_to_raceline(close_threshold_smart) * smart_helper._check_close_to_raceline_heading(20)

        if len(smart_helper.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_SmartMode(state_machine, close_to_raceline)
        else:
            if close_to_raceline and smart_helper._check_free_frenet(state_machine.cur_smart_static_avoidance_wpnts):
                return StateType.SMART_STATIC, StateType.SMART_STATIC

            recovery_availability = smart_helper._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
            if (recovery_availability and smart_helper._check_free_frenet(state_machine.cur_recovery_wpnts)):
                return StateType.RECOVERY, StateType.RECOVERY

            # ===== HJ MODIFIED: Disable dynamic overtaking in Smart mode =====
            # In Smart mode: only static overtaking allowed (dynamic overtaking disabled)
            if smart_helper._check_static_overtaking_mode():
                return StateType.OVERTAKE, StateType.OVERTAKE
            # Note: _check_overtaking_mode() (dynamic) is intentionally disabled in Smart mode
            # ===== HJ MODIFIED END =====
            else:
                return StateType.FTGONLY, StateType.FTGONLY
    else:
        close_to_raceline = state_machine._check_close_to_raceline(close_threshold_smart) * state_machine._check_close_to_raceline_heading(20)

        if len(state_machine.cur_obstacles_in_interest) == 0:
            return NonObstacleTransition_GBMode(state_machine, close_to_raceline)
        else:
            if close_to_raceline and state_machine._check_free_frenet(state_machine.cur_gb_wpnts):
                return StateType.GB_TRACK, StateType.GB_TRACK

            recovery_availability = state_machine._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
            if (recovery_availability and state_machine._check_free_frenet(state_machine.cur_recovery_wpnts)):
                return StateType.RECOVERY, StateType.RECOVERY

            # ===== HJ MODIFIED: Disable dynamic overtaking when smart_static_active=true =====
            if state_machine._check_static_overtaking_mode():
                return StateType.OVERTAKE, StateType.OVERTAKE
            # Dynamic overtaking only when smart_static is NOT active
            if state_machine._check_overtaking_mode() and not state_machine.smart_static_active:
                return StateType.OVERTAKE, StateType.OVERTAKE
            # ===== HJ MODIFIED END =====
            else:
                return StateType.FTGONLY, StateType.FTGONLY


def SmartStaticTransition(state_machine: StateMachine) -> Tuple[StateType, StateType]:
    """Transitions for being in `StateType.SMART_STATIC`

    Entry point for Smart mode's closed loop.
    When flag is active: uses Smart Static path only (GB raceline ignored).
    When flag turns off: returns to GB mode transitions.
    """
    # ===== HJ ADDED: Handle smart_static_active flag off - return to GB mode =====
    if not state_machine.smart_static_active:
        # Flag turned off - need to safely return to GB mode
        _logger.warning("[SmartStaticTransition] Flag off, returning to GB mode")

        # Clear all closest_targets to prevent Frenet coordinate mismatch
        # (smart_helper may have set these with Fixed Frenet coordinates)
        state_machine.cur_gb_wpnts.closest_target = None
        state_machine.cur_recovery_wpnts.closest_target = None
        state_machine.cur_avoidance_wpnts.closest_target = None
        state_machine.cur_static_avoidance_wpnts.closest_target = None

        # Check if we're close enough to GB raceline for direct transition
        close_to_gb = state_machine._check_close_to_raceline(state_machine._get_adaptive_close_threshold()) * \
                      state_machine._check_close_to_raceline_heading(20)

        if close_to_gb:
            # Close enough to GB - safe to return immediately
            _logger.warning("[SmartStaticTransition→GB_TRACK] Close to GB, direct return")
            return StateType.GB_TRACK, StateType.GB_TRACK
        else:
            # Not close to GB - delegate to GB mode transitions
            # GB transitions will handle recovery/overtaking/trailing logic
            _logger.warning("[SmartStaticTransition] Not close to GB, using GB transitions")
            if len(state_machine.cur_obstacles_in_interest) == 0:
                return NonObstacleTransition_GBMode(state_machine, close_to_gb)
            else:
                return ObstacleTransition_GBMode(state_machine, close_to_gb)
    # ===== HJ ADDED END =====

    # Original Smart mode logic (flag still active)
    smart_helper = state_machine.smart_helper
    close_to_smart = smart_helper._check_close_to_raceline(close_threshold_smart) * smart_helper._check_close_to_raceline_heading(20)
    num_obstacles = len(smart_helper.cur_obstacles_in_interest)

    debug_log_on_change("SmartStatic",
                       close=close_to_smart,
                       num_obs=num_obstacles)

    # Delegate to Smart mode transitions only
    if num_obstacles == 0:
        return NonObstacleTransition_SmartMode(state_machine, close_to_smart)
    else:
        return ObstacleTransition_SmartMode(state_machine, close_to_smart)


##################################################################################################################
##################################################################################################################
# ===== UNIFIED MODE TRANSITIONS — GB / Smart 통합 (B-2) =====
#
# 두 모드의 NonObstacle/Obstacle transition 을 ModeContext 로 묶어 단일 함수로 통합.
# 기존 *_GBMode / *_SmartMode 함수는 thin wrapper 로 남아 호출자 변경 없음.

def NonObstacleTransition(state_machine: StateMachine, ctx: ModeContext, close_to_base: bool) -> Tuple[StateType, StateType]:
    """장애물 없는 경우의 transition (GB/Smart 통합).

    Priority:
        1. base path 진입 가능 → base_state
           (GB: close 면 OK / Smart: wpnts valid 면 OK, close 무관)
        2. recovery 가용 + on_spline → RECOVERY
        3. fallback → LOSTLINE (wpnts_src=base_state)
    """
    helper = ctx.helper

    # Priority 1: base 진입 — GB 는 close 만, Smart 는 wpnts validity
    if ctx.base_wpnts_msg is None:
        base_ok = close_to_base
    else:
        base_ok = _is_base_path_valid(ctx)
    if base_ok:
        return ctx.base_state, ctx.base_state

    # Priority 2: recovery
    if helper._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts):
        if helper._check_on_spline(state_machine.cur_recovery_wpnts):
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 3: lostline (wpnts_src 는 base_state 로 — 외부에서 trajectory 가 base 따라감)
    return StateType.LOSTLINE, ctx.base_state


def ObstacleTransition(state_machine: StateMachine, ctx: ModeContext, close_to_base: bool) -> Tuple[StateType, StateType]:
    """장애물 있는 경우의 transition (GB/Smart 통합).

    Priority:
        1. base path close + free (+ valid for Smart) → base_state
        2. not close + recovery 가용 + free → RECOVERY
        3. static_overtaking_mode → OVERTAKE
        4. dynamic_overtaking_mode (allow_dynamic_overtaking=True 일 때만) → OVERTAKE
        5. TRAILING:
           - base 진입 가능 (GB: close, Smart: valid) → TRAILING + base_state
           - recovery 가용 → TRAILING + RECOVERY
           - fallback → TRAILING + base_state
    """
    helper = ctx.helper
    base_path_free = helper._check_free_frenet(ctx.base_wpnts_data)
    base_valid = _is_base_path_valid(ctx)

    # Priority 1: base 진입 (close + free + valid)
    if base_valid and close_to_base and base_path_free:
        return ctx.base_state, ctx.base_state

    # Priority 2: recovery (only if not close)
    recovery_availability = False
    if not close_to_base:
        recovery_availability = helper._check_latest_wpnts(state_machine.recovery_wpnts, state_machine.cur_recovery_wpnts)
        if recovery_availability and helper._check_free_frenet(state_machine.cur_recovery_wpnts):
            return StateType.RECOVERY, StateType.RECOVERY

    # Priority 3: overtaking
    if helper._check_static_overtaking_mode():
        return StateType.OVERTAKE, StateType.OVERTAKE
    if ctx.allow_dynamic_overtaking and helper._check_overtaking_mode():
        return StateType.OVERTAKE, StateType.OVERTAKE

    # Priority 4: TRAILING — GB 는 close 면 base_state, Smart 는 valid 면 base_state
    if ctx.base_wpnts_msg is None:
        base_ok_for_trailing = close_to_base
    else:
        base_ok_for_trailing = base_valid
    if base_ok_for_trailing:
        return StateType.TRAILING, ctx.base_state
    if recovery_availability:
        return StateType.TRAILING, StateType.RECOVERY
    # Fallback (GB 의 gb_path_free 분기 + Smart 의 fallback 모두 여기로 흘러감 — 결과 동일)
    return StateType.TRAILING, ctx.base_state


##################################################################################################################
##################################################################################################################
# ===== SMART MODE CLOSED LOOP - Only considers Smart Static path =====

def NonObstacleTransition_SmartMode(state_machine: StateMachine, close_to_smart: bool) -> Tuple[StateType, StateType]:
    """Smart 모드 thin wrapper — 통합 NonObstacleTransition 호출."""
    return NonObstacleTransition(state_machine, _make_smart_context(state_machine), close_to_smart)


def ObstacleTransition_SmartMode(state_machine: StateMachine, close_to_smart: bool) -> Tuple[StateType, StateType]:
    """Smart 모드 thin wrapper — 통합 ObstacleTransition 호출."""
    return ObstacleTransition(state_machine, _make_smart_context(state_machine), close_to_smart)


##################################################################################################################
# ===== GB MODE CLOSED LOOP - Only considers GB raceline =====

def NonObstacleTransition_GBMode(state_machine: StateMachine, close_to_gb: bool) -> Tuple[StateType, StateType]:
    """GB 모드 thin wrapper — 통합 NonObstacleTransition 호출."""
    return NonObstacleTransition(state_machine, _make_gb_context(state_machine), close_to_gb)


def ObstacleTransition_GBMode(state_machine: StateMachine, close_to_gb: bool) -> Tuple[StateType, StateType]:
    """GB 모드 thin wrapper — 통합 ObstacleTransition 호출."""
    return ObstacleTransition(state_machine, _make_gb_context(state_machine), close_to_gb)
