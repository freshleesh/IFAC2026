#!/usr/bin/env python3
"""### HJ : External side-of-passing decision (LEFT / RIGHT / TRAIL / CLEAR).

v3 (2026-04-21): feasibility-aware. "Can the ego physically fit past the
obstacle on this side?" is checked BEFORE the NLP. If neither side fits,
TRAIL is forced; the MPC then sees v_max capped by the obstacle's speed
and the continuity cost pulls the trajectory smoothly back to centerline.

v2 was geometry-only ("which side is wider"), which could choose a side
that required margin_wall < 0.05 m — solver then fought the wall cushion
and the path hugged the wall. v3 refuses such side picks at the decision
layer.

Inputs per tick:
  - ego_v (m/s)
  - obs list, each with (s0, n0, v_s_obs, half_width, d_L_at_obs, d_R_at_obs)
  - ref_v_at_obs (target speed at obstacle station)

Output:
  side_int   int  (SIDE_{CLEAR,LEFT,RIGHT,TRAIL})
  side_str   str  ("clear"/"left"/"right"/"trail")
  scores     dict {'d_free_L', 'd_free_R', 'can_pass_L', 'can_pass_R',
                   'dv', 'reason', ...}

Hysteresis: chosen side persists `hold_ticks` before a LEFT↔RIGHT flip.
TRAIL entry uses `trail_entry_ticks` (smaller) to react faster to narrow
corridors — "못 갈 것 같으면 바로 감속" semantics the user asked for.
"""

SIDE_CLEAR = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2
SIDE_TRAIL = 3

_NAME = {SIDE_CLEAR: 'clear', SIDE_LEFT: 'left',
         SIDE_RIGHT: 'right', SIDE_TRAIL: 'trail'}


class SideDecider:
    def __init__(self,
                 ego_half_width=0.15,
                 gap_lat=0.25,
                 trail_dv_thresh=0.5,
                 hold_ticks=5,
                 min_pass_margin=0.10,
                 trail_entry_ticks=3,
                 wall_safe=0.15,
                 inflation=0.05):
        self.ego_half = float(ego_half_width)
        self.gap_lat = float(gap_lat)
        self.trail_dv_thresh = float(trail_dv_thresh)
        self.hold_ticks = int(hold_ticks)
        # v3: extra safety — the narrower the corridor-side-slot, the more
        # the MPC will drive the trajectory into the wall cushion. Require
        # a positive residual (min_pass_margin) on top of ego_half+gap_lat
        # BEFORE declaring a side viable.
        self.min_pass_margin = float(min_pass_margin)
        self.trail_entry_ticks = int(trail_entry_ticks)
        # ### HJ : v3b — match solver's hard corridor. Solver enforces
        # |n| ≤ d_wall - ego_half - wall_safe - inflation. Decider must
        # use the same effective wall when judging feasibility, otherwise
        # "can_pass" fires on slots the solver will reject as infeasible
        # (or force-slack) → "late-commit to TRAIL just as we hit the wall"
        # symptom the user reported.
        self.wall_safe = float(wall_safe)
        self.inflation = float(inflation)

        self._prev_side = SIDE_CLEAR
        self._pending_side = SIDE_CLEAR
        self._pending_streak = 0
        self.last_scores = {}

    def decide(self, ego_v, obs_list):
        if not obs_list:
            raw = SIDE_CLEAR
            scores = {'reason': 'no_obstacle'}
        else:
            o = obs_list[0]
            n_o = float(o['n0'])
            w_o = float(o['half_width'])
            dL = float(o['d_L'])
            dR = float(o['d_R'])
            v_obs = float(o.get('v_s_obs', 0.0))
            ref_v = float(o.get('ref_v', ego_v))
            dv = ref_v - v_obs

            # v3b: free lateral slot on each side, using the SOLVER's
            # hard-corridor wall (dL - wall_safe - inflation) rather than
            # the raw wall. This makes "can_pass" agree with what the
            # solver will actually let n[k] reach.
            eff_dL = dL - self.wall_safe - self.inflation
            eff_dR = dR - self.wall_safe - self.inflation
            d_free_L = eff_dL - (n_o + w_o) - (self.ego_half + self.gap_lat)
            d_free_R = (n_o - w_o) - (-eff_dR) - (self.ego_half + self.gap_lat)

            # v3 feasibility gate: require a positive min_pass_margin on top.
            # This margin is what prevents the MPC from being pushed into
            # the wall cushion to squeeze through a marginal slot.
            can_pass_L = d_free_L >= self.min_pass_margin
            can_pass_R = d_free_R >= self.min_pass_margin

            scores = {'d_free_L': round(d_free_L, 3),
                      'd_free_R': round(d_free_R, 3),
                      'can_pass_L': bool(can_pass_L),
                      'can_pass_R': bool(can_pass_R),
                      'dv': round(dv, 3),
                      'n_o': round(n_o, 3),
                      'v_obs': round(v_obs, 3),
                      'min_pass_margin': round(self.min_pass_margin, 3)}

            if dv < self.trail_dv_thresh:
                raw = SIDE_TRAIL
                scores['reason'] = 'dv_small'
            elif not can_pass_L and not can_pass_R:
                raw = SIDE_TRAIL
                scores['reason'] = 'no_side_fits'
            elif can_pass_L and not can_pass_R:
                raw = SIDE_LEFT
                scores['reason'] = 'only_left_fits'
            elif can_pass_R and not can_pass_L:
                raw = SIDE_RIGHT
                scores['reason'] = 'only_right_fits'
            elif d_free_L >= d_free_R:
                raw = SIDE_LEFT
                scores['reason'] = 'left_wider'
            else:
                raw = SIDE_RIGHT
                scores['reason'] = 'right_wider'

        # Hysteresis. TRAIL entry fires faster (trail_entry_ticks) to
        # bail out of infeasible overtakes quickly; LEFT↔RIGHT uses the
        # longer hold_ticks to damp flips.
        if raw == self._prev_side:
            self._pending_side = raw
            self._pending_streak = 0
            final = raw
        else:
            if raw == self._pending_side:
                self._pending_streak += 1
            else:
                self._pending_side = raw
                self._pending_streak = 1

            gate = (self.trail_entry_ticks if raw == SIDE_TRAIL
                    else self.hold_ticks)
            if self._pending_streak >= gate:
                final = raw
                self._prev_side = raw
                self._pending_streak = 0
            else:
                final = self._prev_side
                scores['held'] = True
                scores['pending'] = _NAME.get(raw, '?')
                scores['pending_streak'] = self._pending_streak
                scores['gate'] = gate

        self.last_scores = scores
        return final, _NAME.get(final, '?'), scores
