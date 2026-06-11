"""Window-aware avoidance side decision (decide_side_window).

2026-06-11: replaces the single-point top-2 boundary-distance compare whose
centerline tie always returned -1 ("always avoids down" bug). The decision
now looks at the corridor room along a downstream window of the detour tube
and prefers the side whose bottleneck (then mean room) is larger.

Convention (unchanged from decide_side_pref): +1 = pass on the labeled-LEFT
boundary side, -1 = labeled-RIGHT side. The labeled boundaries' signed e_c
projections flip with track orientation (CW/CCW), so gaps are |w - e_c_obs|
— orientation-agnostic.

Run:
    cd src/nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_side_pref_window.py -q
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.model_policy import decide_side_window


class TestDecideSideWindow(unittest.TestCase):
    def test_centerline_tie_picks_roomier_side_downstream(self):
        # THE bug case: obstacle dead-center, symmetric at s_obs, but the
        # left side opens up downstream — old code returned -1 blindly.
        w_left = [0.5, 0.5, 1.2, 1.2]
        w_right = [-0.5, -0.5, -0.5, -0.5]
        self.assertEqual(decide_side_window(0.0, w_left, w_right), +1)

    def test_obstacle_near_right_boundary_passes_left(self):
        # Old-behavior parity: obstacle 0.2 m from right wall → right
        # blocked → pass left.
        w_left = [-0.5] * 4
        w_right = [0.5] * 4
        self.assertEqual(decide_side_window(0.3, w_left, w_right), +1)

    def test_orientation_agnostic(self):
        # Same physical scenario on a track with flipped orientation
        # (labeled-left projects positive): obstacle still 0.2 m from the
        # labeled-right wall → still +1.
        w_left = [0.5] * 4
        w_right = [-0.5] * 4
        self.assertEqual(decide_side_window(-0.3, w_left, w_right), +1)

    def test_left_pinch_downstream_blocks_left(self):
        # Left is wide at the obstacle but pinches below W_CAR_SAFE
        # downstream — single-point compare would wrongly pick left.
        w_left = [1.0, 1.0, 0.15, 1.0]
        w_right = [-0.4, -0.4, -0.4, -0.4]
        self.assertEqual(decide_side_window(0.0, w_left, w_right), -1)

    def test_fully_symmetric_defaults_right(self):
        # True tie → keep the legacy -1 default (deterministic).
        w_left = [0.5] * 4
        w_right = [-0.5] * 4
        self.assertEqual(decide_side_window(0.0, w_left, w_right), -1)

    def test_both_blocked_picks_less_bad(self):
        # Both under W_CAR_SAFE: pick the larger bottleneck instead of the
        # old unconditional -1.
        w_left = [0.18] * 4
        w_right = [-0.10] * 4
        self.assertEqual(decide_side_window(0.0, w_left, w_right), +1)

    def test_bottleneck_beats_mean(self):
        # Right has the better bottleneck even though left has the better
        # mean — the car must fit through the narrowest point.
        w_left = [0.25, 2.0, 2.0, 2.0]   # min 0.25, mean high
        w_right = [-0.6, -0.6, -0.6, -0.6]  # min 0.6
        self.assertEqual(decide_side_window(0.0, w_left, w_right), -1)

    def test_empty_window_defaults_right(self):
        self.assertEqual(decide_side_window(0.0, [], []), -1)


if __name__ == '__main__':
    unittest.main()
