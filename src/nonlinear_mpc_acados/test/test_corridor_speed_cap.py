"""Corridor-width-aware speed cap.

Root cause of the final-map crash corner (s≈60): the speed reference is
curvature-only. A narrow-but-gently-curving section has low |κ|, so the κ-cap
allows ~v_max, the car runs full speed into a ~1.1 m corridor and clips the
wall (gym in_collision freeze). corridor_speed_cap slows the car where the
corridor is narrow — the geometry the κ-cap cannot see.
"""
import numpy as np

from nonlinear_mpc_acados.track_loader import corridor_speed_cap


def test_disabled_when_floor_nonpositive():
    # v_floor<=0 → feature off → always full speed regardless of width
    assert corridor_speed_cap(0.5, v_full=6.0, v_floor=0.0) == 6.0
    assert corridor_speed_cap(0.9, v_full=6.0, v_floor=-1.0) == 6.0


def test_wide_corridor_full_speed():
    assert corridor_speed_cap(2.0, v_full=6.0, v_floor=2.5,
                              w_tight=1.0, w_wide=1.6) == 6.0


def test_tight_corridor_floor_speed():
    assert corridor_speed_cap(0.9, v_full=6.0, v_floor=2.5,
                              w_tight=1.0, w_wide=1.6) == 2.5


def test_linear_ramp_midpoint():
    # width=1.3 is the midpoint of [1.0,1.6] → halfway between floor and full
    v = corridor_speed_cap(1.3, v_full=6.0, v_floor=2.5, w_tight=1.0, w_wide=1.6)
    assert abs(v - (2.5 + 0.5 * (6.0 - 2.5))) < 1e-9   # 4.25


def test_crash_zone_is_slowed_below_full():
    # the observed crash width (~1.07 m) must be slowed well below v_max
    v = corridor_speed_cap(1.07, v_full=6.0, v_floor=2.5, w_tight=1.0, w_wide=1.6)
    assert 2.5 <= v < 6.0
    assert v < 3.5   # meaningfully slowed


def test_monotonic_nondecreasing_in_width():
    widths = np.linspace(0.5, 2.2, 40)
    vs = [corridor_speed_cap(w, v_full=6.0, v_floor=2.5, w_tight=1.0, w_wide=1.6)
          for w in widths]
    assert all(vs[i + 1] >= vs[i] - 1e-9 for i in range(len(vs) - 1))
