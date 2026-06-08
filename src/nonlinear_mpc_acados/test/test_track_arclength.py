"""find_current_arc_length must not snap s to 0 just BEFORE the start/finish seam.

Bug: `if nearest == 0: current_s = 0.0` clobbered the projected arc length even
when the car is just behind the seam (nearest waypoint = index 0, projecting back
onto the last segment → correct s ≈ L). That injected a full-lap discontinuity at
the line every lap (corrupting ref/κ lookups and the LMPC s-window).
"""
import math
import types

import numpy as np

from nonlinear_mpc_acados.track_loader import find_current_arc_length


def _ring(N=20, R=5.0):
    th = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    cl = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    seg = np.linalg.norm(np.diff(cl, axis=0, append=cl[:1]), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])   # (N,) cumulative, s[0]=0
    L = float(np.sum(seg))
    track = types.SimpleNamespace(
        center_lane=cl,
        element_arc_lengths=s,
        element_arc_lengths_orig=np.append(s, L),
    )
    return track, L


def test_arc_length_just_before_seam_is_near_L_not_zero():
    track, L = _ring()
    # Car just behind the seam (angle slightly negative → nearest waypoint idx 0,
    # but physically still on the last segment): correct s is ≈ L, not 0.
    eps = 0.03
    car = np.array([5.0 * math.cos(-eps), 5.0 * math.sin(-eps)])
    s, nearest = find_current_arc_length(track, car, arc_min_dist_tol=0.01)
    assert s > 0.8 * L, f"expected s≈L (~{L:.2f}), got {s:.3f} (seam clobber bug)"


def test_arc_length_at_start_is_zero():
    track, L = _ring()
    car = track.center_lane[0].copy()   # exactly on waypoint 0
    s, nearest = find_current_arc_length(track, car)
    assert abs(s) < 1e-6, f"at start s should be 0, got {s}"
