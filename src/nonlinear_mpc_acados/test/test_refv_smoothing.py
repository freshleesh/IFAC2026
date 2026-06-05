import numpy as np
from nonlinear_mpc_acados.mpc_core.refv_smoothing import distance_tapered_forward_max


def _hard_forward_max(k, n_look):
    n = len(k); out = np.empty(n)
    for i in range(n):
        j = min(i + n_look, n)
        out[i] = k[i:j].max()
    return out


def test_continuous_as_corner_enters_window():
    # Flat |κ| with one sharp corner. Hard-max steps 0.05->0.8 the instant the
    # corner crosses the window edge; the tapered version must ramp smoothly.
    n = 400
    k = np.full(n, 0.05)
    k[300] = 0.8
    n_look, taper = 100, 30
    fwd = distance_tapered_forward_max(k, n_look, taper)
    seg = fwd[150:301]
    max_jump = np.abs(np.diff(seg)).max()
    hard = _hard_forward_max(k, n_look)
    hard_jump = np.abs(np.diff(hard[150:301])).max()
    assert max_jump < 0.1, f"tapered max step {max_jump} not smooth"
    assert hard_jump > 0.5, f"hard-max should step hard, got {hard_jump}"
    assert fwd[300] >= 0.79          # full κ once at the corner


def test_preserves_full_kappa_when_inside_window():
    # A corner well inside the window (past the taper) must report full κ so the
    # controller still brakes early (continuity must not weaken the cap).
    n = 200
    k = np.full(n, 0.05); k[120] = 0.6
    fwd = distance_tapered_forward_max(k, n_look=80, taper=20)
    # corner 50 steps ahead = inside the non-taper region (d < n_look-taper=60) → full κ
    assert fwd[70] >= 0.59


def test_current_point_full_weight():
    k = np.full(50, 0.05); k[10] = 0.4
    fwd = distance_tapered_forward_max(k, n_look=40, taper=10)
    assert fwd[10] >= 0.4            # κ at the current point is full-weighted
