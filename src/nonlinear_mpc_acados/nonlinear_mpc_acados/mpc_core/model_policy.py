"""Backend-agnostic model/LMPC policy helpers.

Kept dependency-free (no casadi / acados_template) so the ROS node can import
it regardless of which solver backend (acados or ipopt) is selected — importing
it must never pull in a heavy/optional solver dependency.
"""
from __future__ import annotations


def effective_lmpc(use_dynamic, use_lmpc):
    """Resolve whether LMPC is actually usable for the selected model.

    2026-06-10 unified layout: kinematic mode now builds the SAME 8-state
    layout [x, y, ψ, vx, vy, r, s, δ_prev] as dynamic (f_expl = f_kin, the
    kinematic single-track branch of the blended model), so slot 3 = vx in
    BOTH modes and the safe-set terminal cost / SS packing assumptions hold
    everywhere. LMPC is therefore allowed regardless of use_dynamic.

    Kept as the single policy point (rather than deleting the gate) so any
    future model whose layout diverges from [.., vx@3, vy@4, r@5, ..] re-gates
    here instead of silently corrupting the lap database again.
    """
    del use_dynamic  # unified 8-state layout — no longer a constraint
    return bool(use_lmpc)


# ─── Grip single source (2026-06-10 friction-ellipse-mu spec) ───────────────
G_GRAV = 9.81
# Solver longitudinal brake limit [m/s²]. MUST stay in sync with
# acados_kinematic lbu[0] (which imports this const — single source).
A_MIN_DYN = -3.0


def grip_a_lat_limit(mu, ellipse_frac=0.95):
    """Physical lateral-accel ceiling a_lim = μ·g·η (η = ellipse headroom)."""
    return float(mu) * G_GRAV * float(ellipse_frac)


def clamp_a_lat_to_grip(a_lat_safe, mu, ellipse_frac=0.95):
    """Clamp a requested a_lat_safe to the physical μ·g·η ceiling.

    Returns (effective_a_lat, clamped). BO/yaml can request any a_lat — the
    speed profile must never be built on grip the tire cannot deliver
    (mu=0.6 BO-best non-reproduction root cause, 2026-06-09/10).
    """
    lim = grip_a_lat_limit(mu, ellipse_frac)
    a = float(a_lat_safe)
    return (min(a, lim), a > lim)


# ─── Avoidance side decision (2026-06-11 window-aware) ──────────────────────
def decide_side_window(e_c_obs, w_left, w_right,
                       w_car_safe=0.21, margin=0.1):
    """Window-aware avoidance side decision (pure, numpy/casadi-free).

    Replaces the single-point top-2 boundary-distance compare whose
    centerline tie always returned -1 ("always avoids down" bug,
    2026-06-11). Looks at the corridor room the detour tube actually
    drives through: per window sample s_k, the gap between the obstacle's
    lateral line and each labeled boundary.

    e_c_obs  — obstacle lateral offset (solver e_c sign convention).
    w_left/w_right — signed e_c projections of the labeled left/right
        boundary at each window sample (sin_t·Δx − cos_t·Δy). The labels'
        sign flips with track orientation (CW/CCW — see the corridor
        smooth-max/min in acados_kinematic), so gaps use |w − e_c_obs|,
        orientation-agnostic.

    Returns +1 (pass on labeled-left side) or -1 (labeled-right side):
      1. one side's bottleneck < w_car_safe → the other side
      2. both blocked → larger bottleneck (less-bad, was unconditional -1)
      3. bottleneck gap differs > margin → larger bottleneck
      4. else mean room differs > margin → larger mean (the "tie at the
         obstacle but one side opens downstream" case)
      5. true tie → -1 (legacy default, deterministic)
    """
    if len(w_left) == 0 or len(w_right) == 0:
        return -1
    e = float(e_c_obs)
    gap_l = [abs(float(w) - e) for w in w_left]
    gap_r = [abs(float(w) - e) for w in w_right]
    min_l, min_r = min(gap_l), min(gap_r)
    left_blocked = min_l < w_car_safe
    right_blocked = min_r < w_car_safe
    if left_blocked and not right_blocked:
        return -1
    if right_blocked and not left_blocked:
        return +1
    if left_blocked and right_blocked:
        return +1 if min_l > min_r else -1
    if abs(min_l - min_r) > margin:
        return +1 if min_l > min_r else -1
    mean_l = sum(gap_l) / len(gap_l)
    mean_r = sum(gap_r) / len(gap_r)
    if abs(mean_l - mean_r) > margin:
        return +1 if mean_l > mean_r else -1
    return -1
