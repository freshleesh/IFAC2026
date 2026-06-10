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
