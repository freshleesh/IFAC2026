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
