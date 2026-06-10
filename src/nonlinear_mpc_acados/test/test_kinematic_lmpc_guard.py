"""kinematic + LMPC policy — unified 8-state layout.

2026-06-10: kinematic mode now builds the SAME 8-state layout
[x, y, ψ, vx, vy, r, s, δ_prev] as dynamic (f_expl = f_kin, the kinematic
single-track branch of the blended model). Slot 3 = vx in BOTH modes, so the
safe-set terminal cost / SS packing assumptions hold and LMPC is allowed for
kinematic too. effective_lmpc() is now a pass-through on use_lmpc — kept as
the single policy point so any future layout divergence re-gates here.

Run:
    cd src/nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest \
        test/test_kinematic_lmpc_guard.py -q
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.model_policy import effective_lmpc


class TestEffectiveLmpc(unittest.TestCase):
    def test_dynamic_lmpc_stays_on(self):
        self.assertTrue(effective_lmpc(use_dynamic=True, use_lmpc=True))

    def test_kinematic_lmpc_allowed(self):
        # unified 8-state layout: kinematic slot 3 = vx too → LMPC valid
        self.assertTrue(effective_lmpc(use_dynamic=False, use_lmpc=True))

    def test_lmpc_off_stays_off_dynamic(self):
        self.assertFalse(effective_lmpc(use_dynamic=True, use_lmpc=False))

    def test_lmpc_off_stays_off_kinematic(self):
        self.assertFalse(effective_lmpc(use_dynamic=False, use_lmpc=False))


if __name__ == "__main__":
    unittest.main()
