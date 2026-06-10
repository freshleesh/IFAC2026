"""Codegen path isolation — switching kinematic<->dynamic<->LMPC must never
reuse a stale acados codegen directory.

We were repeatedly bitten by a FIXED export dir (`/tmp/acados_codegen_evompcc`):
after toggling use_dynamic (nx 8<->5) or use_lmpc (nx 8<->18), acados re-used
the previous codegen and silently ran the WRONG model. The fix keys the export
dir + json to the OCP structure so each config gets its own codegen.

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_codegen_paths -v
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.acados_kinematic import codegen_paths


class TestCodegenPaths(unittest.TestCase):
    def test_kinematic_and_dynamic_differ(self):
        # 2026-06-10 unified layout: kinematic is ALSO 8-state (f_kin) — the
        # dyn/kin tag alone must keep the codegens apart at identical nx.
        kin = codegen_paths(use_dynamic=False, lmpc_joint=False, nx_solver=8)
        dyn = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        self.assertNotEqual(kin[0], dyn[0], "export dir must differ kin vs dyn")
        self.assertNotEqual(kin[1], dyn[1], "json must differ kin vs dyn")

    def test_lmpc_differs_from_plain_dynamic(self):
        dyn = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        lmpc = codegen_paths(use_dynamic=True, lmpc_joint=True, nx_solver=18)
        self.assertNotEqual(dyn[0], lmpc[0], "LMPC (nx=18) must not collide with dynamic (nx=8)")

    def test_kinematic_lmpc_differs_from_dynamic_lmpc(self):
        kin = codegen_paths(use_dynamic=False, lmpc_joint=True, nx_solver=18)
        dyn = codegen_paths(use_dynamic=True, lmpc_joint=True, nx_solver=18)
        self.assertNotEqual(kin[0], dyn[0], "kin18_lmpc must not collide with dyn18_lmpc")

    def test_deterministic_for_same_config(self):
        a = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        b = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        self.assertEqual(a, b, "same config must map to the same paths (warm reuse)")

    def test_returns_dir_and_json_pair(self):
        export_dir, json_path = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        self.assertTrue(export_dir.startswith("/tmp/"), "export dir under /tmp")
        self.assertTrue(json_path.endswith(".json"), "json path ends in .json")


if __name__ == "__main__":
    unittest.main()
