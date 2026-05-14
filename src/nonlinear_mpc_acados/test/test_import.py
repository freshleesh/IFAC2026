"""Smoke test — verify mpc_core can be imported and instantiated without ROS.

Run after `colcon build --packages-select nonlinear_mpc_acados` from the
workspace root, then:
    source install/setup.bash
    python3 -m pytest src/nonlinear_mpc_acados/test/

Or standalone (no ROS env, just casadi):
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_import -v

Acados class needs `acados_template`; if not installed (host env), that
test is skipped. IPOPT class needs only `casadi`.
"""
from __future__ import annotations

import unittest


class TestRosFreeImport(unittest.TestCase):
    def test_compat_helpers(self):
        from nonlinear_mpc_acados.mpc_core._ros_compat import (
            NullLogger, monotonic_now, yaw_to_quat,
        )
        log = NullLogger("[test]")
        log.info("hello %d", 42)
        log.warn_throttle(0.01, "throttled %s", "msg")
        self.assertGreater(monotonic_now(), 0.0)
        q = yaw_to_quat(0.0)
        self.assertEqual(q, (0.0, 0.0, 0.0, 1.0))

    def test_ipopt_instantiates(self):
        from nonlinear_mpc_acados.mpc_core.ipopt_kinematic import MPC
        m = MPC(cost_type=None, system_model=None)
        self.assertIsNone(m.boundary_hook)
        self.assertEqual(m.heading(0.0), (0.0, 0.0, 0.0, 1.0))

    def test_ipopt_with_injected_logger(self):
        from nonlinear_mpc_acados.mpc_core.ipopt_kinematic import MPC
        captured = []
        class CaptureLogger:
            def info(self, msg, *args):  captured.append(("info", msg % args if args else msg))
            def warn(self, msg, *args):  captured.append(("warn", msg % args if args else msg))
            def warn_throttle(self, period, msg, *args):
                captured.append(("warnT", msg % args if args else msg))
        m = MPC(cost_type=None, system_model=None, logger=CaptureLogger())
        m._log.info("test %s", "logger")
        self.assertEqual(captured, [("info", "test logger")])

    def test_acados_instantiates(self):
        try:
            from nonlinear_mpc_acados.mpc_core.acados_kinematic import MPC
        except ImportError as e:
            self.skipTest(f"acados_template not available: {e}")
        m = MPC(cost_type=None, system_model=None)
        self.assertFalse(m.use_dynamic)
        self.assertIsNone(m.boundary_hook)


if __name__ == "__main__":
    unittest.main(verbosity=2)
