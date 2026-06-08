"""LMPC safe-set must not learn from teleported laps.

User rule (2026-06-08): when the car gets stuck and is teleported (safe-reset)
back onto the track mid-lap, that lap's trajectory is corrupted (a discontinuous
jump that never happened under closed-loop control). Such a lap must NOT be
admitted to the LapDatabase / safe set, otherwise LMPC reinforces the crash.

By default a single teleport (n_resets >= 1) is enough to reject the lap.
"""
import numpy as np

from nonlinear_mpc_acados.mpc_core.lmpc.lap_database import LapDatabase
from nonlinear_mpc_acados.mpc_core.lmpc.nominal_dynamics import DT


def _clean_lap(T=60, vx=3.0):
    state = np.zeros((T, 8))
    state[:, 3] = vx
    inp = np.zeros((T - 1, 3))
    t = np.arange(T) * DT
    return state, inp, t


def test_default_db_rejects_lap_with_any_teleport():
    db = LapDatabase(min_lap_steps=10)
    state, inp, t = _clean_lap()
    ok = db.add_lap(3.0, state, inp, t, lap_time=state.shape[0] * DT,
                    n_resets=1, dt=DT)
    assert ok is False
    assert "n_resets" in db.last_reject_reason


def test_default_db_accepts_clean_lap():
    db = LapDatabase(min_lap_steps=10)
    state, inp, t = _clean_lap()
    ok = db.add_lap(3.0, state, inp, t, lap_time=state.shape[0] * DT,
                    n_resets=0, dt=DT)
    assert ok is True
