"""LapDatabase persistence + seed bookkeeping correctness.

1. save_all/load_all must round-trip the per-transition velocity residual
   (B4' error-regression data). It was silently dropped on save, so any
   save→load reverted the learned correction to identity.
2. seed_from_raceline's +5.0 cost-to-go tie-break must apply to the SEED, not
   to whatever happens to be last after buffer eviction (which can be a fast
   real lap if the seed was evicted).
"""
import numpy as np

from nonlinear_mpc_acados.mpc_core.lmpc.lap_database import LapDatabase
from nonlinear_mpc_acados.mpc_core.lmpc.nominal_dynamics import predict_next, DT


def _lap_with_residual(T=60, vx=3.0):
    state = np.zeros((T, 8))
    inp = np.zeros((T - 1, 3))
    state[0] = np.array([0, 0, 0, vx, 0, 0, 0, 0])
    offset = np.array([0.02, -0.01, 0.05])
    for t in range(T - 1):
        u = np.array([0.1, 0.02 * np.sin(t / 5), vx])
        inp[t] = u
        nxt = predict_next(state[t], u, DT)
        nxt[3:6] += offset
        state[t + 1] = nxt
    return state, inp


def test_save_load_round_trips_residual(tmp_path):
    db = LapDatabase(min_lap_steps=10)
    state, inp = _lap_with_residual()
    t = np.arange(state.shape[0]) * DT
    assert db.add_lap(3.0, state, inp, t, lap_time=state.shape[0] * DT, dt=DT)
    e0 = db.get_recent(3.0, 1)[0]
    assert e0.residual.shape == (state.shape[0], 3)
    assert np.abs(e0.residual).sum() > 0.0       # genuinely nonzero residual

    path = tmp_path / "ss.npz"
    db.save_all(path)
    db2 = LapDatabase(min_lap_steps=10)
    db2.load_all(path)
    e1 = db2.get_recent(3.0, 1)[0]
    assert e1.residual.shape == e0.residual.shape, "residual dropped on save/load"
    np.testing.assert_allclose(e1.residual, e0.residual, atol=1e-9)


def test_seed_inflation_not_misapplied_to_real_lap_after_eviction():
    # buffer of 1: a real fast lap, then a slightly-slower raceline seed that
    # passes the ratio filter but is then evicted as the worst. The +5.0 must
    # NOT land on the surviving fast real lap.
    db = LapDatabase(buffer_per_bucket=1, min_lap_steps=5)
    T = 20
    real_state = np.zeros((T, 8)); real_state[:, 3] = 5.0
    real_inp = np.zeros((T - 1, 3))
    tt = np.arange(T) * DT
    assert db.add_lap(3.0, real_state, real_inp, tt, lap_time=1.0, dt=DT)
    real = db.get_recent(3.0, 1)[0]
    cost0 = real.cost_to_go.copy()

    # raceline seed: lap_time = path_len / v_mean = 1.4 (< 1.5×best=1.5 → passes
    # ratio filter, then evicted as worst since 1.4 > 1.0).
    xy = np.stack([np.linspace(0.0, 1.4, T), np.zeros(T)], axis=1)
    psi = np.zeros(T); v = np.ones(T); s = np.linspace(0.0, 1.4, T)
    db.seed_from_raceline(3.0, xy, psi, v, s, dt=DT)

    laps = db.get_recent(3.0, 10)
    survivors = [e for e in laps if not e.metadata.get("synthetic", False)]
    assert len(survivors) == 1
    np.testing.assert_array_equal(
        survivors[0].cost_to_go, cost0,
        err_msg="real lap wrongly inflated by seed +5.0 after eviction")
