import numpy as np
from nonlinear_mpc_acados.mpc_core.lmpc.lap_database import LapDatabase
from nonlinear_mpc_acados.mpc_core.lmpc.nominal_dynamics import predict_next, DT


def _make_lap(T=60):
    # Roll a nominal trajectory, then add a constant velocity offset to the
    # realized states so the stored residual should recover that offset.
    state = np.zeros((T, 8))
    inp = np.zeros((T - 1, 3))
    state[0] = np.array([0, 0, 0, 3.0, 0, 0, 0, 0])
    offset = np.array([0.02, -0.01, 0.05])
    for t in range(T - 1):
        u = np.array([0.1, 0.02 * np.sin(t / 5), 3.0])
        inp[t] = u
        nxt = predict_next(state[t], u, DT)
        nxt[3:6] += offset            # inject known residual
        state[t + 1] = nxt
    return state, inp, offset


def test_lap_entry_stores_velocity_residual():
    state, inp, offset = _make_lap()
    db = LapDatabase(min_lap_steps=10)
    t_arr = np.arange(state.shape[0]) * DT
    ok = db.add_lap(3.0, state, inp, t_arr, lap_time=state.shape[0] * DT, dt=DT)
    assert ok
    entry = db.get_recent(3.0, K_laps=1)[0]
    assert entry.residual.shape == (state.shape[0], 3)
    # interior transitions should recover the injected offset
    assert np.allclose(entry.residual[5:-2], offset, atol=1e-6)
