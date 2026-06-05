import numpy as np
from nonlinear_mpc_acados.mpc_core.lmpc.lap_database import LapDatabase, LapEntry
from nonlinear_mpc_acados.mpc_core.lmpc.safe_set import SafeSetLookup


def test_query_returns_residuals_aligned_with_states():
    T = 40
    state = np.zeros((T, 8))
    state[:, 0] = np.linspace(0, 4, T)      # px sweeps
    state[:, 3] = 3.0                       # vx
    state[:, 6] = np.linspace(0, 8, T)      # s
    residual = np.tile(np.array([0.1, -0.2, 0.3]), (T, 1))
    residual[:, 0] = np.linspace(0, 1, T)   # vx-residual varies -> check alignment
    entry = LapEntry(v_bucket=3.0, v_max_eff=3.0, state=state,
                     input=np.zeros((T - 1, 3)), time_step=np.arange(T) * 0.04,
                     cost_to_go=np.arange(T - 1, -1, -1.0),
                     residual=residual, lap_time=T * 0.04, n_resets=0)
    db = LapDatabase(min_lap_steps=5)
    db._db[3.0] = [entry]
    ss = SafeSetLookup(db, K_points=5, slice_window=50)
    res = ss.query(state[10], 3.0, s_curr=float(state[10, 6]), track_length=8.0)
    assert res.is_ready
    assert res.residuals.shape == (res.K, 3)
    # nearest point should be ~index 10 -> its vx-residual ~= 10/(T-1)
    assert abs(res.residuals[np.argmin(res.distances)][0] - 10.0 / (T - 1)) < 0.05
