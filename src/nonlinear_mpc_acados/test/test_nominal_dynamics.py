import numpy as np
from nonlinear_mpc_acados.mpc_core.lmpc.nominal_dynamics import (
    predict_next, velocity_residual, DT)


def test_zero_input_straight_line_keeps_velocity():
    # Straight, vx=3, no lateral, no accel -> vx unchanged, vy/r stay ~0.
    state = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 3.0])           # a_x=0, delta=0, p_v=3
    nxt = predict_next(state, u, DT)
    assert abs(nxt[3] - 3.0) < 1e-6          # vx held
    assert abs(nxt[4]) < 1e-6                # vy stays 0
    assert abs(nxt[5]) < 1e-6                # r stays 0


def test_velocity_residual_is_actual_minus_predicted():
    state = np.array([0.0, 0.0, 0.0, 3.0, 0.1, 0.2, 0.0, 0.0])
    u = np.array([0.5, 0.05, 3.0])
    pred = predict_next(state, u, DT)
    offset = np.array([0.03, -0.02, 0.10])
    actual_next = pred.copy()
    actual_next[3:6] += offset
    res = velocity_residual(state, u, actual_next, DT)
    assert np.allclose(res, offset, atol=1e-9)
    assert res.shape == (3,)
