import numpy as np
from nonlinear_mpc_acados.mpc_core.lmpc.error_regression import epanechnikov_e_corr


def test_returns_zero_when_too_few_neighbours():
    res = np.array([[1.0, 2.0, 3.0]])
    d = np.array([0.1])
    out = epanechnikov_e_corr(res, d, h=1.0, m_min=3)
    assert np.allclose(out, 0.0)


def test_nearer_neighbours_dominate():
    residuals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    distances = np.array([0.0, 0.9])          # first point exactly at query
    out = epanechnikov_e_corr(residuals, distances, h=1.0, m_min=1)
    assert out[0] > 0.5                         # pulled toward the near point
    assert out.shape == (3,)


def test_clamps_magnitude():
    residuals = np.array([[100.0, 0.0, 0.0]])
    distances = np.array([0.0])
    out = epanechnikov_e_corr(residuals, distances, h=1.0, m_min=1, max_norm=2.0)
    assert np.linalg.norm(out) <= 2.0 + 1e-9
