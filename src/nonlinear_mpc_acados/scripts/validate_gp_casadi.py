#!/usr/bin/env python3
"""validate_gp_casadi.py — numerically validate the CasADi GP-mean export.

Compares the closed-form CasADi posterior mean (gp_casadi_residual.py) against
the original gpytorch predictive mean at ~200 random test inputs drawn within
the training input range. Reports max/mean abs error per output.

OFFLINE ONLY. Requires:
  PYTHONPATH=$HOME/l4acados/src python3 validate_gp_casadi.py
(needs torch + gpytorch + l4acados to load the reference model + casadi.)

Pass/fail: max abs error per output must be < 1e-3 (ideally < 1e-5).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# make the package importable without colcon build
_PKG = Path(__file__).resolve().parents[1] / "nonlinear_mpc_acados"
sys.path.insert(0, str(_PKG.parent))

import casadi as ca  # noqa: E402
import torch  # noqa: E402
import gpytorch  # noqa: E402

from nonlinear_mpc_acados.mpc_core.gp_casadi_residual import (  # noqa: E402
    load_gp_casadi_params,
    make_casadi_function,
    eval_numpy,
    _default_train_data_path,
)

CKPT = os.path.expanduser("~/bo_results/gp_residual_realvy.pt")
N_TEST = 200
SEED = 7


def build_conditioned_reference(ckpt_path):
    """Rebuild the gpytorch SGPR CONDITIONED on its real training data, so its
    posterior mean is the true reference (not the ZeroMean prior)."""
    from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
        BatchIndependentInducingPointGPModel,
    )
    ckpt = torch.load(ckpt_path, weights_only=False)
    out_dim = len(ckpt["output_keys"])
    data_path = _default_train_data_path(ckpt_path)
    blob = torch.load(data_path, weights_only=False)
    Xm, Xs = ckpt["X_mean"].double(), ckpt["X_std"].double()
    Ym, Ys = ckpt["Y_mean"].double(), ckpt["Y_std"].double()
    Xn = (blob["x_train"].double() - Xm) / Xs
    Yn = (blob["y_train"].double() - Ym) / Ys
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=out_dim).double()
    lik.load_state_dict(ckpt["lik_state"])
    gp = BatchIndependentInducingPointGPModel(
        Xn, Yn, lik, inducing_points=ckpt["inducing"], use_ard=True,
        residual_dimension=out_dim,
    ).double()
    gp.load_state_dict(ckpt["gp_state"])
    gp.set_train_data(inputs=Xn, targets=Yn, strict=False)
    gp.eval(); lik.eval()
    return gp


def gpytorch_mean(gp, p, z_raw: np.ndarray) -> np.ndarray:
    """Reference gpytorch predictive mean (real scale). z_raw: (N, D) raw."""
    z_norm = (z_raw - p.X_mean[None, :]) / p.X_std[None, :]
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        pred = gp(torch.from_numpy(z_norm).double())
        mu_norm = pred.mean.cpu().numpy().astype(np.float64)   # (N, out_dim)
    return mu_norm * p.Y_std[None, :] + p.Y_mean[None, :]


def main() -> int:
    print(f"Loading params + reference model from {CKPT}")
    p = load_gp_casadi_params(CKPT)
    gp = build_conditioned_reference(CKPT)

    print(f"  M (inducing points)  = {p.M}")
    print(f"  in_dim / out_dim     = {p.in_dim} / {p.out_dim}")
    print(f"  input_keys           = {p.input_keys}")
    print(f"  output_keys          = {p.output_keys}")
    print(f"  outputscale          = {p.outputscale}")
    print(f"  lengthscale[0]       = {p.lengthscale[0]}")

    # --- test inputs: within training input range -------------------------
    # Draw HALF uniformly in X_mean +- 2.5 std, and HALF as small jitters
    # around random inducing points (de-normalized) so we probe regions where
    # the posterior mean is actually non-trivial (not just ~0 far from data).
    rng = np.random.default_rng(SEED)
    n_half = N_TEST // 2
    lo = p.X_mean - 2.5 * p.X_std
    hi = p.X_mean + 2.5 * p.X_std
    z_unif = rng.uniform(lo[None, :], hi[None, :], size=(n_half, p.in_dim))
    # inducing points are normalized -> de-normalize to raw, add small jitter
    Z_raw = p.Z * p.X_std[None, :] + p.X_mean[None, :]
    pick = rng.integers(0, p.M, size=N_TEST - n_half)
    z_near = Z_raw[pick] + 0.05 * p.X_std[None, :] * rng.standard_normal(
        (N_TEST - n_half, p.in_dim))
    z_test = np.vstack([z_unif, z_near])

    # --- reference (gpytorch) ----------------------------------------------
    ref = gpytorch_mean(gp, p, z_test)                          # (N, out_dim)

    # --- closed-form numpy (sanity, no casadi) -----------------------------
    np_out = eval_numpy(z_test, p)

    # --- CasADi function ----------------------------------------------------
    fun = make_casadi_function(p)
    cas_out = np.zeros_like(ref)
    for i in range(N_TEST):
        cas_out[i] = np.asarray(fun(z_test[i])).reshape(-1)

    # --- report -------------------------------------------------------------
    def report(name, a):
        err = np.abs(a - ref)
        print(f"\n[{name} vs gpytorch]  per-output abs error:")
        ok = True
        for d, key in enumerate(p.output_keys):
            mx, mn = err[:, d].max(), err[:, d].mean()
            ref_mag = np.abs(ref[:, d]).max()
            flag = "OK" if mx < 1e-5 else ("OK(<1e-3)" if mx < 1e-3 else "FAIL")
            if mx >= 1e-3:
                ok = False
            print(f"  {key:6s}: max={mx:.3e}  mean={mn:.3e}  "
                  f"(|ref| max over test = {ref_mag:.3e})  [{flag}]")
        print(f"  overall max abs error = {err.max():.3e}")
        return ok

    ok_np = report("numpy closed-form", np_out)
    ok_ca = report("CasADi", cas_out)

    # --- expression size ----------------------------------------------------
    print(f"\nCasADi expression size:")
    print(f"  function n_instructions = {fun.n_instructions()}")
    print(f"  function n_nodes        = {fun.n_nodes()}")
    print(f"  (M={p.M} inducing points x {p.out_dim} outputs)")

    passed = ok_np and ok_ca
    print(f"\n{'PASS' if passed else 'FAIL'}: "
          f"CasADi export {'matches' if passed else 'does NOT match'} gpytorch.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
