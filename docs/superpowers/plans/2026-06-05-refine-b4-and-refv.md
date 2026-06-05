# Refine ①(B4' robust) + ②(max_speed decouple + ref_v κ smooth) — Plan

> Execute via subagent-driven-development (TDD where pure-python, sim-validate for acados/integration). Follows the B4' session (worklog §7). User-approved design 2026-06-05.

**Goal:** Make B4' robustly reduce prediction error (not noise-range), and remove the high-speed trajectory shake at its root, so higher max_speed becomes harmless.

**Why (from B4' §7 data):** B4' gain was −2.3%~+6.3% (noise) because SS neighbours are position-near but velocity-far (W position-heavy) and e_corr jitters cycle-to-cycle. Trajectory shake at high speed traces to ref_v's forward-max κ stepping as corners enter/exit the (v²-scaled) lookahead window, plus max_speed actively driving the progress target + window length (not a passive cap).

**Baseline to preserve:** a_lat=7.14/max=6 = 23.7s/STUCK0/shake0.05. Every task must keep this clean (sim regression check).

---

## R1 — B4' velocity-space Epanechnikov weighting (mpc_node)
**File:** `nonlinear_mpc_acados/mpc_node.py` — the B4'.3 block in `_lmpc_update_per_cycle` (currently computes `_h_mult * max(res.distances)`).
**Change:** weight the K neighbours' residuals by **velocity-space distance** (current state's vx,vy,r ↔ each neighbour's vx,vy,r = `res.states[:,3:6]`), NOT the position-heavy SS `res.distances`. This reuses the same K neighbours but makes the kernel pick velocity-local ones (the regime e_corr corrects). Does NOT touch the SS query (LMPC keeps its position-near terminal).
```python
# in B3 block, replace res.distances usage for the kernel:
_qv = np.asarray(state8, float)[3:6]              # current velocity (vx,vy,r)
_nv = np.asarray(res.states, float)[:, 3:6]        # neighbour velocities (K,3)
_vel_d = np.linalg.norm(_nv - _qv[None, :], axis=1)  # (K,)
_h = float(self.get_parameter('err_regr_bandwidth').value) * max(float(_vel_d.max()), 1e-6)
self.mpc._e_corr = epanechnikov_e_corr(res.residuals, _vel_d, h=_h) / max(_dt, 1e-6)
```
(`state8` is in scope in `_lmpc_update_per_cycle`; `res.states` is (K, n_state).)
**Verify (sim):** under baseline (a_lat=7.14/max=6, B4'on), windowed B4'-pred corrected < nominal **consistently** across 2+ runs (no longer flipping sign). STUCK=0 preserved.

## R2 — e_corr EMA temporal smoothing (mpc_node + yaml)
**Change:** after computing the new e_corr (R1), low-pass it:
```python
_beta = float(self.get_parameter('err_regr_ema').value)   # default 0.8
_prev = np.asarray(getattr(self.mpc, '_e_corr', np.zeros(3)), float)
_newc = epanechnikov_e_corr(...) / max(_dt,1e-6)
self.mpc._e_corr = _beta * _prev + (1.0 - _beta) * _newc
```
Declare `err_regr_ema` (default 0.8) near `err_regr_bandwidth`. Zero-bail paths set `_e_corr=0` (keep). 
**Verify (sim):** pred-consistency rms with B4'on drops toward B4'off level (was 0.35 vs 0.20 at max=8). Baseline still clean.

## R3 — decouple max_speed into cap / target / lookahead (acados)
**File:** `mpc_core/acados_kinematic.py`. `self.v_max` currently drives THREE things; split:
- **hard cap** (keep `v_max`): vx ubx `v_max+0.5` (line ~1236), ubu `v_max` (~1245).
- **progress target** → new `self.speed_target` (default = v_max): line ~935 `sqrt_q_p_scale * (p_v - self.v_max)` → `(p_v - self.speed_target)`. Also terminal/any `(... - self.v_max)` cost target (NOT the caps).
- **lookahead window** → new `self.lookahead_m` (default = current formula): line 336 `LOOKAHEAD_M = max(6.0, v_max²/6.0)` → use `self.lookahead_m if set else max(6.0, speed_target²/6.0)`.
Add params `speed_target`, `lookahead_m` (set from yaml in mpc_node before setup_MPC; default to v_max / formula so behavior unchanged when unset). 
**Verify (sim):** with speed_target=6, lookahead_m=6 fixed, raising `max_speed`(cap) 6→8 keeps STUCK low and pred-shake low (cap↑ now harmless — the user's hypothesis). Baseline (all defaults) unchanged.

## R4 — ref_v forward-κ distance-weighted soft-max (acados track setup)
**File:** `acados_kinematic.py` line ~338-342 (the `abs_k_fwd[i] = abs_k_arr[i:j].max()` hard max over window).
**Change:** replace hard max with a **distance-weighted soft-max** so a corner entering the far window edge contributes continuously (weight ramps 0→1 as it approaches), removing the step that jumps ref_v:
```python
beta_k = 8.0   # soft-max sharpness
dvec = np.arange(n_look + 1) * self.kappa_ds
wdist = np.maximum(0.0, 1.0 - dvec / max(LOOKAHEAD_M, 1e-6))   # linear distance decay, 0 at edge
for i in range(n_grid):
    j = min(i + n_look + 1, n_grid)
    seg = abs_k_arr[i:j]; w = wdist[:j - i]
    # weighted soft-max ≈ tightest-corner-ahead but continuous at the window edge
    m = seg.max()
    abs_k_fwd[i] = m if m <= 1e-9 else (1.0/beta_k) * np.log(np.sum(w * np.exp(beta_k * (seg - m))) + 1e-12) + m
```
**Verify:** (offline) plot/inspect `abs_k_fwd` is continuous (no steps) vs the old hard-max — unit-testable on a synthetic κ profile (a single corner sweeping into the window → abs_k_fwd ramps smoothly, not steps). (sim) pred-consistency rms drops at higher speed; baseline lap unchanged.

---

## Execution order & checkpoints
R2 → R1 (both mpc_node B3, do together) → sim-validate B4' robust. Then R3 → R4 (acados) → sim-validate high-speed harmless + shake↓. Each: keep baseline a_lat=7.14/max=6 clean. Commit per task. Update worklog §8 with results.

## Deferred: ③ real-car (user gives direction after ①②). Needs EKF vy/r observability gate + safety fallback.
