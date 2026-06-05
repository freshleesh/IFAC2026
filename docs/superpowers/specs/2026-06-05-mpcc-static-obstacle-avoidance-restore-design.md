# MPCC Static Obstacle Avoidance — Restore (Option 2)

**Date:** 2026-06-05
**Branch:** `avoidance-restore` (forked from `lmpc-joint-alpha`)
**Status:** design approved, spec for review

## Problem

The MPCC controller (`acados_kinematic.py`, EVO-MPCC acados port) is supposed to avoid
static obstacles, but the worklog flags it as "exists but unverified". Investigation of
the reference repos shows why:

- **EVO-MPCC** (our base) and **`ifac_mpcc`** (our direct IPOPT ancestor) were *built* for
  avoidance/overtaking. `ifac_mpcc` has a complete, tuned design: a soft side-pull cost,
  adaptive lane-tracking attenuation near the obstacle, a hard half-plane keep-out, side
  preference, and a warm-start lateral offset.
- During the **"Phase B (VPMPCC simplify)"** pass our acados port kept the *wiring*
  (`select_front_obstacle`, `D_detour`/`R_safe`/`side_pref`, `p_sym` slots, the hard
  half-plane `h_obs`) but **hard-coded off the two pieces that make a detour smooth and
  anticipatory**:
  - `acados_kinematic.py:886` → `side_term = ca.SX(0.0)` (soft side pull disabled)
  - `acados_kinematic.py:905` → `attenuation = ca.SX(1.0)` (lane-tracking attenuation disabled)

So today only the **reactive hard half-plane** `h_obs` (line 1071, still active) is available.

## Goal

Restore the two disabled cost pieces so MPCC produces a smooth, anticipatory static-obstacle
detour in sim — **while mathematically guaranteeing Phase B racing is unchanged** and proving
it with an A/B regression gate.

## Scope (surgical — zero new OCP decision variables)

| # | Change | Location |
|---|--------|----------|
| A | `attenuation` = `1 − 0.95·exp(−d²/(2·σ²))`, σ_obs = 1.0 | `acados_kinematic.py:905` |
| B | `side_term` = `√(abs_side · proximity_side) · (e_c − side_pref·D_detour_p)` (PSD residual) | `acados_kinematic.py:886` |
| C | *(optional)* tiny `/external_obstacles` (`PoseArray`) publisher node for repeatable scripted tests | new file |

Notes:
- The CONL cost is `ψ = ½·rᵀ·W·r`. The 7th residual (`side_term`) already has a **baked,
  sensibly-tuned weight `q_side_def = 3.0`** in `W_mat` (`:1138`, `:1166`); it currently
  multiplies zero. Restoring the residual expression is all that is needed — no codegen
  weight change.
- All variables needed for A/B are **already computed above the disabled lines**: `d2`,
  `proximity_side` (`:869`, σ_side = 0.5), `side_pref` (`:745`), `D_detour_p` (`:746`),
  `e_c`, `e_c_ref`. `abs_side` = `√(side_pref² + 1e-3)` (smooth |·|, mirrors `:1068`).
- The hard half-plane `h_obs` (`:1071`) is **left as-is** (already active).

## Regression protection (why Phase B cannot break)

With no obstacle, `self._obstacles = []` → `select_front_obstacle` returns the sentinel
`[1e6, 1e6, 1e6]` → predicted-to-obstacle distance `d² ≈ 1e12`. Therefore:

- `proximity_side = exp(−d²/2σ²) = 0` → `side_term = 0` (independent of `side_pref`).
- `attenuation = 1 − 0.95·exp(−d²/2σ²) = 1.0` → lane-tracking residual unchanged.

So the **no-obstacle cost function is bit-identical to current Phase B**. This is the
invariant the A/B gate below verifies empirically.

## Incremental plan (each step has a regression gate AND an avoidance gate)

- **Phase 0 — characterize current state (no code change).** `final` map, click one
  obstacle on the racing line in RViz (`/clicked_point` → `self._obstacles`, already wired).
  Observe whether the active hard `h_obs` alone produces a detour, and whether it is late or
  jerky. Establishes the baseline avoidance behavior and resolves "unverified".
- **Phase 1 — restore attenuation (change A).** Expect the centerline pull to relax near the
  obstacle so the detour is smoother (less brake-to-release-tension, lower jerk).
- **Phase 2 — restore side_term (change B).** Expect an anticipatory pull that commits to the
  detour earlier.

(Optional change C can be added before Phase 0 if scripted/repeatable obstacle placement is
preferred over manual RViz clicks.)

## Success criteria

For **every** phase, both gates must pass:

1. **Identity gate (mandatory, regression guard).** `final`, dynamic, deploy config, **zero
   obstacles** → median ≈ **21 s, 0 contacts, shake ≈ 0.05**, matching the current baseline
   within noise. If it diverges, a gate has leaked → stop, do not merge.
2. **Avoidance gate.** One obstacle on the racing line → no contact, the trajectory detours
   around it and returns to the line after passing.

Telemetry to watch: `[dbg] side_pref`, `obs_dmin`, `[pred-consistency] rms` (jerk proxy),
`STUCK`/`stuck-recover`, and closest center-to-center approach vs `R_safe + R_car`.

## Reproduction

```bash
# sim (regen codegen if model changed):
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final
# place obstacle: RViz "Publish Point" tool → click on racing line
```
Kill background sims by PID (pattern kill self-terminates — see `alat_sweep_final` memory).

## Isolation & out of scope

- All work on branch `avoidance-restore`; deploy racing config and B0–B4' code untouched.
- **Out of scope (YAGNI):** multi-obstacle (keep single front-select), dynamic overtaking
  (separate large workstream), Cartesian slack-relaxed constraint (Option 1 — adds N OCP
  slack vars, risks the B3 joint-α / CONL structure), BO (each eval launches the gym sim, so
  it cannot run concurrently with avoidance sim under the "one sim at a time" rule — separate
  session).

## Reference formulas (ancestor `ifac_mpcc/mpc_core/mpc_core/ipopt_kinematic.py`)

- attenuation: `1.0 - 0.95 * exp(-dist2_0 / (2*sigma_obs²))` (`:202`)
- side cost: `W_SIDE * |side_pref| * proximity_side * (e_c - side_pref*D_DETOUR)²` (`:217-221`)

In our CONL form the residual carries `√` of the above so that `½·q_side·side_term²`
reproduces the penalty (PSD), as the `acados_kinematic.py:852-856` comment already prescribes.
