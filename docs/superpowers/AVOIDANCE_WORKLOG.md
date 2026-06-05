# Avoidance Restore — Worklog

Branch `avoidance-restore`. Plan: `docs/superpowers/plans/2026-06-05-mpcc-static-obstacle-avoidance-restore.md`.

## Task 0 — characterize current state (2026-06-05)

**Setup:** `final` map, dynamic, deploy config (cap=8 / speed_target=6 / lookahead_m=6),
`acados_kinematic.py` unmodified. Sim via `full_sim.launch.py mode:=mpcc map:=final`.

### Baseline (no obstacle) — identity-gate reference for Tasks 1–2
- Lap times: 20.20 / 21.24 / 20.40 / 21.28 / 21.88 s → **median ≈ 21.2 s** (range 20.2–21.9).
- STUCK: 3 single-cycle `stuck-recover` over 5 laps (in family with deploy "STUCK6").
- shake (`[pred-consistency] rms`): < 0.02 (≈0.003–0.017).
- feasibility: `feas=Y` throughout.

### Current-state avoidance probe (3 obstacles clicked: (-4.41,1.04), (-0.12,3.00), (-0.99,-4.50))
- **Avoidance DOES engage** (hard `h_obs` path is functional, contrary to "unverified"):
  `[MPC] committed obs=… side=-1`, consistent cached side (no flips), `[MPC] passed committed
  obs (Δs≈1.5 m) — release` after passing. Selection + commit + side-decision all work.
- **But it is rough** (expected for reactive-only, no soft pull / attenuation):
  cost spikes 27 → **405–428**, speed drops to ~2.4 m/s, stuck-recovers accumulate
  (~13 extra over the obstacle period; 16 total run). On the tight `final` map with 3
  obstacles the car is in a near-continuous commit→pass cycle.

### Gotchas for measurement
- The CSV `side_pref` column (mpc_*.csv col 16) reads **0 even while committed** — unreliable.
  Measure engagement from the **`[MPC] committed` / `passed` log lines**, and roughness from
  the `[dbg]` `cost=`/`v=` fields + `stuck-recover` count.
- 3 obstacles on a 76 m tight loop is a stress test, not a clean A/B. **Next: use a single
  scripted obstacle (Task 3 publisher) at a fixed coordinate** for reproducible before/after
  comparison of Tasks 1 (attenuation) and 2 (side_term).

### Plan adjustment
Reorder: do **Task 3 (scriptable single-obstacle publisher) before Tasks 1–2** so the
avoidance gate is a controlled, repeatable single-obstacle detour rather than manual
multi-clicks. Identity gate (no-obstacle == baseline) is unchanged.

## Task 3 — scriptable single-obstacle publisher (2026-06-05)

Created `src/nonlinear_mpc_acados/scripts/pub_static_obstacle.py` (standalone rclpy node,
publishes one `PoseArray` pose to `/external_obstacles` at 10 Hz; default coord = Task-0-
verified on-line point `(-4.41, 1.04)`, s_obs≈29.5). Verified: topic echo OK, and mpc_node
picks it up → `[MPC] committed obs=(-4.41,1.04) s_obs=29.50 e_c_obs=-0.09 side=-1`.

### Single-obstacle BEFORE reference (current code, hard-`h_obs` only) — the A/B baseline
- Lap times: 20.54 / 21.36 / 20.88 / 21.60 s → **median ≈ 21.1 s** — essentially unchanged
  vs no-obstacle baseline (21.2 s). A single on-line obstacle costs ~no lap time.
- STUCK: **2** (≤ no-obstacle baseline). Single obstacle is handled cleanly.
- **Obstacle-pass cost spike (s≈25–34):** climbs over laps 26 → 52 → 64 → **85**, overall
  peak **114–193** (vs no-obstacle ~27–47). Speed dips to ~2.26 m/s by later laps.
- Commit/side/release all correct (side=-1, cached, no flips).

**Conclusion:** the Task 0 "roughness" (cost 405, 16 stucks) was the 3-obstacle stress case,
not a single obstacle. The real improvement target for Tasks 1–2 is the **obstacle-pass cost
spike + speed dip (detour smoothness)**, not lap time or stuck. A/B metric = peak cost during
the pass (BEFORE peak ~114–193) and min speed (BEFORE ~2.26).
