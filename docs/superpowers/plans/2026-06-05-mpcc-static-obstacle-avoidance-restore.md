# MPCC Static Obstacle Avoidance Restore — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable the two Phase-B-disabled MPCC cost pieces (lane-tracking attenuation + soft side-pull) so the controller produces a smooth, anticipatory static-obstacle detour in sim, gated so that with no obstacle the racing behavior is identical to current Phase B.

**Architecture:** Two surgical edits to the CONL cost residual in `acados_kinematic.py` (no new OCP decision variables). The hard half-plane keep-out `h_obs` is already active. Both restored terms are gated by `d²` to the obstacle, so the sentinel obstacle (no obstacle present) zeroes them exactly. An optional standalone ROS node publishes a static obstacle for scriptable/repeatable tests; otherwise RViz "Publish Point" is used.

**Tech Stack:** Python, CasADi (symbolic cost), acados (codegen MPC), ROS2 (Foxy/Humble), f1tenth_gym_ros sim.

---

## Verification approach (read before starting)

This is a controller-cost change verified by **sim behavior**, not pytest. Default TDD with a
unit test is intentionally skipped: the only meaningful unit test would re-implement the two
formulas in numpy and assert their values, which duplicates the formula (DRY violation) and
does **not** test the actual acados codegen. The real guarantees are two sim gates run on
every code task:

- **Identity gate (regression guard):** sim with **zero obstacles** must reproduce the Task 0
  baseline lap time / STUCK / shake within noise. This empirically proves the restored terms
  are inert with no obstacle.
- **Avoidance gate:** sim with **one obstacle on the racing line** must detour without contact
  and return to the line.

Every code edit changes the cost expression, so **codegen must be regenerated** each run:
`rm -rf /tmp/acados_codegen_evompcc`.

Operational rules (from project memory): only one sim at a time; kill sims by **PID**
(`pkill -f` patterns can self-terminate the launching shell); run long waits in background.

### Shared commands

```bash
# Launch sim (regen codegen first — REQUIRED after any cost edit):
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final

# Telemetry greps (run against the launch stdout or the captured log):
#   lap / speed / steer:      grep "\[dbg\] lap="
#   trajectory jerk proxy:    grep "\[pred-consistency\] rms"
#   stuck/wedge events:       grep "stuck-recover"
#   obstacle engagement:      grep -E "side_pref|obs_dmin"

# Place an obstacle on the racing line: in RViz use the "Publish Point" tool
# and click a point on the global racing line (the click is logged so you can
# record its x,y). mpc_node._clicked_point_cb appends it to self._obstacles.
```

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` | MPCC CONL cost — restore `attenuation` (`:905`) and `side_term` (`:886`) | Modify |
| `src/nonlinear_mpc_acados/scripts/pub_static_obstacle.py` | *(optional)* standalone node publishing one static obstacle to `/external_obstacles` for scriptable tests | Create |
| `docs/superpowers/B4_WORKLOG.md` (or a new `AVOIDANCE_WORKLOG.md`) | Record per-task sim results | Append |

---

## Task 0: Characterize current state (no code change)

Establishes the identity-gate baseline AND resolves the "unverified" status of the existing
hard `h_obs`-only avoidance.

**Files:** none (observation only).

- [ ] **Step 1: Confirm clean tree on the branch**

Run: `git status && git branch --show-current`
Expected: branch `avoidance-restore`; no edits to `acados_kinematic.py`.

- [ ] **Step 2: Run the no-obstacle baseline**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task0_baseline.log
```
Let it run ~4 laps, then kill by PID.

- [ ] **Step 3: Record baseline metrics**

```bash
grep -c "stuck-recover" /tmp/avoid_task0_baseline.log          # STUCK count
grep "\[pred-consistency\] rms" /tmp/avoid_task0_baseline.log | tail -5
grep "\[dbg\] lap=" /tmp/avoid_task0_baseline.log | tail -10    # lap time / v
```
Expected (deploy config cap=8/speed_target=6/lookahead_m=6): ≈ **21–22 s/lap, STUCK low,
shake ≈ 0.05**. Write these exact numbers into the worklog — they are the identity-gate
reference for Tasks 1–2.

- [ ] **Step 4: Run the current-state avoidance probe (hard `h_obs` only)**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task0_obstacle.log
```
When the car is lapping cleanly, in RViz use "Publish Point" to click one point on the racing
line ~5–10 m ahead of the car. Observe: does the car detour, brake, or hit it? Kill by PID.

- [ ] **Step 5: Record current avoidance behavior**

```bash
grep -E "side_pref|obs_dmin" /tmp/avoid_task0_obstacle.log | tail -20
grep "stuck-recover" /tmp/avoid_task0_obstacle.log | wc -l
```
Note in the worklog whether `side_pref` becomes ±1, the min `obs_dmin` reached, and whether
the detour was late/jerky or absent. This is the qualitative baseline the restored terms must
improve.

- [ ] **Step 6: Commit the worklog note**

```bash
git add docs/superpowers/*WORKLOG*.md
git commit -m "docs: avoidance Task 0 — baseline (no-obstacle) + current hard-h_obs probe"
```

---

## Task 1: Restore lane-tracking attenuation (change A)

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py:905`

- [ ] **Step 1: Make the edit**

Replace this line (905):

```python
        attenuation = ca.SX(1.0)
```

with:

```python
        # Restored (avoidance Option 2): lane-tracking attenuation near the
        # selected obstacle, mirrors ifac_mpcc ipopt_kinematic.py:202. Fades
        # the e_c/e_l contouring cost to ~5% at the obstacle center so the
        # detour is "free" and not fighting the centerline pull. Gated by d²
        # to the obstacle (line 837): no obstacle → sentinel → d²≈1e12 →
        # proximity_atten=0 → attenuation=1.0 (Phase-B-identical).
        sigma_atten     = 1.0
        proximity_atten = ca.exp(-d2 / (2.0 * sigma_atten * sigma_atten))
        attenuation     = 1.0 - 0.95 * proximity_atten
```

Leave line 906 (`att_kappa = ca.SX(1.0)`) unchanged.

- [ ] **Step 2: Identity gate — run no-obstacle sim**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task1_identity.log
```
Run ~4 laps, kill by PID.

- [ ] **Step 3: Identity gate — verify match**

```bash
grep "\[dbg\] lap=" /tmp/avoid_task1_identity.log | tail -10
grep -c "stuck-recover" /tmp/avoid_task1_identity.log
grep "\[pred-consistency\] rms" /tmp/avoid_task1_identity.log | tail -5
```
Expected: **matches Task 0 baseline within noise** (lap time ±0.3 s, STUCK same, shake ±0.01).
If it diverges, the gate leaked — STOP, do not proceed, investigate (the sentinel gate is not
zeroing the term).

- [ ] **Step 4: Avoidance gate — run obstacle sim**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task1_obstacle.log
```
Click one point on the racing line ahead of the car (RViz "Publish Point"). Kill by PID.

- [ ] **Step 5: Avoidance gate — verify smoother detour**

```bash
grep -E "side_pref|obs_dmin" /tmp/avoid_task1_obstacle.log | tail -20
grep "\[pred-consistency\] rms" /tmp/avoid_task1_obstacle.log | tail -10
```
Expected vs Task 0: no contact (`obs_dmin` stays above `R_safe`≈0.3 + car), and the detour is
**smoother** — lower `pred-consistency rms` during the pass and less hard braking than the
hard-`h_obs`-only Task 0 probe.

- [ ] **Step 6: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py docs/superpowers/*WORKLOG*.md
git commit -m "nonlinear_mpc_acados: avoidance A — restore lane-tracking attenuation (gated)"
```

---

## Task 2: Restore soft side-pull (change B)

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py:886`

- [ ] **Step 1: Make the edit**

Replace this line (886):

```python
        side_term = ca.SX(0.0)
```

with:

```python
        # Restored (avoidance Option 2): soft side-pull toward the detour line
        # e_c = side_pref·D_detour, mirrors ifac_mpcc ipopt_kinematic.py:217-221.
        # PSD residual form: ½·q_side·side_term² reproduces the penalty
        # q_side·|side_pref|·proximity_side·(e_c − side_pref·D_detour)²/2.
        # Gated by proximity_side (σ_side=0.5, line 869) AND |side_pref|, so
        # no obstacle → sentinel → proximity_side=0 → side_term=0 (Phase-B-
        # identical). q_side baked weight = q_side_def=3.0 (W_mat, line 1166).
        # NOTE: detour line is centerline-relative (side_pref·D_detour); the
        # hard half-plane h_obs (line 1071) guarantees actual clearance using
        # the obstacle's Frenet offset e_c_obs. If a later off-centerline-
        # obstacle test shows the soft pull fighting the obstacle, switch
        # side_target to (e_c_obs_p + side_pref·R_safe_p) — out of scope now.
        abs_side    = ca.sqrt(side_pref * side_pref + 1e-3)
        side_target = side_pref * D_detour_p
        side_term   = ca.sqrt(abs_side * proximity_side) * (e_c - side_target)
```

- [ ] **Step 2: Identity gate — run no-obstacle sim**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task2_identity.log
```
Run ~4 laps, kill by PID.

- [ ] **Step 3: Identity gate — verify match**

```bash
grep "\[dbg\] lap=" /tmp/avoid_task2_identity.log | tail -10
grep -c "stuck-recover" /tmp/avoid_task2_identity.log
grep "\[pred-consistency\] rms" /tmp/avoid_task2_identity.log | tail -5
```
Expected: **matches Task 0 baseline within noise**. If it diverges, STOP and investigate.

- [ ] **Step 4: Avoidance gate — run obstacle sim**

```bash
rm -rf /tmp/acados_codegen_evompcc
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final 2>&1 | tee /tmp/avoid_task2_obstacle.log
```
Click one point on the racing line ahead of the car. Kill by PID.

- [ ] **Step 5: Avoidance gate — verify anticipatory detour**

```bash
grep -E "side_pref|obs_dmin" /tmp/avoid_task2_obstacle.log | tail -20
grep "\[pred-consistency\] rms" /tmp/avoid_task2_obstacle.log | tail -10
```
Expected vs Task 1: detour begins **earlier** (the prediction bends laterally before the
obstacle enters the dynamics horizon), no contact, returns to line after passing. If the IPM
blows up (cost spikes, `S_MINSTEP`, teleport), reduce engagement by lowering `sigma_side`
(0.5 → 0.4) or the baked `q_side_def` (3.0 → 2.0) at line 1166 and re-run from Step 1.

- [ ] **Step 6: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py docs/superpowers/*WORKLOG*.md
git commit -m "nonlinear_mpc_acados: avoidance B — restore soft side-pull (PSD, gated)"
```

---

## Task 3 (optional): Scriptable static-obstacle publisher

Only do this if you want repeatable obstacle placement instead of manual RViz clicks (e.g. for
regression runs). Standalone script — no `setup.py`/colcon rebuild needed.

**Files:**
- Create: `src/nonlinear_mpc_acados/scripts/pub_static_obstacle.py`

- [ ] **Step 1: Create the publisher**

```python
#!/usr/bin/env python3
"""Publish a single static obstacle to /external_obstacles (PoseArray) for
avoidance testing. mpc_node overwrites self._obstacles from this topic each
message, so the obstacle persists while this runs.

Usage: python3 pub_static_obstacle.py --x 1.23 --y 4.56 [--frame map]
Pick (x, y) from one RViz 'Publish Point' click (it is logged) or from the
global racing-line CSV.
"""
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class StaticObstaclePub(Node):
    def __init__(self, x, y, frame):
        super().__init__('static_obstacle_pub')
        self.x, self.y, self.frame = x, y, frame
        self.pub = self.create_publisher(PoseArray, '/external_obstacles', 1)
        self.create_timer(0.1, self._tick)  # 10 Hz
        self.get_logger().info(f'publishing obstacle at ({x:.2f}, {y:.2f}) on {frame}')

    def _tick(self):
        msg = PoseArray()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        p = Pose()
        p.position.x = self.x
        p.position.y = self.y
        msg.poses = [p]
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, required=True)
    ap.add_argument('--y', type=float, required=True)
    ap.add_argument('--frame', type=str, default='map')
    a = ap.parse_args()
    rclpy.init()
    node = StaticObstaclePub(a.x, a.y, a.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify it publishes**

With a sim running, in a second terminal:
```bash
python3 src/nonlinear_mpc_acados/scripts/pub_static_obstacle.py --x <X> --y <Y> &
ros2 topic echo /external_obstacles --once
```
Expected: a `PoseArray` with one pose at (X, Y). Confirm `mpc_node` log shows `side_pref`
becoming ±1 as the car approaches.

- [ ] **Step 3: Commit**

```bash
git add src/nonlinear_mpc_acados/scripts/pub_static_obstacle.py
git commit -m "nonlinear_mpc_acados: add scriptable static-obstacle publisher for avoidance tests"
```

---

## Final verification (after Tasks 1–2, and 3 if done)

- [ ] **Identity gate, final:** one more no-obstacle run reproduces Task 0 baseline within noise.
- [ ] **Avoidance gate, final:** one obstacle on the racing line → clean detour, no contact, line resumed.
- [ ] **Isolation check:** `git log --oneline lmpc-joint-alpha..HEAD` shows only avoidance commits; no deploy-config or B0–B4' files touched.
- [ ] **Memory update:** update `current_status_next.md` / `MEMORY.md` with the avoidance result.

---

## Self-Review

**Spec coverage:** attenuation (A) → Task 1; side_term (B) → Task 2; optional publisher (C) →
Task 3; Phase 0 characterization → Task 0; identity gate + avoidance gate → every code task;
branch isolation → Final verification. All spec sections covered.

**Placeholder scan:** the only `<X>/<Y>` placeholders are runtime coordinates the operator
picks from an RViz click — unavoidable and explained. No TODO/TBD/"handle edge cases".

**Type/name consistency:** `attenuation`, `att_kappa`, `sqrt_att` (consumed at line 907),
`proximity_side`, `sigma_side`, `d2`, `side_pref`, `D_detour_p`, `abs_side`, `side_target`,
`side_term` (consumed in the `y_expr` residual at line 951), `q_side_def` (line 1166) — all
match existing identifiers in `acados_kinematic.py`. The publisher topic `/external_obstacles`
matches the `mpc_node` subscription.
