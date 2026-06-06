# Real-Car MPCC Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sync the evolved `nonlinear_mpc_acados` package (post-2026-05-28) into the car's `nonlinear_mpcc` package on the T7 SSD so the car can run MPCC, while preserving the car's real-car safety config and topic conventions, with staged on-car validation.

**Architecture:** The car's `nonlinear_mpcc` is a 2026-05-28 snapshot of our package with identical structure. Porting = applying our since-05-28 delta (6 changed core files + 12 new files) into the car package under its own name, resolving one missing dependency (`osuf1_common`), and **merging** config rather than overwriting it (the car config holds deliberate safety caps). First bring-up runs the minimal torch-free path (all heavy features OFF, LMPC optional); GP/ML are deferred behind their default-OFF flags.

**Tech Stack:** ROS2 Jazzy, Python (rclpy), CasADi, acados (Mac-rebuilt), f110_msgs, osuf1_common. Car compute = Mac (CPU/MPS only, no CUDA per [[mac_realcar_deploy]]).

**Paths (constants used throughout):**
- `OUR` = `/home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados` (source of truth, this Linux box)
- `OURPY` = `$OUR/nonlinear_mpc_acados` (our python package dir)
- `CAR_WS` = `/media/hmcl/T7/creating_autonomous_car_ws`
- `CAR` = `$CAR_WS/src/creating_autonomous_car/nonlinear_mpcc` (target package)
- `CARPY` = `$CAR/nonlinear_mpcc` (car python package dir)
- `OSU` = `osuf1_common` package location (resolved in Phase 1)

> **Important environment note:** This Linux box can do file sync + Python *import* smoke tests, but **cannot codegen acados or validate driving** — that happens on the Mac/car. acados C codegen must be rebuilt on the Mac. Tasks are tagged **[LINUX]** (do here) or **[CAR/MAC]** (do on the vehicle).

---

## Evidence Summary (why each decision was made)

Confirmed by inspection on 2026-06-06:

| Fact | Evidence | Implication |
|------|----------|-------------|
| Car pkg = 05-28 snapshot, identical structure | file tree match; all car mtimes = 05-28 13:27 (copy time) | port = delta apply, file-by-file |
| Changed core files: `mpc_node.py` (+692/-94), `acados_kinematic.py` (+534/-95), `track_loader.py` (+51/-7), `mpc_debug_logger.py` (+17/-2), `gp_residual_wrapper.py` (+18/-6), `mpc_core/__init__.py` (+22/0) | line-count diff | overwrite these 6 |
| Identical (skip): `ipopt_kinematic.py`, `_ros_compat.py`, `__init__.py` | 0/0 diff | do not touch |
| 12 new files: `ftg_fallback_node.py`, `pp_fallback_node.py`, `ml/{inference,model,train,__init__}.py`, `mpc_core/gp_casadi_residual.py`, `mpc_core/lmpc/{error_regression,lap_database,nominal_dynamics,safe_set,__init__}.py`, `mpc_core/refv_smoothing.py` | `comm -23` of file sets | add these |
| `osuf1_common` NOT on car; eager import at `mpc_node.py:67`; used at lines 331/1851/1854 for `/mpc/prediction` (viz only) | `find` empty; grep | **hard startup blocker** — Phase 1 resolves |
| All heavy features default OFF: `use_ml_scale`, `use_gp_residual`, `use_gp_casadi`, `use_lmpc`, `use_error_regression` | `declare_parameter(..., False)` | first bring-up needs no torch, no ml/ |
| LMPC + refv_smoothing are torch-free | `grep -L import torch` | LMPC can run on car without torch |
| torch imported lazily (`gp_residual_wrapper` only inside `if use_gp_residual`; `ml.inference` at line 1320) | grep line context | torch optional for bring-up |
| track_loader field-access identical car↔ours | empty field diff | no new Wpnt field deps; car's 9-field Wpnt OK |
| Our Wpnt adds only `z_m`, `is_observed` over car's 9 fields | `Wpnt.msg` diff | not read by ported code paths; safe |
| Car config holds safety caps: `max_speed 4.0`, `a_lat 15`, `steering 0.3`, `N_horizon 40`, `track_source centerline`; ours is sim-aggressive (`max_speed 12`, `steering 0.45`, `N 20`, `raceline`, `use_lmpc true`) | yaml diff | **must merge, never overwrite config** |
| `setup.py` entry_points: car has only `mpc_node`; ours has 4 | grep | add 3 entry_points (renamed pkg) |
| `package.xml`: ours adds `osuf1_common`, `sensor_msgs`, `tf_transformations`; uses `<exec_depend>` | diff | update deps |
| Only `ml/train.py` uses absolute pkg-name import (`from nonlinear_mpc_acados.ml.model`) | grep | one rename fix; ml/ deferred anyway |
| Car ws `creating_autonomous_car` IS a git repo | `git rev-parse` | commit-based rollback |
| Car `mpcc.launch.xml` already wired (waypoint_publisher + mpc_node) | grep | keep car launch; do not overwrite |

---

## DECISIONS (recommended defaults — confirm or override before executing)

1. **First bring-up feature set = minimal/safe.** All heavy flags OFF (`use_lmpc=false`, `use_gp_*=false`, `use_ml_scale=false`). Rationale: torch-free, fewest moving parts, matches car's safe baseline. LMPC (`use_lmpc=true`, our sim-deploy best) is enabled *later* in Phase 5 after a clean baseline lap. → **Skips `ml/` entirely from the required sync.**
2. **Config strategy = merge into car config, preserve car safety caps.** We add the *new parameter keys* our code needs (with safe values) but keep car's `max_speed 4.0 / a_lat 15 / steering 0.3 / N_horizon 40 / track_source centerline`. Speeds raised incrementally on-car.
3. **`osuf1_common` = port the package to the car ws** (clean; it is just msg defs). Fallback if porting is undesirable: guard the import (Task 1b-alt). Recommended: port the package.

If you want LMPC on from the first lap, or a different config posture, say so and Tasks 8 + 7 adjust.

---

## File Structure

Files created/modified by this plan:

**Car package code (overwrite from our source):**
- `$CARPY/mpc_node.py`, `$CARPY/track_loader.py`, `$CARPY/mpc_debug_logger.py`
- `$CARPY/mpc_core/acados_kinematic.py`, `$CARPY/mpc_core/gp_residual_wrapper.py`, `$CARPY/mpc_core/__init__.py`

**Car package code (new files):**
- `$CARPY/ftg_fallback_node.py`, `$CARPY/pp_fallback_node.py`
- `$CARPY/mpc_core/gp_casadi_residual.py`, `$CARPY/mpc_core/refv_smoothing.py`
- `$CARPY/mpc_core/lmpc/{__init__,error_regression,lap_database,nominal_dynamics,safe_set}.py`
- (Deferred, NOT in required sync) `$CARPY/ml/*`

**Car package metadata:**
- `$CAR/setup.py` (add 3 entry_points), `$CAR/package.xml` (add deps)
- `$CAR/config/ddrx_unified_params.yaml` (merge new keys, keep safe values)

**New package on car ws:**
- `$CAR_WS/src/creating_autonomous_car/osuf1_common/` (ported msg package)

---

## Phase 0 — Pre-flight & Safety Net [LINUX]

### Task 0: Backup car package state and record safe config

**Files:**
- Read: `$CAR/config/ddrx_unified_params.yaml`
- Create: backup commit in car git repo

- [ ] **Step 1: Confirm no ROS/sim is running and T7 is mounted**

Run:
```bash
mountpoint -q /media/hmcl/T7 && echo MOUNTED || echo "ABORT: T7 not mounted"
pgrep -af "mpc_node|full_sim|gym_bridge" | grep -v pgrep || echo "no ros running — good"
```
Expected: `MOUNTED` and `no ros running — good`. If T7 not mounted, STOP.

- [ ] **Step 2: Commit current car package state as a restore point**

Run:
```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add -A nonlinear_mpcc
git commit -m "checkpoint: nonlinear_mpcc pre-port (2026-05-28 snapshot) restore point" || echo "nothing to commit (already clean)"
git rev-parse HEAD
```
Expected: a commit hash printed (the rollback point). Record it.

- [ ] **Step 3: Snapshot the car's safe config values to a reference file**

Run:
```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc
grep -E "max_speed|max_speed_p|speed_target|a_lat_safe_live|mpc_max_steering|N_horizon|track_source|mpc_corridor_half_width|lookahead_m" config/ddrx_unified_params.yaml | tee /tmp/car_safe_config.txt
```
Expected: prints car safety values (`max_speed: 4.0`, `mpc_max_steering: 0.3`, `N_horizon: 40`, `track_source: centerline`, etc.). Keep `/tmp/car_safe_config.txt` — Task 7 reconciles against it.

- [ ] **Step 4: Commit the plan + reference into our repo**

```bash
cd /home/hmcl/IFAC2026_SH
git add docs/superpowers/plans/2026-06-06-realcar-mpcc-port.md
git commit -m "plan: real-car MPCC port (nonlinear_mpcc sync)"
```

---

## Phase 1 — Resolve the `osuf1_common` Blocker [LINUX]

### Task 1: Port the `osuf1_common` message package to the car ws

`mpc_node.py:67` does `from osuf1_common.msg import MPCTrajectory, MPCPrediction` at module top. Without it, the node fails to import. We port the package.

**Files:**
- Locate: our `osuf1_common` source
- Create: `$CAR_WS/src/creating_autonomous_car/osuf1_common/` (copy)

- [ ] **Step 1: Locate our osuf1_common source**

Run:
```bash
OSU=$(find /home/hmcl/IFAC2026_SH/src -maxdepth 2 -type d -name osuf1_common | head -1); echo "OSU=$OSU"
ls "$OSU"; ls "$OSU/msg" 2>/dev/null
```
Expected: a path printed and `MPCTrajectory.msg`, `MPCPrediction.msg` listed. If empty, STOP — the message package is missing from our tree and Task 1b-alt (guard the import) must be used instead.

- [ ] **Step 2: Copy the package into the car ws (exclude build artifacts)**

Run:
```bash
OSU=$(find /home/hmcl/IFAC2026_SH/src -maxdepth 2 -type d -name osuf1_common | head -1)
DEST=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/osuf1_common
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$OSU/" "$DEST/"
ls "$DEST/msg"
```
Expected: `MPCTrajectory.msg  MPCPrediction.msg` (plus any others). 

- [ ] **Step 3: Verify msg field compatibility with our mpc_node usage**

Run:
```bash
cd /home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/nonlinear_mpc_acados
grep -E "MPCTrajectory\(|MPCPrediction\(|\.trajectory|pred\.[a-z_]+ =" mpc_node.py | head -20
cat $(find /home/hmcl/IFAC2026_SH/src -name MPCPrediction.msg | head -1)
cat $(find /home/hmcl/IFAC2026_SH/src -name MPCTrajectory.msg | head -1)
```
Expected: every field assigned to `pred.*` / `traj_msg.*` in mpc_node exists in the `.msg` files. (Since both come from our repo, they match by construction — this is a sanity check.)

- [ ] **Step 4: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add osuf1_common
git commit -m "port: add osuf1_common msg package (MPCTrajectory/MPCPrediction) for mpc_node"
```

> **Task 1b-alt (only if Step 1 found no osuf1_common in our tree):** Instead of porting, make the import optional in `mpc_node.py` after Task 3: wrap `from osuf1_common.msg import ...` in `try/except ImportError`, set a `_HAS_OSU` flag, and guard the publisher creation (line ~331) and publish calls (lines ~1851/1854) behind `if _HAS_OSU:`. This drops viz but keeps control. Document the divergence so it can be re-synced upstream.

---

## Phase 2 — Code Sync [LINUX]

> All copies preserve the car's **package name** (`nonlinear_mpcc`). Our files use **relative imports** (`from .mpc_core...`), so they are name-agnostic — verified: only `ml/train.py` (deferred) uses an absolute pkg import.

### Task 2: Overwrite the 6 changed core files

**Files:**
- Modify (overwrite): `$CARPY/mpc_node.py`, `track_loader.py`, `mpc_debug_logger.py`, `mpc_core/acados_kinematic.py`, `mpc_core/gp_residual_wrapper.py`, `mpc_core/__init__.py`

- [ ] **Step 1: Copy the 6 changed files**

Run:
```bash
OURPY=/home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/nonlinear_mpc_acados
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
for f in mpc_node.py track_loader.py mpc_debug_logger.py mpc_core/acados_kinematic.py mpc_core/gp_residual_wrapper.py mpc_core/__init__.py; do
  cp "$OURPY/$f" "$CARPY/$f" && echo "copied $f"
done
```
Expected: 6 `copied ...` lines.

- [ ] **Step 2: Verify no absolute pkg-name imports leaked in**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
grep -rn "nonlinear_mpc_acados" "$CARPY" --include=*.py || echo "CLEAN — no stray pkg-name refs"
```
Expected: `CLEAN — no stray pkg-name refs`. If any appear (other than in deferred `ml/`), edit them to relative imports.

- [ ] **Step 3: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add nonlinear_mpcc/nonlinear_mpcc
git commit -m "sync: overwrite 6 changed core files (mpc_node +692, acados_kinematic +534, ...)"
```

### Task 3: Add the new (non-ml) files

**Files:**
- Create: `$CARPY/ftg_fallback_node.py`, `pp_fallback_node.py`, `mpc_core/gp_casadi_residual.py`, `mpc_core/refv_smoothing.py`, `mpc_core/lmpc/{__init__,error_regression,lap_database,nominal_dynamics,safe_set}.py`

- [ ] **Step 1: Copy the LMPC package and standalone new files**

Run:
```bash
OURPY=/home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/nonlinear_mpc_acados
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
rsync -a --exclude '__pycache__' "$OURPY/mpc_core/lmpc/" "$CARPY/mpc_core/lmpc/"
cp "$OURPY/mpc_core/gp_casadi_residual.py" "$CARPY/mpc_core/"
cp "$OURPY/mpc_core/refv_smoothing.py" "$CARPY/mpc_core/"
cp "$OURPY/ftg_fallback_node.py" "$CARPY/"
cp "$OURPY/pp_fallback_node.py" "$CARPY/"
ls "$CARPY/mpc_core/lmpc"
```
Expected: `__init__.py  error_regression.py  lap_database.py  nominal_dynamics.py  safe_set.py`.

- [ ] **Step 2: Confirm LMPC path is torch-free (no hidden torch dep pulled into bring-up)**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
grep -rln "import torch" "$CARPY/mpc_core/lmpc" "$CARPY/mpc_core/refv_smoothing.py" && echo "WARN: torch in lmpc" || echo "OK — lmpc/refv torch-free"
```
Expected: `OK — lmpc/refv torch-free`.

- [ ] **Step 3: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add nonlinear_mpcc/nonlinear_mpcc
git commit -m "sync: add new files (lmpc/, gp_casadi_residual, refv_smoothing, ftg/pp fallback nodes)"
```

> `ml/` (4 files) is intentionally **not** copied — it is only imported when `use_ml_scale=true`, pulls in torch, and is offline-training oriented. Deferred to Phase 5 optional enable.

### Task 4: Update `setup.py` entry_points

**Files:**
- Modify: `$CAR/setup.py`

- [ ] **Step 1: Read current car setup.py entry_points**

Run:
```bash
grep -A8 "console_scripts" /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/setup.py
```
Expected: only `'mpc_node = nonlinear_mpcc.mpc_node:main'`.

- [ ] **Step 2: Add the 3 new entry_points (note pkg name `nonlinear_mpcc`)**

Edit `$CAR/setup.py` so `console_scripts` reads exactly:
```python
        'console_scripts': [
            'mpc_node = nonlinear_mpcc.mpc_node:main',
            'mpc_debug_logger = nonlinear_mpcc.mpc_debug_logger:main',
            'ftg_fallback_node = nonlinear_mpcc.ftg_fallback_node:main',
            'pp_fallback_node = nonlinear_mpcc.pp_fallback_node:main',
        ],
```

- [ ] **Step 3: Verify each entry_point target defines `main`**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
for f in mpc_node mpc_debug_logger ftg_fallback_node pp_fallback_node; do
  grep -q "def main" "$CARPY/$f.py" && echo "$f: main OK" || echo "$f: MISSING main"
done
```
Expected: 4 `main OK` lines.

- [ ] **Step 4: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add nonlinear_mpcc/setup.py
git commit -m "sync: add mpc_debug_logger/ftg_fallback/pp_fallback entry_points"
```

### Task 5: Update `package.xml` dependencies

**Files:**
- Modify: `$CAR/package.xml`

- [ ] **Step 1: Add missing exec deps**

Edit `$CAR/package.xml` to ensure these `<exec_depend>` entries exist (keep existing `<depend>` lines; just add what is missing): `osuf1_common`, `sensor_msgs`, `tf_transformations`. Example block to add inside `<package>`:
```xml
  <exec_depend>osuf1_common</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>tf_transformations</exec_depend>
```

- [ ] **Step 2: Verify deps resolve against the car ws / system**

Run:
```bash
CAR_WS=/media/hmcl/T7/creating_autonomous_car_ws
find $CAR_WS/src -maxdepth 3 -name package.xml -path "*osuf1_common*" && echo "osuf1_common pkg present" || echo "WARN: osuf1_common missing (did Task 1 run?)"
python3 -c "import sensor_msgs.msg" 2>&1 | head -1 || true
ros2 pkg prefix tf_transformations 2>/dev/null || echo "tf_transformations: verify on car (ros-jazzy-tf-transformations)"
```
Expected: `osuf1_common pkg present`. `sensor_msgs` import OK. `tf_transformations` may need `apt install ros-jazzy-tf-transformations` on the car — note it for Phase 4.

- [ ] **Step 3: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add nonlinear_mpcc/package.xml
git commit -m "sync: add osuf1_common/sensor_msgs/tf_transformations exec deps"
```

---

## Phase 3 — Linux Import Smoke Test [LINUX]

### Task 6: Static import + syntax check of the synced package

This catches name/relative-import/syntax errors *before* the Mac build, where iteration is slow. It does NOT need acados or a running ROS graph — pure `py_compile` + targeted import.

**Files:**
- Test: ad-hoc (no file created)

- [ ] **Step 1: Byte-compile every synced python file**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
python3 -m py_compile $(find "$CARPY" -name '*.py' -not -path '*/ml/*') && echo "PY_COMPILE OK" || echo "SYNTAX ERROR — fix before proceeding"
```
Expected: `PY_COMPILE OK`.

- [ ] **Step 2: Verify the import graph has no missing local modules (excluding ROS/acados/torch externals)**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
grep -rn "^from \.\|^from \.\.\|^import \." "$CARPY" --include=*.py | grep -v "/ml/" | \
  awk -F: '{print $3}' | sort -u | head -40
echo "--- check each referenced local module file exists ---"
# manual scan: every 'from .mpc_core.lmpc.X import' must have CARPY/mpc_core/lmpc/X.py
ls "$CARPY/mpc_core/lmpc"
```
Expected: every `from .mpc_core.lmpc.<X>` target exists in the `lmpc` dir listing (`lap_database`, `safe_set`, `nominal_dynamics`, `error_regression`). No reference to a missing local module.

- [ ] **Step 3: Confirm torch is NOT required to import mpc_node (bring-up path)**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
grep -n "import torch" "$CARPY/mpc_node.py" && echo "WARN: eager torch in mpc_node" || echo "OK — mpc_node has no module-level torch"
grep -n "from .mpc_core.gp_residual_wrapper\|from .ml" "$CARPY/mpc_node.py"
```
Expected: `OK — no module-level torch`, and the gp_residual_wrapper / ml.inference imports appear only inside method bodies (lazy), not at top of file.

---

## Phase 4 — Config Reconciliation [LINUX]

### Task 7: Merge new config keys into car config WITHOUT overwriting safety caps

The car config has deliberate real-car safety values. Our sim config is aggressive. We add only the *new keys* our synced code reads, set to safe values.

**Files:**
- Modify: `$CAR/config/ddrx_unified_params.yaml`

- [ ] **Step 1: Diff to find NEW keys our code needs that the car config lacks**

Run:
```bash
OUR=/home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados
CAR=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc
echo "=== keys in OUR config but not in CAR config (candidates to add) ==="
comm -13 \
  <(grep -oE "^\s*[a-z_0-9]+:" "$CAR/config/ddrx_unified_params.yaml" | tr -d ' :' | sort -u) \
  <(grep -oE "^\s*[a-z_0-9]+:" "$OUR/config/ddrx_unified_params.yaml" | tr -d ' :' | sort -u)
```
Expected: a list including (at least) `use_lmpc`, `lmpc_alpha`, `use_gp_casadi`, `use_error_regression`, `auto_step_enable`, etc. These are the keys to add.

- [ ] **Step 2: Cross-check which new keys mpc_node actually reads (avoid adding dead keys — YAGNI)**

Run:
```bash
CARPY=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc/nonlinear_mpcc
for k in use_lmpc lmpc_alpha use_gp_casadi use_error_regression use_gp_residual use_ml_scale auto_step_enable; do
  grep -q "declare_parameter('$k'" "$CARPY/mpc_node.py" && echo "$k: READ -> add (safe value)" || echo "$k: not read -> SKIP"
done
```
Expected: each new key labeled READ or SKIP. Only add READ keys.

- [ ] **Step 3: Append the new keys with SAFE / OFF values to the car config**

Edit `$CAR/config/ddrx_unified_params.yaml`, adding (under the same `ros__parameters:` block as existing keys) the READ keys from Step 2 with these first-bring-up values:
```yaml
    # --- ported keys (2026-06-06), real-car safe defaults; raise after validation ---
    use_lmpc: false            # bring-up: OFF. Enable in Phase 5 after a clean baseline lap.
    lmpc_alpha: 1.0            # only active when use_lmpc=true
    use_gp_casadi: false       # torch-free MPCC baseline
    use_error_regression: false
    use_gp_residual: false     # requires torch on Mac; deferred
    use_ml_scale: false        # requires torch + ml/ package; deferred
    auto_step_enable: false    # no autoreg BO on car
```
(Only include the keys Step 2 reported as READ. Do NOT change `max_speed`, `a_lat_safe_live`, `mpc_max_steering`, `N_horizon`, `track_source`, `mpc_corridor_half_width` — keep car's safe values from `/tmp/car_safe_config.txt`.)

- [ ] **Step 4: Verify safety caps are unchanged from the Task 0 snapshot**

Run:
```bash
CAR=/media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car/nonlinear_mpcc
grep -E "max_speed:|max_speed_p:|a_lat_safe_live:|mpc_max_steering:|N_horizon:|track_source:|mpc_corridor_half_width:" "$CAR/config/ddrx_unified_params.yaml" | tee /tmp/car_config_after.txt
diff <(grep -E "max_speed|a_lat_safe_live|mpc_max_steering|N_horizon|track_source|mpc_corridor_half_width" /tmp/car_safe_config.txt | sort) \
     <(grep -E "max_speed|a_lat_safe_live|mpc_max_steering|N_horizon|track_source|mpc_corridor_half_width" /tmp/car_config_after.txt | sort) \
  && echo "SAFETY CAPS UNCHANGED — good" || echo "STOP: a safety cap changed; revert it"
```
Expected: `SAFETY CAPS UNCHANGED — good`.

- [ ] **Step 5: Commit**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git add nonlinear_mpcc/config/ddrx_unified_params.yaml
git commit -m "config: add ported param keys with real-car-safe defaults (all heavy features OFF)"
```

---

## Phase 5 — Car / Mac Bring-Up & Staged Enable [CAR/MAC]

> These steps run **on the Mac/car**, where acados codegen and real driving happen. Do not attempt on this Linux box.

### Task 8: Build and baseline-drive on the car

**Files:** none (build + run)

- [ ] **Step 1: Resolve system deps on the car**

Run (on Mac/car):
```bash
sudo apt install -y ros-jazzy-tf-transformations 2>/dev/null || pip3 install transforms3d
# acados: rebuild C codegen on this machine (NOT copied from Linux)
cd ~/acados && git submodule update --recursive --init
```
Expected: tf_transformations available; acados source present (see `$OUR/scripts/install_acados.sh` for the canonical build).

- [ ] **Step 2: colcon build the two packages**

Run (on Mac/car):
```bash
cd /media/hmcl/T7/creating_autonomous_car_ws   # (or the car's local ws path)
colcon build --packages-select osuf1_common nonlinear_mpcc --symlink-install
source install/setup.bash
```
Expected: both packages build with no errors. If `acados_template` import fails, confirm `ACADOS_SOURCE_DIR` / `LD_LIBRARY_PATH` per `install_acados.sh`.

- [ ] **Step 3: Import smoke test on the car**

Run:
```bash
python3 -c "import nonlinear_mpcc.mpc_node; print('mpc_node import OK')"
python3 -c "from osuf1_common.msg import MPCTrajectory, MPCPrediction; print('osu msgs OK')"
```
Expected: both print OK. (No torch needed — confirms the bring-up path.)

- [ ] **Step 4: Launch with the car's existing launch + safe config, car STATIONARY/on-stand first**

Run:
```bash
ros2 launch nonlinear_mpcc mpcc.launch.xml map:=<your_map>
```
Verify: node starts, no traceback; acados solver initializes (first solve does codegen). Check `/mpc/prediction` publishes and the commanded velocity respects `max_speed 4.0`. Wheels off ground or e-stop ready.

- [ ] **Step 5: First slow lap (wheels down), watch for safety**

Verify on car: clean tracking at ≤4 m/s, no oscillation, `solve_ms` within budget, no STUCK. Confirm odom topic (`/car_state/odom`) is what mpc_node subscribes to (`odom_topic_name` param). Record a baseline lap time.

- [ ] **Step 6: Commit the validated build state**

```bash
cd /media/hmcl/T7/creating_autonomous_car_ws/src/creating_autonomous_car
git tag mpcc-port-baseline-validated
git commit --allow-empty -m "validated: MPCC baseline drives on car at safe config (4 m/s)"
```

### Task 9: Staged capability enable (one change at a time, lap between each)

Only after Task 8 baseline is clean. Each sub-step = change one thing, drive, verify, commit/revert.

- [ ] **Step 1: Raise speed incrementally** — bump `max_speed`/`speed_target` (e.g. 4→6→8), one step per lap, watching slip/contact. Stop at the car's safe envelope (recall sim a_lat sweep favored moderate caps per [[alat_sweep_final]]).
- [ ] **Step 2: Enable LMPC** — set `use_lmpc: true`, `lmpc_alpha: 1.0`. Needs warm-up laps to fill the lap database. Verify no drift (our sim-deploy used this; [[current_status_next]]).
- [ ] **Step 3 (optional): Enable GP residual** — requires torch on Mac (CPU/MPS, no CUDA per [[mac_realcar_deploy]]) + a trained checkpoint + porting `ml/`. Set `use_gp_residual` or `use_gp_casadi` true; validate solve-time budget holds.
- [ ] **Step 4: Update memory** — record the final validated on-car config and any car-specific divergences in `realcar_port_scope.md`.

---

## Self-Review

**Spec coverage** (against `realcar_port_scope.md` checklist):
1. Package name mapping → Task 2 Step 2 (verify no stray refs), Task 4 (entry_points use `nonlinear_mpcc`). ✓
2. Wpnt 9 vs 14 fields → Evidence Summary: track_loader field-access identical, no new deps; documented. ✓
3. Topic/odom remap → Task 8 Step 5 verifies `odom_topic_name`/`/car_state/odom`. ✓
4. acados Mac rebuild → Task 8 Step 1–2. ✓
5. Avoidance pipeline difference → covered implicitly (obstacle features off in bring-up); **not** force-ported. ✓ (NOTE: avoidance/`/external_obstacles` is not part of bring-up scope — flagged here as out-of-scope, matching memory's "차는 dynamic spliner 별도".)
6. Validation on car/Mac only → Phase 5 is all [CAR/MAC]. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete commands/values. `<your_map>` in Task 8 Step 4 is a required user input, not a placeholder. ✓

**Type/name consistency:** Package name `nonlinear_mpcc` used consistently in setup.py/launch/imports. Config keys in Task 7 match `declare_parameter` names verified in Step 2. ✓

**Open items surfaced (not gaps — decisions):** osuf1_common port vs guard (Task 1 vs 1b-alt); first-lap LMPC on/off (Decision 1). Both have a recommended default and an override path.
