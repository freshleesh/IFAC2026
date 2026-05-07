#!/usr/bin/env python3
"""
### HJ : State-aware variant of mpc_planner_node.py.

`~state ∈ {overtake, recovery, observe}` routes the MPC solver output to a
state-machine-facing topic in addition to the always-on debug outputs. Phase
1~6 run with `_observation`-suffixed topics so the state_machine keeps
consuming the legacy spliner/recovery_spliner outputs — this node publishes
only for RViz/comparison. Phase X (gate) flips `~attach_to_statemachine`.

Role outputs (after launch remaps):
  overtake → OTWpntArray on /planner/avoidance/otwpnts_observation
  recovery → WpntArray   on /planner/recovery/wpnts_observation
  observe  → WpntArray   on ~best_trajectory_observation (debug only)

Debug outputs (all states):
  ~best_sample/markers     (MarkerArray — trajectory LINE_STRIP + spheres)
  ~debug/markers           (MarkerArray — corridor walls, obstacle blobs,
                            ref-slice centerline, tier/status text label)
  ~debug/tick_json         (std_msgs/String, JSON payload per tick — used by
                            Claude-side `rostopic echo -c` monitoring loop.
                            Contains tier, ipopt status, solve_ms, cost,
                            ego (s,n,v), margins, jitter, obstacles, weights.)
  ~debug_log/live_summary  (std_msgs/String, rolling-window stats every N ticks)
  ~status                  (std_msgs/String, latched)
  ~timing_ms               (std_msgs/Float32)
  ~best_trajectory_observation (WpntArray; mirror of role output for observe)

Phases 2/3/4/5 extend this skeleton — 3D lift via Track3D, state-specific
presets, obstacle injection, 4-tier fallback, dynamic_reconfigure.
"""

import os
import sys
import time
import yaml
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, Float32, Header, String
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import (WpntArray, Wpnt, OTWpntArray,
                           ObstacleArray, OpponentTrajectory)

# ### HJ : Phase 5 — dynamic_reconfigure (per-instance server).
# MPCCost       : legacy MPCC / Frenet-D backends (rebuild on weight change)
# FrenetKinCost : FrenetKin v3+ (opti.parameter() hot-swap; JIT stays warm)
try:
    from mpc_planner.cfg import FrenetKinCostConfig
except ImportError:
    FrenetKinCostConfig = None

# Import solver from sibling src/ directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.normpath(os.path.join(_this_dir, '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from mpcc_solver import MPCCSolver  # noqa: E402
from frenet_d_solver import FrenetDSolver  # noqa: E402 — Frenet-d perturbation backend
from frenet_kin_solver import (FrenetKinMPC,  # noqa: E402 — new frenet-kin backend
                               SIDE_CLEAR, SIDE_LEFT, SIDE_RIGHT, SIDE_TRAIL)
from side_decider import SideDecider  # noqa: E402 — rule-based side selection
from mpc_raceline_lifter import MPCRacelineLifter  # noqa: E402
### HJ : 2026-04-28 Stage 2 Phase 2-1 — plan library
from plan_library import (pick_plan as _pick_plan, plan_ot_line as _plan_ot_line,
                          make_target_n_profile as _make_target_n_profile,
                          PLAN_LEFT_PASS, PLAN_RIGHT_PASS, PLAN_TRAIL, PLAN_RACELINE)  # noqa: E402
from plan_scorer import (pick_plan_scored as _pick_plan_scored,
                         horizon_aware_d_free as _horizon_d_free,
                         filter_feasible_plans as _filter_feasible_plans)  # noqa: E402
from geometric_fallback import build_quintic_fallback, build_recovery_path  # noqa: E402

# ### HJ : debug_log — optional structured logger (CSV + NPZ per tick).
_dbg_dir = os.path.normpath(os.path.join(_this_dir, '..', 'debug_log'))
if _dbg_dir not in sys.path:
    sys.path.insert(0, _dbg_dir)
try:
    from debug_logger import DebugLogger  # noqa: E402
except Exception as _e:  # pragma: no cover
    DebugLogger = None
    _debug_logger_import_err = _e
else:
    _debug_logger_import_err = None

_VALID_STATES = ('overtake', 'recovery', 'observe', 'auto')

# ### HJ : Phase X (refactored) — 2-state MPC FSM for state:=auto.
#   WITH_OBS : obstacle within horizon. Solver runs with full obstacle cost
#              + SideDecider (LEFT/RIGHT/TRAIL/CLEAR). OVERTAKE AND TRAIL
#              behaviour both emerge from this cost mix — SideDecider picks
#              TRAIL when passing isn't feasible, capping v_max to follow
#              instead of force-passing.
#   NO_OBS   : no obstacle in horizon. Solver runs with obstacle cost off +
#              strong n→0 / GB tracking weights (converge-to-raceline).
#
# Mode transitions are DISCRETE (obs enters/exits horizon) but output is
# kept CONTINUOUS via:
#   (a) weight alpha ramp over K_trans ticks (w_t = (1-α)·W_with_obs + α·W_no_obs)
#   (b) warm-start seed carry-over (mode flip does not reset x_sol / u_sol)
#   (c) tick-to-tick path continuity guard on published wpnts
#
# Single MPC output — always publishes to /planner/mpc/wpnts (attach mode).
# The solver output trajectory inherently converges to n≈0 in NO_OBS and
# naturally deflects around obstacles in WITH_OBS. SM picks "GB or MPC"
# downstream; this node never "stops" solving.
MPC_MODE_WITH_OBS = 'WITH_OBS'
MPC_MODE_NO_OBS = 'NO_OBS'
_MPC_MODES = (MPC_MODE_WITH_OBS, MPC_MODE_NO_OBS)

# Legacy aliases (deprecated; kept so any stale reference maps to the nearest
# current mode rather than crashing). Remove once all consumers are migrated.
MPC_MODE_IDLE = MPC_MODE_NO_OBS
MPC_MODE_OVERTAKE = MPC_MODE_WITH_OBS
MPC_MODE_TRANSITION_OT2RC = MPC_MODE_WITH_OBS
MPC_MODE_RECOVERY = MPC_MODE_NO_OBS


class MPCPlannerStateNode:

    def __init__(self):
        rospy.init_node('mpc_planner_state_node', anonymous=False)

        # -- Role -------------------------------------------------------------
        state = str(self._get_param_or_default('~state', 'observe')).lower().strip()
        if state not in _VALID_STATES:
            self.get_logger().warning('[mpc][%s] invalid ~state=%r — falling back to "observe".',
                          rospy.get_name(), state)
            state = 'observe'
        self.state = state

        # -- Solver params ----------------------------------------------------
        self.freq = self._get_param_or_default('~planner_freq', 30)
        params = {
            'N':                 self._get_param_or_default('~N', 20),
            'dT':                self._get_param_or_default('~dT', 0.05),
            'vehicle_L':         self._get_param_or_default('~vehicle_L', 0.33),
            'max_speed':         self._get_param_or_default('~max_speed', 12.0),
            'min_speed':         self._get_param_or_default('~min_speed', 0.5),
            'max_steering':      self._get_param_or_default('~max_steering', 0.6),
            'w_contour':         self._get_param_or_default('~w_contour', 3.9),
            'w_lag':             self._get_param_or_default('~w_lag', 2.0),
            'w_velocity':        self._get_param_or_default('~w_velocity', 3.0),
            'v_bias_max':        self._get_param_or_default('~v_bias_max', 1.0),
            'w_dv':              self._get_param_or_default('~w_dv', 9.5),
            'w_dsteering':       self._get_param_or_default('~w_dsteering', 14.0),
            'boundary_inflation': self._get_param_or_default('~boundary_inflation', 0.1),
            'w_slack':           self._get_param_or_default('~w_slack', 1000.0),
            'ipopt_max_iter':    self._get_param_or_default('~ipopt_max_iter', 500),
            'ipopt_print_level': self._get_param_or_default('~ipopt_print_level', 0),
            # ### HJ : stage-wise contour/lag ramp (1.0 = flat legacy). <1.0
            # softens near-car lateral pull to kill snap-back overshoot.
            'contour_ramp_start': self._get_param_or_default('~contour_ramp_start', 1.0),
            'lag_ramp_start':     self._get_param_or_default('~lag_ramp_start', 1.0),
        }
        self.N = params['N']
        self.dT = float(params['dT'])  # ### HJ : Phase 2 — needed for ax from u_sol

        # ### HJ : solver backend switch.
        # `frenet_kin` (default): Frenet kinematic bicycle MPC. State
        #                [n,mu,v], 4-term cost, obstacle as hard half-plane
        #                via external SideDecider, corridor box with
        #                wall_safe baked in. See src/frenet_kin_solver.py.
        # `frenet_d`  : legacy n(s)-perturbation on fixed s-grid (backup).
        # `xy`        : legacy Cartesian MPCC (backup).
        self.solver_backend = str(
            self._get_param_or_default('~solver_backend', 'frenet_kin')).lower()
        if self.solver_backend == 'frenet_kin':
            params['q_n']           = float(self._get_param_or_default('~q_n', 3.0))
            params['gamma_progress']= float(self._get_param_or_default('~gamma_progress', 10.0))
            params['r_a']           = float(self._get_param_or_default('~r_a', 0.5))
            params['r_delta']       = float(self._get_param_or_default('~r_delta', 5.0))
            params['r_steer_reg']   = float(self._get_param_or_default('~r_steer_reg', 0.1))
            params['v_min']         = float(self._get_param_or_default('~min_speed', 0.5))
            params['v_max']         = float(self._get_param_or_default('~max_speed', 8.0))
            params['a_min']         = float(self._get_param_or_default('~a_min', -4.0))
            params['a_max']         = float(self._get_param_or_default('~a_max', 3.0))
            params['delta_max']     = float(self._get_param_or_default('~max_steering', 0.6))
            params['mu_max']        = float(self._get_param_or_default('~mu_max', 0.9))
            params['inflation']     = float(self._get_param_or_default('~boundary_inflation', 0.05))
            params['wall_safe']     = float(self._get_param_or_default('~wall_safe', 0.15))
            params['gap_lat']       = float(self._get_param_or_default('~gap_lat', 0.25))
            params['gap_long']      = float(self._get_param_or_default('~gap_long', 0.8))
            params['w_slack']       = float(self._get_param_or_default('~w_slack', 2000.0))
            params['n_obs_max']     = int(self._get_param_or_default('~n_obs_max', 2))
            params['ipopt_max_iter']= int(self._get_param_or_default('~ipopt_max_iter', 200))
            # ### HJ : v2 redesign — soft obstacle bubble + side bias + wall cushion
            params['w_obs']         = float(self._get_param_or_default('~w_obs', 180.0))
            params['sigma_s_obs']   = float(self._get_param_or_default('~sigma_s_obs', 0.7))
            params['sigma_n_obs']   = float(self._get_param_or_default('~sigma_n_obs', 0.18))
            params['w_side_bias']   = float(self._get_param_or_default('~w_side_bias', 25.0))
            params['w_wall_buf']    = float(self._get_param_or_default('~w_wall_buf', 2500.0))
            params['wall_buf']      = float(self._get_param_or_default('~wall_buf', 0.30))
            # ### HJ : v3 — C^1 curvature (δ as state) + continuity + terminal
            params['r_dd']          = float(self._get_param_or_default('~r_dd', 5.0))
            params['r_dd_rate']     = float(self._get_param_or_default('~r_dd_rate', 1.0))
            params['w_cont']        = float(self._get_param_or_default('~w_cont', 20.0))
            params['q_n_term']      = float(self._get_param_or_default('~q_n_term', 10.0))
            params['q_v_term']      = float(self._get_param_or_default('~q_v_term', 0.5))
            params['delta_rate_max'] = float(self._get_param_or_default('~delta_rate_max', 3.0))
            # ### HJ : v3b — solver-side ego half-width (was only in decider).
            params['ego_half_width'] = float(
                self._get_param_or_default('~ego_half_width', 0.15))
            # ### HJ : v3c+ — HSL ma27 swap. set '~linear_solver:=mumps' to revert.
            params['linear_solver'] = str(
                self._get_param_or_default('~linear_solver', 'ma27'))
            # ### HJ : v3c+ — CasADi JIT. compile-time cost ~10-20s, runtime
            # function-eval gain on top of ma27's linear-solve gain.
            params['ipopt_jit'] = bool(self._get_param_or_default('~ipopt_jit', True))
            self.solver = FrenetKinMPC(**params)
            self.n_obs_max = params['n_obs_max']
            self.wall_safe = params['wall_safe']
            self.w_wall = 0.0
        elif self.solver_backend == 'frenet_d':
            params['obstacle_sigma'] = float(self._get_param_or_default(
                '~obstacle_sigma', 0.5))
            params['n_obs_max'] = int(self._get_param_or_default('~n_obs_max', 2))
            params['w_wall'] = float(self._get_param_or_default('~w_wall', 0.0))
            params['wall_safe'] = float(self._get_param_or_default('~wall_safe', 0.15))
            self.solver = FrenetDSolver(params)
        elif self.solver_backend == 'xy':
            self.solver = MPCCSolver(params)
        else:
            self.get_logger().warning(
                '[mpc][%s] unknown ~solver_backend=%r — using frenet_kin',
                rospy.get_name(), self.solver_backend)
            self.solver_backend = 'frenet_kin'
            self.solver = FrenetKinMPC(**params)
        self.get_logger().info('[mpc][%s] solver_backend=%s',
                      rospy.get_name(), self.solver_backend)

        # ### HJ : External side decider (rule-based, with hysteresis). Only
        # meaningful for frenet_kin backend — legacy backends ignore side.
        self.side_decider = SideDecider(
            ego_half_width=float(self._get_param_or_default('~ego_half_width', 0.15)),
            gap_lat=float(self._get_param_or_default('~gap_lat', 0.25)),
            trail_dv_thresh=float(self._get_param_or_default('~trail_dv_thresh', 0.5)),
            hold_ticks=int(self._get_param_or_default('~side_hold_ticks', 10)),
            min_pass_margin=float(self._get_param_or_default('~min_pass_margin', 0.10)),
            trail_entry_ticks=int(self._get_param_or_default('~trail_entry_ticks', 3)),
            # ### HJ : v3b — mirror solver's hard-corridor wall inset so
            # decider's can_pass matches what the solver can actually do.
            wall_safe=float(self._get_param_or_default('~wall_safe', 0.15)),
            inflation=float(self._get_param_or_default('~boundary_inflation', 0.05)),
        )
        ### HJ : obstacle half_width source for SideDecider feasibility math.
        ###      "fixed" — use _obs_half_fixed unconditionally (conservative)
        ###      "msg"   — use obstacle.size / 2 from prediction (per-obstacle)
        ###      Switch in real-time via yaml + reload.
        self._obs_size_source = str(
            self._get_param_or_default('~obstacle_size_source', 'fixed')).lower()
        self._obs_half_fixed = float(
            self._get_param_or_default('~obstacle_half_width_fixed', 0.15))
        self._obs_half_min = float(
            self._get_param_or_default('~obstacle_half_min', 0.05))
        if self._obs_size_source not in ('fixed', 'msg'):
            self.get_logger().warning(
                '[mpc] obstacle_size_source=%r invalid; using "fixed"',
                self._obs_size_source)
            self._obs_size_source = 'fixed'
        ### HJ : end
        self._last_side_int = SIDE_CLEAR
        self._last_side_str = 'clear'
        self._last_side_scores = {}
        # ### HJ : v3 — bias ramp REMOVED. Solution continuity cost + stable
        # side decision (feasibility gate) make ramp unnecessary / harmful.
        # Kept attributes as stubs so tick_json stays backward compatible.
        self._ticks_since_flip = 0
        self._last_bias_scale = 1.0
        # ### HJ : v3 — obstacle s/n EMA filter. Opponent predictor jitter
        # makes the bubble centre wobble tick-to-tick, which shakes the cost
        # landscape and induces tiny trajectory jitter. EMA smoothes it.
        self.obs_ema_alpha = float(self._get_param_or_default('~obs_ema_alpha', 0.30))
        self._obs_arr_ema = None
        # ### HJ : v3 — TRAIL velocity ramp. On TRAIL commit, v_target is
        # ramped from current down to v_obs*0.95 over trail_vel_ramp_ticks
        # ticks. Softens the "deceleration feel" during fallback.
        self.trail_vel_ramp_ticks = int(
            self._get_param_or_default('~trail_vel_ramp_ticks', 8))
        self._trail_ticks_since_enter = 0

        # ### HJ : Phase 3 — obstacle cost params (read; honored by solver in
        # Phase 3.5). Kept on the node side so dynamic_reconfigure (Phase 5)
        # can update them without rebuilding the NLP.
        self.collision_mode = str(self._get_param_or_default('~collision_mode', 'none')).lower()
        self.w_obstacle     = float(self._get_param_or_default('~w_obstacle', 0.0))
        self.obstacle_sigma = float(self._get_param_or_default('~obstacle_sigma', 0.35))
        self.n_obs_max      = int(self._get_param_or_default('~n_obs_max', 2))
        self.w_wall         = float(self._get_param_or_default('~w_wall', 0.0))
        self.wall_safe      = float(self._get_param_or_default('~wall_safe', 0.15))
        if self.collision_mode not in ('none', 'soft', 'hard'):
            self.get_logger().warning('[mpc][%s] unknown collision_mode=%r — falling back to "none"',
                          rospy.get_name(), self.collision_mode)
            self.collision_mode = 'none'

        # -- Ego state --------------------------------------------------------
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_z = 0.0  # ### HJ : need z to disambiguate 3D overpass layers in _nearest_idx
        self.car_yaw = 0.0
        self.car_vx = 0.0
        self.pose_received = False

        # ### HJ : canonical 3D-aware Frenet from frenet_odom_republisher
        # (stack standard, used by state_machine/controllers). Prefer this
        # over a local 2D xy-nearest that flips between overpass layers.
        self.ego_s = None
        self.ego_n = None
        self.ego_s_idx = None
        self._frenet_t = None
        self._frenet_stale_s = 0.2

        # ### HJ : Phase 4.3 — 4-tier fallback state.
        # tier 0 성공 시 trajectory + u_sol 캐시 → tier 1 에서 s-shift 로 재발행.
        # streak > H 이면 tier 2 (Frenet quintic) 진입. tier 2 실패 시 tier 3.
        self._fail_streak = 0
        ### HJ : 2026-04-28 (S1-5) — was 5; 3 tick cap 으로 축소.
        ###      이전 5 + obs_in_horizon OR 무한 연장이 51 tick (3.62s) HOLD_LAST
        ###      유발 → stale traj → inter-msg jump 1.683m → 충돌. 짧은 cap +
        ###      cap 초과 + obstacle 시 publish 생략으로 구조적 robust.
        self._fail_tier_H = int(self._get_param_or_default('~fail_streak_H', 3))
        self._last_good_traj = None    # (N+1, 5)  [x, y, psi, s, z] (lifted)
        self._last_good_frenet_traj = None  # (N+1, 4) raw frenet [s, n, mu, v]
        self._last_good_u = None       # (N, 2)    [v, δ]
        self._last_good_s = None       # s at which it was anchored
        self._last_good_time = None    # rospy.Time
        ### HJ : 2026-04-27 — was 8.0m (too long, user feedback). 3.0m
        ###      gives a tight, snappy recovery curve.
        self._quintic_delta_s = float(self._get_param_or_default('~quintic_delta_s', 3.0))
        ### HJ : 2026-04-27 — recovery path caching. Once tier-2 succeeds,
        ###      cache the path in world s-coordinates. Subsequent fallback
        ###      ticks SAMPLE the cache instead of recomputing — solves the
        ###      "goalpost moves with ego, never converges" issue user
        ###      flagged. Cache invalidates on WITH_OBS entry, ego past
        ###      s_end, or |ego_n| < exit threshold.
        self._recovery_cache_xy = None      # (K, 5)
        self._recovery_cache_sn = None      # (K, 4)
        self._recovery_cache_s_start = None
        self._recovery_cache_s_end = None
        self._recovery_cache_committed_at = None
        # ### HJ : 2026-04-29 — single-use cache flag.
        # User directive: cache is "build once, follow once, then drop".
        # While in_use=True, every tick samples the cache and the NLP
        # output is suppressed for publishing. When the cache window
        # ends (ego past s_end OR ego_n converged OR mode flip to
        # WITH_OBS) the flag is cleared AND the cache itself wiped, so
        # the same cache cannot be re-used after a gap.
        self._recovery_cache_in_use = False
        self._recovery_cache_exit_n = float(
            self._get_param_or_default('~recovery_cache_exit_n', 0.10))
        ### HJ : end
        self._last_status = None       # for recovery log
        # ### HJ : v3c — track fallback tier so RViz marker colors stay in
        # sync with solver health. _publish_debug_markers picks the colour
        # from (_viz_tier, _viz_status, _viz_pass) set by the owning handler
        # right before the publish path runs.
        self._viz_tier = 0
        self._viz_status = 'OK'
        self._viz_pass = 1

        # ### HJ : Phase 5 — instance YAML for save/reset triggers. Default
        # resolves to config/state_<state>.yaml (same file the launch loaded).
        # Launch can override with ~instance_yaml.
        default_yaml = os.path.join(
            os.path.dirname(_this_dir), 'config', 'state_%s.yaml' % self.state)
        self.instance_yaml_path = self._get_param_or_default('~instance_yaml', default_yaml)
        self._suppress_dynreg_cb = False
        self._dyn_srv = None  # set after solver.setup() in _global_wpnts_cb

        # ### HJ : debug_log — structured per-tick CSV + on-anomaly NPZ dump.
        # All knobs are ROS params, runtime-writable via rosparam set + the
        # logger reads them on every tick (see _refresh_debug_cfg).
        self._debug_logger = None
        self._dbg_params_snapshot = None
        self._dbg_cfg = None
        self._dbg_tick_counter = 0
        if DebugLogger is not None:
            default_dbg_dir = os.path.join(_dbg_dir, 'runs')
            self._dbg_cfg = {
                'enable':  bool(self._get_param_or_default('~debug_log_enable', False)),
                'dir':     str(self._get_param_or_default('~debug_log_dir', default_dbg_dir)),
                'tier_b':  str(self._get_param_or_default('~debug_log_tier_b', 'on_anomaly')),
                'every_n': int(self._get_param_or_default('~debug_log_every_n', 10)),
                'anomaly_kappa_max': float(self._get_param_or_default(
                    '~debug_log_anomaly_kappa_max', 3.0)),
                'summary_every':  int(self._get_param_or_default('~debug_log_summary_every', 10)),
                'summary_window': int(self._get_param_or_default('~debug_log_summary_window', 50)),
                'state':   self.state,
                'node_name': rospy.get_name(),
            }
            self._dbg_params_snapshot = dict(params)
            self._dbg_params_snapshot.update({
                'obstacle_sigma': self.obstacle_sigma,
                'n_obs_max':      self.n_obs_max,
                'w_steer_reg':    float(params.get('w_steer_reg', 1e-3)),
                'solver_backend': self.solver_backend,
                'w_wall':         float(params.get('w_wall', self.w_wall)),
                'wall_safe':      float(params.get('wall_safe', self.wall_safe)),
                'contour_ramp_start': float(params.get('contour_ramp_start', 1.0)),
            })
            if self._dbg_cfg['enable']:
                self._spawn_debug_logger(reason='launch')
        else:
            rospy.logwarn_once(
                '[mpc] DebugLogger unavailable (%s) — logging disabled',
                _debug_logger_import_err)

        # Publisher for live rolling-window summary (latched; consumed by me
        # during live runs via `rostopic echo` or simply by reading the file).
        self.pub_debug_summary = rospy.Publisher(
            '~debug_log/live_summary', String, queue_size=1, latch=True)

        # ### HJ : per-tick JSON for live Claude-side `rostopic echo -c`
        # monitoring (CLAUDE.md "디버깅은 Claude가 직접 터미널에서 실시간
        # rostopic echo로 한다"). Not latched, published every tick.
        self.pub_debug_tick = rospy.Publisher(
            '~debug/tick_json', String, queue_size=1)
        # Extended debug markers (corridor walls + obstacles + ref-slice +
        # tier/status text). Kept separate from ~best_sample/markers so RViz
        # can toggle the "trajectory only" vs "context" layers independently.
        self.pub_debug_markers = rospy.Publisher(
            '~debug/markers', MarkerArray, queue_size=1)

        # Monotonic tick counter — independent of DebugLogger availability so
        # tick_json numbering stays consistent across enable/disable cycles.
        self._tick_counter = 0
        # Previous tick trajectory xy for jitter RMS computation.
        self._prev_traj_xy = None
        self._prev_traj_ego_s = None

        # -- Global waypoint cache -------------------------------------------
        self.global_cached = False
        self.g_s = None
        self.g_x = None
        self.g_y = None
        self.g_z = None        # ### HJ : z_m for 3D marker lift
        self.g_psi = None
        self.g_kappa = None    # ### HJ : raceline curvature (fallback source)
        self.g_dleft = None
        self.g_dright = None
        self.g_vx = None
        self.g_mu = None       # ### HJ : raceline pitch (fallback source)
        self.track_length = None

        self._timer = None

        # -- Publishers (debug — always) -------------------------------------
        self.pub_best_trajectory = rospy.Publisher(
            '~best_trajectory_observation', WpntArray, queue_size=1)
        # ### HJ : Path publisher removed — RViz was resolving ~best_sample to
        # MarkerArray from an overlapping config; keep MarkerArray as the only
        # pose-list output so no consumer gets type-confused.
        self.pub_best_markers = rospy.Publisher(
            '~best_sample/markers', MarkerArray, queue_size=1)
        self.pub_status = rospy.Publisher(
            '~status', String, queue_size=1, latch=True)
        self.pub_timing = rospy.Publisher(
            '~timing_ms', Float32, queue_size=1)

        # -- Publishers (role-specific) --------------------------------------
        # ### HJ : Phase X (refactored) — unified MPC output topic.
        #   ~out/mpc_wpnts → /planner/mpc/wpnts                (attach mode)
        #                 → /planner/mpc/wpnts_observation     (observation mode)
        # Single publisher for all roles (overtake / recovery / auto). The
        # OTWpntArray.ot_line field carries the semantic tag so SM can label
        # by behaviour without needing multiple topics.
        # Legacy ~out/otwpnts and ~out/wpnts are kept as optional second
        # publishers for backward-compat rollout but the standard path is
        # the unified topic.
        self.pub_mpc = None
        self.pub_ot = None
        self.pub_rc = None
        if self.state in ('overtake', 'recovery', 'auto'):
            self.pub_mpc = rospy.Publisher(
                '~out/mpc_wpnts', OTWpntArray, queue_size=1)
        # Legacy dual topics — left unremapped by default (no-op unless a
        # launch explicitly re-enables them). Kept so a rollback to Phase 1/2
        # observation plumbing is possible without touching code.
        if self.state == 'overtake':
            self.pub_ot = rospy.Publisher(
                '~out/otwpnts', OTWpntArray, queue_size=1)
        elif self.state == 'recovery':
            self.pub_rc = rospy.Publisher(
                '~out/wpnts', WpntArray, queue_size=1)
        self.pub_out = self.pub_mpc   # primary role publisher

        # ### HJ : Phase X (refactored) — initial FSM state pinned by launch.
        if self.state == 'overtake':
            self._mpc_mode = MPC_MODE_WITH_OBS
        elif self.state == 'recovery':
            self._mpc_mode = MPC_MODE_NO_OBS
        else:
            # 'auto' and 'observe' start in NO_OBS (safe convergent solve).
            self._mpc_mode = MPC_MODE_NO_OBS
        self._prev_mpc_mode = self._mpc_mode
        self._mode_dwell = 0
        self._alpha_ramp = (0.0 if self._mpc_mode == MPC_MODE_WITH_OBS else 1.0)

        # ### HJ : Phase X (refactored) FSM knobs.
        # Only mode_dwell_min_ticks + K_trans + horizon_s_pred remain used.
        # n_recovery_trigger / n_recovery_exit / n_idle_ok are kept (read-only
        # legacy) — the MPC itself no longer branches on ego_n; that decision
        # now lives in the SM (path source picker). We expose them so yaml
        # stays backward-compat but default to unused values.
        self._n_recovery_trigger = float(
            self._get_param_or_default('~n_recovery_trigger', 0.15))
        self._n_recovery_exit = float(
            self._get_param_or_default('~n_recovery_exit', 0.08))
        self._n_idle_ok = float(
            self._get_param_or_default('~n_idle_ok', 0.08))
        self._mode_dwell_min_ticks = int(
            self._get_param_or_default('~mode_dwell_min_ticks', 2))
        self._K_trans = int(
            self._get_param_or_default('~K_trans', 8))
        self._ttc_critical_s = float(
            self._get_param_or_default('~ttc_critical_s', 1.5))
        self._horizon_s_pred = float(
            self._get_param_or_default('~horizon_s_pred', 8.0))
        # ### HJ : 2026-04-24 — asymmetric distance hysteresis on the mode
        # switch. Previously a single `horizon_s_pred` controlled BOTH cost
        # scope and mode entry → MPC entered WITH_OBS too early (at 8m) and
        # crawled compared to GB which flies through. User directive:
        # enter WITH_OBS only when obstacle is close (≤ 5m), exit back to
        # NO_OBS when it fades past 10m or tracking is stale.
        self._obs_enter_dist_m = float(
            self._get_param_or_default('~obs_enter_dist_m', 5.0))
        self._obs_exit_dist_m = float(
            self._get_param_or_default('~obs_exit_dist_m', 10.0))

        # ### HJ : 2026-04-24 — adaptive recovery q_n boost.
        # When ego is close to GB (|ego_n| < thresh), boost stage q_n
        # beyond the NO_OBS profile value so EARLY horizon also converges
        # to GB (not just the ramp-enforced terminal). Zero boost when
        # |ego_n| >= thresh so large deviations still get a smooth curve.
        # Only active in MPC_MODE_NO_OBS (recovery) — doesn't interfere
        # with WITH_OBS obstacle avoidance.
        self._q_n_near_boost = float(
            self._get_param_or_default('~q_n_near_boost', 30.0))
        self._q_n_near_thresh = float(
            self._get_param_or_default('~q_n_near_thresh', 0.15))
        self._last_q_n_boost_applied = 0.0
        ### HJ : 2026-04-26 (A4-c) — wall entry ramp tuning knobs.
        # K_entry = clip(overshoot / step_m, K_min, K_max)
        self._wall_ramp_step_m = float(
            self._get_param_or_default('~wall_ramp_step_m', 0.05))
        self._wall_ramp_K_min = int(
            self._get_param_or_default('~wall_ramp_K_min', 3))
        self._wall_ramp_K_max = int(
            self._get_param_or_default('~wall_ramp_K_max', 10))
        ### HJ : end
        ### HJ : 2026-04-26 (A6) — post-OT q_n boost (decay over K ticks).
        ###      On WITH_OBS → NO_OBS transition, kick q_n above the NO_OBS
        ###      baseline for K ticks (linear decay) to pull ego back to GB
        ###      decisively before any wobble can compound. Combined with A5
        ###      (warm-start reset) and the existing _apply_recovery_near_boost.
        self._post_ot_boost_amount = float(
            self._get_param_or_default('~post_ot_boost_amount', 50.0))
        self._post_ot_boost_total_ticks = int(
            self._get_param_or_default('~post_ot_boost_total_ticks', 15))
        self._post_ot_boost_ticks_left = 0
        ### HJ : R1 — proactive recovery cache build trigger
        self._just_transitioned_to_no_obs = False
        ### HJ : end
        ### HJ : 2026-04-27 (C1'+) — per-obstacle sigma safety clearance.
        ###      sigma_n[o] = obs_half[o] + ego_half + safety_clearance.
        ###      User asked for at least 0.2m clearance.
        self._obs_safety_clearance = float(
            self._get_param_or_default('~obs_safety_clearance', 0.20))
        ### HJ : end
        ### HJ : 2026-04-27 — rear safety window. When obstacle just slipped
        ###      behind ego (ds slightly negative), keep treating it as "in
        ###      horizon" so OT mode lingers long enough for ego to clear
        ###      the obstacle laterally before raceline pull-back kicks in.
        self._obs_rear_window_m = float(
            self._get_param_or_default('~obs_rear_window_m', 3.0))
        ### HJ : end

        # ### HJ : 2026-04-24 — prediction variance aware obstacle bubble.
        # When enabled, σ_s_obs and σ_n_obs are inflated per tick based on
        # max(vs_var, vd_var) over active obstacles using a worst-case end-
        # of-horizon propagation (σ_eff = √(σ_base² + var · (N·dT)²)).
        # Wider bubble when predictor is uncertain → more defensive.
        # Launch arg `use_pred_variance:=true` to turn on.
        self._use_pred_variance = bool(
            self._get_param_or_default('~use_pred_variance', False))
        # Snapshot YAML-defined base sigmas so we can invert the inflation
        # cleanly if use_pred_variance is toggled at runtime.
        self._sigma_s_obs_base = float(
            self._get_param_or_default('~sigma_s_obs', 0.7))
        self._sigma_n_obs_base = float(
            self._get_param_or_default('~sigma_n_obs', 0.35))
        # Populated each tick by _build_obstacle_array_frenet.
        self._last_obs_max_vs_var = 0.0
        self._last_obs_max_vd_var = 0.0
        self._last_sigma_s_eff = self._sigma_s_obs_base
        self._last_sigma_n_eff = self._sigma_n_obs_base

        # Densified output step (match GB waypoints_dist = 0.1 m so SM slicer
        # and controller lookahead see a familiar spacing).
        self._mpc_output_ds = float(
            self._get_param_or_default('~mpc_output_ds', 0.1))
        # Densify toggle — user directive 2026-04-24 (second iteration): the
        # solver's time-parametrised output produces uneven spatial spacing
        # (Δs varies with v_k per step). Re-enable post-densify so SM +
        # vel_planner_25d see equispaced 0.1 m grid and compute a smoother
        # velocity profile. Set ~mpc_densify_enable:=false to revert to raw.
        self._mpc_densify_enable = bool(
            self._get_param_or_default('~mpc_densify_enable', True))

        # ### HJ : Phase X — weight profile for NO_OBS mode (hot-swap via
        # FrenetKinMPC.update_weights). Loaded from YAML `no_obs_profile:`.
        # WITH_OBS mode uses the top-level YAML (state_overtake.yaml) values
        # exactly as before — that's the baseline captured lazily.
        no_obs_profile = self._get_param_or_default('~no_obs_profile', {}) or {}
        if not isinstance(no_obs_profile, dict):
            self.get_logger().warning('[mpc] ~no_obs_profile must be a dict — ignored.')
            no_obs_profile = {}
        # Accept legacy key name for one release so existing YAMLs keep working.
        if not no_obs_profile:
            legacy = self._get_param_or_default('~recovery_nlp_profile', {}) or {}
            if isinstance(legacy, dict) and legacy:
                no_obs_profile = legacy
                self.get_logger().info('[mpc] using legacy ~recovery_nlp_profile as ~no_obs_profile')
        # ### HJ : 2026-04-24 — filter unknown keys BEFORE building the
        # profile dict. Stale rosparams (e.g. `r_steer_reg` from prior
        # launches) can linger on the ROS master even after the YAML is
        # cleaned up, since <rosparam load> MERGES rather than replaces.
        # Filtering here avoids the update_weights UserWarning spam and
        # makes the node resilient to stale master state.
        _allowed_keys = set(getattr(FrenetKinMPC, 'LIVE_TUNABLE_WEIGHTS', ()))
        _allowed_keys.add('gamma_progress')   # alias → gamma
        filtered = {}
        dropped = []
        for k, v in no_obs_profile.items():
            if k in _allowed_keys:
                filtered[str(k)] = float(v)
            else:
                dropped.append(k)
        if dropped:
            self.get_logger().warning(
                '[mpc] ~no_obs_profile has unknown keys %s (likely stale rosparam); dropping.',
                sorted(dropped))
        self._no_obs_weights = filtered
        # Lazy snapshot of WITH_OBS baseline (user-tuned values from rqt).
        self._with_obs_baseline_weights = None
        self._last_applied_weight_alpha = None

        # Speed painter parameters (Phase 3).
        self._painter_enable = bool(
            self._get_param_or_default('~painter_enable', True))
        self._painter_n_near = float(
            self._get_param_or_default('~painter_n_near', 0.15))
        self._painter_K_blend = int(
            self._get_param_or_default('~painter_K_blend', 10))
        self._painter_a_max = float(
            self._get_param_or_default('~painter_a_max', 3.0))

        # Path continuity guard parameters (Phase 1.5).
        self._continuity_guard_enable = bool(
            self._get_param_or_default('~continuity_guard_enable', True))
        self._continuity_K_guard = int(
            self._get_param_or_default('~continuity_K_guard', 5))
        self._continuity_threshold_m = float(
            self._get_param_or_default('~continuity_threshold_m', 0.15))
            # 2026-04-28: tested 0.08 (more aggressive blend) but it slowed
            # MPC's lateral reaction during plan transitions and produced
            # 3 collisions in the post_final bag. Reverted to 0.15. Plan
            # transition smoothness comes from the sticky-fix + feasibility
            # filter (fewer flips) + solver's own w_cont=300, not from
            # head-wpnt over-blending.
        self._last_published_wpnts = None   # list[Wpnt] cache for L2 check
        self._last_published_mode = None
        self._last_continuity_L2 = 0.0
        self._last_path_blend_applied = False
        self._last_painter_seam_idx = -1
        self._last_painter_blend_applied = False
        self._last_painter_vx_first_delta = 0.0

        # GB vx cache for speed painter (s-based lookup; xy-roundtrip forbidden).
        self._gb_vx_by_s_s = None   # sorted 1-D array of s_m values
        self._gb_vx_by_s_v = None   # matching vx_mps values

        # Mode transition bookkeeping for tick_json.
        self._last_ttc_min = float('inf')
        self._last_obs_in_horizon = False
        self._last_min_obs_ds = float('inf')

        # -- Subscribers ------------------------------------------------------
        self.create_subscription(WpntArray, '/global_waypoints', self._global_wpnts_cb, queue_size=1, 10)
        self.create_subscription(PoseStamped, '/car_state/pose', self._pose_cb, queue_size=1, 10)
        self.create_subscription(Odometry, '/car_state/odom', self._odom_cb, queue_size=1, 10)
        # ### HJ : frenet_odom_republisher — 3D-aware (z included in nearest search)
        rospy.Subscriber('/car_state/odom_frenet', Odometry,
                         self._frenet_odom_cb, queue_size=1)

        # ### HJ : Phase 3.5 — SQP-compatible obstacle ingestion.
        # prediction primary → tracking fallback → neither (cost disabled).
        # 100ms staleness threshold (SQP has none — this is an improvement).
        self._obs_stale_s = 0.1
        self._obs_predict = None
        self._obs_predict_t = None
        self._obs_track = None
        self._obs_track_t = None
        self._opp_wpnts = None
        self._opp_wpnts_t = None
        rospy.Subscriber('/opponent_prediction/obstacles', ObstacleArray,
                         self._obs_predict_cb, queue_size=1)
        rospy.Subscriber('/tracking/obstacles', ObstacleArray,
                         self._obs_track_cb, queue_size=1)
        rospy.Subscriber('/opponent_trajectory', OpponentTrajectory,
                         self._opp_wpnts_cb, queue_size=1)

        self._publish_status('INIT_WAITING_GLOBAL')
        self.get_logger().info(
            '[mpc][%s] ready — state=%s N=%d dT=%.3f  waiting for /global_waypoints...',
            rospy.get_name(), self.state, self.N, params['dT'],
        )

    # ---------------------------------------------------------------- callbacks
    def _global_wpnts_cb(self, msg):
        if self.global_cached:
            return
        wpnts = msg.wpnts
        n = len(wpnts)
        if n < 10:
            self.get_logger().warning('[mpc][%s] too few global waypoints: %d', rospy.get_name(), n)
            return

        self.g_s = np.array([w.s_m for w in wpnts], dtype=float)
        self.g_x = np.array([w.x_m for w in wpnts], dtype=float)
        self.g_y = np.array([w.y_m for w in wpnts], dtype=float)
        self.g_z = np.array([getattr(w, 'z_m', 0.0) for w in wpnts], dtype=float)
        self.g_psi = np.array([w.psi_rad for w in wpnts], dtype=float)
        self.g_kappa = np.array([getattr(w, 'kappa_radpm', 0.0) for w in wpnts], dtype=float)
        ### HJ : 2026-04-27 — inflection points (sign change of g_kappa) for
        ###      recovery_spliner-style lookahead in fallback.
        try:
            self._inflection_points = np.where(
                np.diff(np.sign(self.g_kappa)) != 0)[0]
        except Exception:
            self._inflection_points = np.array([], dtype=int)
        ### HJ : end
        self.g_dleft = np.array([w.d_left for w in wpnts], dtype=float)
        self.g_dright = np.array([w.d_right for w in wpnts], dtype=float)
        ### HJ : centerline tangent at the foot used to measure d_left/d_right.
        ###      d_left/d_right are now centerline-normal distances (raceline-anchored).
        ###      Solver's |n[k]| <= corridor expects RACELINE-normal limits, so we apply
        ###      d_eff = d / cos(psi_rad - psi_centerline_rad) at slice time.
        self.g_psi_center = np.array(
            [getattr(w, 'psi_centerline_rad', w.psi_rad) for w in wpnts], dtype=float)
        ### HJ : end
        self.g_vx = np.array([w.vx_mps for w in wpnts], dtype=float)
        self.g_mu = np.array([getattr(w, 'mu_rad', 0.0) for w in wpnts], dtype=float)
        self.track_length = float(self.g_s[-1])

        # ### HJ : Phase X — GB vx cache (s-based lookup for speed painter).
        # 3D rule: never xy→frenet round-trip. Use solver/lifter's s directly.
        self._gb_vx_by_s_s = self.g_s.copy()
        self._gb_vx_by_s_v = self.g_vx.copy()

        # ### HJ : Phase 2 — raceline-base lifter (no Track3D; see module
        # docstring for the centerline/raceline mismatch rationale).
        self.lifter = MPCRacelineLifter(
            g_s=self.g_s, g_x=self.g_x, g_y=self.g_y, g_z=self.g_z,
            g_psi=self.g_psi, g_kappa=self.g_kappa, g_mu=self.g_mu,
            g_dleft=self.g_dleft, g_dright=self.g_dright, g_vx=self.g_vx,
            track_length=self.track_length,
        )

        self.solver.setup()
        self.global_cached = True

        # ### HJ : Phase 5 — start dynreg server after solver is up so hot
        # updates have a valid NLP to touch. Server(...) constructor calls
        # _weight_cb once with .cfg defaults to prime the server — we must
        # suppress rebuild BEFORE constructing, then push our YAML-loaded
        # values back into rqt in one shot.
        # frenet_kin uses FrenetKinCostConfig (hot-swap via opti.parameter);
        # legacy backends keep MPCCostConfig (rebuild on weight change).
        self._suppress_dynreg_cb = True
        try:
            if self.solver_backend == 'frenet_kin':
                if FrenetKinCostConfig is None:
                    self.get_logger().warning(
                        '[mpc][%s] FrenetKinCost cfg not built yet — '
                        'rqt tuning disabled. Run catkin build mpc_planner.',
                        rospy.get_name())
                    self._dyn_srv = None
                else:
                    self._dyn_srv = Server(
                        FrenetKinCostConfig, self._weight_cb_kin)
                    self._push_params_to_dynreg_kin()
                    self.get_logger().info(
                        '[mpc][%s] FrenetKinCost dynreg server ready — '
                        'live tuning enabled (JIT-safe).', rospy.get_name())
            else:
                self._dyn_srv = Server(MPCCostConfig, self._weight_cb)
                self._push_params_to_dynreg()
        finally:
            self._suppress_dynreg_cb = False

        self._publish_status('INIT_OK')
        self.get_logger().info('[mpc][%s] solver ready: %d waypoints, track=%.1fm',
                      rospy.get_name(), n, self.track_length)

        if self._timer is None:
            self._timer = rospy.Timer(rospy.Duration(1.0 / self.freq), self._plan_loop)

    def _pose_cb(self, msg):
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        self.car_z = msg.pose.position.z  # ### HJ : 3D overpass disambiguation
        q = msg.pose.orientation
        _, _, self.car_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_received = True

    def _odom_cb(self, msg):
        self.car_vx = msg.twist.twist.linear.x

    def _frenet_odom_cb(self, msg):
        # ### HJ : frenet_odom_republisher packs s into pose.x, n into pose.y,
        # and closest waypoint index into child_frame_id (string). See
        # f110_utils/nodes/frenet_odom_republisher/.../frenet_odom_republisher_node.cc
        self.ego_s = float(msg.pose.pose.position.x)
        self.ego_n = float(msg.pose.pose.position.y)
        try:
            self.ego_s_idx = int(msg.child_frame_id)
        except (TypeError, ValueError):
            self.ego_s_idx = None
        self._frenet_t = self.get_clock().now().to_msg()

    # ---------------------------------------------------------------- obstacle inputs
    def _obs_predict_cb(self, msg):
        self._obs_predict = msg
        self._obs_predict_t = self.get_clock().now().to_msg()

    def _obs_track_cb(self, msg):
        self._obs_track = msg
        self._obs_track_t = self.get_clock().now().to_msg()

    def _opp_wpnts_cb(self, msg):
        self._opp_wpnts = msg
        self._opp_wpnts_t = self.get_clock().now().to_msg()

    def _is_fresh(self, t_stamp):
        if t_stamp is None:
            return False
        return (self.get_clock().now().to_msg() - t_stamp).to_sec() <= self._obs_stale_s

    def _pick_obs_source(self):
        """Return (ObstacleArray, 'predict'|'track'|'none'). Prediction wins
        when fresh; tracking is fallback; otherwise signal cost-off.

        DEPRECATED for planning: use _merge_obs_sources instead so we don't
        lose static obstacles (which only appear in tracking) when dynamic
        prediction is fresh. Kept for legacy xy-backend path.
        """
        if self._is_fresh(self._obs_predict_t) and \
           self._obs_predict is not None and self._obs_predict.obstacles:
            return self._obs_predict, 'predict'
        if self._is_fresh(self._obs_track_t) and \
           self._obs_track is not None and self._obs_track.obstacles:
            return self._obs_track, 'track'
        return None, 'none'

    def _merge_obs_sources(self):
        """### HJ : 2026-04-27 (C1') — tracking-anchored merge.

        Previous bug: prediction publishes N=20 Obstacle msgs per opponent
        (one per horizon timestep). Earlier merge dedupe-by-id failed
        because prediction's `id` is the timestep index, not opponent id.
        N msgs all kept → n_obs_max=2 slots filled by prediction
        snapshots → static obstacles dropped → collision risk.

        Fix: tracking is canonical source (1 msg per real obstacle).
        For each tracked obstacle:
          - find the prediction msg with (s, d) closest to (t.s, t.d)
            AND smallest stamp (= timestep 0 of that opponent's sequence)
          - if matched: copy vs / vd / vs_var / vd_var onto the tracked
            obstacle (so solver gets propagation info), mark dynamic
          - if no match: treat as static (vs=vd=0)
        Result: 1 entry per real obstacle. Static + dynamic coexist
        cleanly within n_obs_max slots.
        """
        pred_fresh = (self._is_fresh(self._obs_predict_t)
                      and self._obs_predict is not None
                      and len(self._obs_predict.obstacles) > 0)
        track_fresh = (self._is_fresh(self._obs_track_t)
                       and self._obs_track is not None
                       and len(self._obs_track.obstacles) > 0)
        if not track_fresh:
            # No tracking — fall back to legacy behaviour: emit prediction
            # if available, dedup by (s,d) bucket. Won't see static-only.
            if not pred_fresh:
                return [], 'none'
            out = []
            seen = []
            for p in self._obs_predict.obstacles:
                key = (round(float(p.s_center), 0), round(float(p.d_center), 1))
                if key in seen:
                    continue
                seen.append(key)
                out.append(p)
            return out, 'pred-only:%d' % len(out)

        ### Tracking-anchored loop
        out = []
        tl = float(self.track_length) if self.track_length else 1e6
        n_matched = 0
        for t in self._obs_track.obstacles:
            match = None
            best_score = float('inf')
            best_stamp = None
            if pred_fresh:
                t_s = float(t.s_center)
                t_d = float(t.d_center)
                for p in self._obs_predict.obstacles:
                    ds = abs(float(p.s_center) - t_s)
                    if ds > 0.5 * tl:
                        ds = tl - ds
                    dn = abs(float(p.d_center) - t_d)
                    # Match if within reasonable thresholds
                    if ds > 1.5 or dn > 0.5:
                        continue
                    # Prefer earliest stamp (= timestep 0 of opponent's seq)
                    p_stamp = (p.header.stamp.to_sec()
                               if hasattr(p, 'header') else 0.0)
                    score = ds + 2.0 * dn  # weighted distance
                    if (score < best_score
                            or (abs(score - best_score) < 0.05
                                and (best_stamp is None
                                     or p_stamp < best_stamp))):
                        best_score = score
                        best_stamp = p_stamp
                        match = p
            if match is not None:
                # Use TRACKED obstacle as canonical, attach prediction info.
                # Copy by attribute (Obstacle msg is mutable).
                try:
                    t.vs = float(match.vs)
                    t.vd = float(match.vd)
                    t.vs_var = float(getattr(match, 'vs_var', 0.0) or 0.0)
                    t.vd_var = float(getattr(match, 'vd_var', 0.0) or 0.0)
                    if abs(t.vs) > 0.05 or abs(t.vd) > 0.05:
                        t.is_static = False
                except Exception:
                    pass
                n_matched += 1
            else:
                # Tracked obs without prediction → static (or stale).
                try:
                    t.vs = 0.0
                    t.vd = 0.0
                    t.vs_var = 0.0
                    t.vd_var = 0.0
                    t.is_static = True
                except Exception:
                    pass
            out.append(t)
        tag = 'tanchor:T%d+P%d' % (len(out), n_matched)
        return out, tag

    def _extrapolate_obs_traj(self, obs, N, dT):
        """Expand one Obstacle into (N, 2) Cartesian trajectory.

        Static obstacle → repeat (x_m, y_m). Dynamic → constant-velocity on
        (s_center, d_center) using (vs, vd), then sn→xy via the lifter.
        """
        if obs.is_static or abs(obs.vs) < 1e-3:
            return np.tile([obs.x_m, obs.y_m], (N, 1))
        out = np.zeros((N, 2), dtype=np.float64)
        s0 = float(obs.s_center)
        d0 = float(obs.d_center)
        vs = float(obs.vs)
        vd = float(obs.vd)
        for k in range(N):
            s_k = (s0 + vs * k * dT) % self.track_length
            d_k = d0 + vd * k * dT
            x_k, y_k = self.lifter.sn_to_xy(s_k, d_k)
            out[k, 0] = x_k
            out[k, 1] = y_k
        return out

    def _build_obstacle_array(self, ego_s):
        """### HJ : Phase 3.5 — SQP-compatible obstacle → (n_obs_max, *, 3).

        Dispatch by solver backend.
          xy:         (n_obs_max, N,   3) Cartesian [x, y, w_obs].
          frenet_d:   (n_obs_max, N+1, 3) Frenet    [s, n, w_obs].
          frenet_kin: (n_obs_max, N+1, 3) Frenet    [s, n, w_obs]  — same shape
                      as frenet_d; FrenetKinMPC._build_corridor_bounds reads
                      obs_arr[o, :, 1] (n) and obs_arr[o, :, 2] (w gate).
        All paths share the SQP-style prediction-first / tracking-fallback /
        staleness-off fusion.
        """
        if self.solver_backend in ('frenet_d', 'frenet_kin'):
            return self._build_obstacle_array_frenet(ego_s)

        N = self.N
        dT = self.dT
        far = self.solver.FAR_XY
        obs_arr = np.zeros((self.n_obs_max, N, 3), dtype=np.float64)
        obs_arr[:, :, 0] = far
        obs_arr[:, :, 1] = far
        # w_obs = 0 already.

        if self.collision_mode == 'none' or self.w_obstacle <= 0.0:
            return obs_arr, 'disabled'

        src, tag = self._pick_obs_source()
        if tag == 'none':
            return obs_arr, 'stale'

        # Pick obstacles ahead of ego_s (shortest forward s-distance first).
        def fwd_dist(o):
            ds = (o.s_center - ego_s) % self.track_length
            return ds
        obs_list = sorted(src.obstacles, key=fwd_dist)
        # Drop anything too far away (more than half track ahead = behind).
        obs_list = [o for o in obs_list if fwd_dist(o) < 0.5 * self.track_length]

        n_used = 0
        for o in obs_list:
            if n_used >= self.n_obs_max:
                break
            traj = self._extrapolate_obs_traj(o, N, dT)
            obs_arr[n_used, :, 0] = traj[:, 0]
            obs_arr[n_used, :, 1] = traj[:, 1]
            obs_arr[n_used, :, 2] = self.w_obstacle
            n_used += 1

        if n_used == 0:
            return obs_arr, 'empty'
        return obs_arr, '%s:%d' % (tag, n_used)

    def _build_obstacle_array_frenet(self, ego_s):
        """### HJ : Frenet-space obstacle array for FrenetDSolver.

        Returns (n_obs_max, N+1, 3) with columns [s_o, n_o, w_obs]. Obstacle
        s is unwrapped relative to ego_s (same ±½·L domain as ref_s) so the
        flat-Frenet distance `ds = ref_s[k] - s_o[k]` is correct even near a
        lap boundary. Static/low-vs obstacles are held constant; dynamic
        obstacles propagate linearly in (s, d) without modulus — slicer keeps
        ref_s monotone so a linear s_o stays consistent.
        """
        N_plus_1 = self.N + 1
        dT = self.dT
        tl = self.track_length
        far = FrenetDSolver.FAR_SN
        obs_arr = np.zeros((self.n_obs_max, N_plus_1, 3), dtype=np.float64)
        obs_arr[:, :, 0] = far  # s
        obs_arr[:, :, 1] = far  # n (far lateral too, harmless)
        ### HJ : per-slot half_width sidecar — populated according to
        ###      obstacle_size_source (fixed | msg). Fixed default keeps the
        ###      pre-2026-04-26 behaviour.
        self._obs_half_arr = np.full(self.n_obs_max,
                                     self._obs_half_fixed, dtype=np.float64)
        ### HJ : 2026-04-27 — per-slot static/dynamic flag for SideDecider.
        ###      True = static (must avoid laterally), False = dynamic
        ###      (trail-able). Default True (conservative — treat unknown
        ###      slots as static).
        self._obs_static_arr = np.full(self.n_obs_max, True, dtype=bool)
        ### HJ : end

        if self.collision_mode == 'none' or self.w_obstacle <= 0.0:
            return obs_arr, 'disabled'

        # ### HJ : 2026-04-25 — use merged source (prediction + tracking) so
        # dynamic + static obstacles coexist in one obs_arr. Was: prediction
        # XOR tracking, which silently dropped static obstacles whenever an
        # opponent was being predicted.
        merged_obs, tag = self._merge_obs_sources()
        if tag == 'none' or len(merged_obs) == 0:
            return obs_arr, 'stale'

        ### HJ : 2026-04-27 — signed lap distance with rear window. Modulo
        ###      alone makes ds=-0.06 wrap to ~tl-0.06 → filtered as "almost
        ###      full lap behind", which DROPPED an obstacle that just slipped
        ###      under ego's rear (still alongside!) → mode flips to NO_OBS
        ###      → recovery cache fires while ego is still next to the
        ###      obstacle. _obs_in_horizon_and_ttc has rear_window logic but
        ###      it can't run if obs_arr is empty here. Apply the same window
        ###      at the build step. signed_dist returns ds in (-tl/2, +tl/2].
        rear_window = float(getattr(self, '_obs_rear_window_m', 3.0))

        def signed_dist(o):
            ds = (o.s_center - ego_s) % tl
            if ds > 0.5 * tl:
                ds -= tl
            return ds

        obs_list = sorted(merged_obs,
                          key=lambda o: max(signed_dist(o), 0.0))
        obs_list = [o for o in obs_list
                    if -rear_window <= signed_dist(o) <= 0.5 * tl]
        ### HJ : end
        # ### HJ : dedup near-duplicate obstacles (prediction+detection race can
        # emit the same object twice within ~0.5m s and ~0.1m d). Keeping both
        # loads two half-plane constraints on the same target → solver pinches
        # the trajectory and thrashes side decisions.
        n_obs_raw = len(obs_list)
        dedup = []
        for o in obs_list:
            keep = True
            for o2 in dedup:
                if (abs(o.s_center - o2.s_center) < 0.5 and
                        abs(o.d_center - o2.d_center) < 0.1):
                    keep = False
                    break
            if keep:
                dedup.append(o)
        obs_list = dedup
        self._last_n_obs_raw = n_obs_raw

        n_used = 0
        max_vs_var = 0.0
        max_vd_var = 0.0
        for o in obs_list:
            if n_used >= self.n_obs_max:
                break
            s0 = float(o.s_center)
            d0 = float(o.d_center)
            # Unwrap s0 into the ego's lap domain (± tl/2 around ego_s).
            while s0 - ego_s > 0.5 * tl:
                s0 -= tl
            while s0 - ego_s < -0.5 * tl:
                s0 += tl

            ### HJ : 2026-04-27 — capture static/dynamic flag in sidecar.
            ###      User mandate: side decision must be driven by STATIC
            ###      obstacles (must avoid laterally), dynamic obstacles can
            ###      be trailed. SideDecider needs this classification to
            ###      separate.
            is_static_flag = bool(o.is_static or abs(o.vs) < 1e-3)
            self._obs_static_arr[n_used] = is_static_flag
            ### HJ : end
            if is_static_flag:
                obs_arr[n_used, :, 0] = s0
                obs_arr[n_used, :, 1] = d0
            else:
                vs = float(o.vs)
                vd = float(o.vd)
                for k in range(N_plus_1):
                    obs_arr[n_used, k, 0] = s0 + vs * k * dT
                    obs_arr[n_used, k, 1] = d0 + vd * k * dT
            obs_arr[n_used, :, 2] = self.w_obstacle
            ### HJ : per-slot half_width selection.
            if self._obs_size_source == 'msg':
                size_raw = float(getattr(o, 'size', 0.0) or 0.0)
                self._obs_half_arr[n_used] = max(size_raw * 0.5, self._obs_half_min)
            else:  # 'fixed'
                self._obs_half_arr[n_used] = self._obs_half_fixed
            ### HJ : end
            # ### HJ : 2026-04-24 — collect prediction variance for
            # use_pred_variance. `s_var/d_var/vs_var/vd_var` may be zero
            # (static obs, or predictor doesn't fill them). getattr defaults
            # to 0 → no inflation contribution.
            max_vs_var = max(max_vs_var, float(getattr(o, 'vs_var', 0.0) or 0.0))
            max_vd_var = max(max_vd_var, float(getattr(o, 'vd_var', 0.0) or 0.0))
            n_used += 1

        self._last_obs_max_vs_var = max_vs_var
        self._last_obs_max_vd_var = max_vd_var

        ### HJ : 2026-04-27 (C1'+) — push per-obstacle sigma to solver.
        ###      sigma[o] = obs_half[o] + ego_half + safety_clearance.
        ###      Inactive slots get scalar fallback (sigma_n_obs).
        try:
            ego_half = float(getattr(self.solver, 'ego_half', 0.15))
            sigma_arr = np.full(self.n_obs_max,
                                max(float(self.solver.sigma_n_obs), 1e-3),
                                dtype=np.float64)
            for o in range(min(n_used, self.n_obs_max)):
                obs_half_o = float(self._obs_half_arr[o])
                sigma_arr[o] = max(obs_half_o + ego_half
                                   + self._obs_safety_clearance, 1e-3)
            self.solver.set_sigma_n_obs_per(sigma_arr)
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] sigma_n_obs_per push failed: %s',
                rospy.get_name(), e)
        ### HJ : end
        ### HJ : 2026-04-27 — push static/dynamic flag so solver applies
        ###      σ_n inflation ONLY to dynamic obstacles in TRAIL mode.
        ###      Static obstacles keep full lateral bubble for steering ego
        ###      around them even while trailing dynamic ones.
        try:
            if hasattr(self.solver, 'set_obs_static'):
                self.solver.set_obs_static(self._obs_static_arr.copy())
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] obs_static push failed: %s',
                rospy.get_name(), e)
        ### HJ : end
        ### HJ : 2026-04-27 (Option A) — pre-compute obstacle xy at every (o, k)
        ###      and push to solver. Inactive slots = far placeholder.
        try:
            obs_xy_mat = np.full(
                (2, self.n_obs_max, N_plus_1), 1.0e3, dtype=np.float64)
            for o in range(min(n_used, self.n_obs_max)):
                for k in range(N_plus_1):
                    s_o = float(obs_arr[o, k, 0])
                    n_o = float(obs_arr[o, k, 1])
                    s_o_w = s_o % tl if tl > 0 else s_o
                    try:
                        x, y = self.lifter.sn_to_xy(s_o_w, n_o)
                    except Exception:
                        x = 1e3; y = 1e3
                    obs_xy_mat[0, o, k] = x
                    obs_xy_mat[1, o, k] = y
            self.solver.set_obs_xy(obs_xy_mat)
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] obs_xy push failed: %s',
                rospy.get_name(), e)
        ### HJ : end

        if n_used == 0:
            return obs_arr, 'empty'
        return obs_arr, '%s:%d' % (tag, n_used)

    # ---------------------------------------------------------------- dynamic_reconfigure
    # ---- FrenetKin live-tune (opti.parameter() hot-swap, JIT-safe) ---------
    def _push_params_to_dynreg_kin(self):
        """Seed FrenetKinCost sliders with the current solver / node values
        so the first rqt connect doesn't snap to .cfg defaults."""
        if self._dyn_srv is None:
            return
        self._suppress_dynreg_cb = True
        try:
            w = self.solver.get_weights()
            cfg_vals = {
                # cost weights (live)
                'q_n':            w['q_n'],
                'q_n_term':       w['q_n_term'],
                'q_v_term':       w['q_v_term'],
                'gamma_progress': w['gamma'],
                'r_a':            w['r_a'],
                'r_steer_reg':    w['r_reg'],
                'r_dd':           w['r_dd'],
                'r_dd_rate':      w['r_dd_rate'],
                'w_obs':          w['w_obs'],
                'sigma_s_obs':    w['sigma_s_obs'],
                'sigma_n_obs':    w['sigma_n_obs'],
                'w_side_bias':    w['w_side_bias'],
                'gap_lat':        w['gap_lat'],
                'w_wall_buf':     w['w_wall_buf'],
                'wall_buf':       w['wall_buf'],
                'w_slack':        w['w_slack'],
                'w_cont':         w['w_cont'],
                # fallback (node-side)
                'fail_streak_H':   int(self._fail_tier_H),
                'quintic_delta_s': float(self._quintic_delta_s),
                # ### HJ : per-obstacle sigma extra clearance (node-side)
                'obs_safety_clearance': float(self._obs_safety_clearance),
                'obs_rear_window_m':    float(self._obs_rear_window_m),
                # one-shot triggers off by default
                'save_params':  False,
                'reset_params': False,
            }
            self._dyn_srv.update_configuration(cfg_vals)
        finally:
            self._suppress_dynreg_cb = False

    def _weight_cb_kin(self, config, level):
        """rqt callback for FrenetKinCostConfig. Updates solver weights
        via self.solver.update_weights() — no NLP rebuild."""
        # one-shot save/reset, same pattern as legacy callback.
        if config.save_params:
            self._save_yaml_kin(config)
            config.save_params = False
        if config.reset_params:
            new_cfg = self._reload_yaml_kin()
            if new_cfg is not None:
                for k, v in new_cfg.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
                self.get_logger().info('[mpc][%s][reset] reloaded YAML: %s',
                              rospy.get_name(), self.instance_yaml_path)
            config.reset_params = False

        # Hot-swap cost weights. Maps rqt keys -> solver attribute names.
        updates = {
            'q_n':         float(config.q_n),
            'q_n_term':    float(config.q_n_term),
            'q_v_term':    float(config.q_v_term),
            'gamma':       float(config.gamma_progress),
            'r_a':         float(config.r_a),
            'r_reg':       float(config.r_steer_reg),
            'r_dd':        float(config.r_dd),
            'r_dd_rate':   float(config.r_dd_rate),
            'w_obs':       float(config.w_obs),
            'sigma_s_obs': float(config.sigma_s_obs),
            'sigma_n_obs': float(config.sigma_n_obs),
            'w_side_bias': float(config.w_side_bias),
            'gap_lat':     float(config.gap_lat),
            'w_wall_buf':  float(config.w_wall_buf),
            'wall_buf':    float(config.wall_buf),
            'w_slack':     float(config.w_slack),
            'w_cont':      float(config.w_cont),
        }
        changed = self.solver.update_weights(**updates)

        # Fallback tuning (node-side, no solver touch).
        self._fail_tier_H = int(config.fail_streak_H)
        self._quintic_delta_s = float(config.quintic_delta_s)
        ### HJ : 2026-04-27 (C1'+) — per-obstacle sigma extra clearance.
        ###      Node-side state; consumed by next _build_obstacle_array
        ###      tick when sigma per slot is recomputed.
        if hasattr(config, 'obs_safety_clearance'):
            self._obs_safety_clearance = float(config.obs_safety_clearance)
        if hasattr(config, 'obs_rear_window_m'):
            self._obs_rear_window_m = float(config.obs_rear_window_m)
        ### HJ : end

        if changed and not self._suppress_dynreg_cb:
            rospy.loginfo_throttle(
                1.0,
                '[mpc][%s][kin-tune] %d weight(s) hot-swapped: %s',
                rospy.get_name(), len(changed),
                ', '.join('%s=%.3g' % (k, v[1]) for k, v in changed.items()))
        return config

    def _save_yaml_kin(self, config):
        """Dump current rqt slider values to self.instance_yaml_path. Keeps
        unrelated YAML keys untouched (non-dynreg node params survive)."""
        path = getattr(self, 'instance_yaml_path', None)
        if not path:
            self.get_logger().warning('[mpc][%s][save] instance_yaml_path unset',
                          rospy.get_name())
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warning('[mpc][%s][save] failed to read %s: %s',
                          rospy.get_name(), path, exc)
            data = {}

        # keys we own in FrenetKinCost.cfg -> YAML keys (identical names
        # except gamma_progress which stays gamma_progress in YAML).
        dump_keys = (
            'q_n', 'q_n_term', 'q_v_term', 'gamma_progress',
            'r_a', 'r_steer_reg', 'r_dd', 'r_dd_rate',
            'w_obs', 'sigma_s_obs', 'sigma_n_obs',
            'w_side_bias', 'gap_lat',
            'w_wall_buf', 'wall_buf',
            'w_slack', 'w_cont',
            'fail_streak_H', 'quintic_delta_s',
        )
        for key in dump_keys:
            if hasattr(config, key):
                val = getattr(config, key)
                data[key] = (int(val) if isinstance(val, bool) or
                             key == 'fail_streak_H' else float(val))
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info('[mpc][%s][save] wrote %d keys to %s',
                          rospy.get_name(), len(dump_keys), path)
        except Exception as exc:
            self.get_logger().warning('[mpc][%s][save] failed to write %s: %s',
                          rospy.get_name(), path, exc)

    def _reload_yaml_kin(self):
        """Re-read the instance YAML and return a dict of rqt-slider values.
        Missing keys fall back to the live solver state so the UI doesn't
        get wiped to zero when the YAML is partial."""
        path = getattr(self, 'instance_yaml_path', None)
        if not path:
            return None
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warning('[mpc][%s][reset] failed to read %s: %s',
                          rospy.get_name(), path, exc)
            return None

        w = self.solver.get_weights()
        def pick(yaml_key, fallback):
            v = data.get(yaml_key, None)
            return float(v) if v is not None else float(fallback)

        out = {
            'q_n':            pick('q_n', w['q_n']),
            'q_n_term':       pick('q_n_term', w['q_n_term']),
            'q_v_term':       pick('q_v_term', w['q_v_term']),
            'gamma_progress': pick('gamma_progress', w['gamma']),
            'r_a':            pick('r_a', w['r_a']),
            'r_steer_reg':    pick('r_steer_reg', w['r_reg']),
            'r_dd':           pick('r_dd', w['r_dd']),
            'r_dd_rate':      pick('r_dd_rate', w['r_dd_rate']),
            'w_obs':          pick('w_obs', w['w_obs']),
            'sigma_s_obs':    pick('sigma_s_obs', w['sigma_s_obs']),
            'sigma_n_obs':    pick('sigma_n_obs', w['sigma_n_obs']),
            'w_side_bias':    pick('w_side_bias', w['w_side_bias']),
            'gap_lat':        pick('gap_lat', w['gap_lat']),
            'w_wall_buf':     pick('w_wall_buf', w['w_wall_buf']),
            'wall_buf':       pick('wall_buf', w['wall_buf']),
            'w_slack':        pick('w_slack', w['w_slack']),
            'w_cont':         pick('w_cont', w['w_cont']),
            'fail_streak_H':   int(data.get('fail_streak_H',
                                            self._fail_tier_H)),
            'quintic_delta_s': float(data.get('quintic_delta_s',
                                              self._quintic_delta_s)),
        }
        # Apply reloaded cost weights to the live solver so the next solve
        # picks them up even before rqt pushes them back to the sliders.
        self.solver.update_weights(
            q_n=out['q_n'], q_n_term=out['q_n_term'],
            q_v_term=out['q_v_term'], gamma=out['gamma_progress'],
            r_a=out['r_a'], r_reg=out['r_steer_reg'],
            r_dd=out['r_dd'], r_dd_rate=out['r_dd_rate'],
            w_obs=out['w_obs'], sigma_s_obs=out['sigma_s_obs'],
            sigma_n_obs=out['sigma_n_obs'],
            w_side_bias=out['w_side_bias'], gap_lat=out['gap_lat'],
            w_wall_buf=out['w_wall_buf'], wall_buf=out['wall_buf'],
            w_slack=out['w_slack'], w_cont=out['w_cont'],
        )
        self._fail_tier_H = out['fail_streak_H']
        self._quintic_delta_s = out['quintic_delta_s']
        return out

    # ---- Legacy MPCC / Frenet-D callback (rebuild on weight change) -------
    def _push_params_to_dynreg(self):
        """Seed rqt sliders with the rosparam-loaded values so the first
        connect doesn't snap to .cfg defaults and silently overwrite YAML."""
        if self._dyn_srv is None:
            return
        self._suppress_dynreg_cb = True
        try:
            self._dyn_srv.update_configuration({
                'w_contour':          float(self.solver.w_contour),
                'w_lag':              float(self.solver.w_lag),
                'w_velocity':         float(self.solver.w_velocity),
                'v_bias_max':         float(self.solver.v_bias_max),
                'w_dv':               float(self.solver.w_dv),
                'w_dsteering':        float(self.solver.w_dsteering),
                'w_slack':            float(self.solver.w_slack),
                'contour_ramp_start': float(self.solver.contour_ramp_start),
                'lag_ramp_start':     float(self.solver.lag_ramp_start),
                'max_speed':          float(self.solver.v_max),
                'min_speed':          float(self.solver.v_min),
                'max_steering':       float(self.solver.theta_max),
                'boundary_inflation': float(self.solver.inflation),
                'w_obstacle':         float(self.w_obstacle),
                'obstacle_sigma':     float(self.obstacle_sigma),
                'w_wall':             float(self.w_wall),
                'wall_safe':          float(self.wall_safe),
                'fail_streak_H':      int(self._fail_tier_H),
                'quintic_delta_s':    float(self._quintic_delta_s),
                'ipopt_max_iter':     int(self.solver.ipopt_max_iter),
                'save_params':        False,
                'reset_params':       False,
            })
        finally:
            self._suppress_dynreg_cb = False

    def _weight_cb(self, config, level):
        # ### HJ : Phase 5 — save/reset one-shot triggers (pattern from
        # sampling_planner — mutate `config` in place on reset so the outer
        # return replays the reloaded values into rqt).
        if config.save_params:
            self._save_yaml(config)
            config.save_params = False
        if config.reset_params:
            new_cfg = self._reload_yaml()
            if new_cfg is not None:
                for k, v in new_cfg.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
                self.get_logger().info('[mpc][%s][reset] reloaded YAML: %s',
                              rospy.get_name(), self.instance_yaml_path)
            config.reset_params = False

        # ### HJ : Snapshot current weights BEFORE mutation, to log actual
        # diffs into events.jsonl. Only when not suppressing boot-time.
        before_snap = self._current_weights_snapshot() \
            if not self._suppress_dynreg_cb else None

        # --- Hot cost-weight swap (no NLP rebuild) --------------------------
        w_changed = False
        for k in ('w_contour', 'w_lag', 'w_velocity', 'v_bias_max',
                  'w_dv', 'w_dsteering', 'w_slack',
                  'contour_ramp_start', 'lag_ramp_start'):
            new = float(getattr(config, k))
            if abs(getattr(self.solver, k, None) - new) > 1e-12:
                w_changed = True
        # Cost weights ARE baked into the symbolic NLP — hot swap alone would
        # not take effect. We rebuild if any weight actually changed; cheap
        # compared to the planner tick.
        rebuild_needed = False

        # --- Box bounds (truly hot) -----------------------------------------
        if (abs(self.solver.v_max - float(config.max_speed)) > 1e-12 or
                abs(self.solver.v_min - float(config.min_speed)) > 1e-12 or
                abs(self.solver.theta_max - float(config.max_steering)) > 1e-12):
            self.solver.update_box_bounds(
                v_min=float(config.min_speed),
                v_max=float(config.max_speed),
                theta_max=float(config.max_steering),
            )

        # --- Corridor inflation (picked up on next ref slice) ---------------
        self.solver.update_inflation(float(config.boundary_inflation))

        # --- Obstacle params ------------------------------------------------
        self.w_obstacle = float(config.w_obstacle)
        # ### HJ : Gaussian obstacle cost + convex wall buffer soft cost.
        # Both w_wall and wall_safe are baked into the frenet_d NLP, so a
        # change requires a rebuild.
        new_w_wall = float(config.w_wall)
        new_wall_safe = float(config.wall_safe)
        if self.solver_backend == 'frenet_d':
            if abs(getattr(self.solver, 'w_wall', new_w_wall) - new_w_wall) > 1e-9:
                rebuild_needed = True
            if abs(getattr(self.solver, 'wall_safe', new_wall_safe) - new_wall_safe) > 1e-9:
                rebuild_needed = True
        self.w_wall = new_w_wall
        self.wall_safe = new_wall_safe
        new_sigma = float(config.obstacle_sigma)
        if abs(self.solver.obstacle_sigma - new_sigma) > 1e-9:
            rebuild_needed = True
            self.obstacle_sigma = new_sigma

        # --- IPOPT iterations (rebuild) -------------------------------------
        new_iter = int(config.ipopt_max_iter)
        if new_iter != self.solver.ipopt_max_iter:
            rebuild_needed = True

        # --- Fallback tuning (pure node state, no rebuild) ------------------
        self._fail_tier_H = int(config.fail_streak_H)
        self._quintic_delta_s = float(config.quintic_delta_s)

        # ### HJ : Suppress rebuild during the boot-time push. The values
        # going in are identical to what the solver was constructed with
        # (node reads rosparam → builds solver → pushes same values into
        # rqt), so "changed" here is a float-equality false positive.
        if (w_changed or rebuild_needed) and not self._suppress_dynreg_cb:
            t0 = self.get_clock().now().to_msg()
            self.solver.rebuild_nlp(
                w_contour=float(config.w_contour),
                w_lag=float(config.w_lag),
                w_velocity=float(config.w_velocity),
                v_bias_max=float(config.v_bias_max),
                w_dv=float(config.w_dv),
                w_dsteering=float(config.w_dsteering),
                w_slack=float(config.w_slack),
                contour_ramp_start=float(config.contour_ramp_start),
                lag_ramp_start=float(config.lag_ramp_start),
                ipopt_max_iter=new_iter,
                obstacle_sigma=new_sigma,
                w_wall=new_w_wall,
                wall_safe=new_wall_safe,
            )
            dt_ms = (self.get_clock().now().to_msg() - t0).to_sec() * 1000.0
            self.get_logger().info(
                '[mpc][%s] rebuilt NLP in %.1f ms (weights/sigma/iter changed)',
                rospy.get_name(), dt_ms)

        if not self._suppress_dynreg_cb:
            rospy.loginfo_throttle(
                2.0,
                '[mpc][%s] tune w(c/l/v/dv/dδ/slack)=%.2f/%.2f/%.2f/%.1f/%.1f/%.0f  '
                'v[%.1f,%.1f] δ=%.2f inf=%.2f  obs(w=%.0f σ=%.2f) wall(w=%.0f s=%.2f) H=%d Δs=%.1f iter=%d',
                rospy.get_name(),
                self.solver.w_contour, self.solver.w_lag, self.solver.w_velocity,
                self.solver.w_dv, self.solver.w_dsteering, self.solver.w_slack,
                self.solver.v_min, self.solver.v_max, self.solver.theta_max,
                self.solver.inflation,
                self.w_obstacle, self.obstacle_sigma,
                self.w_wall, self.wall_safe,
                self._fail_tier_H, self._quintic_delta_s,
                self.solver.ipopt_max_iter,
            )

        # ### HJ : events.jsonl — per-rqt-change timeline for live analysis.
        if before_snap is not None and self._debug_logger is not None:
            after_snap = self._current_weights_snapshot()
            diff = {}
            for k, v_new in after_snap.items():
                v_old = before_snap.get(k)
                if v_old is None:
                    continue
                try:
                    if abs(float(v_new) - float(v_old)) > 1e-9:
                        diff[k] = {'old': float(v_old), 'new': float(v_new)}
                except (TypeError, ValueError):
                    if v_new != v_old:
                        diff[k] = {'old': v_old, 'new': v_new}
            if diff:
                self._debug_logger.log_event('config_change', diff)
        return config

    def _save_yaml(self, config):
        """Write current dynreg state back to the instance YAML, preserving
        fields the cfg doesn't know about (e.g., N, dT, vehicle_L, n_obs_max,
        collision_mode — structural or launch-once params)."""
        if not self.instance_yaml_path:
            self.get_logger().warning('[mpc][%s][save] ~instance_yaml not set — skipping',
                          rospy.get_name())
            return
        try:
            if os.path.exists(self.instance_yaml_path):
                with open(self.instance_yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}

            # Cost weights
            data['w_contour']    = float(config.w_contour)
            data['w_lag']        = float(config.w_lag)
            data['w_velocity']   = float(config.w_velocity)
            data['v_bias_max']   = float(config.v_bias_max)
            data['w_dv']         = float(config.w_dv)
            data['w_dsteering']  = float(config.w_dsteering)
            data['w_slack']      = float(config.w_slack)
            data['contour_ramp_start'] = float(config.contour_ramp_start)
            data['lag_ramp_start']     = float(config.lag_ramp_start)
            # Box bounds + inflation
            data['max_speed']    = float(config.max_speed)
            data['min_speed']    = float(config.min_speed)
            data['max_steering'] = float(config.max_steering)
            data['boundary_inflation'] = float(config.boundary_inflation)
            # Obstacle
            data['w_obstacle']     = float(config.w_obstacle)
            data['obstacle_sigma'] = float(config.obstacle_sigma)
            data['w_wall']         = float(config.w_wall)
            data['wall_safe']      = float(config.wall_safe)
            # Fallback
            data['fail_streak_H']  = int(config.fail_streak_H)
            data['quintic_delta_s'] = float(config.quintic_delta_s)
            # IPOPT
            data['ipopt_max_iter'] = int(config.ipopt_max_iter)

            with open(self.instance_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info('[mpc][%s][save] YAML updated: %s',
                          rospy.get_name(), self.instance_yaml_path)
        except Exception as e:
            self.get_logger().error('[mpc][%s][save] failed: %s', rospy.get_name(), e)

    def _reload_yaml(self):
        """Return dict of dynreg-field values from YAML, or None on error."""
        if not self.instance_yaml_path or not os.path.exists(self.instance_yaml_path):
            self.get_logger().warning('[mpc][%s][reset] ~instance_yaml missing — skip',
                          rospy.get_name())
            return None
        try:
            with open(self.instance_yaml_path, 'r') as f:
                d = yaml.safe_load(f) or {}
            return {
                'w_contour':          float(d.get('w_contour',    self.solver.w_contour)),
                'w_lag':              float(d.get('w_lag',        self.solver.w_lag)),
                'w_velocity':         float(d.get('w_velocity',   self.solver.w_velocity)),
                'v_bias_max':         float(d.get('v_bias_max',   self.solver.v_bias_max)),
                'w_dv':               float(d.get('w_dv',         self.solver.w_dv)),
                'w_dsteering':        float(d.get('w_dsteering',  self.solver.w_dsteering)),
                'w_slack':            float(d.get('w_slack',      self.solver.w_slack)),
                'contour_ramp_start': float(d.get('contour_ramp_start', self.solver.contour_ramp_start)),
                'lag_ramp_start':     float(d.get('lag_ramp_start',     self.solver.lag_ramp_start)),
                'max_speed':          float(d.get('max_speed',    self.solver.v_max)),
                'min_speed':          float(d.get('min_speed',    self.solver.v_min)),
                'max_steering':       float(d.get('max_steering', self.solver.theta_max)),
                'boundary_inflation': float(d.get('boundary_inflation', self.solver.inflation)),
                'w_obstacle':         float(d.get('w_obstacle',     self.w_obstacle)),
                'obstacle_sigma':     float(d.get('obstacle_sigma', self.solver.obstacle_sigma)),
                'w_wall':             float(d.get('w_wall',         self.w_wall)),
                'wall_safe':          float(d.get('wall_safe',      self.wall_safe)),
                'fail_streak_H':      int(d.get('fail_streak_H',   self._fail_tier_H)),
                'quintic_delta_s':    float(d.get('quintic_delta_s', self._quintic_delta_s)),
                'ipopt_max_iter':     int(d.get('ipopt_max_iter',  self.solver.ipopt_max_iter)),
                'save_params':        False,
                'reset_params':       False,
            }
        except Exception as e:
            self.get_logger().error('[mpc][%s][reset] parse failed: %s', rospy.get_name(), e)
            return None

    # ---------------------------------------------------------------- helpers
    def _publish_status(self, s):
        self.pub_status.publish(String(data=s))

    def _get_ego_frenet(self):
        """Return (s, n) for the ego. Prefer /car_state/odom_frenet (3D-aware,
        stack-standard), fall back to local 3D xy+z argmin + lifter projection.

        Fallback is never silent — logs a throttled warning so we notice if the
        frenet republisher is down in a real run.
        """
        now = self.get_clock().now().to_msg()
        fresh = (
            self.ego_s is not None
            and self._frenet_t is not None
            and (now - self._frenet_t).to_sec() < self._frenet_stale_s
        )
        if fresh:
            return float(self.ego_s), float(self.ego_n)

        rospy.logwarn_throttle(
            1.0,
            '[mpc] /car_state/odom_frenet stale/missing — falling back to '
            'local 3D nearest (overpass layers may alias).',
        )
        idx = self._nearest_idx(self.car_x, self.car_y, self.car_z)
        s_fb = float(self.g_s[idx])
        n_fb = 0.0
        try:
            _, n_fb = self.lifter.project_xy_to_sn(
                self.car_x, self.car_y, idx_hint=idx,
            )
        except Exception:
            pass
        return s_fb, n_fb

    def _nearest_idx(self, x, y, z=None):
        # ### HJ : 3D distance (xy + z) so overpass layers don't collide.
        # In 2D, xy=(3.28,-2.08) matches both s=11.7 (z=0.14) and s=53.7
        # (z=0.56) on the gazebo_wall_2 map, causing ego_s to flip between
        # floors and feeding the NLP a 90°-rotated reference.
        if z is None:
            z = getattr(self, 'car_z', 0.0)
        d2 = ((self.g_x - x) ** 2
              + (self.g_y - y) ** 2
              + (self.g_z - z) ** 2)
        return int(np.argmin(d2))

    # ---------------------------------------------------------------- side
    def _decide_side(self, obs_arr, ref_slice):
        """Rule-based side-of-passing decision (LEFT / RIGHT / TRAIL / CLEAR).
        Kept external to the NLP so the MPC enforces one-sided hard
        constraints without ambiguity. See src/side_decider.py."""
        if obs_arr is None:
            return SIDE_CLEAR, 'clear', {'reason': 'no_obs_arr'}

        rs = np.asarray(ref_slice['ref_s'])
        dL_ref = np.asarray(ref_slice['d_left_arr'])
        dR_ref = np.asarray(ref_slice['d_right_arr'])
        N_plus_1 = rs.shape[0]

        ### HJ : 2026-04-27 — side decision filters TRULY PASSED obstacles
        ###      only. Earlier draft filtered any obs with s0 < ego_s, which
        ###      was wrong: a dynamic opponent in mid-overtake (ego at same
        ###      s within cm) is the SIDE-DECIDE TARGET, not "rear". Bag
        ###      2026-04-27-10-55-57 t=2.5 had ego_s=80.31 vs opponent
        ###      s=80.25 (alongside, OT in progress) — that obstacle MUST
        ###      drive side decision. Real "rear" only when ego physically
        ###      cleared the obstacle: signed_dist < -fully_passed_m.
        ###      fully_passed_m = ego_length + obs_half so ego's tail is
        ###      past obstacle's head. Within that window, obstacle stays
        ###      in side decision as the alongside OT target.
        ego_s_for_side = float(rs[0]) if rs.size > 0 else 0.0
        ego_length = float(getattr(self.solver, 'ego_half', 0.15)) * 2.0
        obs_half_default = float(getattr(self, '_obs_half_fixed', 0.05))
        fully_passed_m = ego_length + obs_half_default + 0.10  # ~0.45m

        obs_list = []
        for o in range(obs_arr.shape[0]):
            w_ts = obs_arr[o, :, 2]
            if float(np.max(w_ts)) <= 0.0:
                continue
            s0 = float(obs_arr[o, 0, 0])
            n0 = float(obs_arr[o, 0, 1])
            sN = float(obs_arr[o, -1, 0])
            nN = float(obs_arr[o, -1, 1])
            ### HJ : skip ONLY obstacles ego has fully cleared (longer than
            ###      ego_length + obs_half behind). Alongside obstacles
            ###      (within ±fully_passed_m) are kept — they're the active
            ###      OT target, even if s0 is marginally < ego_s.
            if s0 < ego_s_for_side - fully_passed_m:
                continue
            ### HJ : end
            v_s_obs = (sN - s0) / max(self.N * self.dT, 1e-3)
            ### HJ : 2026-04-27 — d_L/d_R lookup at obstacle's EXACT world s
            ###      via lifter (np.interp wraps s%L). For a static obstacle
            ###      this MUST be constant tick-to-tick. No fallback path —
            ###      previous try/except silently routed to ref_slice[k_near]
            ###      which moved with ego_s in tight corners and produced the
            ###      d_free swing the user observed.
            d_L_raw = float(self.lifter._interp(s0, self.lifter.g_dleft))
            d_R_raw = float(self.lifter._interp(s0, self.lifter.g_dright))
            cos_d = 1.0
            if (self.g_psi_center is not None
                    and len(self.g_psi_center) == len(self.lifter.g_s)):
                psi = float(self.lifter._interp_psi(s0))
                psi_c = float(self.lifter._interp(s0, self.g_psi_center))
                cos_d = float(max(np.cos(psi - psi_c), 0.5))
            d_L_eff = d_L_raw / cos_d
            d_R_eff = d_R_raw / cos_d
            ref_v_at_obs = float(self.lifter._interp(s0, self.lifter.g_vx))
            ### HJ : per-slot half_width from obs_size_source. Falls back to
            ###      _obs_half_fixed when sidecar unavailable (e.g. legacy backend).
            try:
                half_w = float(self._obs_half_arr[o])
            except (AttributeError, IndexError):
                half_w = float(self._obs_half_fixed)
            ### HJ : 2026-04-27 — propagate static/dynamic flag.
            try:
                is_static = bool(self._obs_static_arr[o])
            except (AttributeError, IndexError):
                # Fallback: classify by computed v_s_obs.
                is_static = abs(v_s_obs) < 0.3
            ### HJ : end
            obs_list.append({
                's0': s0, 'n0': n0, 'v_s_obs': v_s_obs,
                'half_width': half_w,
                'd_L': d_L_eff,
                'd_R': d_R_eff,
                'ref_v': ref_v_at_obs,
                'is_static': is_static,
            })
            ### HJ : end
        # sort by forward distance (smallest s0 first assuming ref_s is
        # monotone forward from ego — which _slice_local_ref guarantees)
        obs_list.sort(key=lambda d: d['s0'])

        ego_v = float(getattr(self, 'car_vx', 0.0))
        ### HJ : 2026-04-27 (v4) — pass ego state so SideDecider can do
        ###      ego-aware reach feasibility + switching-safety check.
        ego_n_for_decider = float(ref_slice.get('n_ego', 0.0))
        return self.side_decider.decide(
            ego_v, obs_list,
            ego_n=ego_n_for_decider,
            ego_s=ego_s_for_side)
        ### HJ : end

    # ---------------------------------------------------------------- Phase X FSM
    def _obs_in_horizon_and_ttc(self, obs_arr, ego_s, ego_v):
        """Return (min_ds, ttc_min) over active obstacles.

        min_ds : minimum FORWARD s-distance to any active obstacle (infinity
                 if no active obstacle or tracking stale). This replaces the
                 previous boolean "in horizon" gate — the caller
                 (_update_mpc_mode) applies asymmetric enter/exit distance
                 thresholds for the mode-switch hysteresis.
        ttc_min: same as before (min time-to-collision).
        """
        if obs_arr is None or obs_arr.shape[0] == 0:
            return float('inf'), float('inf')
        ### HJ : 2026-04-27 — rear window. When ego just passed an obstacle,
        ###      ds = s_obs - ego_s is small NEGATIVE. Old code wrapped it
        ###      to ~track_length (huge) → mode flips to NO_OBS immediately
        ###      → q_n pulls ego to raceline while obstacle still beside it.
        ###      Now: keep ds as small negative (= "alongside / just past")
        ###      until ego is more than rear_window m past — only then wrap.
        ###      Mode logic naturally treats small ds as still in horizon.
        rear_window = float(getattr(self, '_obs_rear_window_m', 3.0))
        ### HJ : end
        min_ds = float('inf')
        ttc_min = float('inf')
        for o in range(obs_arr.shape[0]):
            w_ts = obs_arr[o, :, 2]
            if float(np.max(w_ts)) <= 0.0:
                continue   # inactive slot (stale / dedup / out-of-range)
            s0 = float(obs_arr[o, 0, 0])
            sN = float(obs_arr[o, -1, 0])
            ds = s0 - ego_s
            if ds < -rear_window:
                ds += self.track_length   # wrap (genuinely far behind = full lap)
            if ds < min_ds:
                min_ds = ds
            v_obs = (sN - s0) / max(self.N * self.dT, 1e-3)
            v_rel = max(ego_v - v_obs, 0.1)
            ttc = max(ds, 0.0) / v_rel
            if ttc < ttc_min:
                ttc_min = ttc
        return min_ds, ttc_min

    def _update_mpc_mode(self, ego_n, side_int, min_obs_ds, ttc_min):
        """Run the MPC internal 2-state FSM (WITH_OBS / NO_OBS).

        Uses asymmetric distance hysteresis on `min_obs_ds` (min forward
        s-distance to any active obstacle):
            enter_dist < exit_dist    (e.g. 5 m / 10 m)
            NO_OBS → WITH_OBS:  min_obs_ds < enter_dist
            WITH_OBS → NO_OBS:  min_obs_ds > exit_dist   (or tracking stale
                                                          → min_obs_ds = inf)
        This lets GB tracking run at full speed until the opponent is
        actually close, while preventing flip-flop at the boundary.

        alpha_ramp keeps the solver cost blend continuous across the flip.
        """
        if self.state != 'auto':
            self._mode_dwell += 1
            self._alpha_ramp = (0.0 if self._mpc_mode == MPC_MODE_WITH_OBS
                                else 1.0)
            self._last_ttc_min = float(ttc_min)
            self._last_obs_in_horizon = bool(min_obs_ds < self._obs_enter_dist_m)
            self._last_min_obs_ds = float(min_obs_ds)
            return

        prev = self._mpc_mode

        # ### HJ : 2026-04-24 — asymmetric enter/exit hysteresis (RESTORED).
        # Approach: stays NO_OBS until obs reaches obs_enter_dist_m (5 m).
        # Recede:  stays WITH_OBS until obs exceeds obs_exit_dist_m (10 m).
        # Stale / no obstacle: min_obs_ds = inf → WITH_OBS cannot be entered,
        # and if already WITH_OBS, exits immediately (inf > exit_dist).
        if prev == MPC_MODE_WITH_OBS:
            target = (MPC_MODE_NO_OBS if min_obs_ds > self._obs_exit_dist_m
                      else MPC_MODE_WITH_OBS)
        else:
            target = (MPC_MODE_WITH_OBS if min_obs_ds < self._obs_enter_dist_m
                      else MPC_MODE_NO_OBS)

        # Anti-chatter dwell guard.
        if (target != prev
                and self._mode_dwell < self._mode_dwell_min_ticks):
            target = prev

        if target != prev:
            self._prev_mpc_mode = prev
            self._mpc_mode = target
            self._mode_dwell = 0
            ### HJ : 2026-04-27 — invalidate recovery cache on mode flip
            ###      to WITH_OBS (obstacle near, need real planning).
            if target == MPC_MODE_WITH_OBS and prev == MPC_MODE_NO_OBS:
                self._invalidate_recovery_cache('NO_OBS→WITH_OBS')
            ### HJ : end
            ### HJ : 2026-04-26 (A5) — warm-start reset on WITH_OBS → NO_OBS.
            ###      Right after overtake, the warm-start carries the OT
            ###      avoidance shape (n[k] biased to one side around obstacle
            ###      position). New cost balance (no obstacle) wants ego on
            ###      raceline. With limited iterations, ipopt may settle near
            ###      the OT shape's local minimum → trajectory comes out
            ###      wobbly for several ticks until warm-start drifts to GB.
            ###      Resetting forces a fresh seed so the new convergence-only
            ###      cost lands a clean GB-aligned trajectory immediately.
            ###      WITH_OBS direction kept normal (warm-start continuity
            ###      matters for OT path smoothness).
            if (prev == MPC_MODE_WITH_OBS and target == MPC_MODE_NO_OBS
                    and self.solver_backend == 'frenet_kin'):
                try:
                    self.solver.reset_warm_start()
                    rospy.loginfo_throttle(
                        2.0, '[mpc][%s] warm-start reset on OT→NO_OBS',
                        rospy.get_name())
                except Exception:
                    pass
                ### HJ : 2026-04-26 (A6) — kick post-OT q_n boost.
                self._post_ot_boost_ticks_left = self._post_ot_boost_total_ticks
                ### HJ : end
                ### HJ : 2026-04-27 (R1) — flag for proactive recovery cache
                ###      build on this tick. _plan_loop reads this flag
                ###      AFTER _update_mpc_mode and forces a cache build so
                ###      the very next sample hijacks NLP with a clean
                ###      ego→GB recovery path. Eliminates the post-OT
                ###      "wobble for several ticks while NLP unwinds OT
                ###      shape from warm-start" problem.
                self._just_transitioned_to_no_obs = True
                ### HJ : end
                ### HJ : 2026-04-27 (R2) — clear SideDecider commit on
                ###      OT exit. SideDecider's _prev_side / _pending_side
                ###      hold_ticks=10 carries the OT's left/right
                ###      selection into NO_OBS, biasing the solver. Reset
                ###      them so first NO_OBS tick has SIDE_CLEAR (no
                ###      bias). Also zero bias_scale immediately.
                try:
                    self.side_decider._prev_side = SIDE_CLEAR
                    self.side_decider._pending_side = SIDE_CLEAR
                    self.side_decider._pending_streak = 0
                except Exception:
                    pass
                self._last_side_int = SIDE_CLEAR
                self._last_side_str = 'clear'
                self._last_bias_scale = 0.0
                ### HJ : end
            ### HJ : end
        else:
            self._mode_dwell += 1

        # Alpha ramp unchanged.
        K = max(self._K_trans, 1)
        step = 1.0 / K
        goal_alpha = (0.0 if self._mpc_mode == MPC_MODE_WITH_OBS else 1.0)
        if self._alpha_ramp < goal_alpha:
            self._alpha_ramp = float(min(self._alpha_ramp + step, goal_alpha))
        elif self._alpha_ramp > goal_alpha:
            self._alpha_ramp = float(max(self._alpha_ramp - step, goal_alpha))

        self._last_ttc_min = float(ttc_min)
        self._last_min_obs_ds = float(min_obs_ds)
        # "in_horizon" is now defined as "within the enter threshold" for
        # backward-compat with debug consumers that expected the old boolean.
        self._last_obs_in_horizon = bool(min_obs_ds < self._obs_enter_dist_m)
        self._last_side_int_for_mode = int(side_int)

    # ---------------------------------------------------------------- Adaptive recovery q_n
    def _apply_recovery_near_boost(self, ego_n):
        """### HJ : 2026-04-24 — when in NO_OBS (recovery) and ego is close
        to GB, boost stage q_n so the early horizon also converges quickly.
        q_n_eff = q_n_mode_blended + q_n_near_boost · max(0, 1 - |ego_n|/thresh)
        Only meaningful in auto mode. Applied AFTER mode weights + variance
        so it overrides the q_n portion while sigmas stay as variance set.
        """
        if self.state != 'auto':
            return
        if self._mpc_mode != MPC_MODE_NO_OBS:
            # outside recovery: clear any residual boost
            target_boost = 0.0
            # A6: also clear post-OT countdown (we're not in NO_OBS).
            self._post_ot_boost_ticks_left = 0
        else:
            n_abs = abs(float(ego_n))
            thresh = max(self._q_n_near_thresh, 1e-3)
            factor = max(0.0, 1.0 - n_abs / thresh)
            near_boost = self._q_n_near_boost * factor
            ### HJ : 2026-04-26 (A6) — transient post-OT boost, linear decay.
            if self._post_ot_boost_ticks_left > 0:
                decay = (float(self._post_ot_boost_ticks_left)
                         / float(max(self._post_ot_boost_total_ticks, 1)))
                post_ot_boost = self._post_ot_boost_amount * decay
                self._post_ot_boost_ticks_left -= 1
            else:
                post_ot_boost = 0.0
            ### HJ : end
            target_boost = near_boost + post_ot_boost

        if abs(target_boost - self._last_q_n_boost_applied) < 1e-3:
            return
        # Current solver q_n already reflects mode + alpha blend from
        # _apply_mode_weights. We add the boost ON TOP (remove prior boost,
        # add new). Track last applied so we don't double-add.
        try:
            base_q_n = float(self.solver.q_n) - self._last_q_n_boost_applied
            new_q_n = base_q_n + target_boost
            self.solver.update_weights(q_n=new_q_n)
            self._last_q_n_boost_applied = target_boost
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] recovery near-boost failed: %s',
                rospy.get_name(), e)


    ### HJ : 2026-04-26 — tick_json diagnostic helpers (A1, A4).
    def _build_a1_vmax_debug(self):
        """Sample solver-built vmax[k] to check ramp shape per tick."""
        try:
            vmax = getattr(self.solver, '_last_vmax', None)
            if vmax is None:
                return {'k0': None, 'k1': None, 'k_mid': None, 'k_end': None,
                        'monotone_dec': None, 'cap_min': None}
            arr = np.asarray(vmax, dtype=float)
            N1 = arr.shape[0]
            mid = N1 // 2
            return {
                'k0': round(float(arr[0]), 3),
                'k1': round(float(arr[1]), 3) if N1 > 1 else None,
                'k_mid': round(float(arr[mid]), 3),
                'k_end': round(float(arr[-1]), 3),
                'cap_min': round(float(arr.min()), 3),
                'monotone_dec': bool(np.all(np.diff(arr) <= 1e-6)),
            }
        except Exception:
            return {'k0': None, 'k1': None, 'k_mid': None, 'k_end': None,
                    'monotone_dec': None, 'cap_min': None}

    def _build_a4_wall_ramp_debug(self):
        """Report current wall_ramp[k] state and entry detection."""
        try:
            ramp = getattr(self.solver, '_wall_ramp_arr', None)
            if ramp is None:
                return {'active': False, 'k0': None, 'k1': None, 'k2': None,
                        'first_full_k': None, 'K_entry': None}
            arr = np.asarray(ramp, dtype=float)
            idx_full = np.where(arr >= 0.999)[0]
            first_full = int(idx_full[0]) if idx_full.size > 0 else None
            active = bool(arr[0] < 0.999)
            return {
                'active': active,
                'k0': round(float(arr[0]), 3),
                'k1': round(float(arr[1]), 3) if arr.shape[0] > 1 else None,
                'k2': round(float(arr[2]), 3) if arr.shape[0] > 2 else None,
                'first_full_k': first_full,
                'K_entry': first_full,
            }
        except Exception:
            return {'active': False, 'k0': None, 'k1': None, 'k2': None,
                    'first_full_k': None, 'K_entry': None}
    ### HJ : end

    ### HJ : 2026-04-26 (A4-c) — graceful corridor entry ramp.
    ###      Detects if ego is inside the wall_buf cushion or outside the
    ###      hard corridor. Sets `wall_ramp[k]` so J_wall and J_slack costs
    ###      are zero at k=0..K_entry-1 and ramp to 1 at K_entry+. This lets
    ###      the solver "accept" the current wall-violation as a fait
    ###      accompli at k=0 and produce a smooth escape trajectory; only
    ###      from K_entry onward is wall enforcement at full strength.
    ###      K_entry chosen so escape rate ≤ a_lat_max ≈ 5 m/s² is feasible.
    def _apply_wall_entry_ramp(self, ego_n, ref_slice):
        if ref_slice is None:
            return
        N1 = self.N + 1
        # Default: full enforcement at every step.
        ramp = np.ones(N1, dtype=np.float64)

        # Cushion threshold: |ego_n| inside (d - cushion) means ego in cushion.
        try:
            dL_arr = np.asarray(ref_slice['d_left_arr'], dtype=float)
            dR_arr = np.asarray(ref_slice['d_right_arr'], dtype=float)
        except KeyError:
            self.solver.set_wall_ramp(ramp)
            return

        wall_buf = float(getattr(self.solver, 'wall_buf', 0.30))
        ego_half = float(getattr(self.solver, 'ego_half', 0.15))
        buf_c = ego_half + wall_buf  # cushion total depth from wall

        # Use k=0 corridor for entry detection
        nub_k0 = float(dL_arr[0]) - buf_c          # cushion outer (left)
        nlb_k0 = -(float(dR_arr[0]) - buf_c)       # cushion outer (right)

        # Compute overshoot beyond cushion
        if ego_n > nub_k0:
            overshoot = float(ego_n) - nub_k0
        elif ego_n < nlb_k0:
            overshoot = nlb_k0 - float(ego_n)
        else:
            # Inside cushion or comfortably inside corridor; no relaxation
            self.solver.set_wall_ramp(ramp)
            return

        # K_entry: how many steps to bring ramp to 1.0.
        # Heuristic: 1 step per `_wall_ramp_step_m` of overshoot, clamped.
        step_m = max(self._wall_ramp_step_m, 0.02)
        K_entry = int(np.ceil(overshoot / step_m))
        K_entry = int(np.clip(K_entry, self._wall_ramp_K_min,
                              self._wall_ramp_K_max))

        # Linear ramp 0 → 1 over K_entry steps; 1.0 from K_entry onward.
        for k in range(N1):
            if k < K_entry:
                ramp[k] = float(k) / float(K_entry)
            else:
                ramp[k] = 1.0

        self.solver.set_wall_ramp(ramp)

    ### HJ : end

    # ---------------------------------------------------------------- Variance-aware bubble
    def _apply_variance_inflation(self):
        """### HJ : 2026-04-24 — If ~use_pred_variance is True, inflate
        σ_s_obs / σ_n_obs based on current tick's max vs_var / vd_var
        over active obstacles.

          σ_eff = √(σ_base² + var · (N·dT)²)

        Applied AFTER _apply_mode_weights so it overrides. When disabled,
        restores the YAML base sigmas (in case a prior tick inflated them).
        No-op when sigma difference is below 1e-4 (avoid update churn).
        """
        if not self._use_pred_variance:
            # Ensure base sigmas are active (in case user toggled param OFF mid-run).
            sigma_s_target = self._sigma_s_obs_base
            sigma_n_target = self._sigma_n_obs_base
        else:
            t_h = float(self.N) * float(self.dT)
            sigma_s_target = float(np.sqrt(
                self._sigma_s_obs_base ** 2 + self._last_obs_max_vs_var * t_h * t_h))
            sigma_n_target = float(np.sqrt(
                self._sigma_n_obs_base ** 2 + self._last_obs_max_vd_var * t_h * t_h))

        # Skip call if negligible change
        if (abs(sigma_s_target - self._last_sigma_s_eff) < 1e-4 and
                abs(sigma_n_target - self._last_sigma_n_eff) < 1e-4):
            return
        try:
            self.solver.update_weights(
                sigma_s_obs=sigma_s_target,
                sigma_n_obs=sigma_n_target)
            self._last_sigma_s_eff = sigma_s_target
            self._last_sigma_n_eff = sigma_n_target
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] variance inflation failed: %s',
                rospy.get_name(), e)

    # ---------------------------------------------------------------- Phase X weight switch
    def _capture_with_obs_baseline(self):
        """Snapshot current solver weights as the WITH_OBS baseline. Called
        lazily on the first update so user's rqt-tuned values (if any)
        become the baseline rather than hard-coded YAML values."""
        if self._with_obs_baseline_weights is None:
            try:
                self._with_obs_baseline_weights = dict(self.solver.get_weights())
            except Exception:
                self._with_obs_baseline_weights = {}

    def _apply_mode_weights(self, alpha_ramp):
        """Phase X — push a blended weight profile to the solver.

        alpha_ramp semantics:
            0.0 → pure WITH_OBS weights (baseline)
            1.0 → pure NO_OBS weights (obstacle cost off, stronger n→0 pull)
            in-between → linear blend per-key

        Called every tick — idempotent on no-change.
        """
        if not self._no_obs_weights:
            return  # no NO_OBS profile configured; solver keeps baseline
        self._capture_with_obs_baseline()
        if not self._with_obs_baseline_weights:
            return
        alpha = float(np.clip(alpha_ramp, 0.0, 1.0))
        if (self._last_applied_weight_alpha is not None
                and abs(self._last_applied_weight_alpha - alpha) < 1e-3):
            return

        base = self._with_obs_baseline_weights
        target = self._no_obs_weights
        blend = {}
        for key, base_val in base.items():
            no_obs_val = target.get(key, base_val)
            blend[key] = (1.0 - alpha) * base_val + alpha * no_obs_val
        # Target-only keys → scale linearly.
        for key, no_obs_val in target.items():
            if key not in base:
                blend[key] = alpha * no_obs_val

        try:
            self.solver.update_weights(**blend)
            self._last_applied_weight_alpha = alpha
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] update_weights failed: %s',
                rospy.get_name(), e)

    ### HJ : 2026-04-28 Stage 2 Phase 2-1 + plan transition smooth ramp.
    def _apply_plan_weights(self, plan):
        """plan 의 weight overlay 를 solver 에 push. Plan 변경 시 5 tick
        (0.10s @ 50Hz) 동안 prev plan 의 weight 와 alpha-blend.

        User wants smooth trajectory on plan change but slowing alpha to
        10 ticks made MPC react too slowly to obstacles entering horizon
        (post_smooth bag: 1 collision, 2mm lateral). 5-tick ramp keeps
        reaction speed; trajectory smoothness is delivered by the solver's
        own continuity cost (w_cont=300) and the head-blending continuity
        guard with the lowered 0.08m threshold.
        """
        if plan is None:
            return
        try:
            weights = dict(plan.get('weights', {}))
            if not weights:
                return
            cur_name = plan.get('name', '?')
            prev_name = getattr(self, '_prev_applied_plan_name', None)
            prev_weights = getattr(self, '_prev_applied_plan_weights', None)
            # Plan 변경 시 alpha ramp 시작 (5 tick 동안 0→1)
            if prev_name is None or prev_name == cur_name:
                self._plan_transition_alpha = 1.0
            elif prev_name != cur_name:
                # 새 transition 시작 또는 진행 중
                if not hasattr(self, '_plan_transition_alpha') or self._plan_transition_alpha >= 1.0:
                    self._plan_transition_alpha = 0.0
                else:
                    self._plan_transition_alpha = min(1.0, self._plan_transition_alpha + 0.20)
            alpha = float(getattr(self, '_plan_transition_alpha', 1.0))
            if alpha < 1.0 and prev_weights is not None:
                # Blend: alpha=0 → prev, alpha=1 → cur
                blended = {}
                for k, v in weights.items():
                    pv = prev_weights.get(k, v)
                    blended[k] = (1.0 - alpha) * pv + alpha * v
                self.solver.update_weights(**blended)
            else:
                self.solver.update_weights(**weights)
                self._prev_applied_plan_name = cur_name
                self._prev_applied_plan_weights = weights.copy()
            self._last_applied_weight_alpha = None
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] plan weights apply failed (%s): %s',
                rospy.get_name(), plan.get('name', '?'), e)
    ### HJ : end

    # ---------------------------------------------------------------- Phase X speed painter
    def _lookup_gb_vx(self, s):
        """Binary-search GB raceline vx at arc-length s. s-based only, no xy
        roundtrip. Returns vx_mps at the closest raceline waypoint (linear
        interp between neighbours)."""
        if self._gb_vx_by_s_s is None:
            return 0.0
        s_arr = self._gb_vx_by_s_s
        v_arr = self._gb_vx_by_s_v
        # Handle wrap-around
        s_mod = float(s) % float(self.track_length) if self.track_length > 0 else float(s)
        idx = int(np.searchsorted(s_arr, s_mod))
        if idx <= 0:
            return float(v_arr[0])
        if idx >= len(s_arr):
            return float(v_arr[-1])
        s_lo = s_arr[idx - 1]
        s_hi = s_arr[idx]
        if s_hi - s_lo < 1e-9:
            return float(v_arr[idx])
        alpha = (s_mod - s_lo) / (s_hi - s_lo)
        return float((1.0 - alpha) * v_arr[idx - 1] + alpha * v_arr[idx])

    ### HJ : 2026-04-26 — smoothed kappa for SPEED CAP ONLY.
    ###      Computes a smooth, continuous curvature profile from wp xy. The
    ###      raw kappa stored on each wp (from solver) can ring tick-to-tick;
    ###      using it directly for v_curv = sqrt(mu·g/|kappa|) yields jagged
    ###      speed profiles. We re-derive kappa from xy via arc-length finite
    ###      differences and then savgol-smooth the heading (window 7,
    ###      polyorder 2 by default — small window so we don't bleed corner
    ###      curvature into straight regions). Path xy is NOT modified.
    ###      Returns np.array of len(wp_list).
    def _smooth_kappa_for_speed(self, wp_list):
        n = len(wp_list)
        if n < 3:
            return np.zeros(n, dtype=np.float64)
        xs = np.array([float(w.x_m) for w in wp_list], dtype=np.float64)
        ys = np.array([float(w.y_m) for w in wp_list], dtype=np.float64)
        dxs = np.diff(xs)
        dys = np.diff(ys)
        seg = np.sqrt(dxs * dxs + dys * dys)
        seg = np.where(seg < 1e-6, 1e-6, seg)
        # Heading per segment (n-1), then per-vertex by averaging neighbours.
        psi_seg = np.arctan2(dys, dxs)
        # Unwrap so atan2 jumps don't fake a huge curvature.
        psi_seg = np.unwrap(psi_seg)
        psi = np.zeros(n, dtype=np.float64)
        psi[1:-1] = 0.5 * (psi_seg[:-1] + psi_seg[1:])
        psi[0] = psi_seg[0]
        psi[-1] = psi_seg[-1]
        # Savgol smooth on psi if window fits.
        try:
            from scipy.signal import savgol_filter
            win = 7 if n >= 7 else (5 if n >= 5 else 3)
            if win % 2 == 0:
                win -= 1
            poly = 2 if win >= 5 else 1
            if win >= 3 and n >= win:
                psi_smooth = savgol_filter(psi, window_length=win,
                                           polyorder=poly, mode='nearest')
            else:
                psi_smooth = psi
        except Exception:
            psi_smooth = psi
        # kappa = dpsi/ds at each vertex (centered diff).
        # ds at vertex i ≈ (seg[i-1] + seg[i]) / 2 for interior; ends use neighbour seg.
        ds = np.zeros(n, dtype=np.float64)
        ds[1:-1] = 0.5 * (seg[:-1] + seg[1:])
        ds[0] = seg[0]
        ds[-1] = seg[-1]
        ds = np.where(ds < 1e-6, 1e-6, ds)
        dpsi = np.zeros(n, dtype=np.float64)
        dpsi[1:-1] = 0.5 * (
            (psi_smooth[2:] - psi_smooth[1:-1]) +
            (psi_smooth[1:-1] - psi_smooth[:-2]))
        dpsi[0] = psi_smooth[1] - psi_smooth[0]
        dpsi[-1] = psi_smooth[-1] - psi_smooth[-2]
        kappa = dpsi / ds
        # Optional second-pass low-pass on kappa itself for extra smoothness.
        try:
            from scipy.signal import savgol_filter
            win2 = 5 if n >= 5 else 3
            if win2 % 2 == 0:
                win2 -= 1
            if win2 >= 3 and n >= win2:
                kappa = savgol_filter(kappa, window_length=win2,
                                      polyorder=1, mode='nearest')
        except Exception:
            pass
        return kappa
    ### HJ : end

    def _post_process_speed(self, wp_list, mpc_mode):
        """Paint speed on MPC output waypoints and apply seam blend.

        Modifies wp_list[*].vx_mps in place. Returns (seam_idx, blend_applied,
        vx_first_step_delta) for tick_json.

        ### HJ : 2026-04-26 — curvature smoothing for speed cap ONLY.
        ###      The path xy (and stored kappa_radpm) is NOT modified — that
        ###      remains exactly what the solver produced, so the controller
        ###      sees the original geometry. Just for the v_curv cap we build
        ###      a separate smoothed-kappa view via:
        ###        (1) re-derive kappa from xy with arc-length finite diffs
        ###            (avoids stale per-step kappa from solver)
        ###        (2) savgol low-pass on the heading sequence (window 7,
        ###            polyorder 2) so per-wp kappa noise → smooth profile
        ###      Speed cap then uses smooth_kappa[k] in place of wp.kappa.
        ###      Result: vx profile no longer jitters with raw solver kappa
        ###      ringing, but path xy is untouched.

        Backward compat: painter is only active under state:='auto'. Legacy
        overtake/recovery/observe launches preserve original solver vx.
        """
        if self.state != 'auto':
            return -1, False, 0.0
        if not self._painter_enable or not wp_list:
            return -1, False, 0.0

        # 1) baseline: GB raceline vx at each wp.s_m
        baseline_vx = [self._lookup_gb_vx(w.s_m) for w in wp_list]

        # ---- smooth kappa view (speed-only, path xy untouched) ----
        smooth_kappa = self._smooth_kappa_for_speed(wp_list)

        # 2) curvature/mu cap — simple grip-limited v^2 ≤ mu*g/|kappa|
        g = 9.81
        capped = []
        for k, (w, v_gb) in enumerate(zip(wp_list, baseline_vx)):
            kappa = abs(float(smooth_kappa[k]))
            mu = max(abs(float(getattr(w, 'mu_rad', 0.0))), 0.6)
            if kappa < 1e-4:
                v_curv = v_gb
            else:
                v_curv = float(np.sqrt(max(mu * g / kappa, 0.0)))
            capped.append(min(v_gb, v_curv))

        # ### HJ : 2026-04-28 — plan-aware painter override tried and
        # reverted. User bag (2026-04-28-14-54-57) showed cmd_v0 stuck
        # at ~3.95 m/s while obs_vs was 1-3 m/s (TRAILING ego 가 obstacle
        # 따라잡으며 충돌). Root cause: even when the cap was set to
        # obs.vs - 0.3, the immediately-following ego_v continuity ramp
        # (`capped[0] = clip(capped[0], ego_v - a_max*dT, ego_v + a_max*dT)`,
        # ±0.15 m/s/tick) clipped capped[0] back to ego_v - 0.15, so
        # the brake intent never reached the controller's first
        # waypoint. The override needs a plan-aware a_max (TRAIL allowing
        # larger deceleration) before it can take effect — see Step 4
        # in HJ_docs/mpc_overtake_redesign_plan_20260428.md.

        # 3) ego v continuity — clip wp0 then forward acc limit
        ego_v = float(getattr(self, 'car_vx', 0.0))
        dT = self.dT
        a_max = self._painter_a_max
        capped[0] = float(np.clip(capped[0], ego_v - a_max * dT,
                                  ego_v + a_max * dT))
        for k in range(1, len(capped)):
            capped[k] = min(capped[k], capped[k - 1] + a_max * dT)

        # 4) seam blend
        K = max(self._painter_K_blend, 1)
        if mpc_mode == MPC_MODE_NO_OBS:
            # NO_OBS mode is the convergent solve — tail is always near GB,
            # so blend the final K waypoints to GB vx smoothly.
            seam_idx = max(len(capped) - K, 0)
        else:
            # WITH_OBS: scan forward for first waypoint that is near GB
            # (|d| < n_near); seam blend begins there. If none found, skip.
            seam_idx = -1
            for k, w in enumerate(wp_list):
                if abs(float(getattr(w, 'd_m', 0.0))) < self._painter_n_near:
                    seam_idx = k
                    break
        blend_applied = False
        if seam_idx >= 0:
            blend_end = min(seam_idx + K, len(capped))
            for k in range(seam_idx, blend_end):
                alpha = (k - seam_idx + 1) / K
                vx_gb = self._lookup_gb_vx(wp_list[k].s_m)
                capped[k] = (1.0 - alpha) * capped[k] + alpha * vx_gb
            blend_applied = True

        # Write back
        for k, w in enumerate(wp_list):
            w.vx_mps = float(capped[k])

        vx_first_step_delta = float(abs(capped[0] - ego_v))
        return seam_idx, blend_applied, vx_first_step_delta

    def _apply_continuity_guard(self, wp_list):
        """Blend first K_guard waypoints of wp_list with the previously
        published wp_list (s-shifted) if their wp-wise L2 exceeds threshold.
        Modifies wp_list[*].{x_m, y_m, z_m, d_m, psi_rad} in place.

        Returns (L2, applied) for tick_json.

        Backward compat: active only under state:='auto'.
        """
        if self.state != 'auto':
            return 0.0, False
        if (not self._continuity_guard_enable
                or self._last_published_wpnts is None
                or len(self._last_published_wpnts) == 0
                or len(wp_list) == 0):
            return 0.0, False
        K = max(self._continuity_K_guard, 1)
        prev = self._last_published_wpnts
        # Align by s: for each current wp, pick the previous wp with closest s.
        # Simpler: use shifted index (1-step forward shift assumption).
        cur_head = wp_list[:K]
        prev_shifted = prev[1:1 + K] if len(prev) > 1 else prev[:K]
        n = min(len(cur_head), len(prev_shifted))
        if n == 0:
            return 0.0, False
        sq_sum = 0.0
        for k in range(n):
            dx = cur_head[k].x_m - prev_shifted[k].x_m
            dy = cur_head[k].y_m - prev_shifted[k].y_m
            dz = getattr(cur_head[k], 'z_m', 0.0) - getattr(prev_shifted[k], 'z_m', 0.0)
            sq_sum += dx * dx + dy * dy + dz * dz
        L2 = float(np.sqrt(sq_sum / n))
        if L2 <= self._continuity_threshold_m:
            return L2, False
        # Blend xy/z/d_m/psi_rad (keep s_m monotone).
        for k in range(n):
            beta = (k + 1) / K   # 0 → 1; weight of new path grows
            cur_head[k].x_m = float((1 - beta) * prev_shifted[k].x_m + beta * cur_head[k].x_m)
            cur_head[k].y_m = float((1 - beta) * prev_shifted[k].y_m + beta * cur_head[k].y_m)
            if hasattr(cur_head[k], 'z_m'):
                cur_head[k].z_m = float((1 - beta) * getattr(prev_shifted[k], 'z_m', 0.0)
                                        + beta * cur_head[k].z_m)
            cur_head[k].d_m = float((1 - beta) * prev_shifted[k].d_m + beta * cur_head[k].d_m)
            cur_head[k].psi_rad = float((1 - beta) * prev_shifted[k].psi_rad
                                        + beta * cur_head[k].psi_rad)
        return L2, True

    # ---------------------------------------------------------------- lift
    def _lift_frenet_to_xy(self, traj_frenet, ref_slice):
        """Convert solver (s, n, mu, v) trajectory into (x, y, psi, s, z).
        Uses ref_slice center_points + tangent (ref_dx, ref_dy) so no
        xy→frenet round-trip is needed (CLAUDE.md: 3D 트랙에서 Frenet xy
        round-trip 금지). `s` is the solver's Frenet station (raceline
        arc-length) which uniquely identifies the overpass floor; `z` is
        interpolated from g_z at that s. Downstream viz/markers MUST use
        the carried s/z instead of re-projecting (x, y) — otherwise the
        2D nearest-xy lookup aliases overpass layers at crossing points.
        Returned shape: (M, 5) columns [x, y, psi, s, z]."""
        N1 = traj_frenet.shape[0]
        rc = ref_slice['center_points']          # (N+1, 2)
        rdx = ref_slice['ref_dx']                # cos(psi_ref)
        rdy = ref_slice['ref_dy']                # sin(psi_ref)
        M = int(min(N1, rc.shape[0]))
        # Left (+n) normal in xy = (-sin(psi_ref), cos(psi_ref)) = (-rdy, rdx)
        lnx = -rdy[:M]
        lny = rdx[:M]
        out = np.zeros((M, 5), dtype=np.float64)
        for k in range(M):
            s_k = float(traj_frenet[k, 0])
            n_k = float(traj_frenet[k, 1])
            mu_k = float(traj_frenet[k, 2])
            out[k, 0] = float(rc[k, 0] + n_k * lnx[k])
            out[k, 1] = float(rc[k, 1] + n_k * lny[k])
            psi_ref = float(np.arctan2(rdy[k], rdx[k]))
            out[k, 2] = psi_ref + mu_k
            out[k, 3] = s_k
            out[k, 4] = float(self.lifter._interp(s_k, self.lifter.g_z))
        return out

    def _slice_local_ref(self, s_cur):
        """Build N+1 local reference from /global_waypoints around s_cur.
        Port of the original mpc_planner_node._slice_local_ref — unchanged."""
        N = self.N
        dT = self.solver.dT
        tl = self.track_length
        inflation = self.solver.inflation

        center_pts, left_pts, right_pts = [], [], []
        d_left_arr, d_right_arr = [], []
        d_left_raw_arr, d_right_raw_arr = [], []
        cos_delta_arr = []
        ref_v, ref_s, ref_dx, ref_dy, ref_psi, ref_kappa = [], [], [], [], [], []
        s_offset = 0.0
        prev_s = None

        target_s = s_cur
        for k in range(N + 1):
            s_wrap = target_s % tl if tl > 0 else target_s
            idx = int(np.argmin(np.abs(self.g_s - s_wrap)))

            x = self.g_x[idx]
            y = self.g_y[idx]
            psi = self.g_psi[idx]
            dl_raw = self.g_dleft[idx]
            dr_raw = self.g_dright[idx]
            kappa = self.g_kappa[idx]

            ### HJ : centerline-normal corridor → raceline-normal corridor.
            ###      d_left/d_right are measured along centerline normal at the foot.
            ###      Solver's hard wall |n[k]| <= eff_d expects RACELINE-normal limits.
            ###      Linear conversion: eff_d_rn = d_cn / cos(psi_R - psi_C).
            ###      Cap cos to 0.5 (60 deg) as a safety net against numerical glitches
            ###      — F1tenth tracks never sustain |Delta| beyond that.
            psi_c = self.g_psi_center[idx] if self.g_psi_center is not None else psi
            cos_delta = float(max(np.cos(psi - psi_c), 0.5))
            dl = float(dl_raw) / cos_delta
            dr = float(dr_raw) / cos_delta
            ### HJ : end

            ### HJ : marker now drawn along centerline normal (matches the actual
            ###      walls that walls_d was measured against). center stays at the
            ###      raceline waypoint; the wall point is raceline_xy ± n_C * d_raw.
            normal_R = np.array([-np.sin(psi), np.cos(psi)])
            normal_C = np.array([-np.sin(psi_c), np.cos(psi_c)])
            center = np.array([x, y])
            center_pts.append(center)
            left_pts.append(center + normal_C * max(dl_raw - inflation, 0.0))
            right_pts.append(center - normal_C * max(dr_raw - inflation, 0.0))
            d_left_arr.append(float(dl))
            d_right_arr.append(float(dr))
            d_left_raw_arr.append(float(dl_raw))
            d_right_raw_arr.append(float(dr_raw))
            cos_delta_arr.append(cos_delta)
            ### HJ : end
            ref_v.append(float(self.g_vx[idx]))
            ref_dx.append(np.cos(psi))
            ref_dy.append(np.sin(psi))
            ref_psi.append(float(psi))
            ref_kappa.append(float(kappa))

            s_val = float(self.g_s[idx])
            if prev_s is not None and s_val + s_offset < prev_s - 1.0:
                s_offset += max(tl or 100.0, 1.0)
            s_val += s_offset
            ref_s.append(s_val)
            prev_s = s_val

            local_v = max(float(self.g_vx[idx]), 1.0)
            target_s += local_v * dT

        ### HJ : Option A — explicit ref_x/y arrays for xy-based obstacle cost.
        cp_arr = np.array(center_pts)
        return {
            'center_points': cp_arr,
            'ref_x_arr': cp_arr[:, 0],
            'ref_y_arr': cp_arr[:, 1],
            'left_points': np.array(left_pts),
            'right_points': np.array(right_pts),
            'd_left_arr': np.array(d_left_arr),       # raceline-normal eff (post cos(Δ))
            'd_right_arr': np.array(d_right_arr),     # raceline-normal eff (post cos(Δ))
            'd_left_raw_arr': np.array(d_left_raw_arr),     # centerline-normal as published
            'd_right_raw_arr': np.array(d_right_raw_arr),
            'cos_delta_arr': np.array(cos_delta_arr),
            'ref_v': np.array(ref_v),
            'ref_s': np.array(ref_s),
            'ref_dx': np.array(ref_dx),
            'ref_dy': np.array(ref_dy),
            'ref_psi': np.array(ref_psi),
            'kappa_ref': np.array(ref_kappa),
        }

    # ---------------------------------------------------------------- main loop
    def _plan_loop(self, event):
        if not self.solver.ready or not self.pose_received:
            return

        # Runtime debug_log toggles (rosparam poll — cheap, ~μs).
        self._refresh_debug_runtime()

        # ### HJ : prefer canonical 3D-aware Frenet from odom_frenet_republisher.
        # Only fall back to local 3D xy+z nearest if odom_frenet is stale/missing
        # (startup, publisher crash). The local fallback can still alias overpass
        # layers if |Δz| < xy noise — odom_frenet uses proper frenet_converter
        # with z-aware segment projection which is more robust.
        s_cur, ego_n = self._get_ego_frenet()
        ref_slice = self._slice_local_ref(s_cur)
        # ### HJ : FrenetDSolver needs current lateral offset as the fixed
        # initial constraint n_0 = n_ego. Harmless for xy backend (ignored).
        ref_slice['n_ego'] = float(ego_n)

        # ### HJ : Phase 3.5 — fuse obstacle inputs and hand to solver.
        obs_arr, obs_tag = self._build_obstacle_array(s_cur)

        # ### HJ : v3 — obstacle EMA filter. Blend new obs_arr with prev
        # tick's (same shape (n_slot, N+1, 3)). Only blend (s, n) columns;
        # keep active-weight (col 2) as a hard on/off to avoid zombie slots.
        if self.solver_backend == 'frenet_kin':
            ### HJ : 2026-04-28 (S1-8) — ego_s 점프 (lap-wrap, sim reset)
            ###      감지 시 EMA reset. 점프 후 stale EMA 와 새 obs_arr blend
            ###      하면 nonsense s 값 (예: 61, 69) 으로 mode flip 유발.
            ###      구조적으로 robust — staleness check 자체는 기존 EMA 가
            ###      처리, 추가는 ego_s 점프 한 번 만 detect.
            try:
                cur_es = float(getattr(self, 'ego_s', 0.0) or 0.0)
                prev_es = float(getattr(self, '_prev_ego_s_for_ema', cur_es))
                if abs(cur_es - prev_es) > 30.0:  # >30m 점프 (lap-wrap or sim reset)
                    self._obs_arr_ema = None
                self._prev_ego_s_for_ema = cur_es
            except Exception:
                pass
            ### HJ : end (S1-8)
            if obs_arr is None:
                # no obstacles this tick — drop stale EMA so a future
                # re-acquisition doesn't blend against 5-sec-old data.
                self._obs_arr_ema = None
            elif (self._obs_arr_ema is not None
                    and self._obs_arr_ema.shape == obs_arr.shape):
                a = float(self.obs_ema_alpha)
                blend = self._obs_arr_ema.copy()
                # carry active gate from new (fresh activation decisions)
                blend[:, :, 2] = obs_arr[:, :, 2]
                # EMA on (s, n) only where both prev and new are active
                active_mask = (obs_arr[:, :, 2] > 0.0) \
                              & (self._obs_arr_ema[:, :, 2] > 0.0)
                for col in (0, 1):
                    blend[:, :, col] = np.where(
                        active_mask,
                        a * obs_arr[:, :, col] + (1.0 - a) * self._obs_arr_ema[:, :, col],
                        obs_arr[:, :, col])
                self._obs_arr_ema = blend
                obs_arr = blend
            else:
                self._obs_arr_ema = obs_arr.copy()

        # ### HJ : Build initial_state + side decision per backend.
        if self.solver_backend == 'frenet_kin':
            # ψ_ref at s_cur (ref_psi[0]).
            psi_ref0 = float(ref_slice['ref_psi'][0])
            mu0 = float(self.car_yaw - psi_ref0)
            # wrap to [-pi, pi]
            mu0 = (mu0 + np.pi) % (2.0 * np.pi) - np.pi
            v0 = float(getattr(self, 'car_vx', 0.5))
            initial_state = np.array([float(ego_n), mu0, v0])
            # side decision (rule-based, external, now feasibility-aware)
            side_int, side_str, side_scores = self._decide_side(obs_arr, ref_slice)
            ### HJ : 2026-04-27 — log every side decision change with full
            ###      score breakdown so user can immediately see WHY decider
            ###      picked LEFT vs RIGHT (numbers, not guesses).
            if side_int != self._last_side_int:
                self.get_logger().warning(
                    '[mpc][%s] SIDE DECISION → %s ego_n=%.3f obs_n=%.3f '
                    'd_L=%.3f d_R=%.3f d_free_L=%.3f d_free_R=%.3f '
                    'can_pass_L=%s can_pass_R=%s reason=%s',
                    rospy.get_name(),
                    side_str.upper(), float(ego_n),
                    float(side_scores.get('n_o', 0.0)),
                    float(ref_slice['d_left_arr'][0]),
                    float(ref_slice['d_right_arr'][0]),
                    float(side_scores.get('d_free_L', 0.0)),
                    float(side_scores.get('d_free_R', 0.0)),
                    side_scores.get('can_pass_L'),
                    side_scores.get('can_pass_R'),
                    side_scores.get('reason'))
            ### HJ : end
            # ### HJ : v3 — bias_scale ramp REMOVED. Continuity cost in
            # solver handles tick-to-tick smoothness directly. Fixed 1.0.
            if side_int != self._last_side_int:
                self._ticks_since_flip = 0
            else:
                self._ticks_since_flip += 1
            bias_scale = 1.0
            self._last_bias_scale = bias_scale
            # TRAIL entry tick counter (for debug only; velocity ramp lives
            # in the solver via v_max cap when side==TRAIL)
            if side_int == SIDE_TRAIL and self._last_side_int != SIDE_TRAIL:
                self._trail_ticks_since_enter = 0
            elif side_int == SIDE_TRAIL:
                self._trail_ticks_since_enter += 1
            else:
                self._trail_ticks_since_enter = 0
            self._last_side_int = side_int
            self._last_side_str = side_str
            self._last_side_scores = side_scores
        else:
            initial_state = np.array([self.car_x, self.car_y, self.car_yaw])
            side_int = SIDE_CLEAR; side_str = 'n/a'; side_scores = {}
            bias_scale = 0.0

        # ### HJ : Phase X — MPC internal FSM (auto mode). In legacy
        # overtake/recovery/observe, this is a no-op that only bumps dwell.
        ego_v_now = float(getattr(self, 'car_vx', 0.0))
        min_obs_ds, ttc_min = self._obs_in_horizon_and_ttc(
            obs_arr, s_cur, ego_v_now)
        self._update_mpc_mode(ego_n, side_int, min_obs_ds, ttc_min)
        ### HJ : 2026-04-27 (R1) — proactive recovery cache on OT→NO_OBS.
        if getattr(self, '_just_transitioned_to_no_obs', False):
            self._just_transitioned_to_no_obs = False
            self._force_recovery_cache_build(s_cur, ego_n)
        ### HJ : end

        # ### HJ : 2026-04-24 — corridor sanity diagnostic. If ego is
        # physically outside the corridor at solve-time (dragged in Gazebo,
        # spawned over a wall, localisation glitch), the NLP's n_0=ego_n
        # initial constraint violates the hard bounds → slack absorbs but
        # first few stages may still be outside. Log so we can distinguish
        # "solver bad" vs "ego was spawned outside corridor". Throttled.
        try:
            wall_safe = float(getattr(self.solver, 'wall_safe', 0.15))
            d_L_ego = float(self.lifter._interp(s_cur, self.lifter.g_dleft))
            d_R_ego = float(self.lifter._interp(s_cur, self.lifter.g_dright))
            n_lo_ego = -(d_R_ego - wall_safe)
            n_hi_ego = +(d_L_ego - wall_safe)
            if ego_n < n_lo_ego or ego_n > n_hi_ego:
                rospy.logwarn_throttle(
                    0.5,
                    '[mpc][%s] ego OUTSIDE corridor @ s=%.2f: n=%.3f bounds=[%.3f, %.3f]'
                    ' — NLP will use slack; first-tick path may still violate walls.',
                    rospy.get_name(), s_cur, ego_n, n_lo_ego, n_hi_ego)
                self._last_ego_outside_corridor = True
            else:
                self._last_ego_outside_corridor = False
        except Exception:
            self._last_ego_outside_corridor = False

        # ### HJ : Phase X (refactored) — single NLP path. The solver always
        # runs; no quintic bypass, no skip. FSM mode just decides the weight
        # profile, blended smoothly via alpha_ramp. Obstacle presence flips
        # the target profile; the alpha ramp keeps the trajectory continuous.
        self._apply_mode_weights(self._alpha_ramp)
        # ### HJ : 2026-04-24 — variance-aware sigma inflation (AFTER mode
        # blend so variance has the final word on σ).
        self._apply_variance_inflation()
        # ### HJ : 2026-04-24 — adaptive recovery q_n near boost (AFTER
        # mode blend so we can read the blended q_n baseline from solver).
        self._apply_recovery_near_boost(ego_n)
        ### HJ : 2026-04-26 (A4-c) — graceful corridor entry ramp.
        ###      When ego starts near/past wall, ramp wall_buf weight from
        ###      0 at k=0 → 1 over K_entry steps. K_entry scales with how
        ###      deep ego is in the cushion / outside corridor.
        if self.solver_backend == 'frenet_kin':
            try:
                self._apply_wall_entry_ramp(ego_n, ref_slice)
            except Exception as _e:
                rospy.logwarn_throttle(
                    2.0, '[mpc][%s] wall_ramp apply failed: %s',
                    rospy.get_name(), _e)
        ### HJ : end

        ### HJ : 2026-04-28 Stage 2 Phase 2-1 — plan_picker + weight overlay.
        ###      side_decider 의 결정 + obs_in_horizon 으로 plan 선택.
        ###      plan 의 weight overlay 를 solver 에 push (앞의 _apply_*_weights
        ###      위에 덮어쓰기 → plan 이 최종 결정).
        try:
            obs_in_horizon = (self._mpc_mode == MPC_MODE_WITH_OBS)
            ### HJ : 2026-04-28 (a) — fully plan-based picker.
            ###      SideDecider 의 결정 무시. d_free_L/R / dv 정보만 활용.
            ###      모든 5 plan (LEFT/RIGHT/TRAIL/RACELINE) score 비교 → best.
            ###      사용자 명시: "MPC 가 전략 짠다" 의 진짜 구현.
            ###      sticky bonus + risk 강화 → plan 변경 빈도 제한.
            if obs_in_horizon:
                candidates = [PLAN_LEFT_PASS, PLAN_RIGHT_PASS, PLAN_TRAIL]
            else:
                candidates = [PLAN_RACELINE, PLAN_LEFT_PASS, PLAN_RIGHT_PASS]
            current_name = getattr(self, '_last_plan_name_logged', None)
            ### HJ : 2026-04-28 root-cause fix — physical-feasibility filter.
            ###      User feedback rejected both (a) hysteresis-only locks
            ###      ("그냥 고정하는거잖나") and (b) worst-case obs.n inflation
            ###      ("저정도 트래킹 에러는 갖고 있어야지"). Tracking noise on
            ###      d_free is normal; MPC w_obs + target_n cost handle it.
            ###
            ###      The actual root cause of LEFT↔RIGHT flips: scorer compares
            ###      both passes even when ego_n is committed (e.g. +0.30).
            ###      Cross-over RIGHT_PASS is physically infeasible from there
            ###      → flip is just score-noise, not strategy.
            ###
            ###      Drop infeasible candidates BEFORE scoring. Not a time-lock,
            ###      no state — purely structural. LEFT↔TRAIL, RACELINE↔LEFT,
            ###      etc. all stay open and respond to legitimate signals.
            candidates = _filter_feasible_plans(candidates, ego_n)
            ### HJ : end
            ### HJ : 2026-04-28 (a)+근원 (1)+(2) — horizon-aware d_free + EMA.
            ###      사용자 명시 근본 해결. snapshot d_free 진동의 근원 차단.
            ###      Horizon-aware: k=0..N min d_free → ego ≈ obs 시점도 안정.
            ###      EMA: 추가 smoothing (잔여 변동).
            ### HJ : 2026-04-28 — adaptive gap_lat from runtime topic state
            ###      (user feedback: 토픽 상태 기반 유동적). Pulls predicted
            ###      lateral velocity variance + rolling obs.n stddev. When
            ###      both report low, gap_lat barely grows (no over-margin);
            ###      when predictor uncertainty or recent obs.n variation is
            ###      high, ramp grows accordingly.
            #
            # Rolling history of current-tick obs.n (closest active obs) for
            # local stddev estimate. Buffer length ~20 ticks (~0.4s @ 50Hz).
            if not hasattr(self, '_obs_n_history'):
                from collections import deque
                self._obs_n_history = deque(maxlen=20)
            obs_n_std = 0.0
            try:
                if obs_arr is not None and obs_arr.size > 0:
                    active = obs_arr[:, 0, 2] > 0
                    if np.any(active):
                        self._obs_n_history.append(float(obs_arr[np.where(active)[0][0], 0, 1]))
                if len(self._obs_n_history) >= 4:
                    obs_n_std = float(np.std(np.array(self._obs_n_history)))
            except Exception:
                obs_n_std = 0.0
            ### HJ : 2026-04-28 — snapshot d_free only (no horizon, no EMA).
            ###      long_baseline 10min bag analysis: 10/12 collisions during
            ###      TRAIL plan. In 4 of those, snapshot d_free showed a clear
            ###      pass side (d_free_L=0.47, 0.49 etc.) but the smoothed
            ###      values used by the scorer were small enough that TRAIL's
            ###      long_term bonus dominated. The horizon-min and EMA chain
            ###      collapses confidence in legitimate pass margins. Trust
            ###      the snapshot — the MPC solver handles future obstacle
            ###      motion via its own w_obs cost.
            smoothed_scores = side_scores  # alias kept for downstream code
            score_hist = getattr(self, '_plan_score_history', None)
            best_plan, all_scores, new_hist = _pick_plan_scored(
                candidates, ego_n, smoothed_scores, obs_in_horizon,
                current_plan_name=current_name,
                score_history=score_hist,
                score_gap_threshold=0.15)  # was 0.40 — feasibility filter handles
                                            # main flip source, light gap is enough
            self._plan_score_history = new_hist
            ### HJ : 2026-04-28 — passing-freeze + 5-tick dwell removed.
            ###      User: "두가지 방법 말고(저건 그냥 고정하는거잖나)".
            ###      Plan stability now comes from feasibility filter +
            ###      smooth quadratic risk (no cliff) + score EMA + light gap.
            ###      No time-locks.
            self._last_plan = best_plan
            self._last_plan_scores = all_scores
            self._apply_plan_weights(self._last_plan)
            ### HJ : end (a)
            ### HJ : 2026-04-28 (trailing 강화 — 사용자 정정 정확 구현):
            ###      TRAIL plan 시 ref_v 를 obs_v - margin 으로 override.
            ###      "차 뒤 + 일정 거리/속도, 빈틈 노림" 의 'target_v' 명시화.
            ###      Solver progress cost (gamma) 가 자연스럽게 ego_v → target_v.
            ###      단순 vmax cap 보다 active trailing 효과.
            try:
                if self._last_plan is not None and self._last_plan.get('name') == 'trail':
                    obs_vs = float(side_scores.get('v_obs', 0.0))
                    target_v = max(obs_vs - 0.3, 0.5)  # margin 0.3 m/s, floor 0.5
                    ref_slice['ref_v'] = np.full_like(ref_slice['ref_v'], target_v)
            except Exception as _e:
                rospy.logwarn_throttle(2.0,
                    '[mpc][%s] TRAIL ref_v override failed: %s',
                    rospy.get_name(), _e)
            ### HJ : end (trailing target_v)
            self._apply_plan_weights(self._last_plan)
            ### HJ : 2026-04-28 — plan-aligned side_int tried (long_side_align).
            ###      Forcing side_int from plan name made things worse (7→10
            ###      events). When plan_picker chose RIGHT_PASS while ego was
            ###      slightly on the LEFT side of the obstacle (ego_n=+0.12,
            ###      under filter threshold 0.15), the forced side_int=RIGHT
            ###      drove the solver across the obstacle bubble — collisions
            ###      followed. Reverted; side_decider's side_int stays
            ###      authoritative for the solver.
            ### HJ : 2026-04-28 Phase 2-2 — target_n profile push.
            ###      Plan 의 'where to go' 를 명시적 cost 로 강제. obs_n =
            ###      가장 가까운 (s 기준) obstacle 의 n. None 이면 fallback.
            try:
                obs_n = None
                if obs_arr is not None and obs_arr.size > 0:
                    # 가장 가까운 (s_close) obstacle 의 n0 추출
                    active_mask = (obs_arr[:, 0, 2] > 0)
                    if active_mask.any():
                        # k=0 의 n 평균 (multi-obs 시 closest 사용 가능하나 단순)
                        active_idx = np.where(active_mask)[0][0]
                        obs_n = float(obs_arr[active_idx, 0, 1])
                target_profile = _make_target_n_profile(
                    self._last_plan, ego_n, obs_n, self.N)
                self.solver.set_n_target_profile(target_profile)
                ### HJ : 2026-04-28 — target_n profile blend tried (option C)
                ###      and removed. 5x60s aggregate trial_c showed 12
                ###      collisions vs trial_b's 4 — blending the target
                ###      across plan transitions slowed MPC's lateral
                ###      response just like the alpha=0.10 weight ramp did.
                ###      Smoothness must come from solver's w_cont=300 +
                ###      fewer transitions (sticky fix + filter), not from
                ###      blending the planner's intent toward the previous
                ###      tick's intent.
            except Exception as _e:
                rospy.logwarn_throttle(2.0,
                    '[mpc][%s] target_n profile push failed: %s',
                    rospy.get_name(), _e)
            ### HJ : end (Phase 2-2)
            # log plan transition
            if getattr(self, '_last_plan_name_logged', None) != self._last_plan['name']:
                self.get_logger().info(
                    '[mpc][%s] PLAN → %s (side=%s obs_in_h=%s)',
                    rospy.get_name(),
                    self._last_plan['name'].upper(), side_str, obs_in_horizon)
                self._last_plan_name_logged = self._last_plan['name']
        except Exception as _e:
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] plan_picker apply failed: %s',
                rospy.get_name(), _e)
            self._last_plan = None
        ### HJ : end (Stage 2 Phase 2-1)

        warm_used = int(bool(getattr(self.solver, 'warm', False)))

        ### HJ : 2026-04-27 — cache HIJACKS NLP. Once a recovery cache is
        ###      committed, it is sampled EVERY tick (including when NLP
        ###      would otherwise succeed) until invalidation. Per user:
        ###      "리커버리 한번 되면 그거 끝까지 따라가고 사이에 mpc 풀려도
        ###       장애물 생기는거 아니면 그거 따라가게". Solver still runs
        ###      so warm-start stays warm and we have a fresh "intent"
        ###      even if we don't publish it — keeps tier-0 ready when
        ###      cache invalidates.
        ###
        ###      Cache invalidation when obstacle re-enters horizon is
        ###      handled inside _sample_recovery_cache (mode flip to
        ###      WITH_OBS triggers invalidate). Cache is intentionally
        ###      allowed when obstacle is far (NO_OBS mode + ego far from
        ###      GB) because that's the smooth-return-to-GB use case the
        ###      cache was built for.
        ###
        ### HJ : 2026-04-29 — Cache pre-NLP hijack REMOVED.
        # Previous behaviour: when obs was outside horizon, the cache
        # could be sampled BEFORE the NLP call and published as the MPC
        # output, skipping the NLP entirely. With cache build firing on
        # every R1 OT→NO_OBS transition (and on every NLP-fail quintic),
        # the cache could repeatedly hijack the publish path so that
        # only "RECOVERY_CACHED" trajectories appeared in RViz, even
        # when the NLP would have produced a fresh, more accurate
        # trajectory. User report (2026-04-29): "obstacle in horizon
        # 아니어도 지금 MPC recovery 만 나오고 있어".
        # Fix: always run the NLP first. Cache is now exclusively a
        # fallback (used in the post-fail ladder only — see below).
        _obs_in_h_pre_nlp = bool(
            obs_arr is not None
            and obs_arr.size > 0
            and (obs_arr[:, :, 2] > 0).any())
        ### HJ : end

        t0 = time.time()
        if self.solver_backend == 'frenet_kin':
            speed, steering, trajectory, success = self.solver.solve(
                initial_state, ref_slice, obstacles=obs_arr, side=side_int,
                bias_scale=bias_scale)
        else:
            speed, steering, trajectory, success = self.solver.solve(
                initial_state, ref_slice, obstacles=obs_arr)
        solve_ms = (time.time() - t0) * 1000.0
        self.pub_timing.publish(Float32(data=solve_ms))

        u_sol = getattr(self.solver, 'last_u_sol', None)
        ipopt_status = getattr(self.solver, 'last_return_status', '-')
        iter_count = getattr(self.solver, 'last_iter_count', -1)
        slack_max = float(getattr(self.solver, 'last_slack_max', 0.0))

        # ### HJ : frenet_kin returns trajectory in (s, n, mu, v). The rest
        # of the pipeline (_publish_outputs, _publish_debug_markers) expects
        # (x, y, psi). Lift before handing off. Also stash the original
        # frenet trajectory for the tick_json payload (margins etc.) AND for
        # the downstream Wpnt filler (`fill_wpnt_from_s`, 3D-safe) — do NOT
        # reset to None on failure. Failure paths (tier1/2/3) assign their
        # own synthetic frenet trajectory below so `_publish_outputs` never
        # falls back to the 2D xy→s round-trip that aliases overpass layers.
        if (self.solver_backend == 'frenet_kin' and success
                and trajectory is not None):
            self._last_frenet_traj = np.array(trajectory, copy=True)
            trajectory = self._lift_frenet_to_xy(trajectory, ref_slice)
        else:
            # Keep the last successful frenet traj so tier1 can re-publish it
            # via fill_wpnt_from_s. Tiers 2/3 overwrite this with their own
            # synthetic (s, n) array before calling _publish_outputs.
            pass

        # ### HJ : Phase 4.3 — 4-tier fallback. Even at tier 3 we still
        # publish a sane Wpnt[] so controller has zero-gap input.
        # ### HJ : 2026-04-29 — single-use cache follow-through.
        # Once a recovery cache is in_use (committed by R1 OT→NO_OBS
        # pre-build or by tier 2 quintic), we follow it to the end.
        # Even if the NLP starts succeeding mid-follow, we keep
        # publishing the cache so the controller doesn't suddenly
        # switch from a smooth recovery curve to a fresh NLP path.
        # The cache invalidates itself when:
        #   - mode flips to WITH_OBS (obstacle entered horizon)
        #   - ego_n converged (raceline reached)
        #   - ego past s_end (cache window exhausted)
        # so this branch only fires while the cache is genuinely active.
        if self._recovery_cache_in_use:
            cached_xy, cached_sn = self._sample_recovery_cache(s_cur, ego_n)
            if cached_xy is not None:
                self._handle_tier2_geometric(
                    cached_xy, cached_sn, ego_n,
                    'cache_follow', solve_ms, obs_tag + ':cached')
                self._debug_log(
                    tier=2, status='RECOVERY_CACHED',
                    ipopt_status='cache_follow',
                    iter_count=0, solve_ms=solve_ms, slack_max=0.0,
                    trajectory=cached_xy, u_sol=None, obs_arr=obs_arr,
                    obs_tag=obs_tag + ':cached', ref_slice=ref_slice,
                    initial_state=initial_state, ego_s=s_cur, ego_n=ego_n,
                    warm_used=warm_used,
                )
                return
            # _sample_recovery_cache returned None (one of the
            # invalidation conditions fired). in_use was already cleared
            # inside _invalidate_recovery_cache; fall through to publish
            # the fresh NLP / fallback below.

        if success and trajectory is not None:
            self._handle_tier0_success(trajectory, s_cur, solve_ms, obs_tag,
                                       speed, steering)
            self._debug_log(
                tier=0, status='OK', ipopt_status=ipopt_status,
                iter_count=iter_count, solve_ms=solve_ms, slack_max=slack_max,
                trajectory=trajectory, u_sol=u_sol, obs_arr=obs_arr,
                obs_tag=obs_tag, ref_slice=ref_slice,
                initial_state=initial_state, ego_s=s_cur, ego_n=ego_n,
                warm_used=warm_used, speed0=speed, steer0=steering,
            )
            return

        # NLP failed — move down the ladder.
        self._fail_streak += 1
        status = ipopt_status
        self.solver.reset_warm_start()

        ### HJ : 2026-04-28 — "MPC must always publish" directive.
        # Previous version (S1-5) capped HOLD_LAST at 3 ticks and dropped
        # to EMERGENCY_NO_TRAJ (publish nothing) when the cap was exceeded
        # with an obstacle in horizon. That was the source of the
        # multi-second publish blackouts the user observed in bag
        # 2026-04-28-15-08-36 (3.7s gap, SM kept using a stale OT path
        # from a previous lap → local_wpnts jumping by ~15m). User
        # mandate (2026-04-28): MPC trajectory must come out every tick
        # regardless of state — global tracking, overtake, recovery —
        # because the controller / SM treat publish absence as brake and
        # produce far worse behaviour than a slightly-stale path.
        # New ladder:
        #   tier 0 — NLP success (above)
        #   tier 1 — HOLD_LAST: republish last good trajectory while NLP
        #            recovers. No fail_streak cap. Always publish if a
        #            last_good_traj exists.
        #   tier 2 — recovery cache (still gated by obs_in_horizon to
        #            avoid driving an OT-shaped cached path through a new
        #            obstacle).
        #   tier 3 — quintic fallback (recovery curve).
        #   tier 4 — raceline slice (last resort, always available).
        # EMERGENCY_NO_TRAJ removed — every fail must reach a publish.
        obs_in_horizon_strict = bool(
            obs_arr is not None
            and obs_arr.size > 0
            and (obs_arr[:, :, 2] > 0).any())

        ### HJ : 2026-04-29 — Fallback ladder REORDERED.
        # Previous order put HOLD_LAST first. That meant on every NLP
        # fail the system republished the last NLP-success trajectory,
        # which is typically a raceline-shaped path (because the prior
        # tick the obstacle was farther / NLP solved easily). The
        # painter then overwrote vx_mps with vel-planner GB raceline
        # speeds. Net effect: when the obstacle got close enough to
        # break the NLP, the car kept driving the previous raceline
        # path at full GB speed → drove straight into the obstacle.
        # User report (2026-04-29): "냅다 GB 경로 나오던데".
        #
        # New order — recovery-first, hold-last as last-ditch:
        #   tier 2 cache       (committed recovery path; obs-aware self-invalidate)
        #   tier 2 quintic     (fresh ego_n→0 recovery curve, obs-aware skip)
        #   tier 3 short quintic (aggressive convergence)
        #   tier 3 raceline slice (last resort; controller might jump)
        #   tier 1 HOLD_LAST   (only if every fresh option above failed)
        #
        # The HOLD_LAST now exists to keep the publish stream alive in
        # the rare case where every recovery primitive raised, not as
        # the default fail handler.

        # Tier 2: cache (committed recovery path)
        cached_xy, cached_sn = self._sample_recovery_cache(s_cur, ego_n)
        if cached_xy is not None:
            self._handle_tier2_geometric(
                cached_xy, cached_sn, ego_n, status, solve_ms,
                obs_tag + ':cached')
            self._debug_log(
                tier=2, status='RECOVERY_CACHED', ipopt_status=status,
                iter_count=iter_count, solve_ms=solve_ms, slack_max=slack_max,
                trajectory=cached_xy, u_sol=None, obs_arr=obs_arr,
                obs_tag=obs_tag + ':cached', ref_slice=ref_slice,
                initial_state=initial_state, ego_s=s_cur, ego_n=ego_n,
                warm_used=warm_used,
            )
            return

        # Tier 2: quintic recovery (fresh path). Skip when obstacle is
        # in horizon — quintic targets ego_n→0 (raceline) and would
        # drive through the obstacle. Fall through to short-quintic /
        # raceline-slice / HOLD_LAST in that case.
        if obs_in_horizon_strict:
            traj_fb, frenet_fb = None, None
        else:
            traj_fb, frenet_fb = self._try_quintic_fallback(
                s_cur, ego_n, delta_s=self._quintic_delta_s, clip_walls=True)
        if traj_fb is not None:
            self._cache_recovery_path(traj_fb, frenet_fb)
            self._handle_tier2_geometric(
                traj_fb, frenet_fb, ego_n, status, solve_ms, obs_tag)
            self._debug_log(
                tier=2, status='GEOMETRIC_FALLBACK', ipopt_status=status,
                iter_count=iter_count, solve_ms=solve_ms, slack_max=slack_max,
                trajectory=traj_fb, u_sol=None, obs_arr=obs_arr,
                obs_tag=obs_tag, ref_slice=ref_slice,
                initial_state=initial_state, ego_s=s_cur, ego_n=ego_n,
                warm_used=warm_used,
            )
            return

        # Tier 3: short-Δs quintic (aggressive convergence)
        traj_short, frenet_short = self._try_quintic_fallback(
            s_cur, ego_n, delta_s=max(self._quintic_delta_s * 0.5, 3.0),
            clip_walls=True)
        if traj_short is not None:
            self._handle_tier3_convergence_quintic(
                traj_short, frenet_short, ego_n, status, solve_ms, obs_tag)
            self._debug_log(
                tier=3, status='CONVERGENCE_QUINTIC', ipopt_status=status,
                iter_count=iter_count, solve_ms=solve_ms, slack_max=slack_max,
                trajectory=traj_short, u_sol=None, obs_arr=obs_arr,
                obs_tag=obs_tag, ref_slice=ref_slice,
                initial_state=initial_state, ego_s=s_cur, ego_n=ego_n,
                warm_used=warm_used,
            )
            return

        # ### HJ : 2026-04-29 — HOLD_LAST removed entirely.
        # User directive: "HOLD_LAST 없애고, 항상 무조건 valid 한 경로가
        # 나오게". Stale-path republishing was producing GB-shaped paths
        # right when an obstacle was bearing down (the prior NLP success
        # had been raceline-tracking before the obstacle came close), so
        # the car drove the old path straight into the new obstacle.
        # The recovery primitives above (cache, quintic, short quintic)
        # plus the raceline-slice last resort all generate fresh,
        # geometrically-valid trajectories (finite x/y, bounded kappa,
        # n=0 raceline path). Publish is therefore always guaranteed
        # without needing to fall back to a stale snapshot.

        # Tier 3-final: raceline slice. Fresh raceline path from s_cur
        # for N+1 waypoints. Always succeeds (depends only on the global
        # raceline lifter, no NLP / quintic dependence).
        traj_last = self._handle_tier3_raceline(
            s_cur, status, solve_ms, obs_tag)
        # ### HJ : v3b — feed the raceline-slice trajectory to debug_log so
        # the RViz marker keeps showing a trajectory line even in absolute
        # worst-case (both quintics raised).
        self._debug_log(
            tier=3, status='RACELINE_SLICE', ipopt_status=status,
            iter_count=iter_count, solve_ms=solve_ms, slack_max=slack_max,
            trajectory=traj_last, u_sol=None, obs_arr=obs_arr, obs_tag=obs_tag,
            ref_slice=ref_slice, initial_state=initial_state,
            ego_s=s_cur, ego_n=ego_n, warm_used=warm_used,
        )

    # ---------------------------------------------------------------- debug_log
    def _spawn_debug_logger(self, reason='runtime'):
        """(Re)create the DebugLogger → new run directory + fresh files."""
        if DebugLogger is None or self._dbg_cfg is None:
            return
        if self._debug_logger is not None:
            try:
                self._debug_logger.close()
            except Exception:
                pass
            self._debug_logger = None
        try:
            self._debug_logger = DebugLogger(
                dict(self._dbg_cfg, enable=True),
                self._dbg_params_snapshot,
                repo_hint_path=os.path.dirname(_dbg_dir))
            self._debug_logger.log_event('run_start', {
                'reason': reason,
                'state': self.state,
                'cfg': {k: v for k, v in self._dbg_cfg.items()
                        if k not in ('node_name',)},
            })
            self.get_logger().info('[mpc][%s] debug_log run started (%s): %s',
                          rospy.get_name(), reason,
                          getattr(self._debug_logger, '_run_dir', '?'))
        except Exception as e:
            self.get_logger().warning('[mpc][%s] debug_log init failed: %s',
                          rospy.get_name(), e)
            self._debug_logger = None

    def _current_weights_snapshot(self):
        """Dict of every tunable the logger records per tick. Mirrors rqt.
        Backend-aware: frenet_kin exposes a different set of knobs than
        the legacy frenet_d / xy backends."""
        if self.solver_backend == 'frenet_kin':
            s = self.solver
            return {
                'backend':        'frenet_kin',
                'q_n':            float(s.q_n),
                'q_n_ramp':       float(getattr(s, 'q_n_ramp', 0.0)),
                'gamma_progress': float(s.gamma),
                'r_a':            float(s.r_a),
                'r_steer_reg':    float(s.r_reg),
                'w_slack':        float(s.w_slack),
                'v_min':          float(s.v_min),
                'v_max':          float(s.v_max),
                'a_min':          float(s.a_min),
                'a_max':          float(s.a_max),
                'delta_max':      float(s.delta_max),
                'delta_rate_max': float(s.delta_rate_max),
                'mu_max':         float(s.mu_max),
                'inflation':      float(s.inflation),
                'wall_safe':      float(s.wall_safe),
                'gap_lat':        float(s.gap_lat),
                'gap_long':       float(s.gap_long),
                'w_obs':          float(s.w_obs),
                'sigma_s_obs':    float(s.sigma_s_obs),
                'sigma_n_obs':    float(s.sigma_n_obs),
                'w_side_bias':    float(s.w_side_bias),
                'w_wall_buf':     float(s.w_wall_buf),
                'wall_buf':       float(s.wall_buf),
                # ### HJ : v3 — C^1 / continuity / terminal weights
                'r_dd':           float(s.r_dd),
                'r_dd_rate':      float(s.r_dd_rate),
                'w_cont':         float(s.w_cont),
                'q_n_term':       float(s.q_n_term),
                'q_v_term':       float(s.q_v_term),
                # feasibility gate + obstacle filter (node-side)
                'min_pass_margin':    float(self.side_decider.min_pass_margin),
                'trail_entry_ticks':  int(self.side_decider.trail_entry_ticks),
                'obs_ema_alpha':      float(self.obs_ema_alpha),
                'trail_vel_ramp_ticks': int(self.trail_vel_ramp_ticks),
                'ipopt_max_iter': int(s.ipopt_max_iter),
            }
        # Legacy frenet_d / xy (kept for A/B rollback).
        return {
            'backend':      self.solver_backend,
            'w_contour':    float(self.solver.w_contour),
            'w_lag':        float(self.solver.w_lag),
            'w_velocity':   float(self.solver.w_velocity),
            'v_bias_max':   float(self.solver.v_bias_max),
            'w_dv':         float(self.solver.w_dv),
            'w_dsteering':  float(self.solver.w_dsteering),
            'w_slack':      float(self.solver.w_slack),
            'contour_ramp_start': float(self.solver.contour_ramp_start),
            'lag_ramp_start':     float(self.solver.lag_ramp_start),
            'max_speed':    float(self.solver.v_max),
            'min_speed':    float(self.solver.v_min),
            'max_steering': float(self.solver.theta_max),
            'boundary_inflation': float(self.solver.inflation),
            'w_obstacle':     float(self.w_obstacle),
            'obstacle_sigma': float(self.solver.obstacle_sigma),
            'w_wall':         float(self.w_wall),
            'wall_safe':      float(self.wall_safe),
            'ipopt_max_iter': int(self.solver.ipopt_max_iter),
        }

    def _refresh_debug_runtime(self):
        """### HJ : Runtime rosparam polling — lets me toggle logging on/off
        and rotate the run directory without restarting the node. Called at
        the top of every _plan_loop tick; ~1 μs when params haven't changed."""
        if DebugLogger is None or self._dbg_cfg is None:
            return
        try:
            want_enable = bool(self._get_param_or_default(
                '~debug_log_enable', self._dbg_cfg['enable']))
            want_reset = bool(self._get_param_or_default('~debug_log_reset_run', False))
        except Exception:
            return
        currently_on = self._debug_logger is not None

        if want_reset:
            rospy.set_param('~debug_log_reset_run', False)
            self._spawn_debug_logger(reason='reset_run')
            return

        if want_enable and not currently_on:
            self._dbg_cfg['enable'] = True
            self._spawn_debug_logger(reason='rosparam_enable')
        elif (not want_enable) and currently_on:
            self._dbg_cfg['enable'] = False
            try:
                self._debug_logger.close()
            except Exception:
                pass
            self._debug_logger = None
            self.get_logger().info('[mpc][%s] debug_log paused (rosparam disable)',
                          rospy.get_name())

    def _debug_log(self, **fields):
        """### HJ : Single entry point into the per-tick logger. No-op when
        the logger is disabled or failed to init. Exceptions are swallowed —
        logging must never crash the planner.

        Even when the file logger is disabled, we still publish the live
        `~debug/tick_json` + `~debug/markers` so Claude-side `rostopic echo`
        monitoring (CLAUDE.md HJ mode) keeps working.
        """
        # Live topics first (independent of file-logging state).
        try:
            self._publish_tick_live(fields)
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                5.0, '[mpc][%s] tick publish error: %s',
                rospy.get_name(), e)

        if self._debug_logger is None:
            return
        try:
            t_ros = self.get_clock().now().to_msg().to_sec()
            rec = {
                't_ros': t_ros,
                'state': self.state,
                'car_x': self.car_x, 'car_y': self.car_y,
                'car_yaw': self.car_yaw, 'car_vx': self.car_vx,
                'car_vy': float(getattr(self, 'car_vy', 0.0)),
                'fail_streak': self._fail_streak,
                'weights': self._current_weights_snapshot(),
            }
            rec.update(fields)
            self._debug_logger.log_tick(rec)
            self._dbg_tick_counter += 1
            # Publish live summary every N ticks (cheap, file is already
            # written — we just echo it to ROS as well).
            if (self._dbg_tick_counter %
                    max(self._dbg_cfg.get('summary_every', 10), 1)) == 0:
                summary = self._debug_logger.get_live_summary()
                if summary is not None:
                    import json as _json
                    self.pub_debug_summary.publish(
                        String(data=_json.dumps(summary)))
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                5.0, '[mpc][%s] debug_log error: %s',
                rospy.get_name(), e)

    # ---- Tier 0 ------------------------------------------------------------
    def _handle_tier0_success(self, trajectory, s_cur, solve_ms, obs_tag,
                              speed, steering):
        if self._last_status and self._last_status != 'OK':
            self.get_logger().info('[mpc][%s] recovered after tier=%s  streak=%d',
                          rospy.get_name(), self._last_status, self._fail_streak)
        self._fail_streak = 0
        self._viz_tier = 0
        self._viz_status = 'OK'
        self._viz_pass = int(getattr(self.solver, 'last_pass', 1) or 1)
        ### HJ : 2026-04-27 — clear recovery cache when ego converged & NLP works.
        ###      NLP success in NO_OBS with ego_n small means recovery is over.
        if (self._mpc_mode == MPC_MODE_NO_OBS
                and self._recovery_cache_sn is not None):
            try:
                ego_n = float(getattr(self, 'ego_n', 0.0) or 0.0)
                if abs(ego_n) < self._recovery_cache_exit_n:
                    self._invalidate_recovery_cache('tier0+converged')
            except Exception:
                pass
        ### HJ : end

        self._publish_outputs(trajectory)
        self._last_good_traj = np.array(trajectory, copy=True)
        # ### HJ : v3b — cache raw frenet too so HOLD_LAST can re-publish
        # via fill_wpnt_from_s (3D-safe) instead of fill_wpnt (2D xy lookup).
        if self._last_frenet_traj is not None:
            self._last_good_frenet_traj = np.array(
                self._last_frenet_traj, copy=True)
        u_sol = getattr(self.solver, 'last_u_sol', None)
        self._last_good_u = np.array(u_sol, copy=True) if u_sol is not None else None
        self._last_good_s = s_cur
        self._last_good_time = self.get_clock().now().to_msg()

        self._publish_status('OK obs=%s' % obs_tag)
        self._last_status = 'OK'
        rospy.loginfo_throttle(
            2.0,
            '[mpc][%s] state=%s solve=%.1fms v0=%.2f steer=%.3f obs=%s',
            rospy.get_name(), self.state, solve_ms, speed, steering, obs_tag,
        )

    # ---- Tier 1 ------------------------------------------------------------
    def _handle_tier1_hold_last(self, ipopt_status, solve_ms, obs_tag):
        """Re-publish last good trajectory. No s-shift yet (Phase 4 minimal);
        controller lookahead handles small ego progress within one tick."""
        self._viz_tier = 1
        self._viz_status = 'HOLD_LAST'
        # ### HJ : v3b — restore the cached raw frenet trajectory so
        # _publish_outputs takes the s-direct path (fill_wpnt_from_s) and
        # bypasses xy→s projection that would alias overpass floors.
        self._last_frenet_traj = (
            np.array(self._last_good_frenet_traj, copy=True)
            if self._last_good_frenet_traj is not None else None)
        self._publish_outputs(self._last_good_traj,
                              u_sol_override=self._last_good_u)
        self._publish_status('HOLD_LAST streak=%d obs=%s' %
                             (self._fail_streak, obs_tag))
        self._last_status = 'HOLD_LAST'
        rospy.logwarn_throttle(
            0.5,
            '[mpc fallback tier1][%s] HOLD_LAST streak=%d ipopt=%s solve=%.1fms obs=%s',
            rospy.get_name(), self._fail_streak, ipopt_status, solve_ms, obs_tag,
        )

    ### HJ : 2026-04-28 (S1-5) — emergency no-publish.
    def _handle_emergency_no_traj(self, ipopt_status, solve_ms, obs_tag):
        """Cap 초과 + obstacle in horizon 시 publish 생략.

        사용자 룰 1: 장애물 옆 recovery/GB curve 금지. Stale HOLD_LAST,
        raceline_slice, geometric quintic 모두 obstacle 충돌 위험. publish
        안 함이 가장 안전. Controller 의 latest_threshold (200ms) 가 자체
        brake 트리거. Status 토픽만 발행해서 SM 등이 신호 받음.
        """
        self._viz_tier = -1
        self._viz_status = 'EMERGENCY_NO_TRAJ'
        self._publish_status('EMERGENCY_NO_TRAJ streak=%d obs=%s' %
                             (self._fail_streak, obs_tag))
        self._last_status = 'EMERGENCY'
        rospy.logerr_throttle(
            0.5,
            '[mpc][%s] EMERGENCY (no traj) streak=%d ipopt=%s solve=%.1fms obs=%s',
            rospy.get_name(), self._fail_streak, ipopt_status, solve_ms, obs_tag,
        )
    ### HJ : end (S1-5)

    ### HJ : 2026-04-27 — Recovery path cache (commit-once).
    def _cache_recovery_path(self, xy_traj, sn_traj):
        """Cache a tier-2 recovery path. Subsequent fallback ticks sample
        from this cache rather than recompute, so the goalpost stays
        s-fixed instead of moving with ego."""
        if xy_traj is None or sn_traj is None:
            return
        self._recovery_cache_xy = np.array(xy_traj, copy=True)
        self._recovery_cache_sn = np.array(sn_traj, copy=True)
        self._recovery_cache_s_start = float(sn_traj[0, 0])
        self._recovery_cache_s_end = float(sn_traj[-1, 0])
        self._recovery_cache_committed_at = self.get_clock().now().to_msg()
        # 2026-04-29 single-use: any commit starts a new follow window.
        self._recovery_cache_in_use = True
        rospy.loginfo_throttle(
            1.0,
            '[mpc][%s] recovery path CACHED: s=[%.2f, %.2f] L=%.2f',
            rospy.get_name(),
            self._recovery_cache_s_start, self._recovery_cache_s_end,
            self._recovery_cache_s_end - self._recovery_cache_s_start)

    ### HJ : 2026-04-27 (R1) — proactive recovery cache build on OT→NO_OBS.
    ###      Called from _plan_loop right after _update_mpc_mode detects
    ###      WITH_OBS → NO_OBS transition. Builds + caches a recovery path
    ###      synchronously so the rest of THIS tick's cache-hijack picks it
    ###      up. Skips work if cache already present.
    def _force_recovery_cache_build(self, s_cur, ego_n):
        if self._recovery_cache_sn is not None:
            self.get_logger().warning(
                '[mpc][%s] R1 OT→NO_OBS: recovery cache already exists '
                '(s=[%.2f, %.2f]), skipping pre-build',
                rospy.get_name(),
                self._recovery_cache_s_start,
                self._recovery_cache_s_end)
            return
        if self.solver_backend != 'frenet_kin':
            return
        try:
            n_samples = self.N + 1
            traj_fb, frenet_fb = self._try_quintic_fallback(
                s_cur, ego_n, delta_s=self._quintic_delta_s,
                n_samples=int(n_samples), clip_walls=True)
            if traj_fb is not None:
                self._cache_recovery_path(traj_fb, frenet_fb)
                self.get_logger().warning(
                    '[mpc][%s] R1 OT→NO_OBS: recovery cache PRE-BUILT '
                    '(ego_n=%.3f, s_start=%.2f, s_end=%.2f, L=%.2f)',
                    rospy.get_name(),
                    float(ego_n),
                    self._recovery_cache_s_start,
                    self._recovery_cache_s_end,
                    self._recovery_cache_s_end - self._recovery_cache_s_start)
            else:
                self.get_logger().warning(
                    '[mpc][%s] R1 OT→NO_OBS: cache pre-build FAILED '
                    '(build_recovery_path returned None at ego_n=%.3f)',
                    rospy.get_name(), float(ego_n))
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, '[mpc][%s] proactive recovery build failed: %s',
                rospy.get_name(), e)
    ### HJ : end

    def _invalidate_recovery_cache(self, reason='unspecified'):
        if self._recovery_cache_sn is None:
            self._recovery_cache_in_use = False
            return
        rospy.loginfo_throttle(
            1.0, '[mpc][%s] recovery cache cleared (%s)',
            rospy.get_name(), reason)
        self._recovery_cache_xy = None
        self._recovery_cache_sn = None
        self._recovery_cache_s_start = None
        self._recovery_cache_s_end = None
        self._recovery_cache_committed_at = None
        self._recovery_cache_in_use = False

    def _sample_recovery_cache(self, s_cur, ego_n):
        """If cache valid, sample at current s_cur. Returns (xy, sn) of
        shape (N+1, ...) or (None, None) if cache should be ignored.
        """
        if self._recovery_cache_sn is None:
            return None, None
        # ### HJ : 2026-04-29 — TTL removed. Cache is single-use:
        # follow-once until window end / convergence / obstacle entry.
        # Invalidate if mode flipped to WITH_OBS (obstacle near)
        if self._mpc_mode == MPC_MODE_WITH_OBS:
            self._invalidate_recovery_cache('mode→WITH_OBS')
            return None, None
        # Invalidate if ego converged (raceline reached)
        if abs(float(ego_n)) < self._recovery_cache_exit_n:
            self._invalidate_recovery_cache('converged')
            return None, None
        # Invalidate if ego went past cache end (path exhausted)
        if s_cur > self._recovery_cache_s_end - 0.3:
            self._invalidate_recovery_cache('s past end')
            return None, None
        # Invalidate if ego went backward unreasonably
        if s_cur < self._recovery_cache_s_start - 1.0:
            self._invalidate_recovery_cache('s underflow')
            return None, None
        ### HJ : 2026-04-27 v2 — fixed total output length ~15m.
        ###      User: 21m+ too long. Cap at K_target_total wpnts; cache
        ###      portion takes priority, GB tail fills any remainder.
        s_arr = self._recovery_cache_sn[:, 0]
        idx = int(np.searchsorted(s_arr, s_cur))
        idx = max(0, min(idx, len(s_arr) - 1))
        cached_xy = self._recovery_cache_xy
        cached_sn = self._recovery_cache_sn
        wpnt_dist = float(self.g_s[1] - self.g_s[0]) if (
            self.g_s is not None and len(self.g_s) > 1) else 0.10
        K_target_total = int(self._get_param_or_default(
            '~recovery_cache_total_n', 150))   # 150 × 0.1m = 15m default
        K_remaining = min(len(s_arr) - idx, K_target_total)
        K_gb_tail = max(0, K_target_total - K_remaining)
        K_total = K_remaining + K_gb_tail
        out_xy = np.zeros((K_total, cached_xy.shape[1]), dtype=np.float64)
        out_sn = np.zeros((K_total, cached_sn.shape[1]), dtype=np.float64)
        # 1) cached portion
        for k in range(K_remaining):
            out_xy[k] = cached_xy[idx + k]
            out_sn[k] = cached_sn[idx + k]
        # 2) GB blend after cache s_end (n=0 raceline, raceline tangent)
        s_end = float(self._recovery_cache_s_end)
        for k in range(K_gb_tail):
            s_w = (s_end + (k + 1) * wpnt_dist) % self.track_length
            try:
                x, y = self.lifter.sn_to_xy(s_w, 0.0)
                psi = self.lifter._interp_psi(s_w)
                z = float(self.lifter._interp(s_w, self.lifter.g_z))
            except Exception:
                x = y = psi = z = 0.0
            j = K_remaining + k
            # xy_traj layout: [x, y, psi, s, z]
            out_xy[j, 0] = x
            out_xy[j, 1] = y
            out_xy[j, 2] = psi
            if cached_xy.shape[1] > 3:
                out_xy[j, 3] = s_w
            if cached_xy.shape[1] > 4:
                out_xy[j, 4] = z
            # sn_traj layout: [s, n, mu, v]
            out_sn[j, 0] = s_w
            out_sn[j, 1] = 0.0
            if cached_sn.shape[1] > 2:
                out_sn[j, 2] = 0.0
            if cached_sn.shape[1] > 3:
                # placeholder velocity = lifter's GB vx at this s
                try:
                    out_sn[j, 3] = float(self.lifter._interp(s_w, self.lifter.g_vx))
                except Exception:
                    out_sn[j, 3] = 0.0
        return out_xy, out_sn
        ### HJ : end
    ### HJ : end

    # ---- Tier 2 ------------------------------------------------------------
    def _try_quintic_fallback(self, s_cur, ego_n, delta_s=None,
                               n_samples=None, clip_walls=False):
        """### HJ : v3b — recovery-style "ego_n → 0" smooth return primitive.

        Uses (s_cur, ego_n) from /odom_frenet directly so the 3D overpass
        layer is preserved. Previous implementation called
        `lifter.project_xy_to_sn(car_x, car_y)` which is 2D nearest and
        aliases overpass floors at bridge entry — exactly the failure mode
        the user is debugging (solver dies at bridge, tier2 then re-projects
        to the wrong floor and sends the car onto the lower path).

        Parameters
        ----------
        n_samples : int | None
            Samples along s. None → `self.N + 1` (legacy tier 2/3 fallback
            shape). Phase X auto-mode RECOVERY passes a larger value (e.g. 81)
            so SM / controller see a dense, equispaced-in-s trajectory.
        clip_walls : bool
            If True, hard-clip |n| to `[wall_safe - d_right, d_left - wall_safe]`
            at every sample and re-lift xy. Phase X RECOVERY enables this so
            the polynomial doesn't punch through track bounds — legacy tier
            2/3 fallback keeps it off (corridor handled by the solver itself).

        Returns (xy_traj (K,5) [x, y, psi, s, z], frenet (K,4) [s, n, 0, v])
        or (None, None) if the lifter blows up.
        """
        if delta_s is None:
            delta_s = self._quintic_delta_s
        if n_samples is None:
            n_samples = self.N + 1
        try:
            ### HJ : 2026-04-27 — tier-2 algorithm switch.
            ###      Old: pure quintic with optional clip. Visually started
            ###      at "wall edge" when ego was past corridor, no smoothing.
            ###      New (default): build_recovery_path — recovery_spliner
            ###      style BPoly + GB-tangent lookahead + wall-aware pull-in
            ###      + savgol smoothing. clip_walls=True triggers it; False
            ###      keeps the legacy pure-quintic path.
            if clip_walls:
                ### HJ : 2026-04-27 — shrink retry. First attempt uses full
                ###      candidate range; if validation fails, retry with
                ###      progressively smaller max_candidate_len so
                ###      tangent_idx is forced to a closer (= shorter
                ###      spline) endpoint. User: "안될거같으면 더 당겨서
                ###      생성하는거 맞지?" — yes.
                shrink_caps = [None, 60, 30, 15]   # idx caps; None = no cap
                xy_traj, sn_traj = None, None
                for attempt_idx, max_cap in enumerate(shrink_caps):
                    xy_traj, sn_traj = build_recovery_path(
                        self.lifter, s_cur, ego_n,
                        ego_x=float(self.car_x), ego_y=float(self.car_y),
                        ego_yaw=float(self.car_yaw),
                        g_dleft=self.lifter.g_dleft,
                        g_dright=self.lifter.g_dright,
                        g_kappa=getattr(self, 'g_kappa', None),
                        inflection_points=getattr(self, '_inflection_points', None),
                        min_candidates_lookahead_n=int(self._get_param_or_default(
                            '~recovery_min_candidates_lookahead_n', 100)),
                        num_kappas=int(self._get_param_or_default(
                            '~recovery_num_kappas', 20)),
                        max_candidate_len=max_cap,
                        n_additional=int(self._get_param_or_default(
                            '~recovery_n_additional', 100)),
                        delta_s=delta_s, n_samples=int(n_samples),
                        wall_safe=float(getattr(self.solver, 'wall_safe', 0.15)),
                        spline_scale=float(self._get_param_or_default(
                            '~recovery_spline_scale', 0.8)),
                        return_frenet=True)
                    if xy_traj is not None:
                        if attempt_idx > 0:
                            rospy.loginfo_throttle(
                                1.0,
                                '[mpc tier2][%s] recovery valid after shrink (max_cap=%s, attempt=%d)',
                                rospy.get_name(), max_cap, attempt_idx)
                        break
                ### HJ : end
                if xy_traj is None:
                    rospy.logwarn_throttle(
                        1.0, '[mpc fallback tier2][%s] recovery path invalid '
                        '(corridor punch even after pull-in) — drop',
                        rospy.get_name())
                    return None, None
                N1 = xy_traj.shape[0]
                # Skip the legacy clip_walls block below by setting flag off
                clip_walls_already_done = True
            else:
                psi_track = self.lifter._interp_psi(s_cur)
                psi_delta = float(np.arctan2(
                    np.sin(self.car_yaw - psi_track),
                    np.cos(self.car_yaw - psi_track)))
                xy_traj, sn_traj = build_quintic_fallback(
                    self.lifter, s_cur, ego_n, psi_delta,
                    delta_s=delta_s, n_samples=int(n_samples),
                    return_frenet=True)
                N1 = xy_traj.shape[0]
                clip_walls_already_done = False
            ### HJ : end

            # ### HJ : Phase X — wall-bound clip (auto RECOVERY only).
            # Quintic is pure BC fit; without clipping it can punch through
            # walls on tight corners. Clip n to the track corridor with
            # wall_safe buffer, then re-lift xy so the published trajectory
            # stays inside. (Clipping is a 1-D numpy op; re-lift is K calls
            # of lifter.sn_to_xy — ~1 ms @ K=81.)
            if clip_walls and not locals().get('clip_walls_already_done', False):
                ### HJ : 2026-04-27 — recovery_spliner-style validation.
                ###      Drop "clip every sample to corridor edge" (which
                ###      caused the path to visually start at the wall).
                ###      Now: validate the quintic against d_left/d_right.
                ###      If any k>=1 sample is outside corridor (with
                ###      wall_safe margin), DECLARE THE QUINTIC INVALID
                ###      and return None — caller will fall through to
                ###      tier-3 (raceline slice). This matches the
                ###      recovery_spliner Track3DValidator behaviour:
                ###      "publish a clean valid path or none at all".
                ###      k=0 is ego's actual position — never validated.
                wall_safe = float(getattr(self.solver, 'wall_safe', 0.15))
                invalid = False
                for i in range(1, N1):
                    s_i = float(sn_traj[i, 0])
                    d_L = float(self.lifter._interp(s_i, self.lifter.g_dleft))
                    d_R = float(self.lifter._interp(s_i, self.lifter.g_dright))
                    n_lo = -(d_R - wall_safe)
                    n_hi = +(d_L - wall_safe)
                    if n_lo > n_hi:
                        continue  # degenerate corridor — skip
                    n_i = float(sn_traj[i, 1])
                    if n_i < n_lo or n_i > n_hi:
                        invalid = True
                        break
                if invalid:
                    rospy.logwarn_throttle(
                        1.0,
                        '[mpc fallback tier2][%s] quintic invalid (sample past corridor) — dropping; '
                        'caller falls through to tier-3',
                        rospy.get_name())
                    return None, None
                ### HJ : end

            # Augment xy_traj (K,3) → (K,5) with carried [s, z] so the
            # downstream viz/publish path uses 3D-safe z instead of the
            # marker-time xy nearest-index lookup (which would alias floors).
            aug = np.zeros((N1, 5), dtype=np.float64)
            aug[:, :3] = xy_traj
            for i in range(N1):
                s_i = float(sn_traj[i, 0])
                aug[i, 3] = s_i
                aug[i, 4] = float(self.lifter._interp(s_i, self.lifter.g_z))
            # Frenet companion for fill_wpnt_from_s (solver-frenet-shape
            # compatible: (K, 4) [s, n, mu=0, v=ego_v_placeholder]).
            v_fb = float(np.clip(self.car_vx, 0.5,
                                 max(self.solver.v_max, 0.5)))
            frenet = np.zeros((N1, 4), dtype=np.float64)
            frenet[:, 0] = sn_traj[:, 0]
            frenet[:, 1] = sn_traj[:, 1]
            frenet[:, 3] = v_fb
            return aug, frenet
        except Exception as exc:  # pragma: no cover — guarded log only
            rospy.logerr_throttle(
                1.0, '[mpc fallback tier2][%s] quintic build raised: %s',
                rospy.get_name(), exc)
            return None, None

    def _handle_tier2_geometric(self, traj_fb, frenet_fb, ego_n,
                                ipopt_status, solve_ms, obs_tag):
        self._viz_tier = 2
        self._viz_status = 'GEOMETRIC_FALLBACK'
        # Synthetic u_sol — constant-velocity placeholder so the lifter can
        # still fill vx_mps / ax_mps2. v chosen as mean of the raceline-slice
        # local target to stay close to the velocity planner's expectation.
        v_fb = float(np.clip(self.car_vx, 0.5, max(self.solver.v_max, 0.5)))
        u_fb = np.zeros((self.N, 2), dtype=np.float64)
        u_fb[:, 0] = v_fb
        # ### HJ : v3b — stash synthetic frenet so _publish_outputs takes
        # the s-direct path (fill_wpnt_from_s) and does NOT re-project xy.
        self._last_frenet_traj = frenet_fb
        self._publish_outputs(traj_fb, u_sol_override=u_fb)
        self._publish_status('GEOMETRIC_FALLBACK streak=%d n0=%.2f obs=%s' %
                             (self._fail_streak, ego_n, obs_tag))
        self._last_status = 'GEOMETRIC_FALLBACK'
        rospy.logwarn_throttle(
            0.5,
            '[mpc fallback tier2][%s] GEOMETRIC streak=%d n0=%.2f Δs=%.1f '
            'ipopt=%s solve=%.1fms obs=%s',
            rospy.get_name(), self._fail_streak, ego_n, self._quintic_delta_s,
            ipopt_status, solve_ms, obs_tag,
        )

    # ---- Tier 3 (primary) -------------------------------------------------
    def _handle_tier3_convergence_quintic(self, traj_fb, frenet_fb, ego_n,
                                          ipopt_status, solve_ms, obs_tag):
        """### HJ : v3b — "recovery spliner" analogue. Ego_n → 0 over a
        shorter Δs (more aggressive than tier2). Used when tier2 quintic
        would also be feasible but we want a snappier return-to-center
        shape. Controller still sees continuous (s, n) from ego pose."""
        self._viz_tier = 3
        self._viz_status = 'CONVERGENCE_QUINTIC'
        v_fb = float(np.clip(self.car_vx, 0.5, max(self.solver.v_max, 0.5)))
        u_fb = np.zeros((self.N, 2), dtype=np.float64)
        u_fb[:, 0] = v_fb
        self._last_frenet_traj = frenet_fb
        self._publish_outputs(traj_fb, u_sol_override=u_fb)
        self._publish_status('CONVERGENCE_QUINTIC streak=%d n0=%.2f obs=%s' %
                             (self._fail_streak, ego_n, obs_tag))
        self._last_status = 'CONVERGENCE_QUINTIC'
        rospy.logerr_throttle(
            0.5,
            '[mpc fallback tier3][%s] CONVERGENCE_QUINTIC streak=%d n0=%.2f '
            'ipopt=%s solve=%.1fms obs=%s',
            rospy.get_name(), self._fail_streak, ego_n, ipopt_status,
            solve_ms, obs_tag,
        )

    # ---- Tier 3 (absolute last-resort) ------------------------------------
    def _handle_tier3_raceline(self, s_cur, ipopt_status, solve_ms, obs_tag):
        """Only reached when BOTH tier2 and tier3 quintics raised. Pure
        raceline slice (n=0, no ego-anchor). Controller may see a one-tick
        jump but output never drops."""
        self._viz_tier = 3
        self._viz_status = 'RACELINE_SLICE'
        N_total = self.N + 1
        ds_grid = np.linspace(0.0, max(self._quintic_delta_s, 4.0), N_total)
        traj = np.zeros((N_total, 5), dtype=np.float64)
        frenet = np.zeros((N_total, 4), dtype=np.float64)
        for i, ds in enumerate(ds_grid):
            s_i = s_cur + ds
            x, y = self.lifter.sn_to_xy(s_i, 0.0)
            psi = self.lifter._interp_psi(s_i)
            z = float(self.lifter._interp(s_i, self.lifter.g_z))
            traj[i, 0] = x
            traj[i, 1] = y
            traj[i, 2] = psi
            traj[i, 3] = s_i
            traj[i, 4] = z
            frenet[i, 0] = s_i
            frenet[i, 1] = 0.0

        v_fb = float(np.clip(self.car_vx, 0.5, max(self.solver.v_max, 0.5)))
        u_fb = np.zeros((self.N, 2), dtype=np.float64)
        u_fb[:, 0] = v_fb
        frenet[:, 3] = v_fb
        self._last_frenet_traj = frenet
        self._publish_outputs(traj, u_sol_override=u_fb)
        self._publish_status('RACELINE_SLICE streak=%d obs=%s' %
                             (self._fail_streak, obs_tag))
        self._last_status = 'RACELINE_SLICE'
        rospy.logerr_throttle(
            0.5,
            '[mpc fallback tier3-last][%s] RACELINE_SLICE streak=%d ipopt=%s '
            'solve=%.1fms obs=%s — BOTH quintics failed',
            rospy.get_name(), self._fail_streak, ipopt_status, solve_ms, obs_tag,
        )
        return traj

    def _publish_outputs(self, trajectory, u_sol_override=None):
        """trajectory: (N+1, 3) [x, y, psi]. Publish:
         - WpntArray on ~best_trajectory_observation (always)
         - MarkerArray on ~best_sample/markers (debug, always)
         - role-specific output on ~out/... (overtake/recovery only)

        `u_sol_override` allows fallback tiers to supply a synthetic control
        sequence when the solver didn't produce one (tier 2/3) or when the
        last-good snapshot was re-published (tier 1).
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'

        # ### HJ : Phase 2 — vx comes from solver u_sol[k, 0], ax from its
        # forward difference. Last horizon step has no successor → ax=0.
        if u_sol_override is not None:
            u_sol = u_sol_override
        else:
            u_sol = getattr(self.solver, 'last_u_sol', None)
        N_traj = trajectory.shape[0]

        wp_arr = WpntArray()
        wp_arr.header = header
        # ### HJ : v3 — for frenet backends the solver holds s directly
        # (frenet state). Use fill_wpnt_from_s to skip xy→s argmin which
        # fails at overpass crossings in 3D (CLAUDE.md: xy round-trip 금지).
        fren_traj = getattr(self, '_last_frenet_traj', None)
        use_s_direct = (fren_traj is not None
                        and fren_traj.shape[0] == N_traj
                        and self.solver_backend in ('frenet_kin', 'frenet_d'))
        idx_hint = None
        for i in range(N_traj):
            x = float(trajectory[i, 0])
            y = float(trajectory[i, 1])
            psi_mpc = float(trajectory[i, 2])

            if u_sol is not None and i < u_sol.shape[0]:
                vx = float(u_sol[i, 0])
                if i + 1 < u_sol.shape[0]:
                    ax = (float(u_sol[i + 1, 0]) - vx) / self.dT
                else:
                    ax = 0.0
            else:
                # Terminal state (i == N) has no control; hold last u.
                vx = float(u_sol[-1, 0]) if u_sol is not None else 0.0
                ax = 0.0

            if use_s_direct:
                s_i = float(fren_traj[i, 0])
                n_i = float(fren_traj[i, 1])
                fields = self.lifter.fill_wpnt_from_s(
                    s_ref=s_i, n_ref=n_i,
                    x=x, y=y, psi_mpc=psi_mpc,
                    vx_mpc=vx, ax_mpc=ax,
                )
            else:
                fields = self.lifter.fill_wpnt(
                    x, y, psi_mpc, vx_mpc=vx, ax_mpc=ax, idx_hint=idx_hint,
                )
                idx_hint = self.lifter._nearest_idx(x, y)

            w = Wpnt()
            w.id = i
            w.s_m = fields['s_m']
            w.d_m = fields['d_m']
            w.x_m = fields['x_m']
            w.y_m = fields['y_m']
            if hasattr(w, 'z_m'):
                w.z_m = fields['z_m']
            w.psi_rad = fields['psi_rad']
            w.kappa_radpm = fields['kappa_radpm']
            w.vx_mps = fields['vx_mps']
            w.ax_mps2 = fields['ax_mps2']
            if hasattr(w, 'mu_rad'):
                w.mu_rad = fields['mu_rad']
            w.d_left = fields['d_left']
            w.d_right = fields['d_right']
            wp_arr.wpnts.append(w)

        # ### HJ : Phase X — Speed painter + continuity guard BEFORE role pub.
        # Painter rewrites vx_mps using GB s-lookup + curvature cap + seam blend.
        # Continuity guard blends first K_guard waypoints with previous tick.
        seam_idx_used, blend_applied, vx_first_delta = self._post_process_speed(
            wp_arr.wpnts, self._mpc_mode)
        cont_L2, cont_applied = self._apply_continuity_guard(wp_arr.wpnts)
        self._last_continuity_L2 = float(cont_L2)
        self._last_path_blend_applied = bool(cont_applied)
        self._last_painter_seam_idx = int(seam_idx_used)
        self._last_painter_blend_applied = bool(blend_applied)
        self._last_painter_vx_first_delta = float(vx_first_delta)

        # ### HJ : Phase X (refactored) — densify is OPT-IN only.
        # Default: publish solver's native N+1 grid as-is (user directive
        # 2026-04-24 "mpc 출력 그대로"). Set ~mpc_densify_enable:=true to
        # reintroduce linear-interp post-densify to mpc_output_ds spacing.
        if (self._mpc_densify_enable
                and self.state in ('auto', 'overtake', 'recovery')):
            wp_arr.wpnts = self._densify_wpnt_list(
                wp_arr.wpnts, ds=self._mpc_output_ds)

        ### HJ : 2026-05-01 — sparsify pass.
        ### bag 2026-04-29-16-47-24 분석에서 cache splice junction 이
        ### 인접 ds 0.0287/0.0419/0.0481 로 publish gate (ds_min < 0.05)
        ### 에 매번 막혀 18s 동안 21번 path_source GB ↔ MPC_RC 토글 발생.
        ### 게이트 0.02 로 낮추는 동시에, 호출자가 publish 직전 ds < 0.02
        ### 인 점을 drop. trajectory 모양은 유지하되 인접 점 너무 가까운
        ### 케이스는 한 점만 남김. controller 는 길이 가변에 강건.
        wp_arr.wpnts = self._sparsify_wpnts(wp_arr.wpnts, ds_min=0.02)

        ### HJ : 2026-04-28 (S1-4) — publish-side validation chain.
        ###      lift→painter→guard 후 kappa/ds/finite 검사. 위배 시 publish
        ###      생략 (controller latest_threshold 가 자체 brake). bag2 의
        ###      kappa=1.6M trajectory 발행 차단의 안전망.
        valid, reason = self._validate_publish_wpnts(wp_arr.wpnts)
        if not valid:
            self._viz_status = 'PUBLISH_VALIDATE_FAIL'
            self._publish_status('PUBLISH_VALIDATE_FAIL reason=%s' % reason)
            self._last_status = 'PUBLISH_VALIDATE_FAIL'
            rospy.logerr_throttle(
                0.5, '[mpc][%s] publish blocked: %s',
                rospy.get_name(), reason)
            return
        ### HJ : end (S1-4)

        self.pub_best_trajectory.publish(wp_arr)
        self._publish_debug_markers(header, trajectory)

        # ### HJ : Phase X (refactored) — single unified topic.
        # `ot_line` carries the semantic tag ("avoid" when WITH_OBS, "recover"
        # when NO_OBS) so the SM can colour-code without needing multiple
        # topics. Legacy pub_ot / pub_rc kept optional for rollback.
        if self.pub_mpc is not None:
            ot = OTWpntArray()
            ot.header = header
            ot.last_switch_time = self.get_clock().now().to_msg()
            ot.side_switch = False
            mean_d = (float(np.mean([w.d_m for w in wp_arr.wpnts]))
                      if wp_arr.wpnts else 0.0)
            ot.ot_side = 'left' if mean_d >= 0.0 else 'right'
            ot.ot_line = ('avoid' if self._mpc_mode == MPC_MODE_WITH_OBS
                          else 'recover')
            ot.wpnts = wp_arr.wpnts
            self.pub_mpc.publish(ot)
        # Legacy side-publishers (usually disabled via launch remap).
        if self.pub_ot is not None and self.state == 'overtake':
            self._publish_ot_wpnts(header, wp_arr)
        if self.pub_rc is not None and self.state == 'recovery':
            self.pub_rc.publish(wp_arr)

        # Cache for next-tick continuity guard (deep copy of header-less wpnts).
        self._last_published_wpnts = [self._wpnt_copy(w) for w in wp_arr.wpnts]
        self._last_published_mode = self._mpc_mode

    ### HJ : 2026-04-28 (S1-4) — publish-side validation chain.
    def _validate_publish_wpnts(self, wpnts):
        """publish 전 trajectory 안전성 검사. 사용자 룰 1 (이상한 답 금지).

        Reject if:
          - kappa_max > 5 rad/m (lifter degeneracy 또는 numerical 폭발)
          - 인접 점 ds < 0.02m (near-coincident points → kappa explode)
          - x/y/vx NaN/Inf
          - inter-msg first-wpnt jump > 0.5m (단, ego_s 큰 점프 시 skip)

        2026-05-01: ds_min 임계값 0.05 → 0.02 완화 + 호출자가 sparsify
        후처리하므로 ds_min 게이트는 안전망(곡률 폭발의 진짜 게이트는
        kappa_max > 5)으로만 남음. bag 2026-04-29-16-47-24 에서 cache
        splice junction 이 ds 0.0287~0.0481 로 18s 동안 publish 21번
        차단된 케이스를 해결.
        """
        import math
        if not wpnts or len(wpnts) < 2:
            return False, 'empty_or_short'
        kappa_max = 0.0
        ds_min = 1e9
        last_xy = (wpnts[0].x_m, wpnts[0].y_m)
        for i, w in enumerate(wpnts):
            if not (math.isfinite(w.x_m) and math.isfinite(w.y_m)
                    and math.isfinite(w.vx_mps)):
                return False, 'non_finite_at_%d' % i
            ka = abs(getattr(w, 'kappa_radpm', 0.0))
            if ka > kappa_max:
                kappa_max = ka
            if i > 0:
                ds = math.hypot(w.x_m - last_xy[0], w.y_m - last_xy[1])
                if ds < ds_min:
                    ds_min = ds
                last_xy = (w.x_m, w.y_m)
        if kappa_max > 5.0:
            return False, 'kappa_max=%.2f' % kappa_max
        if ds_min < 0.02:
            return False, 'ds_min=%.4f' % ds_min
        # inter-msg jump check — DOWNGRADED to warn (no reject).
        # 2026-04-28 user directive: MPC trajectory must publish every
        # tick regardless of state. The previous reject path (whether the
        # raw form or the ego_travel-compensated form) blocked publishing
        # whenever a side-decision flip or plan_picker toggle shifted the
        # path's first waypoint laterally by >0.5m. That created
        # multi-tick blackouts and is exactly the failure mode the user
        # is rejecting. Trajectory smoothness on big lateral shifts is
        # the continuity_guard's job (it blends the first K=5 wpnts with
        # the previous publish). Just emit a status string for telemetry
        # and let publishing proceed.
        ego_xy_now = (float(getattr(self, 'car_x', 0.0)),
                      float(getattr(self, 'car_y', 0.0)))
        self._ego_xy_at_prev_publish = ego_xy_now
        prev_pub = getattr(self, '_last_published_wpnts', None)
        if prev_pub is not None and len(prev_pub) > 0:
            jump = math.hypot(wpnts[0].x_m - prev_pub[0].x_m,
                              wpnts[0].y_m - prev_pub[0].y_m)
            if jump > 1.0:
                self._last_inter_msg_jump_warn = jump
        return True, 'ok'
    ### HJ : end (S1-4)

    @staticmethod
    def _sparsify_wpnts(wpnts, ds_min=0.02):
        """Drop adjacent wpnts where xy distance < ds_min.

        Keeps the first point. For each subsequent point, only keep it if
        its xy distance from the LAST KEPT point is >= ds_min. The final
        point is forced to be kept (replacing the previous tail if it was
        within ds_min) so horizon end is preserved.
        """
        import math
        if not wpnts or len(wpnts) < 3:
            return wpnts
        out = [wpnts[0]]
        last_xy = (wpnts[0].x_m, wpnts[0].y_m)
        for w in wpnts[1:-1]:
            d = math.hypot(w.x_m - last_xy[0], w.y_m - last_xy[1])
            if d >= ds_min:
                out.append(w)
                last_xy = (w.x_m, w.y_m)
        # always keep the last horizon point
        tail = wpnts[-1]
        d_tail = math.hypot(tail.x_m - last_xy[0], tail.y_m - last_xy[1])
        if d_tail >= ds_min or len(out) < 2:
            out.append(tail)
        else:
            out[-1] = tail
        return out

    @staticmethod
    def _wpnt_copy(w):
        """Lightweight Wpnt shallow-clone for continuity-guard cache."""
        c = Wpnt()
        c.id = w.id
        c.s_m = w.s_m
        c.d_m = w.d_m
        c.x_m = w.x_m
        c.y_m = w.y_m
        if hasattr(w, 'z_m'):
            c.z_m = w.z_m
        c.psi_rad = w.psi_rad
        c.kappa_radpm = w.kappa_radpm
        c.vx_mps = w.vx_mps
        c.ax_mps2 = w.ax_mps2
        if hasattr(w, 'mu_rad'):
            c.mu_rad = w.mu_rad
        c.d_left = w.d_left
        c.d_right = w.d_right
        return c

    @staticmethod
    def _densify_wpnt_list(wpnts, ds=0.1):
        """Resample waypoints to uniform EUCLIDEAN arc length along (x, y, z).

        User directive 2026-04-24: "s 축 기반 등간격이 아니라, 진짜 냅다 x,y,z
        평면 등간격 샘플링이어야해." So this is NOT s-based; we compute cumulative
        xyz Euclidean distance between wpnts and sample the chain uniformly.
        Result: adjacent output wpnts have `||xyz[i+1] - xyz[i]|| ≈ ds` (no
        stretch/compress at corners that s-based densify had).

        Field handling:
          - scalars (s_m, d_m, vx, ax, κ, μ, d_left, d_right): linear interp
            along the same (cumdist, t) param. Note s_m ends up NON-uniform
            in s but uniform in arc length — that's the goal.
          - psi_rad: circular interp (sin/cos avg) to avoid wrap jumps.
        If native spacing is already ≤ ds (rare), returns as-is.
        """
        import math
        if len(wpnts) < 2:
            return wpnts
        # Cumulative arc length over xyz.
        cumdist = [0.0]
        for i in range(1, len(wpnts)):
            dx = float(wpnts[i].x_m - wpnts[i - 1].x_m)
            dy = float(wpnts[i].y_m - wpnts[i - 1].y_m)
            dz = float(getattr(wpnts[i], 'z_m', 0.0)
                       - getattr(wpnts[i - 1], 'z_m', 0.0))
            cumdist.append(cumdist[-1] + math.sqrt(dx*dx + dy*dy + dz*dz))
        total = float(cumdist[-1])
        if total <= 0.0:
            return wpnts
        native_mean = total / max(len(wpnts) - 1, 1)
        if native_mean <= ds * 1.05:
            return wpnts
        n_new = int(math.floor(total / ds)) + 1
        if n_new <= len(wpnts):
            return wpnts
        targets = np.linspace(0.0, total, n_new)
        # Pre-pack cumdist as numpy for searchsorted
        cum_np = np.asarray(cumdist)
        out = []
        for new_id, d_target in enumerate(targets):
            idx = int(np.searchsorted(cum_np, d_target))
            if idx <= 0:
                lo = hi = wpnts[0]; t = 0.0
            elif idx >= len(wpnts):
                lo = hi = wpnts[-1]; t = 1.0
            else:
                lo = wpnts[idx - 1]
                hi = wpnts[idx]
                seg = float(cum_np[idx] - cum_np[idx - 1])
                t = (float(d_target - cum_np[idx - 1]) / seg) if seg > 1e-9 else 0.0
            def mix(a, b):
                return (1.0 - t) * float(a) + t * float(b)
            w = Wpnt()
            w.id = int(new_id)
            w.s_m = mix(lo.s_m, hi.s_m)    # s is interpolated, not re-uniformed
            w.d_m = mix(lo.d_m, hi.d_m)
            w.x_m = mix(lo.x_m, hi.x_m)
            w.y_m = mix(lo.y_m, hi.y_m)
            if hasattr(w, 'z_m'):
                w.z_m = mix(getattr(lo, 'z_m', 0.0),
                            getattr(hi, 'z_m', 0.0))
            sin_v = (1 - t) * math.sin(lo.psi_rad) + t * math.sin(hi.psi_rad)
            cos_v = (1 - t) * math.cos(lo.psi_rad) + t * math.cos(hi.psi_rad)
            w.psi_rad = float(math.atan2(sin_v, cos_v))
            w.kappa_radpm = mix(lo.kappa_radpm, hi.kappa_radpm)
            w.vx_mps = mix(lo.vx_mps, hi.vx_mps)
            w.ax_mps2 = mix(lo.ax_mps2, hi.ax_mps2)
            if hasattr(w, 'mu_rad'):
                w.mu_rad = mix(getattr(lo, 'mu_rad', 0.0),
                               getattr(hi, 'mu_rad', 0.0))
            w.d_left = mix(lo.d_left, hi.d_left)
            w.d_right = mix(lo.d_right, hi.d_right)
            out.append(w)
        return out

    def _publish_ot_wpnts(self, header, wp_arr):
        """Legacy per-role publisher (only called when state=='overtake' AND
        pub_ot is remapped). Not used in Phase X auto mode."""
        ot = OTWpntArray()
        ot.header = header
        ot.last_switch_time = self.get_clock().now().to_msg()
        ot.side_switch = False
        mean_d = float(np.mean([w.d_m for w in wp_arr.wpnts])) if wp_arr.wpnts else 0.0
        ot.ot_side = 'left' if mean_d >= 0.0 else 'right'
        ot.ot_line = 'mpc'
        ot.wpnts = wp_arr.wpnts
        self.pub_ot.publish(ot)

    def _publish_debug_markers(self, header, trajectory):
        arr = MarkerArray()

        # ### HJ : 3D 트랙의 overpass(교차 층)에서 같은 (x, y)에 서로 다른 z가
        # 존재 → project_xy_to_sn(2D 최단)로 s를 역산하면 아래층/위층이
        # 뒤섞임. frenet_kin 백엔드는 _lift_frenet_to_xy에서 (x, y, psi, s, z)
        # 5컬럼을 싣고 내려오므로 컬럼이 ≥5면 담긴 z를 그대로 쓴다. Fallback
        # 타이어(tier2/3)는 3컬럼이라 s를 모르므로 z=0을 사용.
        has_sz = trajectory.shape[1] >= 5

        # LINE_STRIP — predicted path
        line = Marker()
        line.header = header
        line.ns = 'mpc_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        # ### HJ : v3c — rainbow-graded solver health colour (best→worst):
        #   tier0 pass1  → red     (NLP 1-pass clean, primary mode)
        #   tier0 pass2  → orange  (NLP needed the TRAIL retry)
        #   tier1        → yellow  (HOLD_LAST — re-use last good)
        #   tier2        → green   (GEOMETRIC_FALLBACK quintic Δs≈8m)
        #   tier3 conv.  → cyan    (CONVERGENCE_QUINTIC Δs≈4m)
        #   tier3 rline  → blue    (RACELINE_SLICE — absolute last)
        tier = int(getattr(self, '_viz_tier', 0))
        status = str(getattr(self, '_viz_status', 'OK'))
        vpass = int(getattr(self, '_viz_pass', 1))
        if tier == 0 and vpass == 1:
            rgb = (1.0, 0.05, 0.05)   # red
        elif tier == 0:
            rgb = (1.0, 0.55, 0.0)    # orange
        elif tier == 1:
            rgb = (1.0, 0.95, 0.0)    # yellow
        elif tier == 2:
            rgb = (0.1, 0.9, 0.2)     # green
        elif tier == 3 and status == 'CONVERGENCE_QUINTIC':
            rgb = (0.0, 0.85, 0.95)   # cyan
        else:
            rgb = (0.1, 0.3, 1.0)     # blue (RACELINE_SLICE / unknown)
        line.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.95)
        line.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = float(trajectory[i, 4]) if has_sz else 0.0
            line.points.append(pt)
        arr.markers.append(line)

        # SPHERE_LIST — per-step
        pts = Marker()
        pts.header = header
        pts.ns = 'mpc_steps'
        pts.id = 1
        pts.type = Marker.SPHERE_LIST
        pts.action = Marker.ADD
        pts.scale.x = pts.scale.y = pts.scale.z = 0.12
        pts.color = ColorRGBA(r=line.color.r, g=line.color.g, b=line.color.b, a=0.9)
        pts.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = float(trajectory[i, 4]) if has_sz else 0.0
            pts.points.append(pt)
        arr.markers.append(pts)

        self.pub_best_markers.publish(arr)

    # ---- live debug publishers --------------------------------------------
    def _publish_tick_live(self, fields):
        """### HJ : Build and publish the per-tick live debug bundle.

        Two surfaces:
          (1) ~debug/tick_json  — flat JSON for `rostopic echo -c` monitoring.
          (2) ~debug/markers    — RViz MarkerArray with corridor + obstacles
                                   + ref-slice + status text (ego-anchored).

        `fields` mirrors what `_debug_log` receives from `_plan_loop` tier
        handlers (tier, status, trajectory, ref_slice, obs_arr, obs_tag,
        ipopt_status, iter_count, solve_ms, slack_max, ego_s, ego_n, ...).
        Missing entries are handled gracefully so fallback tiers with partial
        info (tier 1/3 may have trajectory=None) still publish a useful tick.
        """
        import json as _json

        self._tick_counter += 1
        tick = int(self._tick_counter)
        t_now = self.get_clock().now().to_msg().to_sec()

        tier = int(fields.get('tier', -1))
        status = str(fields.get('status', '-'))
        ipopt_status = str(fields.get('ipopt_status', '-'))
        iter_count = int(fields.get('iter_count', -1))
        solve_ms = float(fields.get('solve_ms', 0.0))
        slack_max = float(fields.get('slack_max', 0.0))
        warm_used = int(fields.get('warm_used', 0))
        obs_tag = str(fields.get('obs_tag', '-'))
        ego_s = float(fields.get('ego_s', float('nan')))
        ego_n = float(fields.get('ego_n', float('nan')))
        speed0 = fields.get('speed0', None)
        steer0 = fields.get('steer0', None)

        trajectory = fields.get('trajectory', None)
        ref_slice = fields.get('ref_slice', None)
        obs_arr = fields.get('obs_arr', None)

        # ---- trajectory stats (frenet-lateral margins, curvature RMS) ------
        traj_stats = {
            'len': 0, 'n_min': None, 'n_max': None, 'n_end': None,
            'margin_L_min': None, 'margin_R_min': None,
            'kappa_rms': None, 'kappa_max': None,
        }
        jitter_rms = None
        n_traj_signed = None
        if trajectory is not None and ref_slice is not None:
            try:
                rc = ref_slice['center_points']
                rdx = ref_slice['ref_dx']; rdy = ref_slice['ref_dy']
                dL = ref_slice['d_left_arr']; dR = ref_slice['d_right_arr']
                M = int(min(trajectory.shape[0], rc.shape[0]))
                # Left normal in xy (same convention as the inspect scripts).
                lnx = -rdy[:M]; lny = rdx[:M]
                dx_t = trajectory[:M, 0] - rc[:M, 0]
                dy_t = trajectory[:M, 1] - rc[:M, 1]
                n_signed = dx_t * lnx + dy_t * lny
                n_traj_signed = n_signed
                inflation = float(self.solver.inflation)
                marL = (dL[:M] - inflation) - n_signed
                marR = n_signed + (dR[:M] - inflation)
                # Curvature RMS (discrete κ from (x,y) via 2nd diff / ds).
                kappa_rms = None; kappa_max = None
                if M >= 3:
                    dx = np.diff(trajectory[:M, 0])
                    dy = np.diff(trajectory[:M, 1])
                    ds = np.hypot(dx, dy)
                    ds = np.where(ds < 1e-6, 1e-6, ds)
                    psi = np.arctan2(dy, dx)
                    dpsi = np.diff(np.unwrap(psi))
                    kappa = dpsi / ds[:-1]
                    kappa_rms = float(np.sqrt(np.mean(kappa ** 2)))
                    kappa_max = float(np.max(np.abs(kappa)))
                traj_stats = {
                    'len': M,
                    'n_min': float(np.min(n_signed)),
                    'n_max': float(np.max(n_signed)),
                    'n_end': float(n_signed[-1]),
                    'margin_L_min': float(np.min(marL)),
                    'margin_R_min': float(np.min(marR)),
                    'kappa_rms': kappa_rms,
                    'kappa_max': kappa_max,
                }
            except Exception:
                pass

        # Jitter: match against previous trajectory by index (horizon stays
        # the same N+1 and is re-sliced at same ego pace — good enough for
        # smoothness monitoring).
        if trajectory is not None:
            try:
                cur_xy = np.array(trajectory[:, :2], dtype=float, copy=True)
                if (self._prev_traj_xy is not None and
                        self._prev_traj_xy.shape == cur_xy.shape):
                    d = cur_xy - self._prev_traj_xy
                    jitter_rms = float(np.sqrt(np.mean(d[:, 0] ** 2
                                                     + d[:, 1] ** 2)))
                self._prev_traj_xy = cur_xy
                self._prev_traj_ego_s = ego_s
            except Exception:
                pass

        # ---- ref slice stats -----------------------------------------------
        ref_stats = {
            's_start': None, 's_end': None,
            'v_mean': None, 'corridor_width_mean': None,
        }
        if ref_slice is not None:
            try:
                rs = ref_slice['ref_s']
                rv = ref_slice['ref_v']
                dL = ref_slice['d_left_arr']; dR = ref_slice['d_right_arr']
                ref_stats = {
                    's_start': float(rs[0]),
                    's_end': float(rs[-1]),
                    'v_mean': float(np.mean(rv)),
                    'corridor_width_mean': float(np.mean(dL + dR)),
                }
            except Exception:
                pass

        # ---- obstacles: extract slot[0] (closest tick) per active slot -----
        obs_list = []
        if obs_arr is not None:
            try:
                for o in range(obs_arr.shape[0]):
                    w_ts = obs_arr[o, :, 2]
                    if float(np.max(w_ts)) <= 0.0:
                        continue
                    obs_list.append({
                        's0': float(obs_arr[o, 0, 0]),
                        'n0': float(obs_arr[o, 0, 1]),
                        'sN': float(obs_arr[o, -1, 0]),
                        'nN': float(obs_arr[o, -1, 1]),
                        'w': float(np.max(w_ts)),
                    })
            except Exception:
                pass

        # ### HJ : v3b — solver diagnostic snapshot. Populated on every pass
        # inside FrenetKinMPC.solve(); on failure the LAST pass's input is
        # retained so we can see what IPOPT was choking on.
        solver_input = getattr(self.solver, 'last_input', {}) or {}
        solver_infeas = getattr(self.solver, 'last_infeas_info', {}) or {}
        solver_pass = int(getattr(self.solver, 'last_pass', 0))
        pass_hist = getattr(self.solver, 'last_pass_history', []) or []

        # ### HJ : v3b — prediction freshness. Callbacks for /opponent_prediction
        # and /tracking/obstacles live on separate rospy subscriber threads,
        # so they keep updating even while _plan_loop wrestles with infeasible
        # NLPs. Publishing the age here lets the user verify that (e.g. while
        # the solver was dead for 0.87s, did prediction keep flowing?).
        _now = self.get_clock().now().to_msg()
        def _age(t):
            if t is None:
                return None
            try:
                return float((_now - t).to_sec())
            except Exception:
                return None
        predict_age_s = _age(getattr(self, '_obs_predict_t', None))
        track_age_s = _age(getattr(self, '_obs_track_t', None))
        opp_age_s = _age(getattr(self, '_opp_wpnts_t', None))

        payload = {
            'tick': tick,
            't': t_now,
            'state': self.state,
            'tier': tier,
            'status': status,
            'ipopt_status': ipopt_status,
            'iter': iter_count,
            'solve_ms': round(solve_ms, 3),
            'slack_max': round(slack_max, 4),
            'warm_used': warm_used,
            'obs_tag': obs_tag,
            'fail_streak': int(getattr(self, '_fail_streak', 0)),
            # Solver-level diagnostics (new in v3b).
            'solver_pass': solver_pass,
            'solver_pass_hist': pass_hist,
            'solver_input': solver_input,
            'solver_infeas': solver_infeas,
            # Prediction freshness (independent-thread callbacks — these
            # should keep ticking even when the solver is stuck at infeasibility).
            'predict_age_s': (round(predict_age_s, 3)
                              if predict_age_s is not None else None),
            'track_age_s': (round(track_age_s, 3)
                            if track_age_s is not None else None),
            'opp_age_s': (round(opp_age_s, 3)
                          if opp_age_s is not None else None),
            'ego': {
                's': round(ego_s, 4) if ego_s == ego_s else None,
                'n': round(ego_n, 4) if ego_n == ego_n else None,
                'v': round(float(getattr(self, 'car_vx', 0.0)), 3),
                'psi': round(float(getattr(self, 'car_yaw', 0.0)), 4),
                'x': round(float(getattr(self, 'car_x', 0.0)), 3),
                'y': round(float(getattr(self, 'car_y', 0.0)), 3),
            },
            'u0': {
                'v': round(float(speed0), 3) if speed0 is not None else None,
                'steer': round(float(steer0), 4) if steer0 is not None else None,
            },
            'ref': {k: (round(v, 4) if v is not None else None)
                    for k, v in ref_stats.items()},
            'trajectory': {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in traj_stats.items()},
            'jitter_rms_m': (round(jitter_rms, 4)
                             if jitter_rms is not None else None),
            'side': getattr(self, '_last_side_str', 'n/a'),
            'side_scores': {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in (getattr(self, '_last_side_scores', {}) or {}).items()},
            'bias_scale': round(float(getattr(self, '_last_bias_scale', 0.0)), 3),
            'ticks_since_flip': int(getattr(self, '_ticks_since_flip', 0)),
            # ### HJ : v3 — TRAIL entry dwell counter (0 when not trailing)
            'trail_ticks': int(getattr(self, '_trail_ticks_since_enter', 0)),
            'n_obs_raw': int(getattr(self, '_last_n_obs_raw', 0)),
            'n_obs_used': len(obs_list),
            'obstacles': obs_list,
            'cost': {k: round(float(v), 3)
                     for k, v in (getattr(self.solver, 'last_cost_breakdown', {}) or {}).items()},
            'weights': self._current_weights_snapshot(),
            # ### HJ : Phase X — MPC FSM + painter + continuity guard fields.
            'mpc_mode': str(getattr(self, '_mpc_mode', '-')),
            'prev_mpc_mode': str(getattr(self, '_prev_mpc_mode', '-')),
            'mode_dwell': int(getattr(self, '_mode_dwell', 0)),
            'alpha_ramp': round(float(getattr(self, '_alpha_ramp', 0.0)), 3),
            'ttc_min': (round(float(getattr(self, '_last_ttc_min', float('inf'))), 3)
                        if getattr(self, '_last_ttc_min', float('inf')) != float('inf') else None),
            'obs_in_horizon': bool(getattr(self, '_last_obs_in_horizon', False)),
            'min_obs_ds': (round(float(getattr(self, '_last_min_obs_ds', float('inf'))), 3)
                           if getattr(self, '_last_min_obs_ds', float('inf')) != float('inf') else None),
            'obs_enter_dist_m': float(getattr(self, '_obs_enter_dist_m', 5.0)),
            'obs_exit_dist_m': float(getattr(self, '_obs_exit_dist_m', 10.0)),
            'pred_variance': {
                'enabled': bool(getattr(self, '_use_pred_variance', False)),
                'max_vs_var': round(float(getattr(self, '_last_obs_max_vs_var', 0.0)), 5),
                'max_vd_var': round(float(getattr(self, '_last_obs_max_vd_var', 0.0)), 5),
                'sigma_s_eff': round(float(getattr(self, '_last_sigma_s_eff', 0.0)), 4),
                'sigma_n_eff': round(float(getattr(self, '_last_sigma_n_eff', 0.0)), 4),
            },
            'ego_outside_corridor': bool(getattr(self, '_last_ego_outside_corridor', False)),
            'weight_alpha_applied': (
                round(float(getattr(self, '_last_applied_weight_alpha', 0.0)), 3)
                if getattr(self, '_last_applied_weight_alpha', None) is not None else None),
            'q_n_near_boost_applied': round(float(getattr(self, '_last_q_n_boost_applied', 0.0)), 3),
            ### HJ : 2026-04-26 — A1-A6 debug fields.
            'a1_vmax': self._build_a1_vmax_debug(),
            'a4_wall_ramp': self._build_a4_wall_ramp_debug(),
            'a5_a6_transition': {
                'post_ot_boost_ticks_left': int(getattr(self, '_post_ot_boost_ticks_left', 0)),
                'post_ot_boost_total_ticks': int(getattr(self, '_post_ot_boost_total_ticks', 0)),
                'post_ot_boost_amount': round(float(getattr(self, '_post_ot_boost_amount', 0.0)), 2),
                # Decay value at this tick (active boost contribution to q_n)
                'post_ot_boost_active': round(
                    float(self._post_ot_boost_amount
                          * self._post_ot_boost_ticks_left
                          / max(self._post_ot_boost_total_ticks, 1))
                    if self._post_ot_boost_ticks_left > 0 else 0.0, 3),
            },
            ### HJ : end
            'painter': {
                'seam_idx': int(getattr(self, '_last_painter_seam_idx', -1)),
                'blend_applied': bool(getattr(self, '_last_painter_blend_applied', False)),
                'vx_first_delta': round(float(getattr(self, '_last_painter_vx_first_delta', 0.0)), 4),
            },
            'continuity_guard': {
                'L2': round(float(getattr(self, '_last_continuity_L2', 0.0)), 4),
                'applied': bool(getattr(self, '_last_path_blend_applied', False)),
            },
            ### HJ : 2026-04-28 — expose plan name in tick_json for analysis.
            'plan': (self._last_plan.get('name')
                     if getattr(self, '_last_plan', None) else None),
        }

        try:
            self.pub_debug_tick.publish(String(data=_json.dumps(payload)))
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                5.0, '[mpc][%s] tick_json publish failed: %s',
                rospy.get_name(), e)

        # Markers
        try:
            self._publish_debug_extra_markers(
                trajectory=trajectory, ref_slice=ref_slice,
                obs_arr=obs_arr, tier=tier, status=status,
                solve_ms=solve_ms, obs_tag=obs_tag,
                n_traj_signed=n_traj_signed,
            )
        except Exception as e:  # pragma: no cover
            rospy.logwarn_throttle(
                5.0, '[mpc][%s] debug marker publish failed: %s',
                rospy.get_name(), e)

    def _publish_debug_extra_markers(self, trajectory, ref_slice, obs_arr,
                                     tier, status, solve_ms, obs_tag,
                                     n_traj_signed=None):
        """### HJ : Context markers for RViz: corridor walls (L/R), ref-slice
        centerline, obstacle blobs (per slot, propagated over horizon), and a
        tier/status TEXT_VIEW_FACING anchored at ego pose + ~0.8m up.
        All markers live in the ~debug/markers namespace with lifetime=0.2s
        so stale ones fade if the node dies."""
        if ref_slice is None and obs_arr is None and trajectory is None:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        life = rospy.Duration(0.25)

        arr = MarkerArray()

        # Always drop a DELETEALL first — keeps obstacle slot count correct
        # when a tracked obstacle disappears between ticks.
        clear = Marker()
        clear.header = header
        clear.ns = 'mpc_debug_clear'
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # ---- corridor walls -------------------------------------------------
        if ref_slice is not None:
            try:
                lp = ref_slice['left_points']
                rp = ref_slice['right_points']
                cp = ref_slice['center_points']
                # Reuse z from lifter for the midpoint; walls sit at same z.
                zs = [self.lifter._interp(
                    ref_slice['ref_s'][i] % max(self.track_length, 1e-3),
                    self.lifter.g_z)
                      for i in range(lp.shape[0])]

                def _line(ns_, mid, pts_xy, rgba, width=0.04):
                    m = Marker()
                    m.header = header
                    m.ns = ns_
                    m.id = mid
                    m.type = Marker.LINE_STRIP
                    m.action = Marker.ADD
                    m.scale.x = width
                    m.color = ColorRGBA(*rgba)
                    m.pose.orientation.w = 1.0
                    m.lifetime = life
                    for i in range(pts_xy.shape[0]):
                        p = Point()
                        p.x = float(pts_xy[i, 0])
                        p.y = float(pts_xy[i, 1])
                        p.z = float(zs[i])
                        m.points.append(p)
                    return m
                arr.markers.append(_line(
                    'mpc_corridor_left', 10, lp, (1.0, 0.0, 0.0, 0.7), 0.03))
                arr.markers.append(_line(
                    'mpc_corridor_right', 11, rp, (1.0, 0.0, 0.0, 0.7), 0.03))
                arr.markers.append(_line(
                    'mpc_ref_center', 12, cp, (0.5, 0.5, 0.5, 0.5), 0.02))
            except Exception:
                pass

        # ---- obstacle blobs (horizon-propagated) ---------------------------
        if obs_arr is not None and ref_slice is not None:
            try:
                rc = ref_slice['center_points']
                rdx = ref_slice['ref_dx']; rdy = ref_slice['ref_dy']
                rs_ref = ref_slice['ref_s']
                M = int(min(obs_arr.shape[1], rc.shape[0]))
                lnx = -rdy[:M]; lny = rdx[:M]
                mid = 100
                for o in range(obs_arr.shape[0]):
                    w_ts = obs_arr[o, :, 2]
                    if float(np.max(w_ts)) <= 0.0:
                        continue
                    # Current (k=0) + propagated (k=1..N) positions.
                    pts = Marker()
                    pts.header = header
                    pts.ns = 'mpc_obs_slot_%d' % o
                    pts.id = mid; mid += 1
                    pts.type = Marker.SPHERE_LIST
                    pts.action = Marker.ADD
                    pts.scale.x = pts.scale.y = pts.scale.z = 0.18
                    pts.color = ColorRGBA(r=1.0, g=0.2, b=1.0, a=0.85)
                    pts.pose.orientation.w = 1.0
                    pts.lifetime = life
                    # Highlight sphere at k=0 (opaque, larger).
                    head_s = float(obs_arr[o, 0, 0])
                    head_n = float(obs_arr[o, 0, 1])
                    # Map (s,n) to xy using nearest ref index.
                    for k in range(M):
                        s_o = float(obs_arr[o, k, 0])
                        n_o = float(obs_arr[o, k, 1])
                        # nearest reference-slice index by ref_s
                        kref = int(np.argmin(np.abs(rs_ref[:M] - s_o)))
                        x = rc[kref, 0] + n_o * (-rdy[kref])
                        y = rc[kref, 1] + n_o * (rdx[kref])
                        p = Point(); p.x = x; p.y = y
                        p.z = self.lifter._interp(
                            rs_ref[kref] % max(self.track_length, 1e-3),
                            self.lifter.g_z)
                        pts.points.append(p)
                    arr.markers.append(pts)

                    # Big head ball (k=0) for at-a-glance location.
                    head = Marker()
                    head.header = header
                    head.ns = 'mpc_obs_head_%d' % o
                    head.id = mid; mid += 1
                    head.type = Marker.SPHERE
                    head.action = Marker.ADD
                    head.scale.x = head.scale.y = head.scale.z = 0.35
                    head.color = ColorRGBA(r=1.0, g=0.0, b=0.8, a=0.95)
                    kref0 = int(np.argmin(np.abs(rs_ref[:M] - head_s)))
                    head.pose.position.x = float(
                        rc[kref0, 0] + head_n * (-rdy[kref0]))
                    head.pose.position.y = float(
                        rc[kref0, 1] + head_n * (rdx[kref0]))
                    head.pose.position.z = float(self.lifter._interp(
                        rs_ref[kref0] % max(self.track_length, 1e-3),
                        self.lifter.g_z))
                    head.pose.orientation.w = 1.0
                    head.lifetime = life
                    arr.markers.append(head)
            except Exception:
                pass

        # ---- tier / status text floating above ego -------------------------
        try:
            txt = Marker()
            txt.header = header
            txt.ns = 'mpc_tier_status'
            txt.id = 200
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.scale.z = 0.25
            # Color by tier — green=0, yellow=1, orange=2, red=3.
            tier_color = {
                0: (0.1, 0.9, 0.2, 1.0),
                1: (1.0, 0.9, 0.0, 1.0),
                2: (1.0, 0.5, 0.0, 1.0),
                3: (1.0, 0.1, 0.1, 1.0),
            }.get(int(tier), (0.7, 0.7, 0.7, 1.0))
            txt.color = ColorRGBA(*tier_color)
            txt.pose.position.x = float(getattr(self, 'car_x', 0.0))
            txt.pose.position.y = float(getattr(self, 'car_y', 0.0))
            txt.pose.position.z = float(getattr(self, 'car_z', 0.0)) + 0.8
            txt.pose.orientation.w = 1.0
            txt.lifetime = life
            txt.text = ('[%s] tier=%d %s | solve=%.1fms | obs=%s'
                        % (self.state, int(tier), status, float(solve_ms),
                           obs_tag))
            arr.markers.append(txt)
        except Exception:
            pass

        if arr.markers:
            self.pub_debug_markers.publish(arr)


if __name__ == '__main__':
    try:
        _node = MPCPlannerStateNode()
        rospy.on_shutdown(lambda: _node._debug_logger and _node._debug_logger.close())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
