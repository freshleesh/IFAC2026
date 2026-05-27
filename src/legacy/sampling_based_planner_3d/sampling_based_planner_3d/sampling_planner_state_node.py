#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
### HJ : State-aware variant of sampling_planner_node.py.

Mirrors observation-mode SamplingPlannerNode but:
  * ~state ∈ {overtake, recovery, observe} routes the chosen trajectory to
    the state-machine-facing topic — OTWpntArray on ~out/otwpnts for overtake,
    WpntArray on ~out/wpnts for recovery; observe keeps only debug ~best_*.
  * All cost weights + filter / resample / MPPI toggles are exposed through
    dynamic_reconfigure (cfg/SamplingCost.cfg) so rqt can tune them live per
    instance, with save_params / reset_params one-shot triggers writing back
    to ~instance_yaml.
  * Tick-to-tick continuity via (a) continuity_weight post-added to the
    upstream cost_array before re-argmin, (b) 1-pole EMA on (d, V) between
    ticks, (c) optional uniform arc-length resampling on output.
  * Translucent "previous-best" marker published alongside current best so
    tick jitter is visible in RViz.

Upstream core (LocalSamplingPlanner / Track3D / GGManager) is reused unchanged
via import.
"""

import os
import sys
import time
import copy
import json
import math
import threading
import yaml
import numpy as np

from std_msgs.msg import String, Float32, Float32MultiArray, Header
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray, Wpnt, OTWpntArray, PredictionArray, ObstacleArray


# -- Make upstream src importable -------------------------------------------------------------------
_PKG_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPSTREAM_SRC = os.path.join(_PKG_DIR, 'src')
_SHARED_SRC   = os.path.abspath(os.path.join(_PKG_DIR, '..', '3d_gb_optimizer', 'global_line', 'src'))
for _p in (_UPSTREAM_SRC, _SHARED_SRC):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

from track3D import Track3D                              # noqa: E402
from ggManager import GGManager                          # noqa: E402
from sampling_based_planner import LocalSamplingPlanner  # noqa: E402
from sampling_debug_logger import DebugLogger            # noqa: E402


_VALID_STATES = ('overtake', 'recovery', 'observe')

# ### HJ : Phase 2 mode-aware overtake. Each mode picks its own best candidate via the
# same base cost_array + per-mode side/progress post-add. Hysteresis selects which mode
# the chosen trajectory comes from. Recovery/observe states bypass the loop entirely.
_OVERTAKE_MODES = ('follow', 'ot_left', 'ot_right')
# Default opponent prediction step. The publisher (3d_opp_prediction.py) hard-codes
# `self.dt = 0.02` and emits a Prediction[] of N points at i*dt offsets — kept in sync
# here so we don't need a per-message stride parameter.
_OPP_PRED_DT_DEFAULT = 0.02


class SamplingPlannerStateNode:

    # =============================================================================================
    # Init
    # =============================================================================================
    def __init__(self):
        rospy.init_node('sampling_planner_state_node', anonymous=False)

        # -- Role ---------------------------------------------------------------------------------
        state = str(self._get_param_or_default('~state', 'overtake')).lower().strip()
        if state not in _VALID_STATES:
            self.get_logger().warning('[sampling][%s] invalid ~state=%r — falling back to "observe".',
                          rospy.get_name(), state)
            state = 'observe'
        self.state = state
        self.instance_yaml_path = self._get_param_or_default('~instance_yaml', '')

        # -- Paths --------------------------------------------------------------------------------
        # ### HJ : map_name은 3d_base_system.launch가 /map rosparam에 세팅함.
        # 런치에서 경로를 하드코딩하는 대신 /map을 읽어 경로를 동적 구성 (launch 인자 override 가능).
        self.track_csv_path      = self._get_param_or_default('~track_csv_path', '')
        self.gg_dir_path         = self._get_param_or_default('~gg_dir_path', '')
        self.vehicle_params_path = self._get_param_or_default('~vehicle_params_path', '')
        self.raceline_csv_path   = self._get_param_or_default('~raceline_csv_path', '')
        # ### HJ : vehicle_name is the 3D physics-model id, independent from racecar_version
        # (hardware id). No racecar_version fallback — empty means use default in path resolver.
        self.vehicle_name        = self._get_param_or_default('~vehicle_name', '')
        self._resolve_paths_from_map_rosparam()

        # -- Rates / IO ---------------------------------------------------------------------------
        self.rate_hz  = float(self._get_param_or_default('~rate_hz', 30.0))
        self.frame_id = self._get_param_or_default('~frame_id', 'map')

        # -- Upstream planner params --------------------------------------------------------------
        self.horizon             = float(self._get_param_or_default('~horizon', 2.0))
        self.num_samples         = int(self._get_param_or_default('~num_samples', 30))
        self.n_samples           = int(self._get_param_or_default('~n_samples', 11))
        self.v_samples           = int(self._get_param_or_default('~v_samples', 5))
        self.safety_distance     = float(self._get_param_or_default('~safety_distance', 0.20))
        self.gg_abs_margin       = float(self._get_param_or_default('~gg_abs_margin', 0.0))
        self.gg_margin_rel       = float(self._get_param_or_default('~gg_margin_rel', 0.0))
        self.friction_check_2d   = bool(self._get_param_or_default('~friction_check_2d', False))
        self.relative_generation = bool(self._get_param_or_default('~relative_generation', True))
        self.s_dot_min           = float(self._get_param_or_default('~s_dot_min', 1.5))
        self.kappa_thr           = float(self._get_param_or_default('~kappa_thr', 1.5))

        # -- Cost weights (live-tuneable) --------------------------------------------------------
        self.w_raceline   = float(self._get_param_or_default('~cost/raceline_weight',   0.1))
        self.w_velocity   = float(self._get_param_or_default('~cost/velocity_weight',   100.0))
        self.w_prediction = float(self._get_param_or_default('~cost/prediction_weight', 5000.0))
        self.w_continuity = float(self._get_param_or_default('~cost/continuity_weight', 50.0))
        self.w_boundary   = float(self._get_param_or_default('~cost/boundary_weight',   0.0))

        # -- Phase 2: mode-aware overtake + boundary safety --------------------------------------
        # ### HJ : safety_margin_m is the keep-out band INSIDE the half-width.
        # side / progress / hysteresis only act in state=overtake; recovery uses 0.
        self.safety_margin_m   = float(self._get_param_or_default('~cost/safety_margin_m',   0.15))
        self.w_side            = float(self._get_param_or_default('~cost/side_weight',       100.0))
        self.w_progress        = float(self._get_param_or_default('~cost/progress_weight',     5.0))
        self.opp_pred_ttl_s    = float(self._get_param_or_default('~cost/opp_pred_ttl_s',     0.5))
        self.hysteresis_ttl_s  = float(self._get_param_or_default('~hysteresis/ttl_s',        1.0))
        self.hysteresis_margin = float(self._get_param_or_default('~hysteresis/margin',       0.3))
        self.kappa_dot_max     = float(self._get_param_or_default('~hysteresis/kappa_dot_max', 5.0))
        # ### HJ : detour 항 — Σ(n - n_rl)² · dt, OT mode 가 너무 돌면 손해 페널티.
        # endpoint_chi_raceline_only — sampling 의 endpoint heading 을 raceline tangent 로
        # 고정 (default False = 원본 boundary-interp 동작 유지).
        self.w_detour                    = float(self._get_param_or_default('~cost/detour_weight',           0.0))
        self.endpoint_chi_raceline_only  = bool(self._get_param_or_default('~endpoint_chi_raceline_only', False))

        # Mode hysteresis state — only used in state=overtake.
        self._active_mode      = None
        self._mode_lock_until  = rospy.Time(0)
        # Tick counter (used by DebugLogger and snapshot stride).
        self._tick_idx         = 0

        # Opponent prediction cache. Single-opponent today (PredictionArray.id is at the outer
        # level), but kept as a dict so multi-opponent extension is a drop-in change.
        self._opp_predictions  = {}     # id → {'t', 's', 'n', 'stamp'}
        self._opp_obstacles    = {}     # id → {'is_static', 's_var', 'd_var', 'vs', 'stamp'}
        self._opp_lock         = threading.Lock()

        # -- EMA between ticks -------------------------------------------------------------------
        self.filter_alpha = float(self._get_param_or_default('~filter_alpha', 0.7))

        # -- MPPI --------------------------------------------------------------------------------
        self.mppi_enable          = bool(self._get_param_or_default('~mppi/enable',          True))
        self.mppi_temperature_rel = float(self._get_param_or_default('~mppi/temperature_rel', 0.25))
        self.mppi_temporal_weight = float(self._get_param_or_default('~mppi/temporal_weight', 0.0))
        self._prev_blended_s = None
        self._prev_blended_n = None
        # ### HJ : MPPI blending diagnostics (filled by _mppi_blend, read in _collect_tick_fields).
        self._last_mppi_stats = {}

        # -- Output resampling -------------------------------------------------------------------
        self.resample_enable = bool(self._get_param_or_default('~resample_enable', True))
        self.resample_ds     = float(self._get_param_or_default('~resample_ds_m',   0.1))

        # -- Previous-tick best (for continuity cost + EMA) ---------------------------------------
        # Stored in unwrapped / raw form (pre-filter, pre-resample) so subsequent-tick comparisons
        # measure the planner's *own* tick jitter rather than filtered output variance.
        self._prev_best_s     = None
        self._prev_best_n     = None
        self._prev_best_V     = None
        self._prev_best_xyz   = None   # (xs, ys, zs) for the translucent marker

        # ### HJ : n_end rate constraint — enforce |Δn_end| ≤ n_end_rate_cap per tick
        # so candidate selection can't teleport laterally. If no valid candidate
        # meets rate cap, fall back to "closest-to-prev" (move as much as possible
        # toward prev n_end) so trajectory still continuous, not a jump.
        self._prev_chosen_n_end = None
        self.n_end_rate_cap     = float(self._get_param_or_default('~n_end_rate_cap', 0.12))  # m / tick
        # ### HJ : soft-relax rate cap — instead of hard-filtering out candidates
        # outside the cap (which produces best_idx=-1 → rollback to prev traj,
        # visually the car follows a stale path), add a quadratic penalty on
        # |Δn_end| − cap so argmin still picks the "least bad" candidate.
        # Penalty is applied in normalized cost space (rate_penalty = w_rate *
        # (excess/cap)²) so it's commensurate with existing normalized terms.
        self.w_rate_penalty     = float(self._get_param_or_default('~w_rate_penalty', 5.0))

        # -- OT defaults -------------------------------------------------------------------------
        self.ot_side_default = str(self._get_param_or_default('~ot_side_default', 'right'))

        # -- Vehicle params ----------------------------------------------------------------------
        with open(self.vehicle_params_path, 'r') as f:
            vehicle_yml = yaml.safe_load(f)
        self.params = dict(vehicle_yml)
        self.get_logger().info('[sampling][%s] loaded vehicle params from %s',
                      rospy.get_name(), self.vehicle_params_path)

        # -- Upstream core -----------------------------------------------------------------------
        self.get_logger().info('[sampling][%s] loading Track3D from %s', rospy.get_name(), self.track_csv_path)
        self.track = Track3D(path=self.track_csv_path)

        self.get_logger().info('[sampling][%s] loading gg-diagrams from %s', rospy.get_name(), self.gg_dir_path)
        self.gg = GGManager(gg_path=self.gg_dir_path, gg_margin=self.gg_margin_rel)

        self.planner = LocalSamplingPlanner(
            params=self.params,
            track_handler=self.track,
            gg_handler=self.gg,
        )
        self.get_logger().info('[sampling][%s] LocalSamplingPlanner ready (state=%s).',
                      rospy.get_name(), self.state)

        self.get_logger().info(
            '[sampling][%s][track3d] N=%d  s=[%.3f..%.3f]',
            rospy.get_name(), len(self.track.s),
            float(self.track.s[0]), float(self.track.s[-1]),
        )

        # -- Reference raceline ------------------------------------------------------------------
        self.raceline_dict = self._load_raceline_dict()
        _rs = self.raceline_dict['s']
        _rv = self.raceline_dict['V']
        self.get_logger().info(
            '[sampling][%s][raceline] N=%d  s=[%.3f..%.3f]  V=[%.2f..%.2f]',
            rospy.get_name(), len(_rs), float(_rs[0]), float(_rs[-1]),
            float(np.min(_rv)), float(np.max(_rv)),
        )

        # -- State caches -----------------------------------------------------------------------
        self._cur_x = None
        self._cur_y = None
        self._cur_z = None
        self._cur_vs = None
        self._cur_yaw = None   # ### HJ : filled in _cb_odom, used only by ~debug/tick_json
        self._prev_s_cent = None
        self._prev_xyz    = None
        self._prev_stamp  = None
        self.debug_jump_threshold_m = float(self._get_param_or_default('~debug_jump_threshold_m', 2.0))

        # -- Subscribers ------------------------------------------------------------------------
        self.create_subscription(Odometry, '/car_state/odom', self._cb_odom, 10)
        # ### HJ : Phase 2 — opponent prediction wiring. Currently only one opponent is
        # published (3d_opp_prediction.py id-at-outer-level), but the cache keeps it keyed
        # so multi-opponent works with the same _build_prediction_dict.
        rospy.Subscriber('/opponent_prediction/obstacles_pred',
                         PredictionArray, self._cb_opp_pred, queue_size=2)
        rospy.Subscriber('/opponent_prediction/obstacles',
                         ObstacleArray, self._cb_opp_obs,  queue_size=2)

        # -- Publishers (debug — always) ---------------------------------------------------------
        self.pub_wpnts        = self.create_publisher(WpntArray, '~best_trajectory', 1)
        self.pub_best_sample  = self.create_publisher(Path, '~best_sample', 1)
        self.pub_best_markers = self.create_publisher(MarkerArray, '~best_sample/markers', 10)
        self.pub_vel_markers  = self.create_publisher(MarkerArray, '~best_sample/vel_markers', 10)
        self.pub_candidates   = self.create_publisher(MarkerArray, '~candidates', 10)
        self.pub_prev_marker  = self.create_publisher(MarkerArray, '~prev_best/markers', 10)
        self.pub_status       = rospy.Publisher('~status',  String,  queue_size=1, latch=True)
        self.pub_timing       = self.create_publisher(Float32, '~timing_ms', 1)
        # ### HJ : Phase 2 debug topics — observable mode lock state for tuning.
        self.pub_active_mode  = rospy.Publisher('~active_mode', String, queue_size=1, latch=True)
        self.pub_mode_costs   = self.create_publisher(Float32MultiArray, '~mode_costs', 1)
        # ### HJ : single-stream tick diagnostics as JSON-in-String. Consumed by
        # `rostopic echo -n 1 ~/debug/tick_json` (CLI or offline Claude analysis).
        # Contains per-tick cost stats, invalid-reason counts, best-candidate metrics,
        # opp/mode/timing. Schema: see _build_tick_json.
        self.pub_tick_json    = self.create_publisher(String, '~debug/tick_json', 1)

        # -- Publishers (role-specific) ---------------------------------------------------------
        if self.state == 'overtake':
            self.pub_out = self.create_publisher(OTWpntArray, '~out/otwpnts', 1)
        elif self.state == 'recovery':
            self.pub_out = self.create_publisher(WpntArray, '~out/wpnts', 1)
        else:
            self.pub_out = None

        # -- dynamic_reconfigure -----------------------------------------------------------------
        # Seed rqt sliders with the rosparam-loaded values — otherwise they snap to .cfg defaults
        # on first connect and silently overwrite whatever was in the YAML.
        # ### HJ : _gg_lock guards swaps of self.gg / self.planner.gg_handler from the dynreg
        # thread so the planner tick never observes a half-rebuilt handler.
        self._gg_lock = threading.Lock()
        # ### HJ : suppress FIRST callback so the .cfg-defaults-seeded initial invocation
        # doesn't overwrite YAML-loaded self.* (prediction_weight etc.). _weight_cb
        # early-returns while suppressed; _push_params_to_dynreg then seeds the server
        # with our YAML values.
        self._suppress_dynreg_cb = True
        self._dyn_srv = Server(SamplingCostConfig, self._weight_cb)
        self._push_params_to_dynreg()
        self._suppress_dynreg_cb = False

        # -- DebugLogger -------------------------------------------------------------------------
        # ### HJ : per-tick CSV + event JSONL + meta YAML 로 OT 튜닝 분석용 데이터 dump.
        # rqt 슬라이더 변경은 _weight_cb 에서 param_changed 이벤트로 즉시 events.jsonl 에 기록.
        self._logger = None
        self._init_debug_logger()

        self._publish_status('INIT_OK')
        self.get_logger().info(
            '[sampling][%s] ready — state=%s  rl=%.3f v=%.3f p=%.3f c=%.3f b=%.3f  '
            'α=%.2f  mppi=%s/%.3f/%.1f  resample=%s/%.2f',
            rospy.get_name(), self.state,
            self.w_raceline, self.w_velocity, self.w_prediction,
            self.w_continuity, self.w_boundary,
            self.filter_alpha,
            self.mppi_enable, self.mppi_temperature_rel, self.mppi_temporal_weight,
            self.resample_enable, self.resample_ds,
        )

    # =============================================================================================
    # Path resolution (/map rosparam)
    # =============================================================================================
    def _resolve_paths_from_map_rosparam(self):
        """### HJ : 경로 fallback 해석기.
        launch에서 ~track_csv_path / ~gg_dir_path / ~vehicle_params_path / ~raceline_csv_path를
        전부 하드코딩하는 대신, 비어 있으면 /map (3d_base_system.launch line 27)에서 map 이름을
        읽어 stack_master/maps/<map>/<map>_3d_* 및 global_line_3d/data/* 표준 경로로 동적 구성.

        우선순위: 명시 launch arg(~*_path) > /map rosparam 기반 자동 구성.
        """
        rp = rospkg.RosPack()

        need_paths = not all([self.track_csv_path, self.gg_dir_path, self.vehicle_params_path])
        if not need_paths and self.raceline_csv_path:
            return

        map_name = self._get_param_or_default('/map', '')
        if not map_name:
            map_name = self._get_param_or_default('~map_name', '')
        if not map_name:
            raise RuntimeError(
                '[sampling] cannot resolve paths: /map rosparam empty and ~map_name unset. '
                'Either launch 3d_base_system.launch first (sets /map) or pass ~map_name/~*_path.')

        # ### HJ : vehicle_name is a 3D physics-model id, separate from racecar_version
        # (hardware id). Default to 'rc_car_10th' when unset — NO racecar_version fallback.
        vehicle_name = self.vehicle_name or 'rc_car_10th'
        if not self.vehicle_name:
            self.get_logger().warning("[sampling] ~vehicle_name unset — defaulting to 'rc_car_10th'. "
                          "Pass vehicle_name:=<id> on the launch cmdline to override.")
            self.vehicle_name = vehicle_name
        try:
            stack_master_dir = rp.get_path('stack_master')
            global_line_3d_dir = rp.get_path('global_line_3d')
        except rospkg.ResourceNotFound as e:
            raise RuntimeError('[sampling] rospack missing required package: %s' % e)

        map_dir = os.path.join(stack_master_dir, 'maps', map_name)

        if not self.track_csv_path:
            self.track_csv_path = os.path.join(map_dir, '%s_3d_smoothed.csv' % map_name)
        if not self.raceline_csv_path:
            self.raceline_csv_path = os.path.join(
                map_dir, '%s_3d_%s_timeoptimal.csv' % (map_name, vehicle_name))
        if not self.gg_dir_path:
            self.gg_dir_path = os.path.join(
                global_line_3d_dir, 'data', 'gg_diagrams', vehicle_name, 'velocity_frame')
        if not self.vehicle_params_path:
            self.vehicle_params_path = os.path.join(
                global_line_3d_dir, 'data', 'vehicle_params', 'params_%s.yml' % vehicle_name)

        self.get_logger().info('[sampling] resolved paths from /map=%s (vehicle=%s):', map_name, vehicle_name)
        self.get_logger().info('[sampling]   track    = %s', self.track_csv_path)
        self.get_logger().info('[sampling]   raceline = %s', self.raceline_csv_path)
        self.get_logger().info('[sampling]   gg_dir   = %s', self.gg_dir_path)
        self.get_logger().info('[sampling]   params   = %s', self.vehicle_params_path)

    # =============================================================================================
    # dynamic_reconfigure
    # =============================================================================================
    def _push_params_to_dynreg(self):
        """Push our current member values into the dynreg server so rqt's initial view matches
        the YAML-loaded params instead of the .cfg defaults."""
        self._suppress_dynreg_cb = True
        try:
            self._dyn_srv.update_configuration({
                'raceline_weight':      float(self.w_raceline),
                'velocity_weight':      float(self.w_velocity),
                'prediction_weight':    float(self.w_prediction),
                'continuity_weight':    float(self.w_continuity),
                'boundary_weight':      float(self.w_boundary),
                'filter_alpha':         float(self.filter_alpha),
                'mppi_enable':          bool(self.mppi_enable),
                'mppi_temperature':     float(self.mppi_temperature_rel),
                'mppi_temporal_weight': float(self.mppi_temporal_weight),
                'resample_enable':      bool(self.resample_enable),
                'resample_ds_m':        float(self.resample_ds),
                'horizon':              float(self.horizon),
                'num_samples':          int(self.num_samples),
                'n_samples':            int(self.n_samples),
                'v_samples':            int(self.v_samples),
                'safety_distance':      float(self.safety_distance),
                'gg_abs_margin':        float(self.gg_abs_margin),
                'friction_check_2d':    bool(self.friction_check_2d),
                's_dot_min':            float(self.s_dot_min),
                'kappa_thr':            float(self.kappa_thr),
                'relative_generation':  bool(self.relative_generation),
                'gg_margin_rel':        float(self.gg_margin_rel),
                # Phase 2
                'safety_margin_m':      float(self.safety_margin_m),
                'side_weight':          float(self.w_side),
                'progress_weight':      float(self.w_progress),
                'opp_pred_ttl_s':       float(self.opp_pred_ttl_s),
                'hysteresis_ttl_s':     float(self.hysteresis_ttl_s),
                'hysteresis_margin':    float(self.hysteresis_margin),
                'kappa_dot_max':        float(self.kappa_dot_max),
                'detour_weight':                 float(self.w_detour),
                'endpoint_chi_raceline_only':    bool(self.endpoint_chi_raceline_only),
                'save_params':          False,
                'reset_params':         False,
            })
        finally:
            self._suppress_dynreg_cb = False

    def _weight_cb(self, config, level):
        # ### HJ : YAML-overwrite guard. When suppressed we're inside Server init or
        # _push_params_to_dynreg — self.* already hold the authoritative YAML values,
        # so don't let config's (possibly .cfg-default-seeded) fields overwrite them.
        if self._suppress_dynreg_cb:
            return config
        # save / reset one-shot triggers
        if config.save_params:
            self._save_yaml(config)
            config.save_params = False
        if config.reset_params:
            # ### HJ : Mutate `config` in place and let the server publish it on return.
            # Calling self._dyn_srv.update_configuration() from inside the callback
            # re-enters _weight_cb and the outer return path then overwrites those
            # values with the pre-reload `config` — which is why rqt showed no change.
            new_cfg = self._reload_yaml()
            if new_cfg is not None:
                for k, v in new_cfg.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
                self.get_logger().info('[sampling][%s][reset] reloaded YAML: %s',
                              rospy.get_name(), self.instance_yaml_path)
            config.reset_params = False

        # normal parameter updates — cost / filter / mppi / resample (hot, no rebuild)
        self.w_raceline          = float(config.raceline_weight)
        self.w_velocity          = float(config.velocity_weight)
        self.w_prediction        = float(config.prediction_weight)
        self.w_continuity        = float(config.continuity_weight)
        self.w_boundary          = float(config.boundary_weight)
        self.filter_alpha        = float(config.filter_alpha)
        self.mppi_enable         = bool(config.mppi_enable)
        self.mppi_temperature_rel = float(config.mppi_temperature)
        self.mppi_temporal_weight = float(config.mppi_temporal_weight)
        self.resample_enable     = bool(config.resample_enable)
        self.resample_ds         = float(config.resample_ds_m)

        # ### HJ : sampling grid / horizon / safety — forwarded per tick into
        # calc_trajectory(), so just overwriting self.* is enough. No rebuild.
        self.horizon             = float(config.horizon)
        self.num_samples         = int(config.num_samples)
        self.n_samples           = int(config.n_samples)
        self.v_samples           = int(config.v_samples)
        self.safety_distance     = float(config.safety_distance)
        self.gg_abs_margin       = float(config.gg_abs_margin)
        self.friction_check_2d   = bool(config.friction_check_2d)
        self.s_dot_min           = float(config.s_dot_min)
        self.kappa_thr           = float(config.kappa_thr)
        self.relative_generation = bool(config.relative_generation)

        # ### HJ : gg_margin_rel — baked into GGManager's CasADi interpolants at
        # construction time. Detect change and rebuild; swap handler into planner.
        new_gg_margin = float(config.gg_margin_rel)
        if abs(new_gg_margin - self.gg_margin_rel) > 1e-9:
            self._rebuild_gg_manager(new_gg_margin)

        # Phase 2 — boundary / mode-aware / hysteresis (all hot, no rebuild).
        self.safety_margin_m   = float(config.safety_margin_m)
        self.w_side            = float(config.side_weight)
        self.w_progress        = float(config.progress_weight)
        self.opp_pred_ttl_s    = float(config.opp_pred_ttl_s)
        self.hysteresis_ttl_s  = float(config.hysteresis_ttl_s)
        self.hysteresis_margin = float(config.hysteresis_margin)
        self.kappa_dot_max     = float(config.kappa_dot_max)
        self.w_detour                   = float(config.detour_weight)
        self.endpoint_chi_raceline_only = bool(config.endpoint_chi_raceline_only)

        if not self._suppress_dynreg_cb:
            rospy.loginfo_throttle(
                2.0,
                '[sampling][%s] tune rl=%.3f v=%.2f p=%.1f c=%.2f b=%.2f  α=%.2f  '
                'mppi=%s/%.3f/%.1f  resample=%s/%.2f  '
                'H=%.2f N=%d/%d/%d safe=%.2f gg_abs=%.2f f2d=%s sdotmin=%.2f κthr=%.2f rel=%s '
                'gg_rel=%.3f',
                rospy.get_name(),
                self.w_raceline, self.w_velocity, self.w_prediction,
                self.w_continuity, self.w_boundary, self.filter_alpha,
                self.mppi_enable, self.mppi_temperature_rel, self.mppi_temporal_weight,
                self.resample_enable, self.resample_ds,
                self.horizon, self.num_samples, self.n_samples, self.v_samples,
                self.safety_distance, self.gg_abs_margin, self.friction_check_2d,
                self.s_dot_min, self.kappa_thr, self.relative_generation,
                self.gg_margin_rel,
            )
        # ### HJ : rqt 슬라이더 변경 즉시 events.jsonl 에 기록 (init 단계에선 logger 가
        # 아직 없을 수 있고, _push_params_to_dynreg 가 부르면 _suppress_dynreg_cb=True 라
        # 여기까지 안 옴 → 자연스럽게 startup 자기자신 이벤트는 안 찍힘).
        try:
            self._emit_param_diff_events()
        except Exception:
            pass
        return config

    def _rebuild_gg_manager(self, new_gg_margin):
        """### HJ : Rebuild GGManager (= re-JIT CasADi interpolants) with a new relative
        margin and swap it into the live LocalSamplingPlanner. Runs on the dynreg thread.

        Blocks the planner tick via _gg_lock so calc_trajectory never sees a half-built
        handler. Typical cost: ~100-300 ms depending on the gg-diagram grid size.
        """
        t0 = self.get_clock().now().to_msg()
        self.get_logger().info('[sampling][%s] rebuilding GGManager: gg_margin_rel %.3f → %.3f',
                      rospy.get_name(), self.gg_margin_rel, new_gg_margin)
        try:
            new_gg = GGManager(gg_path=self.gg_dir_path, gg_margin=new_gg_margin)
        except Exception as e:
            self.get_logger().error('[sampling][%s] GGManager rebuild failed, keeping old: %s',
                         rospy.get_name(), e)
            return
        with self._gg_lock:
            self.gg = new_gg
            self.planner.gg_handler = new_gg
            self.gg_margin_rel = new_gg_margin
        dt_ms = (self.get_clock().now().to_msg() - t0).to_sec() * 1000.0
        self.get_logger().info('[sampling][%s] GGManager rebuilt in %.1f ms', rospy.get_name(), dt_ms)

    def _save_yaml(self, config):
        if not self.instance_yaml_path:
            self.get_logger().warning('[sampling][%s][save] ~instance_yaml not set — skipping',
                          rospy.get_name())
            return
        try:
            if os.path.exists(self.instance_yaml_path):
                with open(self.instance_yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
            # cost weights
            data.setdefault('cost', {})
            data['cost']['raceline_weight']   = float(config.raceline_weight)
            data['cost']['velocity_weight']   = float(config.velocity_weight)
            data['cost']['prediction_weight'] = float(config.prediction_weight)
            data['cost']['continuity_weight'] = float(config.continuity_weight)
            data['cost']['boundary_weight']   = float(config.boundary_weight)
            # Phase 2 — mode-aware overtake fields live alongside the base cost block.
            data['cost']['safety_margin_m']   = float(config.safety_margin_m)
            data['cost']['side_weight']       = float(config.side_weight)
            data['cost']['progress_weight']   = float(config.progress_weight)
            data['cost']['opp_pred_ttl_s']    = float(config.opp_pred_ttl_s)
            data.setdefault('hysteresis', {})
            data['hysteresis']['ttl_s']         = float(config.hysteresis_ttl_s)
            data['hysteresis']['margin']        = float(config.hysteresis_margin)
            data['hysteresis']['kappa_dot_max'] = float(config.kappa_dot_max)
            data['cost']['detour_weight']       = float(config.detour_weight)
            data['endpoint_chi_raceline_only']  = bool(config.endpoint_chi_raceline_only)
            # sampling grid / horizon / safety (top-level, matches YAML layout)
            data['horizon']             = float(config.horizon)
            data['num_samples']         = int(config.num_samples)
            data['n_samples']           = int(config.n_samples)
            data['v_samples']           = int(config.v_samples)
            data['safety_distance']     = float(config.safety_distance)
            data['gg_abs_margin']       = float(config.gg_abs_margin)
            data['gg_margin_rel']       = float(config.gg_margin_rel)
            data['friction_check_2d']   = bool(config.friction_check_2d)
            data['s_dot_min']           = float(config.s_dot_min)
            data['kappa_thr']           = float(config.kappa_thr)
            data['relative_generation'] = bool(config.relative_generation)
            # filter / mppi / resample
            data['filter_alpha'] = float(config.filter_alpha)
            data.setdefault('mppi', {})
            data['mppi']['enable']          = bool(config.mppi_enable)
            data['mppi']['temperature_rel'] = float(config.mppi_temperature)
            data['mppi']['temporal_weight'] = float(config.mppi_temporal_weight)
            data['resample_enable'] = bool(config.resample_enable)
            data['resample_ds_m']   = float(config.resample_ds_m)
            with open(self.instance_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info('[sampling][%s][save] YAML updated: %s',
                          rospy.get_name(), self.instance_yaml_path)
        except Exception as e:
            self.get_logger().error('[sampling][%s][save] failed: %s', rospy.get_name(), e)

    def _reload_yaml(self):
        if not self.instance_yaml_path or not os.path.exists(self.instance_yaml_path):
            self.get_logger().warning('[sampling][%s][reset] ~instance_yaml missing — skip',
                          rospy.get_name())
            return None
        try:
            with open(self.instance_yaml_path, 'r') as f:
                d = yaml.safe_load(f) or {}
            cost = d.get('cost', {}) or {}
            mppi = d.get('mppi', {}) or {}
            hyst = d.get('hysteresis', {}) or {}
            return {
                'raceline_weight':      float(cost.get('raceline_weight',   self.w_raceline)),
                'velocity_weight':      float(cost.get('velocity_weight',   self.w_velocity)),
                'prediction_weight':    float(cost.get('prediction_weight', self.w_prediction)),
                'continuity_weight':    float(cost.get('continuity_weight', self.w_continuity)),
                'boundary_weight':      float(cost.get('boundary_weight',   self.w_boundary)),
                'safety_margin_m':      float(cost.get('safety_margin_m',   self.safety_margin_m)),
                'side_weight':          float(cost.get('side_weight',       self.w_side)),
                'progress_weight':      float(cost.get('progress_weight',   self.w_progress)),
                'opp_pred_ttl_s':       float(cost.get('opp_pred_ttl_s',    self.opp_pred_ttl_s)),
                'hysteresis_ttl_s':     float(hyst.get('ttl_s',             self.hysteresis_ttl_s)),
                'hysteresis_margin':    float(hyst.get('margin',            self.hysteresis_margin)),
                'kappa_dot_max':        float(hyst.get('kappa_dot_max',     self.kappa_dot_max)),
                'detour_weight':                float(cost.get('detour_weight',           self.w_detour)),
                'endpoint_chi_raceline_only':   bool(d.get('endpoint_chi_raceline_only',  self.endpoint_chi_raceline_only)),
                'filter_alpha':         float(d.get('filter_alpha', self.filter_alpha)),
                'mppi_enable':          bool(mppi.get('enable',          self.mppi_enable)),
                'mppi_temperature':     float(mppi.get('temperature_rel', self.mppi_temperature_rel)),
                'mppi_temporal_weight': float(mppi.get('temporal_weight', self.mppi_temporal_weight)),
                'resample_enable':      bool(d.get('resample_enable', self.resample_enable)),
                'resample_ds_m':        float(d.get('resample_ds_m',   self.resample_ds)),
                'horizon':              float(d.get('horizon',             self.horizon)),
                'num_samples':          int(d.get('num_samples',           self.num_samples)),
                'n_samples':            int(d.get('n_samples',             self.n_samples)),
                'v_samples':            int(d.get('v_samples',             self.v_samples)),
                'safety_distance':      float(d.get('safety_distance',     self.safety_distance)),
                'gg_abs_margin':        float(d.get('gg_abs_margin',       self.gg_abs_margin)),
                'friction_check_2d':    bool(d.get('friction_check_2d',    self.friction_check_2d)),
                's_dot_min':            float(d.get('s_dot_min',           self.s_dot_min)),
                'kappa_thr':            float(d.get('kappa_thr',           self.kappa_thr)),
                'relative_generation':  bool(d.get('relative_generation',  self.relative_generation)),
                'gg_margin_rel':        float(d.get('gg_margin_rel',       self.gg_margin_rel)),
                'save_params':          False,
                'reset_params':         False,
            }
        except Exception as e:
            self.get_logger().error('[sampling][%s][reset] parse failed: %s', rospy.get_name(), e)
            return None

    # =============================================================================================
    # Helpers (copied from sampling_planner_node.py — identical logic)
    # =============================================================================================
    def _load_raceline_dict(self):
        if self.raceline_csv_path and os.path.exists(self.raceline_csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(self.raceline_csv_path, comment='#')
                self.get_logger().info('[sampling][%s] loaded raceline CSV: %d rows, cols=%s',
                              rospy.get_name(), len(df), list(df.columns))

                def _pick(candidates, fallback=None):
                    low = {c.lower(): c for c in df.columns}
                    for k in candidates:
                        if k.lower() in low:
                            return df[low[k.lower()]].to_numpy()
                    return fallback

                s = _pick(['s_opt', 's_m', 's'])
                v = _pick(['v_opt', 'vx_mps', 'v', 'vs'])
                if s is None or v is None:
                    raise KeyError(
                        "required s/v columns not found in CSV; got {}".format(list(df.columns))
                    )
                n   = _pick(['n_opt', 'd_m', 'n'],      fallback=np.zeros_like(s))
                chi = _pick(['chi_opt', 'chi'],          fallback=np.zeros_like(s))
                ax  = _pick(['ax_opt', 'ax_mps2', 'ax'], fallback=np.zeros_like(s))
                ay  = _pick(['ay_opt', 'ay_mps2', 'ay'], fallback=np.zeros_like(s))

                # s[-1] pin — see sampling_planner_node.py for full rationale.
                L_track = float(self.track.s[-1])
                EPS_S   = 1e-6
                target_last = L_track - EPS_S
                if len(s) >= 2 and s[-1] != target_last:
                    self.get_logger().warning(
                        '[sampling][%s] pinning raceline s[-1]: %.9f → %.9f  (track L=%.9f)',
                        rospy.get_name(), float(s[-1]), target_last, L_track,
                    )
                    s = s.copy()
                    s[-1] = target_last

                v_safe = np.clip(v, 1e-3, None)
                ds = np.diff(s, prepend=s[0])
                t  = np.cumsum(ds / v_safe)
                s_ddot = np.gradient(v, s)
                z = np.zeros_like(s)
                return {
                    't':      t,
                    's':      s,
                    's_dot':  v,
                    's_ddot': s_ddot,
                    'n':      n,
                    'n_dot':  z,
                    'n_ddot': z,
                    'V':      v,
                    'chi':    chi,
                    'ax':     ax,
                    'ay':     ay,
                }
            except Exception as e:
                self.get_logger().warning('[sampling][%s] raceline CSV parse failed (%s) — centerline fallback.',
                              rospy.get_name(), e)

        self.get_logger().warning('[sampling][%s] no raceline CSV — centerline + 5 m/s fallback.',
                      rospy.get_name())
        L = float(self.track.s[-1])
        s = np.linspace(0.0, L, 500)
        v = np.full_like(s, 5.0)
        t = s / 5.0
        z = np.zeros_like(s)
        return {
            't': t,
            's': s, 's_dot': v, 's_ddot': z,
            'n': z, 'n_dot': z, 'n_ddot': z,
            'V': v, 'chi': z, 'ax': z, 'ay': z,
        }

    def _publish_status(self, msg):
        self.pub_status.publish(String(data=msg))

    def _cb_odom(self, msg):
        self._cur_x  = float(msg.pose.pose.position.x)
        self._cur_y  = float(msg.pose.pose.position.y)
        self._cur_z  = float(msg.pose.pose.position.z)
        self._cur_vs = float(msg.twist.twist.linear.x)
        # ### HJ : yaw from quaternion — used only for ~debug/tick_json's chi_raceline_rel.
        # Not feeding back into planner boundary conditions (Fix-2 does that separately).
        q = msg.pose.pose.orientation
        # quaternion → yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._cur_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _cb_opp_pred(self, msg):
        """### HJ : Cache one PredictionArray per opponent id. The publisher
        ([3d_opp_prediction.py](../../prediction/gp_traj_predictor/src/3d_opp_prediction.py))
        emits a fixed-stride Prediction[] (i*dt offsets, dt=0.02s) so we just
        materialise (t, s, n) arrays here. Stale predictions are dropped by
        opp_pred_ttl_s in _build_prediction_dict.
        """
        if not msg.predictions:
            return
        n = len(msg.predictions)
        t_arr = np.arange(n, dtype=np.float64) * _OPP_PRED_DT_DEFAULT
        s_arr = np.fromiter((p.pred_s for p in msg.predictions), dtype=np.float64, count=n)
        n_arr = np.fromiter((p.pred_d for p in msg.predictions), dtype=np.float64, count=n)
        with self._opp_lock:
            self._opp_predictions[int(msg.id)] = {
                't':     t_arr,
                's':     s_arr,
                'n':     n_arr,
                'stamp': msg.header.stamp if msg.header.stamp.to_sec() > 0 else self.get_clock().now().to_msg(),
            }

    def _cb_opp_obs(self, msg):
        """### HJ : Cache the static / dynamic flag and Frenet variances per opponent
        for future sigma-aware Gaussian inflation. Today this only seeds opp metadata
        — the cost weights ignore is_static, but mode hysteresis can use it later."""
        if not msg.obstacles:
            return
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else self.get_clock().now().to_msg()
        with self._opp_lock:
            for ob in msg.obstacles:
                self._opp_obstacles[int(ob.id)] = {
                    'is_static': bool(ob.is_static),
                    's_var':     float(getattr(ob, 's_var', 0.0)),
                    'd_var':     float(getattr(ob, 'd_var', 0.0)),
                    'vs':        float(getattr(ob, 'vs', 0.0)),
                    'stamp':     stamp,
                }

    def _build_prediction_dict(self):
        """Drop stale entries (>= opp_pred_ttl_s old) and return the upstream-friendly
        `{id: {'t', 's', 'n'}}` dict consumed by LocalSamplingPlanner.calc_trajectory.
        Returns an empty dict when no fresh predictions are available — preserves the
        existing observation-mode behavior in that case."""
        now = self.get_clock().now().to_msg()
        ttl = max(0.05, float(self.opp_pred_ttl_s))
        out = {}
        dropped = []
        with self._opp_lock:
            for opp_id, entry in list(self._opp_predictions.items()):
                age = (now - entry['stamp']).to_sec()
                if age >= ttl:
                    dropped.append((int(opp_id), float(age)))
                    self._opp_predictions.pop(opp_id, None)
                    continue
                out[opp_id] = {'t': entry['t'], 's': entry['s'], 'n': entry['n']}
        if dropped:
            self._log_event_safe('prediction_stale', {
                't': time.time(), 'dropped': dropped, 'ttl_s': ttl,
            })
        return out

    def _cart_to_cl_frenet_exact(self, x, y, z, debug=False):
        """Project (x,y,z) onto Track3D centerline → (s_cent, n_cent).
        Identical to the observation node implementation."""
        xs    = np.asarray(self.track.x)
        ys    = np.asarray(self.track.y)
        zs    = np.asarray(self.track.z)
        s_arr = np.asarray(self.track.s)
        L = float(s_arr[-1])
        N = len(xs)

        d2 = (xs - x) ** 2 + (ys - y) ** 2 + (zs - z) ** 2
        i = int(np.argmin(d2))

        best_s, best_d2 = None, np.inf
        for ja, jb in (((i - 1) % N, i), (i, (i + 1) % N)):
            xa, ya, za = xs[ja], ys[ja], zs[ja]
            xb, yb, zb = xs[jb], ys[jb], zs[jb]
            dxab = xb - xa; dyab = yb - ya; dzab = zb - za
            len2 = dxab * dxab + dyab * dyab + dzab * dzab
            if len2 < 1e-12:
                continue
            t_seg = ((x - xa) * dxab + (y - ya) * dyab + (z - za) * dzab) / len2
            t_seg = max(0.0, min(1.0, t_seg))
            fx = xa + t_seg * dxab; fy = ya + t_seg * dyab; fz = za + t_seg * dzab
            dseg2 = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
            if dseg2 < best_d2:
                best_d2 = dseg2
                sa = s_arr[ja]; sb = s_arr[jb]
                if sb < sa:
                    sb = sb + L
                best_s = sa + t_seg * (sb - sa)
                if best_s >= L:
                    best_s -= L

        theta = float(np.interp(best_s, s_arr, self.track.theta))
        mu    = float(np.interp(best_s, s_arr, self.track.mu))
        phi   = float(np.interp(best_s, s_arr, self.track.phi))
        normal = Track3D.get_normal_vector_numpy(theta, mu, phi)
        xc = float(np.interp(best_s, s_arr, xs))
        yc = float(np.interp(best_s, s_arr, ys))
        zc = float(np.interp(best_s, s_arr, zs))
        n  = float(np.dot(np.array([x - xc, y - yc, z - zc]), normal))
        return float(best_s), n

    # =============================================================================================
    # Continuity cost post-add (acts on upstream candidates + cost_array)
    # =============================================================================================
    def _apply_continuity_cost(self):
        """Post-add a continuity L2 penalty against the previous tick's best to
        self.planner.cost_array, re-run argmin, then rebuild self.planner.trajectory
        from the new best candidate slice.

        Returns True when the best idx changed (and trajectory was rebuilt)."""
        if self.w_continuity <= 0.0 or self._prev_best_s is None:
            return False
        cands    = getattr(self.planner, 'candidates', None)
        cost_arr = getattr(self.planner, 'cost_array', None)
        if cands is None or cost_arr is None:
            return False

        valid = np.asarray(cands['valid'], dtype=bool)
        if not valid.any():
            return False

        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)

        L = float(self.track.s[-1])
        m = int(min(self._prev_best_s.shape[0], s_all.shape[1]))
        if m < 2:
            return False

        # Wrap-aware s delta: treat shortest-arc difference.
        ds = s_all[:, :m] - self._prev_best_s[np.newaxis, :m]
        ds = np.where(ds >  L / 2.0, ds - L, ds)
        ds = np.where(ds < -L / 2.0, ds + L, ds)
        dn = n_all[:, :m] - self._prev_best_n[np.newaxis, :m]

        # ### HJ : was `/float(m)` — 그 normalize 가 다른 cost 항(시간적분)에 비해 30배
        # 약하게 만들어, mode-안 후보 jitter (n_endpoint Δ>0.20m 12.5%) 의 root cause 였음.
        # continuity_weight 절대값으로 jitter 강도 직접 제어 (재튜닝: 600 → 50 부근).
        L2 = np.sum(ds * ds + dn * dn, axis=1)
        new_cost = np.asarray(cost_arr, dtype=np.float64).copy() + self.w_continuity * L2
        self.planner.cost_array = new_cost

        old_best = int(self.planner.trajectory.get('optimal_idx', -1))
        masked = np.where(valid, new_cost, np.inf)
        new_best = int(np.argmin(masked))
        if new_best == old_best:
            return False

        self.planner.trajectory = self._extract_traj_from_candidate(new_best)
        return True

    # =============================================================================================
    # Boundary cost post-add (track-edge keep-out using Track3D half-widths)
    # =============================================================================================
    def _apply_boundary_cost(self):
        """### HJ : Phase 2 — penalise candidates whose |n| sits inside (half_w - safety_margin_m).
        Track3D stores `w_tr_left` (positive) and `w_tr_right` (negative-signed). For each
        candidate sample point we pick the half-width on the side n is on, integrate
        max(0, safety_margin - margin)² over the horizon, and add to cost_array."""
        if self.w_boundary <= 0.0:
            return False
        cands    = getattr(self.planner, 'candidates', None)
        cost_arr = getattr(self.planner, 'cost_array', None)
        if cands is None or cost_arr is None:
            return False
        valid = np.asarray(cands['valid'], dtype=bool)
        if not valid.any():
            return False

        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        t_all = np.asarray(cands['t'], dtype=np.float64)

        L = float(self.track.s[-1])
        s_track = np.asarray(self.track.s, dtype=np.float64)
        wL_arr  = np.asarray(self.track.w_tr_left, dtype=np.float64)
        wR_arr  = np.abs(np.asarray(self.track.w_tr_right, dtype=np.float64))
        s_mod   = np.mod(s_all, L)
        wL = np.interp(s_mod, s_track, wL_arr)
        wR = np.interp(s_mod, s_track, wR_arr)
        half_w = np.where(n_all >= 0.0, wL, wR)
        margin = half_w - np.abs(n_all)
        pen = np.maximum(0.0, float(self.safety_margin_m) - margin) ** 2  # (N, T)

        diff_t = np.diff(t_all, axis=1)
        pen_int = np.sum(pen[:, :-1] * diff_t, axis=1)

        new_cost = np.asarray(cost_arr, dtype=np.float64).copy() + self.w_boundary * pen_int
        self.planner.cost_array = new_cost

        old_best = int(self.planner.trajectory.get('optimal_idx', -1))
        masked = np.where(valid, new_cost, np.inf)
        new_best = int(np.argmin(masked))
        if new_best == old_best:
            return False
        self.planner.trajectory = self._extract_traj_from_candidate(new_best)
        return True

    # =============================================================================================
    # Phase 2 — mode-aware overtake extras
    # =============================================================================================
    def _compute_mode_extras(self, mode):
        """Per-candidate (side + progress) extras for a given OT mode.
        - follow:   target_n = raceline n at endpoint, no progress reward.
        - ot_left:  target_n = +0.5 * w_tr_left at endpoint, progress reward.
        - ot_right: target_n = -0.5 * |w_tr_right| at endpoint, progress reward.
        Returns shape (N_candidates,)."""
        cands = self.planner.candidates
        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        L = float(self.track.s[-1])
        s_track = np.asarray(self.track.s, dtype=np.float64)
        s_end = np.mod(s_all[:, -1], L)
        n_end = n_all[:, -1]

        if mode == 'follow':
            rl_s = np.asarray(self.raceline_dict['s'], dtype=np.float64)
            rl_n = np.asarray(self.raceline_dict['n'], dtype=np.float64)
            target_n = np.interp(s_end, rl_s, rl_n, period=L)
            prog_w = 0.0
        elif mode == 'ot_left':
            wL = np.interp(s_end, s_track, np.asarray(self.track.w_tr_left, dtype=np.float64))
            target_n = +0.5 * wL
            prog_w = float(self.w_progress)
        elif mode == 'ot_right':
            wR = np.interp(s_end, s_track, np.abs(np.asarray(self.track.w_tr_right, dtype=np.float64)))
            target_n = -0.5 * wR
            prog_w = float(self.w_progress)
        else:
            target_n = np.zeros_like(n_end)
            prog_w = 0.0

        side_pen = float(self.w_side) * (n_end - target_n) ** 2

        # Wrap-aware total Σs over horizon → reward (negative cost) for OT modes only.
        ds_total = s_all[:, -1] - s_all[:, 0]
        ds_total = np.where(ds_total < -L / 2.0, ds_total + L, ds_total)
        ds_total = np.where(ds_total >  L / 2.0, ds_total - L, ds_total)
        prog_reward = -prog_w * ds_total

        return side_pen + prog_reward

    # =============================================================================================
    # Normalized cost assembly (raw-term variants for per-tick min-max norm + weight × sum)
    # =============================================================================================
    def _compute_continuity_raw(self):
        """### HJ : returns shape (N,) raw L2 deviation from prev tick best, no
        weight, no /m normalize. Used by the normalized mode loop. None if disabled."""
        if self._prev_best_s is None:
            return None
        cands = getattr(self.planner, 'candidates', None)
        if cands is None:
            return None
        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        L = float(self.track.s[-1])
        m = int(min(self._prev_best_s.shape[0], s_all.shape[1]))
        if m < 2:
            return None
        ds = s_all[:, :m] - self._prev_best_s[np.newaxis, :m]
        ds = np.where(ds >  L / 2.0, ds - L, ds)
        ds = np.where(ds < -L / 2.0, ds + L, ds)
        dn = n_all[:, :m] - self._prev_best_n[np.newaxis, :m]
        return np.sum(ds * ds + dn * dn, axis=1)

    def _compute_boundary_raw(self):
        """### HJ : returns shape (N,) raw boundary penalty integral, no weight."""
        cands = getattr(self.planner, 'candidates', None)
        if cands is None:
            return None
        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        t_all = np.asarray(cands['t'], dtype=np.float64)
        L = float(self.track.s[-1])
        s_track = np.asarray(self.track.s, dtype=np.float64)
        wL_arr  = np.asarray(self.track.w_tr_left, dtype=np.float64)
        wR_arr  = np.abs(np.asarray(self.track.w_tr_right, dtype=np.float64))
        s_mod   = np.mod(s_all, L)
        wL = np.interp(s_mod, s_track, wL_arr)
        wR = np.interp(s_mod, s_track, wR_arr)
        half_w = np.where(n_all >= 0.0, wL, wR)
        margin = half_w - np.abs(n_all)
        pen = np.maximum(0.0, float(self.safety_margin_m) - margin) ** 2
        diff_t = np.diff(t_all, axis=1)
        return np.sum(pen[:, :-1] * diff_t, axis=1)

    def _compute_detour_raw(self):
        """### HJ : Σ_t (n(t) - n_raceline(s(t)))² · dt — 가성비 페널티.
        OT 모드가 너무 옆으로 돌면 비용 증가, follow 는 raceline 추종이라 ~0.
        weight 안 곱한 raw 적분. 모든 후보에 대해 동일 계산 (mode-agnostic)."""
        cands = getattr(self.planner, 'candidates', None)
        if cands is None or self.raceline_dict is None:
            return None
        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        t_all = np.asarray(cands['t'], dtype=np.float64)
        L = float(self.track.s[-1])
        rl_s = np.asarray(self.raceline_dict['s'], dtype=np.float64)
        rl_n = np.asarray(self.raceline_dict['n'], dtype=np.float64)
        s_mod = np.mod(s_all, L)
        n_rl_at = np.interp(s_mod, rl_s, rl_n, period=L)
        diff = (n_all - n_rl_at) ** 2
        diff_t = np.diff(t_all, axis=1)
        return np.sum(diff[:, :-1] * diff_t, axis=1)

    def _compute_mode_extras_raw(self, mode):
        """### HJ : returns (side_raw, progress_raw) — both shape (N,), unweighted.
        progress_raw is +Σs (caller subtracts since it's a reward)."""
        cands = self.planner.candidates
        s_all = np.asarray(cands['s'], dtype=np.float64)
        n_all = np.asarray(cands['n'], dtype=np.float64)
        L = float(self.track.s[-1])
        s_track = np.asarray(self.track.s, dtype=np.float64)
        s_end = np.mod(s_all[:, -1], L)
        n_end = n_all[:, -1]
        if mode == 'follow':
            rl_s = np.asarray(self.raceline_dict['s'], dtype=np.float64)
            rl_n = np.asarray(self.raceline_dict['n'], dtype=np.float64)
            target_n = np.interp(s_end, rl_s, rl_n, period=L)
            include_progress = False
        elif mode == 'ot_left':
            wL = np.interp(s_end, s_track, np.asarray(self.track.w_tr_left, dtype=np.float64))
            target_n = +0.5 * wL
            include_progress = True
        elif mode == 'ot_right':
            wR = np.interp(s_end, s_track, np.abs(np.asarray(self.track.w_tr_right, dtype=np.float64)))
            target_n = -0.5 * wR
            include_progress = True
        else:
            target_n = np.zeros_like(n_end)
            include_progress = False
        side_raw = (n_end - target_n) ** 2
        if include_progress:
            ds_total = s_all[:, -1] - s_all[:, 0]
            ds_total = np.where(ds_total < -L / 2.0, ds_total + L, ds_total)
            ds_total = np.where(ds_total >  L / 2.0, ds_total - L, ds_total)
            progress_raw = ds_total
        else:
            progress_raw = np.zeros_like(side_raw)
        return side_raw, progress_raw

    @staticmethod
    def _norm_weighted(arr, w, valid):
        """### HJ : per-tick min-max normalize on valid candidates, then × w.
        Returns array shape == arr.shape with 0 outside valid. Skip if w=0 or arr None."""
        if w <= 0.0 or arr is None:
            return None
        v = arr[valid]
        if v.size == 0:
            return None
        lo = float(v.min()); hi = float(v.max())
        rng = hi - lo
        out = np.zeros(arr.shape, dtype=np.float64)
        if rng < 1e-12:
            return out
        out[valid] = float(w) * ((v - lo) / rng)
        return out

    def _lead_opp_n(self, prediction, ego_s):
        # ### HJ : pick smallest forward-gap opp's lateral n (mirrors _collect_tick_fields).
        if not prediction:
            return float('nan')
        try:
            L = float(self.track.s[-1])
        except Exception:
            return float('nan')
        best_gap = float('inf'); best_n = float('nan')
        for pdata in prediction.values():
            if not len(pdata.get('s', [])): continue
            s0 = float(pdata['s'][0])
            gap = (s0 - ego_s) % L
            if gap < best_gap:
                best_gap = gap
                best_n = float(pdata['n'][0]) if len(pdata.get('n', [])) else float('nan')
        return best_n

    def _apply_mode_filter(self, cost_per_mode, opp_n_lead):
        # ### HJ : opp 위치 기반 hard rule. opp 가 한쪽으로 치우치면 그쪽 OT mode 페널티.
        if not np.isfinite(opp_n_lead):
            return cost_per_mode
        thr = float(getattr(self, 'mode_filter_threshold_n', 0.15))
        mult = float(getattr(self, 'mode_filter_strength', 5.0))
        if mult <= 1.0 + 1e-9:
            return cost_per_mode
        out = dict(cost_per_mode)
        if opp_n_lead < -thr and 'ot_right' in out and np.isfinite(out['ot_right']):
            out['ot_right'] *= mult
        elif opp_n_lead > +thr and 'ot_left' in out and np.isfinite(out['ot_left']):
            out['ot_left'] *= mult
        return out

    def _select_mode_with_hysteresis(self, cost_per_mode):
        """Mode lock with TTL + relative cost margin. Returns the chosen mode name.
        cost_per_mode : dict[str, float]  (∞ for modes with no valid candidate)."""
        valid_modes = {m: c for m, c in cost_per_mode.items() if np.isfinite(c)}
        prev_mode = self._active_mode
        if not valid_modes:
            return self._active_mode  # may be None

        best_now = min(valid_modes, key=valid_modes.get)
        now = self.get_clock().now().to_msg()

        if self._active_mode is None or self._active_mode not in valid_modes:
            self._active_mode = best_now
            self._mode_lock_until = now + rospy.Duration(float(self.hysteresis_ttl_s))
            self._log_event_safe('mode_switch', {
                'prev': prev_mode, 'new': best_now, 'reason': 'init_or_invalid_prev',
                'cost_per_mode': {k: float(v) for k, v in cost_per_mode.items()},
            })
            return best_now

        if now < self._mode_lock_until:
            # Lock blocked a potential switch — only emit when best_now would have switched.
            if best_now != self._active_mode:
                self._log_event_safe('mode_locked', {
                    'active': self._active_mode, 'wanted': best_now,
                    'lock_remain_s': max(0.0, (self._mode_lock_until - now).to_sec()),
                    'cost_per_mode': {k: float(v) for k, v in cost_per_mode.items()},
                })
            return self._active_mode

        if best_now == self._active_mode:
            return self._active_mode

        cur_cost = valid_modes[self._active_mode]
        new_cost = valid_modes[best_now]
        # Avoid sign-flip pathologies: only switch when the new mode strictly improves
        # by the configured margin and current cost is positive.
        if cur_cost > 0.0 and new_cost < (1.0 - float(self.hysteresis_margin)) * cur_cost:
            self._active_mode = best_now
            self._mode_lock_until = now + rospy.Duration(float(self.hysteresis_ttl_s))
            self._log_event_safe('mode_switch', {
                'prev': prev_mode, 'new': best_now, 'reason': 'cost_margin',
                'cur_cost': float(cur_cost), 'new_cost': float(new_cost),
                'margin': float(self.hysteresis_margin),
                'cost_per_mode': {k: float(v) for k, v in cost_per_mode.items()},
            })
            return best_now
        # Margin not crossed → keep active.
        self._log_event_safe('mode_locked', {
            'active': self._active_mode, 'wanted': best_now,
            'reason': 'margin_not_crossed',
            'cur_cost': float(cur_cost), 'new_cost': float(new_cost),
            'margin': float(self.hysteresis_margin),
        })
        return self._active_mode

    def _log_event_safe(self, event_type, payload):
        if self._logger is None:
            return
        try:
            self._logger.log_event(event_type, payload)
        except Exception:
            pass

    def _publish_mode_debug(self, cost_per_mode, chosen_mode):
        mc = Float32MultiArray()
        mc.data = [
            float(cost_per_mode.get('follow',   float('inf'))),
            float(cost_per_mode.get('ot_left',  float('inf'))),
            float(cost_per_mode.get('ot_right', float('inf'))),
        ]
        self.pub_mode_costs.publish(mc)
        lock_remain = max(0.0, (self._mode_lock_until - self.get_clock().now().to_msg()).to_sec())
        self.pub_active_mode.publish(String(data='%s|lock_remain=%.2fs' %
                                            (chosen_mode if chosen_mode else 'none', lock_remain)))

    def _extract_traj_from_candidate(self, idx):
        """Rebuild a trajectory dict from candidates[idx]. candidates has s/n/V/chi/ax/ay/kappa/t
        (no n_dot / s_dot) — sufficient for _publish_trajectory."""
        cand = self.planner.candidates
        traj = {
            'traj_cnt':    int(self.planner.traj_cnt),
            'optimal_idx': int(idx),
            't':     np.asarray(cand['t'][idx],     dtype=np.float64),
            's':     np.asarray(cand['s'][idx],     dtype=np.float64),
            'n':     np.asarray(cand['n'][idx],     dtype=np.float64),
            'V':     np.asarray(cand['V'][idx],     dtype=np.float64),
            'chi':   np.asarray(cand['chi'][idx],   dtype=np.float64),
            'ax':    np.asarray(cand['ax'][idx],    dtype=np.float64),
            'ay':    np.asarray(cand['ay'][idx],    dtype=np.float64),
            'kappa': np.asarray(cand['kappa'][idx], dtype=np.float64),
        }
        L = float(self.track.s[-1])
        s_mod = np.clip(np.mod(traj['s'], L), 1e-6, L - 1e-6)
        try:
            xyz = self.track.sn2cartesian(s=s_mod, n=traj['n'])
            traj['x'] = np.asarray(xyz[:, 0], dtype=np.float64)
            traj['y'] = np.asarray(xyz[:, 1], dtype=np.float64)
            traj['z'] = np.asarray(xyz[:, 2], dtype=np.float64)
        except Exception as e:
            rospy.logwarn_throttle(2.0, '[sampling][%s] sn2cartesian rebuild failed: %s',
                                   rospy.get_name(), e)
            traj['x'] = np.zeros_like(traj['s'])
            traj['y'] = np.zeros_like(traj['s'])
            traj['z'] = np.zeros_like(traj['s'])
        return traj

    # =============================================================================================
    # DebugLogger setup + helpers
    # =============================================================================================
    def _param_snapshot_dict(self):
        """현재 self.* 가중치 / 그리드 / 토글 → params_snapshot.yaml 용 dict.
        log_tick PARAM_COLUMNS 와 1:1 매핑 (`p_*` prefix 제외하면 같은 이름)."""
        return {
            'raceline_weight':      float(self.w_raceline),
            'velocity_weight':      float(self.w_velocity),
            'prediction_weight':    float(self.w_prediction),
            'continuity_weight':    float(self.w_continuity),
            'boundary_weight':      float(self.w_boundary),
            'safety_margin_m':      float(self.safety_margin_m),
            'side_weight':          float(self.w_side),
            'progress_weight':      float(self.w_progress),
            'opp_pred_ttl_s':       float(self.opp_pred_ttl_s),
            'hysteresis_ttl_s':     float(self.hysteresis_ttl_s),
            'hysteresis_margin':    float(self.hysteresis_margin),
            'kappa_dot_max':        float(self.kappa_dot_max),
            'filter_alpha':         float(self.filter_alpha),
            'mppi_temperature_rel': float(self.mppi_temperature_rel),
            'mppi_temporal_weight': float(self.mppi_temporal_weight),
            'horizon':              float(self.horizon),
            'num_samples':          int(self.num_samples),
            'n_samples':            int(self.n_samples),
            'v_samples':            int(self.v_samples),
            'safety_distance':      float(self.safety_distance),
            'gg_abs_margin':        float(self.gg_abs_margin),
            'gg_margin_rel':        float(self.gg_margin_rel),
            's_dot_min':            float(self.s_dot_min),
            'kappa_thr':            float(self.kappa_thr),
            'mppi_enable':          bool(self.mppi_enable),
            'resample_enable':      bool(self.resample_enable),
            'resample_ds_m':        float(self.resample_ds),
            'friction_check_2d':    bool(self.friction_check_2d),
            'relative_generation':  bool(self.relative_generation),
            'detour_weight':                float(self.w_detour),
            'endpoint_chi_raceline_only':   bool(self.endpoint_chi_raceline_only),
        }

    def _active_params_for_tick(self):
        """매 tick CSV row 의 PARAM_COLUMNS 부분. 슬라이더 변경이 다음 tick row 에 즉시 반영."""
        return {
            'p_raceline_w':      float(self.w_raceline),
            'p_velocity_w':      float(self.w_velocity),
            'p_prediction_w':    float(self.w_prediction),
            'p_continuity_w':    float(self.w_continuity),
            'p_boundary_w':      float(self.w_boundary),
            'p_safety_margin_m': float(self.safety_margin_m),
            'p_side_w':          float(self.w_side),
            'p_progress_w':      float(self.w_progress),
            'p_opp_pred_ttl_s':  float(self.opp_pred_ttl_s),
            'p_hys_ttl_s':       float(self.hysteresis_ttl_s),
            'p_hys_margin':      float(self.hysteresis_margin),
            'p_kappa_dot_max':   float(self.kappa_dot_max),
            'p_filter_alpha':    float(self.filter_alpha),
            'p_mppi_temp_rel':   float(self.mppi_temperature_rel),
            'p_mppi_temporal_w': float(self.mppi_temporal_weight),
            'p_horizon':         float(self.horizon),
        }

    def _init_debug_logger(self):
        if not bool(self._get_param_or_default('~debug/enable', True)):
            self.get_logger().info('[sampling][%s] debug logger disabled (~debug/enable=false)',
                          rospy.get_name())
            return
        try:
            rp = rospkg.RosPack()
            pkg_dir = rp.get_path('sampling_based_planner_3d')
            default_dir = os.path.join(pkg_dir, 'debug_log')
        except Exception:
            default_dir = os.path.expanduser('~/sampling_debug_log')
        base_dir = str(self._get_param_or_default('~debug/out_dir', default_dir))
        snapshot_every_n = int(self._get_param_or_default('~debug/snapshot_every_n', 0))

        # meta — git_sha, map, vehicle, paths.
        meta = {
            'map':                self._get_param_or_default('/map', ''),
            'vehicle_name':       self.vehicle_name,
            'state':              self.state,
            'rate_hz':            float(self.rate_hz),
            'instance_yaml':      self.instance_yaml_path,
            'track_csv':          self.track_csv_path,
            'raceline_csv':       self.raceline_csv_path,
            'gg_dir':             self.gg_dir_path,
            'vehicle_params':     self.vehicle_params_path,
            'git_sha':            self._git_sha(),
        }
        try:
            self._logger = DebugLogger(
                base_dir=base_dir,
                instance_name=rospy.get_name(),
                state=self.state,
                rate_hz=self.rate_hz,
                params_snapshot=self._param_snapshot_dict(),
                meta=meta,
                snapshot_every_n=snapshot_every_n,
            )
            self._prev_param_snapshot = self._param_snapshot_dict()
            rospy.on_shutdown(self._logger.close)
            self.get_logger().info('[sampling][%s] debug logger session: %s',
                          rospy.get_name(), self._logger.session_dir)
        except Exception as e:
            self.get_logger().warning('[sampling][%s] debug logger init failed: %s',
                          rospy.get_name(), e)
            self._logger = None

    def _git_sha(self):
        try:
            import subprocess
            sha = subprocess.check_output(
                ['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return sha
        except Exception:
            return 'unknown'

    def _emit_param_diff_events(self):
        """_weight_cb 끝에서 호출 — 이전 snapshot 과 비교해서 변경된 필드별로 1 이벤트 발행."""
        if self._logger is None or not hasattr(self, '_prev_param_snapshot'):
            return
        cur = self._param_snapshot_dict()
        diffs = {}
        for k, v_new in cur.items():
            v_old = self._prev_param_snapshot.get(k)
            if isinstance(v_new, float):
                if v_old is None or abs(float(v_old) - v_new) > 1e-9:
                    diffs[k] = {'old': v_old, 'new': v_new}
            elif v_old != v_new:
                diffs[k] = {'old': v_old, 'new': v_new}
        if diffs:
            self._logger.log_event('param_changed', {'t': time.time(), 'diffs': diffs})
        self._prev_param_snapshot = cur

    def _collect_tick_fields(self, ego_s, ego_n, ego_v, prediction, status,
                              chosen_mode, prev_mode, cost_per_mode, best_idx_per_mode,
                              n_valid, n_total, calc_traj_ms, post_add_ms, total_tick_ms,
                              rollback=False):
        """spin() per-tick metrics → DebugLogger.log_tick fields dict.
        실패하지 말 것 (spin 죽이면 안 됨) — 호출 측에서 try/except 으로 감싸 있음."""
        traj = self.planner.trajectory or {}
        s_traj = np.asarray(traj.get('s', []), dtype=np.float64) if traj else np.zeros(0)
        n_traj = np.asarray(traj.get('n', []), dtype=np.float64) if traj else np.zeros(0)
        V_traj = np.asarray(traj.get('V', []), dtype=np.float64) if traj else np.zeros(0)
        kappa_traj = np.asarray(traj.get('kappa', []), dtype=np.float64) if traj else np.zeros(0)
        t_traj = np.asarray(traj.get('t', []), dtype=np.float64) if traj else np.zeros(0)

        # Chosen-trajectory metrics.
        kappa_max = float(np.max(np.abs(kappa_traj))) if kappa_traj.size else float('nan')
        if kappa_traj.size >= 2 and t_traj.size == kappa_traj.size:
            dt = np.diff(t_traj)
            dt_safe = np.where(dt > 1e-6, dt, 1e-6)
            kappa_dot_max = float(np.max(np.abs(np.diff(kappa_traj) / dt_safe)))
        else:
            kappa_dot_max = float('nan')
        n_swing = float(n_traj.max() - n_traj.min()) if n_traj.size else float('nan')

        # Wrap-aware progress.
        L = float(self.track.s[-1])
        if s_traj.size >= 2:
            ds = float(s_traj[-1] - s_traj[0])
            if ds < -L / 2.0:
                ds += L
            elif ds > L / 2.0:
                ds -= L
            total_progress = ds
        else:
            total_progress = float('nan')

        # Boundary min margin on chosen traj.
        boundary_min = float('nan')
        if s_traj.size and n_traj.size == s_traj.size:
            try:
                s_track = np.asarray(self.track.s, dtype=np.float64)
                wL = np.interp(np.mod(s_traj, L), s_track,
                               np.asarray(self.track.w_tr_left, dtype=np.float64))
                wR = np.interp(np.mod(s_traj, L), s_track,
                               np.abs(np.asarray(self.track.w_tr_right, dtype=np.float64)))
                half_w = np.where(n_traj >= 0.0, wL, wR)
                boundary_min = float(np.min(half_w - np.abs(n_traj)))
            except Exception:
                pass

        # Opponent leader (smallest forward gap in s).
        opp_count = len(prediction or {})
        opp_id_lead = -1
        opp_s_lead = float('nan')
        opp_n_lead = float('nan')
        opp_vs_lead = float('nan')
        opp_pred_age_lead = float('nan')
        pred_min_dist = float('nan')
        if prediction:
            best_gap = float('inf')
            now = self.get_clock().now().to_msg()
            with self._opp_lock:
                obs_cache = dict(self._opp_obstacles)
                pred_cache = dict(self._opp_predictions)
            for oid, pdata in prediction.items():
                s0 = float(pdata['s'][0]) if len(pdata['s']) else float('nan')
                gap = (s0 - ego_s) % L
                if gap < best_gap:
                    best_gap = gap
                    opp_id_lead = int(oid)
                    opp_s_lead = s0
                    opp_n_lead = float(pdata['n'][0]) if len(pdata['n']) else float('nan')
                    ob = obs_cache.get(int(oid))
                    if ob is not None:
                        opp_vs_lead = float(ob.get('vs', float('nan')))
                    pe = pred_cache.get(int(oid))
                    if pe is not None and pe.get('stamp') is not None:
                        opp_pred_age_lead = float((now - pe['stamp']).to_sec())
            # Min ego-vs-opp distance over horizon (s/n L2, time-aligned via np.interp).
            # ### HJ : opponent prediction 은 dt=0.02s, ego trajectory 는 horizon/num_samples
            # 간격 (≈0.05s @ 1.5s/30) — 인덱스 단순 비교는 다른 t 끼리 비교가 됨.
            # upstream sampling_based_planner.py:382 처럼 t_traj 에 보간해서 같은 시각의
            # 상대 위치를 뽑아야 정확.
            if opp_id_lead >= 0 and s_traj.size and t_traj.size == s_traj.size:
                pdata = prediction.get(opp_id_lead) or prediction.get(int(opp_id_lead))
                if pdata is not None and len(pdata['t']) >= 2:
                    p_t = np.asarray(pdata['t'], dtype=np.float64)
                    p_s = np.asarray(pdata['s'], dtype=np.float64)
                    p_n = np.asarray(pdata['n'], dtype=np.float64)
                    # Restrict to ego horizon ∩ pred horizon — np.interp clamps outside,
                    # which would silently make far-future ego compare to last opp pos.
                    t_max = min(float(t_traj[-1]), float(p_t[-1]))
                    mask = t_traj <= t_max + 1e-9
                    if mask.any():
                        ts = t_traj[mask]
                        s_op_t = np.interp(ts, p_t, p_s)
                        n_op_t = np.interp(ts, p_t, p_n)
                        ds_op = (s_op_t - s_traj[mask]) % L
                        ds_op = np.where(ds_op > L / 2.0, ds_op - L, ds_op)
                        dn_op = n_op_t - n_traj[mask]
                        pred_min_dist = float(np.min(np.sqrt(ds_op ** 2 + dn_op ** 2)))

        fields = {
            't_ros': float(self.get_clock().now().to_msg().to_sec()),
            'tick_idx': int(self._tick_idx),
            'state': self.state,
            'ego_s': float(ego_s), 'ego_n': float(ego_n), 'ego_v': float(ego_v),
            'opp_count': int(opp_count),
            'opp_id_lead': int(opp_id_lead),
            'opp_s_lead': opp_s_lead, 'opp_n_lead': opp_n_lead,
            'opp_vs_lead': opp_vs_lead, 'opp_pred_age_lead': opp_pred_age_lead,
            'chosen_mode': chosen_mode if chosen_mode else '',
            'prev_mode':   prev_mode if prev_mode else '',
            'mode_changed': int(bool(chosen_mode != prev_mode and prev_mode is not None)),
            'lock_remain_s': max(0.0, float((self._mode_lock_until - self.get_clock().now().to_msg()).to_sec())),
            'cost_follow':   float(cost_per_mode.get('follow',   float('nan'))),
            'cost_ot_left':  float(cost_per_mode.get('ot_left',  float('nan'))),
            'cost_ot_right': float(cost_per_mode.get('ot_right', float('nan'))),
            'best_idx_follow':   int(best_idx_per_mode.get('follow',   -1)),
            'best_idx_ot_left':  int(best_idx_per_mode.get('ot_left',  -1)),
            'best_idx_ot_right': int(best_idx_per_mode.get('ot_right', -1)),
            'n_endpoint_chosen': float(n_traj[-1]) if n_traj.size else float('nan'),
            'v_endpoint_chosen': float(V_traj[-1]) if V_traj.size else float('nan'),
            'total_progress_m':  float(total_progress),
            'kappa_max_chosen':       kappa_max,
            'kappa_dot_max_chosen':   kappa_dot_max,
            'n_swing_chosen':         n_swing,
            'boundary_min_margin_chosen': boundary_min,
            'prediction_min_dist_chosen': pred_min_dist,
            'n_valid_candidates': int(n_valid),
            'n_total_candidates': int(n_total),
            'calc_traj_ms':  float(calc_traj_ms),
            'post_add_ms':   float(post_add_ms),
            'total_tick_ms': float(total_tick_ms),
            'status': str(status),
            'rollback': bool(rollback),
        }
        fields.update(self._active_params_for_tick())
        # ### HJ : MPPI blending stats (set by _mppi_blend; empty if blend skipped).
        ms = getattr(self, '_last_mppi_stats', None) or {}
        fields['mppi_eff_n']   = float(ms.get('mppi_eff_n',   float('nan')))
        fields['mppi_top1_w']  = float(ms.get('mppi_top1_w',  float('nan')))
        fields['mppi_T_used']  = float(ms.get('mppi_T_used',  float('nan')))
        fields['mppi_n_blend'] = int(ms.get('mppi_n_blend',   0))
        return fields

    # =============================================================================================
    # ~debug/tick_json — single-stream JSON diagnostics
    # =============================================================================================
    @staticmethod
    def _jnum(x):
        """### HJ : JSON-safe number — NaN/Inf → None (null), float → float, int → int.
        json.dumps rejects NaN/Inf without allow_nan=False-equivalent handling,
        and downstream parsers (jq, python json.loads) vary on extended-JSON allowance."""
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf

    def _compute_cost_stats(self, cost_array, valid_array, best_idx):
        """### HJ : Summary statistics over cost_array[valid]. spread_ratio indicates
        how *peaked* the distribution is: 1.0 → best is far below everything else
        (clear winner); close to 0 → best is barely better than max (flat cost,
        selection basically random). min/q25/med/q75/max gives the full shape."""
        if cost_array is None or valid_array is None:
            return None
        mask = np.asarray(valid_array, dtype=bool)
        if not mask.any():
            return None
        c = np.asarray(cost_array, dtype=np.float64)[mask]
        if c.size == 0:
            return None
        c_min  = float(c.min())
        c_max  = float(c.max())
        c_mean = float(c.mean())
        c_std  = float(c.std())
        q25, q50, q75 = [float(v) for v in np.quantile(c, [0.25, 0.50, 0.75])]
        rng = c_max - c_min
        if best_idx is not None and 0 <= best_idx < cost_array.shape[0] and mask[best_idx]:
            c_best = float(cost_array[best_idx])
        else:
            c_best = c_min
        spread = float((c_max - c_best) / rng) if rng > 1e-12 else 0.0
        return {
            'n':            int(c.size),
            'min':          self._jnum(c_min),
            'q25':          self._jnum(q25),
            'median':       self._jnum(q50),
            'q75':          self._jnum(q75),
            'max':          self._jnum(c_max),
            'mean':         self._jnum(c_mean),
            'std':          self._jnum(c_std),
            'best':         self._jnum(c_best),
            'spread_ratio': self._jnum(spread),
        }

    def _build_tick_json(self, fields, ego_chi=None):
        """### HJ : Consolidate per-tick diagnostics into a single JSON object.

        Schema (stable; fields may be None/empty but keys stay):
          tick, t_ros, state, status
          ego{s,n,v,chi_raceline_rel}      — chi_raceline_rel = ego heading vs raceline tangent
          candidates{total, valid, killed_curvature, killed_path, killed_friction}
          cost_stats{n,min,q25,median,q75,max,mean,std,best,spread_ratio}
          best{idx, n_end, v_end, total_progress_m, kappa_max, kappa_dot_max,
               n_swing, boundary_min_margin, pred_min_dist}
          mode{chosen, prev, changed, lock_remain_s,
               cost_follow, cost_ot_left, cost_ot_right,
               best_idx_follow, best_idx_ot_left, best_idx_ot_right}
          opp{count, lead_id, lead_s, lead_n, lead_vs, lead_age_s}
          mppi{eff_n, top1_w, T_used, n_blend}
          timing_ms{calc, post_add, total}
          params{...hot-reconfigurable subset...}
        """
        # Upstream check_stats (set in sampling_based_planner.calc_trajectory).
        cs = getattr(self.planner, 'check_stats', None) or {}
        cands_section = {
            'total':            int(cs.get('total',            fields.get('n_total_candidates', 0))),
            'valid_before_any': int(cs.get('valid_before_any', 0)),
            'killed_curvature': int(cs.get('killed_curvature', 0)),
            'killed_path':      int(cs.get('killed_path',      0)),
            'killed_friction':  int(cs.get('killed_friction',  0)),
            'valid_after_all':  int(cs.get('valid_after_all',  fields.get('n_valid_candidates', 0))),
        }

        # Cost distribution over valid candidates (using the FINAL cost_array that the
        # selector used — overtake state uses the chosen mode's mode_cost_arrays entry).
        cost_arr = getattr(self.planner, 'cost_array', None)
        cands    = getattr(self.planner, 'candidates', None)
        if cands is not None and cost_arr is not None:
            stats = self._compute_cost_stats(
                cost_array=np.asarray(cost_arr, dtype=np.float64),
                valid_array=np.asarray(cands['valid'], dtype=bool),
                best_idx=int(self.planner.trajectory.get('optimal_idx', -1))
                          if self.planner.trajectory else -1,
            )
        else:
            stats = None

        out = {
            'tick':    int(fields.get('tick_idx', 0)),
            't_ros':   self._jnum(fields.get('t_ros')),
            'state':   str(fields.get('state', '')),
            'status':  str(fields.get('status', '')),
            'ego': {
                's':                  self._jnum(fields.get('ego_s')),
                'n':                  self._jnum(fields.get('ego_n')),
                'v':                  self._jnum(fields.get('ego_v')),
                'chi_raceline_rel':   self._jnum(ego_chi),
            },
            'candidates': cands_section,
            'cost_stats': stats,
            'best': {
                'idx':                 int(self.planner.trajectory.get('optimal_idx', -1))
                                       if self.planner.trajectory else -1,
                'n_end':               self._jnum(fields.get('n_endpoint_chosen')),
                'v_end':               self._jnum(fields.get('v_endpoint_chosen')),
                'total_progress_m':    self._jnum(fields.get('total_progress_m')),
                'kappa_max':           self._jnum(fields.get('kappa_max_chosen')),
                'kappa_dot_max':       self._jnum(fields.get('kappa_dot_max_chosen')),
                'n_swing':             self._jnum(fields.get('n_swing_chosen')),
                'boundary_min_margin': self._jnum(fields.get('boundary_min_margin_chosen')),
                'pred_min_dist':       self._jnum(fields.get('prediction_min_dist_chosen')),
            },
            'mode': {
                'chosen':              str(fields.get('chosen_mode', '')),
                'prev':                str(fields.get('prev_mode', '')),
                'changed':             int(fields.get('mode_changed', 0)),
                'lock_remain_s':       self._jnum(fields.get('lock_remain_s')),
                'cost_follow':         self._jnum(fields.get('cost_follow')),
                'cost_ot_left':        self._jnum(fields.get('cost_ot_left')),
                'cost_ot_right':       self._jnum(fields.get('cost_ot_right')),
                'best_idx_follow':     int(fields.get('best_idx_follow',   -1)),
                'best_idx_ot_left':    int(fields.get('best_idx_ot_left',  -1)),
                'best_idx_ot_right':   int(fields.get('best_idx_ot_right', -1)),
            },
            'opp': {
                'count':     int(fields.get('opp_count', 0)),
                'lead_id':   int(fields.get('opp_id_lead', -1)),
                'lead_s':    self._jnum(fields.get('opp_s_lead')),
                'lead_n':    self._jnum(fields.get('opp_n_lead')),
                'lead_vs':   self._jnum(fields.get('opp_vs_lead')),
                'lead_age_s':self._jnum(fields.get('opp_pred_age_lead')),
            },
            'mppi': {
                'eff_n':    self._jnum(fields.get('mppi_eff_n')),
                'top1_w':   self._jnum(fields.get('mppi_top1_w')),
                'T_used':   self._jnum(fields.get('mppi_T_used')),
                'n_blend':  int(fields.get('mppi_n_blend', 0)),
            },
            'timing_ms': {
                'calc':     self._jnum(fields.get('calc_traj_ms')),
                'post_add': self._jnum(fields.get('post_add_ms')),
                'total':    self._jnum(fields.get('total_tick_ms')),
            },
            'rollback': bool(fields.get('rollback', False)),
            # ### HJ : hot-reconfigurable params subset — enough to correlate cost shape
            # with a weight change without replaying the YAML snapshot.
            'params': {
                'w_race':       self._jnum(self.w_raceline),
                'w_vel':        self._jnum(self.w_velocity),
                'w_pred':       self._jnum(self.w_prediction),
                'w_cont':       self._jnum(self.w_continuity),
                'w_bound':      self._jnum(self.w_boundary),
                'w_side':       self._jnum(self.w_side),
                'w_progress':   self._jnum(self.w_progress),
                'w_detour':     self._jnum(self.w_detour),
                'safety_m':     self._jnum(self.safety_margin_m),
                'safety_dist':  self._jnum(self.safety_distance),
                'filter_alpha': self._jnum(self.filter_alpha),
                'mppi_enable':  bool(self.mppi_enable),
                'horizon':      self._jnum(self.horizon),
                'num_samples':  int(self.num_samples),
                'n_samples':    int(self.n_samples),
                'v_samples':    int(self.v_samples),
                'endpoint_chi_raceline_only': bool(self.endpoint_chi_raceline_only),
                'relative_generation':        bool(self.relative_generation),
            },
        }
        return out

    def _ego_chi_vs_raceline(self, s_cent):
        """### HJ : ego's current heading minus raceline tangent at current s.
        Uses the latest odom (not a Frenet-derived chi) so the diagnostic reflects the
        real car's yaw. If odom yaw is missing (self._cur_yaw None), return None.
        Value is *signed* — positive = ego pointing left of raceline."""
        yaw = getattr(self, '_cur_yaw', None)
        if yaw is None or s_cent is None:
            return None
        try:
            theta = float(np.interp(float(s_cent), np.asarray(self.track.s),
                                    np.asarray(self.track.theta)))
        except Exception:
            return None
        # wrap to [-pi, pi]
        d = float(yaw) - theta
        while d >  math.pi: d -= 2.0 * math.pi
        while d < -math.pi: d += 2.0 * math.pi
        return d

    # =============================================================================================
    # Main loop
    # =============================================================================================
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self._cur_x is None:
                self._publish_status('WAITING_ODOM')
                rate.sleep()
                continue

            x_cur, y_cur, z_cur = self._cur_x, self._cur_y, self._cur_z
            v_cur = self._cur_vs or 0.0

            s_cent, n_cent = self._cart_to_cl_frenet_exact(x_cur, y_cur, z_cur)
            self._prev_s_cent = s_cent
            self._prev_xyz    = (x_cur, y_cur, z_cur)
            self._prev_stamp  = self.get_clock().now().to_msg().to_sec()

            # ### HJ : L3 fix - feed measured heading into n_dot so samples fan from
            # the real tangent, not the raceline. n_dot = s_dot * tan(chi) * (1 - Ω_z·n).
            s_dot_eff = max(self.s_dot_min, v_cur)
            chi_rel = self._ego_chi_vs_raceline(s_cent)
            if chi_rel is not None and s_cent is not None:
                try:
                    Omega_z = float(np.interp(float(s_cent),
                                              np.asarray(self.track.s),
                                              np.asarray(self.track.Omega_z)))
                except Exception:
                    Omega_z = 0.0
                n_dot_start = s_dot_eff * math.tan(chi_rel) * (1.0 - Omega_z * float(n_cent or 0.0))
            else:
                n_dot_start = 0.0
            state = {
                's':      s_cent,
                'n':      n_cent,
                's_dot':  s_dot_eff,
                's_ddot': 0.0,
                'n_dot':  n_dot_start,
                'n_ddot': 0.0,
            }
            # ### HJ : Phase 2 — feed cached opponent predictions into the upstream
            # gaussian-kernel cost. Empty dict preserves observation-only behavior.
            prediction = self._build_prediction_dict()

            self._tick_idx += 1
            tick_t0_wall = time.time()
            t0 = time.time()
            tick_status = 'OK'
            tick_cost_per_mode = {}
            tick_best_idx_per_mode = {}
            tick_n_valid = 0
            tick_n_total = 0
            tick_rollback = False
            tick_calc_ms = 0.0
            self._last_mppi_stats = {}  # reset; _mppi_blend fills if it runs this tick
            tick_post_add_ms = 0.0
            prev_mode_for_log = self._active_mode
            try:
                # ### HJ : hold _gg_lock so a concurrent rqt-triggered GGManager rebuild
                # cannot swap self.planner.gg_handler mid-calc_trajectory.
                with self._gg_lock:
                    # ### HJ : inject prev-tick n_end + rate cap so the sampler can
                    # center its linspace around prev_chosen_n_end within the rate
                    # window (empty rate-ok pool → ping-pong fix).
                    self.planner.prev_chosen_n_end = self._prev_chosen_n_end
                    self.planner.n_end_rate_cap    = float(self.n_end_rate_cap)
                    # ### HJ : snapshot previous trajectory BEFORE calc_trajectory so
                    # the mode-loop can restore it when all modes are rate-empty (hold).
                    _prev_traj = None
                    try:
                        _pt = getattr(self.planner, 'trajectory', None)
                        if _pt and 'x' in _pt and len(_pt['x']) > 0:
                            _prev_traj = {k: (v.copy() if hasattr(v, 'copy') else v)
                                          for k, v in _pt.items()}
                    except Exception:
                        _prev_traj = None
                    self.planner.calc_trajectory(
                        state=state,
                        prediction=prediction,
                        raceline=self.raceline_dict,
                        relative_generation=self.relative_generation,
                        n_samples=self.n_samples,
                        v_samples=self.v_samples,
                        horizon=self.horizon,
                        num_samples=self.num_samples,
                        safety_distance=self.safety_distance,
                        gg_abs_margin=self.gg_abs_margin,
                        friction_check_2d=self.friction_check_2d,
                        s_dot_min=self.s_dot_min,
                        kappa_thr=self.kappa_thr,
                        raceline_cost_weight=self.w_raceline,
                        velocity_cost_weight=self.w_velocity,
                        prediction_cost_weight=self.w_prediction,
                        endpoint_chi_raceline_only=self.endpoint_chi_raceline_only,
                    )
                dt_ms = (time.time() - t0) * 1000.0
                tick_calc_ms = dt_ms
                self.pub_timing.publish(Float32(data=dt_ms))

                t_post = time.time()
                # ### HJ : Continuity/Boundary 는 overtake state 에서 mode loop 가 raw
                # 단계에서 직접 합산 (per-tick min-max norm). recovery 등 다른 state 에서만
                # 기존 cost_array 후처리 경로 유지.
                changed = False
                if self.state != 'overtake':
                    changed = self._apply_continuity_cost()
                    if self._apply_boundary_cost():
                        changed = True

                # ### HJ : Phase 2 — mode-aware OT selection. Per-tick min-max
                # normalize each raw cost term × its weight (Option C), then sum.
                # Each weight then bounds [0, w_i] contribution, so prediction
                # (10000) no longer drowns continuity/side/etc. base_norm is shared
                # across modes; side/progress are per-mode.
                chosen_mode = None
                if self.state == 'overtake':
                    cands = getattr(self.planner, 'candidates', None)
                    if cands is not None:
                        valid = np.asarray(cands['valid'], dtype=bool)
                        if valid.any():
                            v_raw = getattr(self.planner, 'cost_velocity_raw',   None)
                            r_raw = getattr(self.planner, 'cost_raceline_raw',   None)
                            p_raw = getattr(self.planner, 'cost_prediction_raw', None)
                            c_raw = self._compute_continuity_raw()
                            b_raw = self._compute_boundary_raw()
                            d_raw = self._compute_detour_raw()
                            # ### HJ : detour 는 mode loop 안에서 OT mode 에만 적용
                            # (follow 는 0). 그래야 mode 간 best_idx 가 분리됨.
                            base_terms = [
                                self._norm_weighted(v_raw, self.w_velocity,   valid),
                                self._norm_weighted(r_raw, self.w_raceline,   valid),
                                self._norm_weighted(p_raw, self.w_prediction, valid),
                                self._norm_weighted(c_raw, self.w_continuity, valid),
                                self._norm_weighted(b_raw, self.w_boundary,   valid),
                            ]
                            base_norm = np.zeros(int(valid.size), dtype=np.float64)
                            for t in base_terms:
                                if t is not None:
                                    base_norm = base_norm + t
                            cost_per_mode = {}
                            best_idx_per_mode = {}
                            mode_cost_arrays = {}
                            for mode in _OVERTAKE_MODES:
                                side_raw, prog_raw = self._compute_mode_extras_raw(mode)
                                side_n = self._norm_weighted(side_raw, self.w_side,     valid)
                                prog_n = self._norm_weighted(prog_raw, self.w_progress, valid)
                                # ### HJ : detour — follow=0, OT mode 만 페널티.
                                detour_w_mode = 0.0 if mode == 'follow' else float(self.w_detour)
                                detour_n = self._norm_weighted(d_raw, detour_w_mode, valid)
                                mc = base_norm.copy()
                                if side_n is not None:
                                    mc = mc + side_n
                                if prog_n is not None:
                                    mc = mc - prog_n  # progress is reward
                                if detour_n is not None:
                                    mc = mc + detour_n
                                mode_cost_arrays[mode] = mc
                                # ### HJ : n_end rate constraint as SOFT penalty
                                # (was hard filter → rollback → car follows stale path).
                                # rate_penalty = w_rate * (max(|Δn_end|−cap, 0) / cap)²
                                # added directly to mode cost. Candidates inside the cap
                                # pay nothing; those outside pay quadratically. argmin
                                # still always picks a real index, so rollback never
                                # fires and best_sample stays synchronized with the
                                # actual chosen candidate every tick.
                                n_end_all = np.asarray(cands['n'])[:, -1]
                                if self._prev_chosen_n_end is not None:
                                    cap_safe = max(self.n_end_rate_cap, 1e-6)
                                    excess = np.maximum(
                                        np.abs(n_end_all - self._prev_chosen_n_end) - self.n_end_rate_cap,
                                        0.0,
                                    )
                                    rate_penalty = self.w_rate_penalty * (excess / cap_safe) ** 2
                                    mc_with_rate = mc + rate_penalty
                                else:
                                    mc_with_rate = mc
                                masked = np.where(valid, mc_with_rate, np.inf)
                                bi = int(np.argmin(masked)) if valid.any() else -1
                                cost_per_mode[mode] = (
                                    float(mc_with_rate[bi])
                                    if bi >= 0 and np.isfinite(mc_with_rate[bi])
                                    else float(np.inf)
                                )
                                best_idx_per_mode[mode] = bi

                            # ### HJ : opp 위치 기반 mode hard filter — opp 가 좌/우로 치우치면
                            # 같은 쪽 OT mode 에 multiplicative 페널티. _select_mode_with_hysteresis
                            # 의 cost 비교 + hysteresis 양쪽에 동일하게 작용.
                            opp_n_for_filter = self._lead_opp_n(prediction, s_cent)
                            cost_per_mode_filtered = self._apply_mode_filter(cost_per_mode, opp_n_for_filter)
                            chosen_mode = self._select_mode_with_hysteresis(cost_per_mode_filtered)
                            # ### HJ : if chosen mode's best_idx == -1 (no rate-ok valid
                            # candidate) OR all modes are inf-cost, restore previous
                            # trajectory (don't teleport). calc_trajectory already wrote
                            # a fresh self.planner.trajectory internally via its own
                            # unconstrained argmin, so skipping the extract below is not
                            # enough — we must overwrite with the snapshot.
                            if (chosen_mode is not None
                                    and chosen_mode in mode_cost_arrays
                                    and best_idx_per_mode.get(chosen_mode, -1) >= 0):
                                self.planner.cost_array = mode_cost_arrays[chosen_mode]
                                bi = best_idx_per_mode[chosen_mode]
                                if bi != int(self.planner.trajectory.get('optimal_idx', -1)):
                                    self.planner.trajectory = self._extract_traj_from_candidate(bi)
                                    changed = True
                            elif _prev_traj is not None:
                                # hold: roll back to snapshot so trajectory doesn't jump.
                                # With soft-relax rate penalty above, this branch should
                                # essentially never fire (bi≥0 whenever any candidate is
                                # valid). Track it so we can verify.
                                self.planner.trajectory = _prev_traj
                                tick_rollback = True
                            self._publish_mode_debug(cost_per_mode_filtered, chosen_mode)
                            tick_cost_per_mode = cost_per_mode_filtered
                            tick_best_idx_per_mode = best_idx_per_mode
                            # ### HJ : persist chosen n_end for next-tick rate constraint.
                            if (chosen_mode is not None
                                    and chosen_mode in best_idx_per_mode
                                    and best_idx_per_mode[chosen_mode] >= 0):
                                _bi = best_idx_per_mode[chosen_mode]
                                try:
                                    self._prev_chosen_n_end = float(np.asarray(cands['n'])[_bi, -1])
                                except Exception:
                                    pass
                            tick_n_valid = int(np.count_nonzero(valid))
                            tick_n_total = int(valid.size)
                tick_post_add_ms = (time.time() - t_post) * 1000.0

                traj = self.planner.trajectory
                if traj and 'x' in traj and len(traj['x']) > 0:
                    if self.mppi_enable:
                        blended = self._mppi_blend()
                        if blended is not None:
                            traj = blended
                    self._publish_trajectory(traj)
                    self._publish_candidates(traj.get('optimal_idx', -1))
                    if changed:
                        tick_status = 'CONTINUITY_SWITCH'
                    else:
                        tick_status = 'OK'
                    self._publish_status(tick_status)
                else:
                    tick_status = 'NO_FEASIBLE'
                    self._publish_status(tick_status)
                    self._log_event_safe('no_feasible', {
                        't': time.time(), 'tick_idx': self._tick_idx,
                        'n_valid': tick_n_valid, 'n_total': tick_n_total,
                    })

            except Exception as e:
                import traceback
                rospy.logerr_throttle(2.0, '[sampling][%s] calc_trajectory failed: %s\n%s',
                                     rospy.get_name(), e, traceback.format_exc())
                tick_status = 'EXCEPTION:' + type(e).__name__
                self._publish_status(tick_status)
                self._log_event_safe('exception', {
                    't': time.time(), 'tick_idx': self._tick_idx,
                    'type': type(e).__name__, 'msg': str(e),
                })

            # ### HJ : per-tick fields dict — used by both CSV logger and ~debug/tick_json.
            # Built unconditionally (cheap) so tick_json publishes even when debug logger
            # is disabled (~debug/enable=false) — the JSON stream is the Claude-facing
            # real-time feed, separate from the persistent CSV trace.
            tick_fields = None
            try:
                total_tick_ms = (time.time() - tick_t0_wall) * 1000.0
                tick_fields = self._collect_tick_fields(
                    ego_s=s_cent, ego_n=n_cent, ego_v=v_cur,
                    prediction=prediction, status=tick_status,
                    chosen_mode=self._active_mode, prev_mode=prev_mode_for_log,
                    cost_per_mode=tick_cost_per_mode,
                    best_idx_per_mode=tick_best_idx_per_mode,
                    n_valid=tick_n_valid, n_total=tick_n_total,
                    calc_traj_ms=tick_calc_ms, post_add_ms=tick_post_add_ms,
                    total_tick_ms=total_tick_ms,
                    rollback=tick_rollback,
                )
            except Exception as _e:
                rospy.logwarn_throttle(5.0,
                    '[sampling][%s] tick_fields build failed: %s', rospy.get_name(), _e)

            # ### HJ : publish JSON diagnostic (Claude/offline consumer) even when CSV
            # logger is off — separate sink, same fields.
            if tick_fields is not None:
                try:
                    chi_rel = self._ego_chi_vs_raceline(s_cent)
                    payload = self._build_tick_json(tick_fields, ego_chi=chi_rel)
                    self.pub_tick_json.publish(String(
                        data=json.dumps(payload, separators=(',', ':'))
                    ))
                except Exception as _e:
                    rospy.logwarn_throttle(5.0,
                        '[sampling][%s] tick_json publish failed: %s',
                        rospy.get_name(), _e)

            # Persistent CSV trace (ungated by tick_json).
            if self._logger is not None and tick_fields is not None:
                try:
                    self._logger.log_tick(tick_fields)
                    if self._logger.snapshot_every_n > 0:
                        cands = getattr(self.planner, 'candidates', None)
                        if cands is not None:
                            self._logger.log_snapshot(
                                self._tick_idx,
                                s=np.asarray(cands.get('s')),
                                n=np.asarray(cands.get('n')),
                                V=np.asarray(cands.get('V')),
                                valid=np.asarray(cands.get('valid')),
                                cost=np.asarray(getattr(self.planner, 'cost_array', [])),
                            )
                except Exception:
                    pass

            rate.sleep()

    # =============================================================================================
    # Publishing — trajectory  (EMA + resample + role-specific emission)
    # =============================================================================================
    def _publish_trajectory(self, traj):
        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        L = float(self.track.s[-1])

        # -- Unwrap raw s -----------------------------------------------------------------------
        s_raw = np.asarray(traj['s'], dtype=np.float64).copy()
        n_arr = np.asarray(traj['n'], dtype=np.float64).copy()
        V_arr = np.asarray(traj['V'], dtype=np.float64).copy()
        ax_arr = np.asarray(traj['ax'], dtype=np.float64).copy()
        kappa_arr = np.asarray(traj['kappa'], dtype=np.float64).copy()

        ds = np.diff(s_raw)
        wrap_adj = np.cumsum(
            np.where(ds < -L / 2.0,  L, 0.0) +
            np.where(ds >  L / 2.0, -L, 0.0)
        )
        s_unwr = s_raw.copy()
        s_unwr[1:] += wrap_adj

        # -- Backward-step truncation (upstream guard) -----------------------------------------
        BACKWARD_THR = -0.20
        ds_unwr = np.diff(s_unwr)
        bad = np.where(ds_unwr < BACKWARD_THR)[0]
        if len(bad) > 0:
            cut = int(bad[0]) + 1
            s_unwr    = s_unwr[:cut]
            n_arr     = n_arr[:cut]
            V_arr     = V_arr[:cut]
            ax_arr    = ax_arr[:cut]
            kappa_arr = kappa_arr[:cut]

        # -- Snapshot raw-unwrapped best (pre-filter) for next-tick continuity/EMA baseline ----
        s_raw_snap = s_unwr.copy()
        n_raw_snap = n_arr.copy()
        V_raw_snap = V_arr.copy()

        # -- EMA on (n, V) vs previous-tick raw best -------------------------------------------
        if (self._prev_best_n is not None and
                0.0 < self.filter_alpha < 1.0 and
                len(self._prev_best_n) > 0 and
                len(n_arr) > 0):
            a = self.filter_alpha
            m = int(min(len(self._prev_best_n), len(n_arr)))
            if m > 0:
                n_arr[:m] = a * n_arr[:m] + (1.0 - a) * self._prev_best_n[:m]
                V_arr[:m] = a * V_arr[:m] + (1.0 - a) * self._prev_best_V[:m]

        # -- Uniform arc-length resampling ------------------------------------------------------
        if self.resample_enable and len(s_unwr) >= 2 and self.resample_ds > 1e-3:
            s0, s1 = float(s_unwr[0]), float(s_unwr[-1])
            if (s1 - s0) > self.resample_ds:
                s_uniform = np.arange(s0, s1, self.resample_ds)
                if len(s_uniform) == 0 or s_uniform[-1] < s1 - 1e-6:
                    s_uniform = np.append(s_uniform, s1)
                n_arr     = np.interp(s_uniform, s_unwr, n_arr)
                V_arr     = np.interp(s_uniform, s_unwr, V_arr)
                ax_arr    = np.interp(s_uniform, s_unwr, ax_arr)
                kappa_arr = np.interp(s_uniform, s_unwr, kappa_arr)
                s_unwr    = s_uniform

        # -- Cartesian recomputation (consistent with filtered/resampled (s,n)) ---------------
        s_for_cart = np.clip(np.mod(s_unwr, L), 1e-6, L - 1e-6)
        try:
            xyz = self.track.sn2cartesian(s=s_for_cart, n=n_arr)
            xs_arr = np.asarray(xyz[:, 0], dtype=np.float64)
            ys_arr = np.asarray(xyz[:, 1], dtype=np.float64)
            zs_arr = np.asarray(xyz[:, 2], dtype=np.float64)
        except Exception as e:
            rospy.logwarn_throttle(2.0, '[sampling][%s] sn2cartesian failed: %s',
                                   rospy.get_name(), e)
            n_pts  = len(s_unwr)
            xs_arr = np.asarray(traj['x'][:n_pts], dtype=np.float64)
            ys_arr = np.asarray(traj['y'][:n_pts], dtype=np.float64)
            zs_arr = np.asarray(traj['z'][:n_pts], dtype=np.float64)

        # -- Cart-gap truncation -------------------------------------------------------------
        if len(xs_arr) >= 2:
            cart_d = np.sqrt(np.diff(xs_arr) ** 2 + np.diff(ys_arr) ** 2 + np.diff(zs_arr) ** 2)
            big = np.where(cart_d > 3.0)[0]
            if len(big) > 0:
                k = int(big[0]) + 1
                xs_arr    = xs_arr[:k]
                ys_arr    = ys_arr[:k]
                zs_arr    = zs_arr[:k]
                s_unwr    = s_unwr[:k]
                n_arr     = n_arr[:k]
                V_arr     = V_arr[:k]
                ax_arr    = ax_arr[:k]
                kappa_arr = kappa_arr[:k]

        if len(xs_arr) == 0:
            return

        # -- Build WpntArray --------------------------------------------------------------------
        wp_arr = WpntArray()
        wp_arr.header = header
        for i in range(len(xs_arr)):
            w = Wpnt()
            w.id          = i
            w.s_m         = float(s_unwr[i])
            w.d_m         = float(n_arr[i])
            w.x_m         = float(xs_arr[i])
            w.y_m         = float(ys_arr[i])
            if hasattr(w, 'z_m'):
                w.z_m = float(zs_arr[i])
            w.vx_mps      = float(V_arr[i])
            w.ax_mps2     = float(ax_arr[i])
            w.kappa_radpm = float(kappa_arr[i])
            wp_arr.wpnts.append(w)

        # -- Publish: debug (always) ------------------------------------------------------------
        self.pub_wpnts.publish(wp_arr)

        path = Path()
        path.header = header
        for i in range(len(xs_arr)):
            ps = PoseStamped()
            ps.header = header
            ps.pose.position.x = float(xs_arr[i])
            ps.pose.position.y = float(ys_arr[i])
            ps.pose.position.z = float(zs_arr[i])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_best_sample.publish(path)

        self._publish_best_markers(header, xs_arr, ys_arr, zs_arr, V_arr)
        self._publish_prev_best_marker(header)

        # -- Publish: role-specific output ------------------------------------------------------
        if self.state == 'overtake' and self.pub_out is not None:
            ot = OTWpntArray()
            ot.header           = header
            ot.last_switch_time = self.get_clock().now().to_msg()
            ot.side_switch      = False
            # Dominant side of the chosen trajectory (sign of mean n).
            mean_n = float(np.mean(n_arr)) if len(n_arr) > 0 else 0.0
            if mean_n >= 0.0:
                ot.ot_side = 'left'
            else:
                ot.ot_side = 'right'
            ot.ot_line = 'sampling'
            ot.wpnts = wp_arr.wpnts
            self.pub_out.publish(ot)
        elif self.state == 'recovery' and self.pub_out is not None:
            self.pub_out.publish(wp_arr)

        # -- Update previous-tick best for next iteration --------------------------------------
        self._prev_best_s   = s_raw_snap
        self._prev_best_n   = n_raw_snap
        self._prev_best_V   = V_raw_snap
        self._prev_best_xyz = (xs_arr.copy(), ys_arr.copy(), zs_arr.copy())

    # =============================================================================================
    # Markers
    # =============================================================================================
    def _publish_best_markers(self, header, xs_arr, ys_arr, zs_arr, V_arr):
        n_pts = len(xs_arr)
        if n_pts == 0:
            return
        vx_vals = [float(v) for v in V_arr[:n_pts]]
        vx_min = min(vx_vals) if vx_vals else 0.0
        vx_max = max(vx_vals) if vx_vals else 1.0

        def _vel_color(vx):
            t = (vx - vx_min) / (vx_max - vx_min) if vx_max > vx_min else 0.5
            return (max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5))),
                    max(0.0, min(1.0, 2.0 * t)),
                    0.0)

        # ### HJ : marker lifetime — self-expire after 0.2s so a publish gap
        # (> one tick) doesn't leave stale spheres in RViz (afterimage).
        mk_life = rospy.Duration(0.2)

        loc_markers = MarkerArray()
        clr = Marker(); clr.header = header; clr.action = Marker.DELETEALL
        loc_markers.markers.append(clr)
        for i in range(n_pts):
            mk = Marker()
            mk.header = header
            mk.type = Marker.SPHERE
            mk.id = i + 1
            mk.scale.x = mk.scale.y = mk.scale.z = 0.15
            mk.color.a = 1.0
            mk.color.r, mk.color.g, mk.color.b = _vel_color(vx_vals[i])
            mk.pose.position.x = float(xs_arr[i])
            mk.pose.position.y = float(ys_arr[i])
            mk.pose.position.z = float(zs_arr[i])
            mk.pose.orientation.w = 1.0
            mk.lifetime = mk_life
            loc_markers.markers.append(mk)
        self.pub_best_markers.publish(loc_markers)

        VEL_SCALE = 0.1317
        vel_markers = MarkerArray()
        clr2 = Marker(); clr2.header = header; clr2.action = Marker.DELETEALL
        vel_markers.markers.append(clr2)
        for i in range(n_pts):
            mk = Marker()
            mk.header = header
            mk.type = Marker.CYLINDER
            mk.id = i + 1
            mk.scale.x = mk.scale.y = 0.1
            height = max(vx_vals[i] * VEL_SCALE, 0.02)
            mk.scale.z = height
            mk.color.a = 0.7
            mk.color.r, mk.color.g, mk.color.b = _vel_color(vx_vals[i])
            mk.pose.position.x = float(xs_arr[i])
            mk.pose.position.y = float(ys_arr[i])
            mk.pose.position.z = float(zs_arr[i]) + height * 0.5
            mk.pose.orientation.w = 1.0
            mk.lifetime = mk_life
            vel_markers.markers.append(mk)
        self.pub_vel_markers.publish(vel_markers)

    def _publish_prev_best_marker(self, header):
        """Translucent gray LINE_STRIP of the previous tick's best — eyeball jitter in RViz."""
        ma = MarkerArray()
        clr = Marker(); clr.header = header; clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        if self._prev_best_xyz is not None:
            xs_p, ys_p, zs_p = self._prev_best_xyz
            mk = Marker()
            mk.header = header
            mk.ns = 'prev_best'
            mk.id = 1
            mk.type = Marker.LINE_STRIP
            mk.action = Marker.ADD
            mk.scale.x = 0.05
            mk.color.a = 0.45
            mk.color.r = 0.8; mk.color.g = 0.8; mk.color.b = 0.8
            mk.pose.orientation.w = 1.0
            for i in range(len(xs_p)):
                p = Point(); p.x = float(xs_p[i]); p.y = float(ys_p[i]); p.z = float(zs_p[i])
                mk.points.append(p)
            ma.markers.append(mk)
        self.pub_prev_marker.publish(ma)

    # =============================================================================================
    # Candidate-samples publishing (gray fan) — copied verbatim from sampling_planner_node.py
    # =============================================================================================
    def _publish_candidates(self, optimal_idx):
        cands = getattr(self.planner, 'candidates', None)
        if cands is None:
            return
        s_all     = np.asarray(cands['s'])
        n_all     = np.asarray(cands['n'])
        valid_all = np.asarray(cands['valid'], dtype=bool)
        N = s_all.shape[0]

        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        ma = MarkerArray()
        clr = Marker(); clr.header = header; clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        L = float(self.track.s[-1])
        drawn = 0
        best_marker = None
        _s_arr  = np.asarray(self.track.s)
        _x_arr  = np.asarray(self.track.x)
        _y_arr  = np.asarray(self.track.y)
        _z_arr  = np.asarray(self.track.z)
        _th_arr = np.asarray(self.track.theta)

        for i in range(N):
            is_best  = (i == optimal_idx)
            is_valid = bool(valid_all[i])

            s_row = s_all[i]
            n_row = n_all[i]
            ds = np.diff(s_row)
            wrap_adj = np.cumsum(
                np.where(ds < -L / 2.0,  L, 0.0) +
                np.where(ds >  L / 2.0, -L, 0.0)
            )
            s_unwr = s_row.copy().astype(np.float64)
            s_unwr[1:] += wrap_adj
            s_mod = np.clip(s_unwr % L, 1e-6, L - 1e-6)

            xc = np.interp(s_mod, _s_arr, _x_arr)
            yc = np.interp(s_mod, _s_arr, _y_arr)
            zc = np.interp(s_mod, _s_arr, _z_arr)
            th = np.interp(s_mod, _s_arr, _th_arr)
            xs_ = xc - np.sin(th) * n_row
            ys_ = yc + np.cos(th) * n_row
            zs_ = zc

            mk = Marker()
            mk.header = header
            mk.ns     = 'candidates'
            mk.id     = drawn + 1
            mk.type   = Marker.LINE_STRIP
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0

            if is_best:
                mk.scale.x = 0.07
                mk.color.r = 1.0; mk.color.g = 0.1; mk.color.b = 0.1
                mk.color.a = 1.0
            elif is_valid:
                mk.scale.x = 0.03
                mk.color.r = 0.1; mk.color.g = 0.1; mk.color.b = 0.1
                mk.color.a = 0.45
            else:
                mk.scale.x = 0.015
                mk.color.r = 0.6; mk.color.g = 0.6; mk.color.b = 0.6
                mk.color.a = 0.25

            for k in range(len(xs_)):
                p = Point(); p.x = float(xs_[k]); p.y = float(ys_[k]); p.z = float(zs_[k])
                mk.points.append(p)

            if is_best:
                best_marker = mk
            else:
                ma.markers.append(mk)
            drawn += 1

        if best_marker is not None:
            ma.markers.append(best_marker)
        self.pub_candidates.publish(ma)

    # =============================================================================================
    # MPPI-style weighted blending  (copied from sampling_planner_node.py, uses the continuity-
    # adjusted cost_array so blending inherits the continuity bias for free)
    # =============================================================================================
    def _mppi_blend(self):
        cands = getattr(self.planner, 'candidates', None)
        cost_arr = getattr(self.planner, 'cost_array', None)
        if cands is None or cost_arr is None:
            return None

        valid = np.asarray(cands['valid'], dtype=bool)
        if valid.sum() == 0:
            return None

        cost = np.asarray(cost_arr, dtype=np.float64).copy()
        if self.mppi_temporal_weight > 0.0 and self._prev_blended_s is not None:
            s_arr = np.asarray(cands['s'])
            n_arr = np.asarray(cands['n'])
            m = min(self._prev_blended_s.shape[0], s_arr.shape[1])
            ds = s_arr[:, :m] - self._prev_blended_s[:m]
            dn = n_arr[:, :m] - self._prev_blended_n[:m]
            tempo = np.sum(ds * ds + dn * dn, axis=1)
            cost = cost + self.mppi_temporal_weight * tempo

        valid_costs = cost[valid]
        c_min = float(valid_costs.min())
        c_max = float(valid_costs.max())
        c_range = max(c_max - c_min, 1e-6)
        T = max(self.mppi_temperature_rel * c_range, 1e-6)

        w = np.exp(-(valid_costs - c_min) / T)
        w /= w.sum()

        # ### HJ : MPPI blending diagnostics — effective sample size and top-1 mass.
        # eff_n = 1/Σw² (Kish ESS): N_blend means uniform, 1 means single candidate.
        # top1_w 는 가장 큰 가중치 (≥0.95 면 사실상 argmax). T_used 는 절대 temperature.
        try:
            self._last_mppi_stats = {
                'mppi_eff_n':  float(1.0 / float(np.sum(w * w))),
                'mppi_top1_w': float(np.max(w)),
                'mppi_T_used': float(T),
                'mppi_n_blend': int(w.size),
            }
        except Exception:
            self._last_mppi_stats = {}

        L = float(self.track.s[-1])
        s_valid = np.asarray(cands['s'])[valid].astype(np.float64).copy()
        ds_valid = np.diff(s_valid, axis=1)
        wrap_adj = np.cumsum(
            np.where(ds_valid < -L / 2.0,  L, 0.0) +
            np.where(ds_valid >  L / 2.0, -L, 0.0),
            axis=1,
        )
        s_valid[:, 1:] += wrap_adj
        anchor = float(np.mean(s_valid[:, 0]))
        for k in range(s_valid.shape[0]):
            while s_valid[k, 0] - anchor > L / 2.0:
                s_valid[k] -= L
            while s_valid[k, 0] - anchor < -L / 2.0:
                s_valid[k] += L

        blended = {
            'traj_cnt': self.planner.traj_cnt,
            'optimal_idx': int(np.arange(valid.size)[valid][int(np.argmax(w))]),
        }
        blended['s'] = np.sum(w[:, None] * s_valid, axis=0)
        for key in ('n', 'V', 'chi', 'ax', 'ay', 'kappa', 't'):
            arr = np.asarray(cands[key])
            if arr.ndim == 1:
                continue
            blended[key] = np.sum(w[:, None] * arr[valid], axis=0)

        try:
            s_for_cart = np.clip(np.mod(blended['s'], L), 1e-6, L - 1e-6)
            xyz = self.track.sn2cartesian(s=s_for_cart, n=blended['n'])
            blended['x'] = np.asarray(xyz[:, 0], dtype=np.float64)
            blended['y'] = np.asarray(xyz[:, 1], dtype=np.float64)
            blended['z'] = np.asarray(xyz[:, 2], dtype=np.float64)
        except Exception as e:
            rospy.logwarn_throttle(2.0, '[sampling][%s][mppi] cartesian failed: %s',
                                   rospy.get_name(), e)
            return None

        self._prev_blended_s = np.asarray(blended['s'], dtype=np.float64).copy()
        self._prev_blended_n = np.asarray(blended['n'], dtype=np.float64).copy()
        return blended


def main():
    node = SamplingPlannerStateNode()
    node.spin()


if __name__ == '__main__':
    main()
