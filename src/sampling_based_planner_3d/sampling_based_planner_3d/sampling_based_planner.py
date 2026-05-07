import numpy as np
from track3D import Track3D
import copy
import sys

g_earth = 9.81

# ### HJ : debug/trace helper — stderr when SPLANNER_DEBUG=1, and a persistent
# file log whenever the SPLANNER_LOG env flag is truthy (default off).
# Log destination defaults to <pkg>/debug_log/ so it stays inside the package tree.
import os as _os
import time as _time
_DEBUG = bool(int(_os.environ.get('SPLANNER_DEBUG', '0') or '0'))
_LOG_ENABLED = bool(int(_os.environ.get('SPLANNER_LOG', '0') or '0'))
# Override log dir via SPLANNER_LOG_DIR; otherwise resolve to <pkg>/debug_log.
_PKG_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_LOG_DIR  = _os.environ.get('SPLANNER_LOG_DIR', _os.path.join(_PKG_ROOT, 'debug_log'))
_LOG_FP   = None
def _log_open():
    global _LOG_FP
    if _LOG_FP is not None: return
    if not _LOG_ENABLED: return
    try:
        _os.makedirs(_LOG_DIR, exist_ok=True)
        ts = _time.strftime('%Y%m%d_%H%M%S')
        path = _os.path.join(_LOG_DIR, 'sampling_planner_internal_%s.log' % ts)
        _LOG_FP = open(path, 'w')
        _LOG_FP.write('# sampling planner internal log — opened %s\n' % ts)
        _LOG_FP.flush()
        sys.stderr.write('[splanner-log] writing internals to %s\n' % path)
    except Exception as _e:
        sys.stderr.write('[splanner-log] open failed: %s\n' % _e)

def _flog(msg):
    global _LOG_FP
    if _LOG_FP is None:
        _log_open()
    if _LOG_FP is not None:
        try:
            _LOG_FP.write(msg + '\n')
        except Exception:
            pass

def _hj_dbg(msg):
    # Append to the persistent file log (when enabled) and also emit to stderr
    # when SPLANNER_DEBUG=1. Function kept as `_hj_dbg` so existing call-sites
    # don't need mass-renaming; it's just a short name, not a hardcoded path.
    if _LOG_ENABLED:
        _flog(msg)
    if _DEBUG:
        sys.stderr.write('[splanner-dbg] ' + msg + '\n')
        sys.stderr.flush()

def _log_flush():
    global _LOG_FP
    if _LOG_FP is not None:
        try: _LOG_FP.flush()
        except Exception: pass


class LocalSamplingPlanner():

    def __init__(
            self,
            params,
            track_handler,
            gg_handler
    ):
        self.vehicle_params = params['vehicle_params']
        self.track_handler = track_handler
        self.left_track_bounds, self.right_track_bounds = track_handler.get_track_bounds(margin=0.0) 
        self.gggv_handler = gg_handler

        self.trajectory = {}
        self.traj_cnt = 0

    def calc_trajectory(
            self, 
            state: dict,
            prediction: dict,
            raceline: dict,
            relative_generation: bool,
            n_samples: int,
            v_samples: int,
            horizon: float,
            num_samples: int,
            safety_distance: float,
            gg_abs_margin: float,
            friction_check_2d: bool,
            s_dot_min: float = 1.0,
            kappa_thr: float = 0.1,
            raceline_cost_weight: float = 0.1,
            velocity_cost_weight: float = 100.0,
            prediction_cost_weight: float = 5000.0,
            prediction_s_factor: float = 0.015,
            prediction_n_factor: float = 0.5,
            endpoint_chi_raceline_only: bool = False,
    ):
        self.traj_cnt += 1

        # frenet state
        s_start = state['s']
        s_dot_start = max(s_dot_min, state['s_dot'])
        s_ddot_start = state['s_ddot']
        n_start = state['n']
        n_dot_start = state['n_dot']
        n_ddot_start = state['n_ddot']

        # ### HJ : entry log — full lap capture (no longer gated).
        L_dbg = float(self.track_handler.s[-1])
        near_boundary = True   # log every tick
        if near_boundary:
            _hj_dbg('==== calc_trajectory cnt=%d ====' % self.traj_cnt)
            _hj_dbg('  state: s=%.4f n=%.4f s_dot=%.4f s_ddot=%.4f n_dot=%.4f n_ddot=%.4f'
                    % (s_start, n_start, s_dot_start, s_ddot_start, n_dot_start, n_ddot_start))
            _hj_dbg('  L=%.4f (s_start near boundary: %s)' % (L_dbg, 'YES' if near_boundary else 'no'))
            _hj_dbg('  raceline raw: s=[%.3f..%.3f] (N=%d)  V=[%.2f..%.2f]  t=[%.3f..%.3f]'
                    % (float(raceline['s'][0]), float(raceline['s'][-1]), len(raceline['s']),
                       float(np.min(raceline['V'])), float(np.max(raceline['V'])),
                       float(raceline['t'][0]), float(raceline['t'][-1])))
            _hj_dbg('  params: relative=%s  n_samples=%d  v_samples=%d  horizon=%.3f  num_samples=%d'
                    % (relative_generation, n_samples, v_samples, horizon, num_samples))

        # postprocess raceline
        postprocessed_raceline = self.postprocess_raceline(
            raw_raceline=raceline,
            s_start=s_start,
            horizon=horizon,
            track_handler=self.track_handler
        )

        # ### HJ : log postprocessed raceline state right after postprocess
        if near_boundary:
            _ps  = postprocessed_raceline['s_post']
            _pt  = postprocessed_raceline['t_post']
            _pv  = postprocessed_raceline['s_dot_post']
            _hj_dbg('  POST: s_post=[%.4f..%.4f] (N=%d, monotonic=%s)  t_post=[%.4f..%.4f]  V=[%.3f..%.3f]'
                    % (float(_ps[0]), float(_ps[-1]), len(_ps),
                       'YES' if np.all(np.diff(_ps) >= -1e-9) else 'NO!!',
                       float(_pt[0]), float(_pt[-1]),
                       float(np.min(_pv)), float(np.max(_pv))))
            # if length is short (no attach), warn
            if _pt[-1] < horizon * 1.5 - 1e-6:
                _hj_dbg('  WARN: t_post[-1]=%.4f < horizon*1.5=%.4f  → np.interp will CLAMP'
                        % (float(_pt[-1]), horizon * 1.5))

        # curves are generated relative to raceline for better tracking of the raceline
        # ### HJ : original code did `raceline['s_dot'][0]` which is the value at s=0,
        # NOT at s_start. That's almost surely a bug — comparing the car's current
        # speed against raceline-at-lap-origin instead of raceline-at-current-position.
        # Keep upstream behaviour unchanged but log it so we can see the effect.
        rl_v0 = float(raceline['s_dot'][0])
        raceline_tendency_s = False
        if abs(rl_v0 - s_dot_start) / max(rl_v0, 1e-3) < 0.3 and relative_generation:
            raceline_tendency_s = True
        if near_boundary:
            _hj_dbg('  raceline_tendency_s=%s  (raceline.s_dot[0]=%.3f vs s_dot_start=%.3f)'
                    % (raceline_tendency_s, rl_v0, s_dot_start))

        # time arrays
        t_vector = np.linspace(0.0, horizon, num_samples)
        t_array = np.tile(t_vector, (v_samples * n_samples, 1))

        # generate longitudinal curves
        s_array, s_dot_array, s_ddot_array, s_end_values, s_dot_end_values = self.generate_longitudinal_curves(
            track_handler=self.track_handler,
            s_start=s_start,
            s_dot_start=s_dot_start,
            s_ddot_start=s_ddot_start,
            s_dot_min=s_dot_min,
            t_array=t_array,
            v_samples=v_samples,
            n_samples=n_samples,
            postprocessed_raceline=postprocessed_raceline,
            horizon=horizon,
            raceline_tendency=raceline_tendency_s,
        )

        # ### HJ : pure shifted-raceline sampling — all candidates = n_rl(s) + d_i.
        # No connector variants → factor=1 (candidate count = n_samples * v_samples).
        # `relative_generation` param is no longer meaningful (absolute vs raceline-
        # relative was the old distinction; now ALL candidates are raceline-relative
        # by construction). Kept in signature for API compat.
        raceline_tendency_n = False
        n_array, n_dot_array, n_ddot_array = self.generate_lateral_curves(
            track_handler=self.track_handler,
            s_array=s_array,
            s_dot_array=s_dot_array,
            s_ddot_array=s_ddot_array,
            s_end_values=s_end_values,
            s_dot_end_values=s_dot_end_values,
            n_start=n_start,
            n_dot_start=n_dot_start,
            n_ddot_start=n_ddot_start,
            t_array=t_array,
            n_samples=n_samples,
            postprocessed_raceline=postprocessed_raceline,
            safety_distance=safety_distance,
            raceline_tendency=raceline_tendency_n,
            endpoint_chi_raceline_only=endpoint_chi_raceline_only,
        )

        # transform frenet curves to velocity frame
        V_array, chi_array, ax_vf_array, ay_vf_array, kappa_array = \
            self.transform_to_velocity_frame(
                track_handler=self.track_handler,
                s_array=s_array,
                s_dot_array=s_dot_array,
                s_ddot_array=s_ddot_array,
                n_array=n_array,
                n_dot_array=n_dot_array,
                n_ddot_array=n_ddot_array,
            )

        # valid_array specifies which trajectories are valid in terms of feasibility. Initially all valid.
        # ### HJ : factor=1 always — shifted-raceline refactor removed the
        # absolute-vs-raceline-relative split (all candidates are now raceline-relative).
        factor = 1
        valid_array = np.ones(factor * n_samples * v_samples, dtype=bool)

        # ### HJ : invalid-reason counters — snapshot valid count before each check so the
        # node can report (total, killed_curv, killed_path, killed_fric, valid_after_all).
        # Each check's soft-fallback (re-enable one candidate when all would die) is counted
        # as "not killed" by design — we want the count of candidates that would have been
        # killed by that predicate if no fallback were active.
        # checks modify the valid array. The order of the checks can have influence on the calculation time
        n_before_any = int(valid_array.sum())
        self.check_curvature(
            valid_array=valid_array,
            kappa=kappa_array,
            kappa_thr=kappa_thr,
        )
        n_after_curv = int(valid_array.sum())

        self.check_path_collision(
            track_handler=self.track_handler,
            valid_array=valid_array,
            s_array=s_array,
            n_array=n_array,
            safety_distance=safety_distance
        )
        n_after_path = int(valid_array.sum())

        self.check_friction_limits(
            valid_array=valid_array,
            track_handler=self.track_handler,
            s_array=s_array,
            V_array=V_array,
            n_array=n_array,
            chi_array=chi_array,
            ax_array=ax_vf_array,
            ay_array=ay_vf_array,
            friction_check_2d=friction_check_2d,
            gg_abs_margin=gg_abs_margin
        )
        n_after_fric = int(valid_array.sum())

        # ### HJ : expose counts for ~debug/tick_json. "killed_*" is the net drop at each
        # stage (may be 0 if soft-fallback rescued one). "valid_after_all" = candidates that
        # survived every hard check. Useful to see which stage is the bottleneck per-tick.
        self.check_stats = {
            'total':             int(valid_array.size),
            'valid_before_any':  n_before_any,
            'killed_curvature':  max(0, n_before_any - n_after_curv),
            'killed_path':       max(0, n_after_curv - n_after_path),
            'killed_friction':   max(0, n_after_path - n_after_fric),
            'valid_after_all':   n_after_fric,
        }

        # ### HJ : expose all candidate arrays (and the valid mask) so the ROS node
        # can visualise every sample — upstream only kept the chosen best internally.
        # Shapes: s_array / n_array / V_array = (v_samples*n_samples, num_samples)
        #         valid_array = (v_samples*n_samples,) bool
        self.candidates = {
            's':        s_array,
            'n':        n_array,
            'V':        V_array,
            'chi':      chi_array,
            'ax':       ax_vf_array,
            'ay':       ay_vf_array,
            'kappa':    kappa_array,
            'valid':    valid_array,
            't':        t_array,
        }
        # ### HJ : lightning diagnosis — dump one mid-lap-boundary candidate's full
        # s, n arrays whenever we're near the lap boundary. We want to see if s
        # actually "stops" near L while n diverges.
        if near_boundary:
            valid_idx = np.where(valid_array)[0]
            if len(valid_idx) > 0:
                j = int(valid_idx[0])
                _hj_dbg('  CAND#%d s_start=%.3f' % (j, s_start))
                _hj_dbg('    s_array[j]   = %s' % np.array2string(s_array[j], precision=3, max_line_width=200))
                _hj_dbg('    n_array[j]   = %s' % np.array2string(n_array[j], precision=3, max_line_width=200))
                _hj_dbg('    diff(s_array[j]) = %s' % np.array2string(np.diff(s_array[j]), precision=3, max_line_width=200))

        # choose best trajectory
        optimal_idx = self.get_optimal_trajectory_idx(
            valid_array=valid_array,
            track_handler=self.track_handler,
            s_array=s_array,
            n_array=n_array,
            t_array=t_array,
            V_array=V_array,
            raceline=raceline,
            prediction=prediction,
            raceline_cost_weight=raceline_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
            prediction_cost_weight=prediction_cost_weight,
            prediction_s_factor=prediction_s_factor,
            prediction_n_factor=prediction_n_factor
        )

        # set planned trajectory
        self.trajectory.clear()
        self.trajectory['traj_cnt'] = self.traj_cnt
        self.trajectory['optimal_idx'] = int(optimal_idx)   # ### HJ : expose for viz
        self.trajectory["t"] = t_array[optimal_idx]
        self.trajectory["s"] = s_array[optimal_idx]
        self.trajectory["s_dot"] = s_dot_array[optimal_idx]
        self.trajectory["s_ddot"] = s_ddot_array[optimal_idx]
        self.trajectory["n"] = n_array[optimal_idx]
        self.trajectory["n_dot"] = n_dot_array[optimal_idx]
        self.trajectory["n_ddot"] = n_ddot_array[optimal_idx]
        self.trajectory["V"] = V_array[optimal_idx]
        self.trajectory["chi"] = chi_array[optimal_idx]
        self.trajectory["ax"] = ax_vf_array[optimal_idx]
        self.trajectory["ay"] = ay_vf_array[optimal_idx]
        self.trajectory["kappa"] = kappa_array[optimal_idx]
        xyz_array = self.track_handler.sn2cartesian(s=self.trajectory["s"], n=self.trajectory["n"])
        self.trajectory["x"] = xyz_array[:, 0]
        self.trajectory["y"] = xyz_array[:, 1]
        self.trajectory["z"] = xyz_array[:, 2]

        return self.trajectory


    def get_optimal_trajectory_idx(
            self,
            valid_array: np.ndarray,
            track_handler: Track3D,
            s_array: np.ndarray,
            n_array: np.ndarray,
            t_array: np.ndarray,
            V_array: np.ndarray,
            raceline: dict,
            prediction: dict,
            raceline_cost_weight: float,
            velocity_cost_weight: float,
            prediction_cost_weight: float,
            prediction_s_factor: float,
            prediction_n_factor: float
    ) -> int:
        velocity_cost_array = np.zeros_like(valid_array, dtype=float)
        raceline_cost_array = np.zeros_like(valid_array, dtype=float)
        prediction_cost_array = np.zeros_like(valid_array, dtype=float)

        # ### HJ : raw (unweighted) per-term arrays — exposed so the node can
        # apply per-tick min-max normalization before weight×sum, so each cost
        # term contributes a commensurate magnitude regardless of physical units.
        velocity_cost_raw     = np.zeros_like(valid_array, dtype=float)
        raceline_cost_raw     = np.zeros_like(valid_array, dtype=float)
        prediction_cost_raw   = np.zeros_like(valid_array, dtype=float)

        # time difference array for integration
        diff_time_array = np.diff(t_array[valid_array], axis=1)

        # velocity costs
        V_raceline = np.interp(s_array[valid_array], raceline['s'], raceline['V'], period=track_handler.s[-1])
        velocity_cost_raw[valid_array] = np.add.reduce(
            ((V_array[valid_array, :-1] - V_raceline[:, :-1]) / V_raceline[:, :-1]) ** 2 * diff_time_array,
            axis=1
        )
        velocity_cost_array = velocity_cost_weight * velocity_cost_raw

        # raceline costs
        raceline_deviation = np.interp(s_array[valid_array], raceline['s'], raceline['n'], period=track_handler.s[-1]) - n_array[valid_array]
        raceline_cost_raw[valid_array] = np.add.reduce(
            raceline_deviation[:, :-1] ** 2 * diff_time_array,
            axis=1
        )
        raceline_cost_array = raceline_cost_weight * raceline_cost_raw

        # prediction costs
        for prediction_id in prediction:
            prediction_cur = prediction[prediction_id]

            s_prediction_cur = np.interp(t_array[valid_array], prediction_cur["t"], prediction_cur["s"])
            n_prediction_cur = np.interp(t_array[valid_array], prediction_cur["t"], prediction_cur["n"])
            raw_prediction_costs = np.exp(- prediction_s_factor * (s_array[valid_array] - s_prediction_cur) ** 2 - prediction_n_factor * (n_array[valid_array] - n_prediction_cur) ** 2)
            prediction_cost_raw[valid_array] += np.add.reduce(raw_prediction_costs[:, :-1] * diff_time_array, axis=1)
        prediction_cost_array = prediction_cost_weight * prediction_cost_raw

        # overall costs (legacy weighted sum, kept for backward compat)
        cost_array = velocity_cost_array + raceline_cost_array + prediction_cost_array

        # ### HJ : expose cost breakdown for downstream MPPI-style weighted blending.
        # Node can skip this by ignoring self.cost_* attributes.
        self.cost_array            = cost_array
        self.cost_velocity         = velocity_cost_array
        self.cost_raceline         = raceline_cost_array
        self.cost_prediction       = prediction_cost_array
        # ### HJ : raw (unweighted) versions for normalize+sum assembly in node.
        self.cost_velocity_raw     = velocity_cost_raw
        self.cost_raceline_raw     = raceline_cost_raw
        self.cost_prediction_raw   = prediction_cost_raw

        # return index of trajectory with the lowest cost
        opt_subset_idx = np.argmin(cost_array[valid_array])
        opt_idx = np.arange(cost_array.shape[0])[valid_array][opt_subset_idx]
        return opt_idx

    def check_friction_limits(
            self,
            valid_array: np.ndarray,
            track_handler,
            s_array: np.ndarray,
            V_array: np.ndarray,
            n_array: np.ndarray,
            chi_array: np.ndarray,
            ax_array: np.ndarray,
            ay_array: np.ndarray,
            friction_check_2d: bool,
            gg_abs_margin: float = 0.0,
            soft_check: bool = True
    ):
        ax_tilde = np.zeros_like(s_array)
        ay_tilde = np.zeros_like(s_array)
        g_tilde = np.zeros_like(s_array)

        if friction_check_2d:
            ax_tilde[valid_array] = ax_array[valid_array]
            ay_tilde[valid_array] = ay_array[valid_array]
            g_tilde[valid_array] = 9.81 * np.ones_like(s_array[valid_array])
        else:
            ax_tilde[valid_array], ay_tilde[valid_array], g_tilde[valid_array] = track_handler.calc_apparent_accelerations_numpy(
                s=s_array[valid_array],
                V=V_array[valid_array],
                n=n_array[valid_array],
                chi=chi_array[valid_array],
                ax=ax_array[valid_array],
                ay=ay_array[valid_array]
            )

        gg_exponent, ax_min, ax_max, ay_max = self.gggv_handler.acc_interpolator(
            np.array((V_array[valid_array].flatten(), g_tilde[valid_array].flatten()))
        ).full().squeeze().reshape(4, g_tilde[valid_array].shape[0], g_tilde[valid_array].shape[1])
        ax_avail = np.abs(ax_min) * np.power(
            np.maximum(
                (1.0 - np.power(
                    np.minimum(np.abs(ay_tilde[valid_array]) / ay_max, 1.0),
                    gg_exponent
                )),
                1e-3
            ),
            1.0 / gg_exponent
        )
        valid_tmp = np.all(np.abs(ay_tilde[valid_array]) <= ay_max + gg_abs_margin, axis=1) & \
                    np.all(np.abs(ax_tilde[valid_array]) <= ax_avail + gg_abs_margin, axis=1) & \
                    np.all(ax_tilde[valid_array] <= ax_max + gg_abs_margin, axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                axy_exc = np.max(np.abs(ax_tilde[valid_array]) - ax_avail, axis=1)
                exc_min_idx = np.argmin(axy_exc)
                valid_tmp[exc_min_idx] = True      

        valid_array[valid_array] = valid_tmp

    def check_curvature(
            self,
            valid_array: np.ndarray,
            kappa: np.ndarray,
            kappa_thr: float,
            soft_check: bool = True
    ):
        valid_tmp = np.all(np.abs(kappa[valid_array]) <= kappa_thr, axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                kappa_max = np.abs(kappa[valid_array]).max(axis=1)
                exc_min_idx = np.argmin(kappa_max)
                valid_tmp[exc_min_idx] = True

        valid_array[valid_array] = valid_tmp

    def check_path_collision(
            self,
            track_handler: Track3D,
            valid_array: np.ndarray,
            s_array: np.ndarray,
            n_array: np.ndarray,
            safety_distance: float,
            soft_check: bool = True
    ):
        left_bound = np.interp(s_array[valid_array], track_handler.s, track_handler.w_tr_left, period=track_handler.s[-1]) - self.vehicle_params['total_width'] / 2.0 - safety_distance
        right_bound = np.interp(s_array[valid_array], track_handler.s, track_handler.w_tr_right, period=track_handler.s[-1]) + self.vehicle_params['total_width'] / 2.0 + safety_distance
        valid_tmp = np.all((n_array[valid_array] < left_bound) & (n_array[valid_array] > right_bound), axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                d_exc = np.maximum(np.max(n_array[valid_array] - left_bound, axis=1), np.max(right_bound - n_array[valid_array], axis=1))
                exc_min_idx = np.argmin(d_exc)
                valid_tmp[exc_min_idx] = True

        valid_array[valid_array] = valid_tmp

    def transform_to_velocity_frame(
            self,
            track_handler: Track3D,
            s_array: np.ndarray, 
            s_dot_array: np.ndarray, 
            s_ddot_array: np.ndarray,
            n_array: np.ndarray, 
            n_dot_array: np.ndarray, 
            n_ddot_array: np.ndarray,
    ):
        # angular velocity of road frame with respect to s expressed in road frame
        Omega_z_rf_array = np.interp(s_array, track_handler.s, track_handler.Omega_z, period=track_handler.s[-1])
        dOmega_z_rf_array = np.interp(s_array, track_handler.s, track_handler.dOmega_z, period=track_handler.s[-1])

        # absolute velocity
        v_array = np.sqrt((1.0 - Omega_z_rf_array * n_array) ** 2 * s_dot_array ** 2 + n_dot_array ** 2)

        # orientation of the velocity vector relative to reference line
        chi_array = np.arctan(n_dot_array / (s_dot_array * (1.0 - Omega_z_rf_array * n_array)))

        # x-acceleration in velocity frame
        ax_vf_array = 1 / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                        (
                            s_dot_array * s_ddot_array * (1.0 - Omega_z_rf_array * n_array) ** 2
                            - s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) * (dOmega_z_rf_array * s_dot_array * n_array + Omega_z_rf_array * n_dot_array)
                            + n_dot_array * n_ddot_array
                        )

        # y-acceleration in velocity frame
        ay_vf_array = 1 / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                        (
                            s_dot_array * n_dot_array * (dOmega_z_rf_array * s_dot_array * n_array + 2.0 * Omega_z_rf_array * n_dot_array)
                            - s_ddot_array * n_dot_array * (1.0 - Omega_z_rf_array * n_array)
                            + s_dot_array ** 3 * Omega_z_rf_array * (1 - Omega_z_rf_array * n_array) ** 2
                            + s_dot_array * n_ddot_array * (1.0 - Omega_z_rf_array * n_array)
                        )

        # angular velocity of velocity frame with respect to s
        kappa_array = s_dot_array / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                            (
                                1.0 / s_dot_array * (
                                s_dot_array * (1.0 - Omega_z_rf_array * n_array) * n_ddot_array - n_dot_array * (
                                s_ddot_array * (1.0 - Omega_z_rf_array * n_array) - s_dot_array * (
                                dOmega_z_rf_array * s_dot_array * n_array + Omega_z_rf_array * n_dot_array))) /
                                (s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) +
                                Omega_z_rf_array
                            )

        return v_array, chi_array, ax_vf_array, ay_vf_array, kappa_array
    
    def postprocess_raceline(
            self,
            raw_raceline: dict,
            s_start: float,
            horizon: float,
            track_handler: Track3D
    ):
        postprocessed_raceline = copy.deepcopy(raw_raceline)

        t_rl_raw = raw_raceline['t']
        n_rl_raw = raw_raceline['n']
        n_rl_dot_raw = raw_raceline['n_dot']
        n_rl_ddot_raw = raw_raceline['n_ddot']
        s_rl_raw = raw_raceline['s']
        s_rl_dot_raw = raw_raceline['s_dot']
        s_rl_ddot_raw = raw_raceline['s_ddot']
        v_rl_raw = raw_raceline['V']
        chi_rl_raw = raw_raceline['chi']
        ax_rl_raw = raw_raceline['ax']
        ay_rl_raw = raw_raceline['ay']

        # interpolate data points at s_start
        t_rl_start = np.interp(s_start, s_rl_raw, t_rl_raw, period=track_handler.s[-1])
        n_rl_start = np.interp(s_start, s_rl_raw, n_rl_raw, period=track_handler.s[-1])
        n_rl_dot_start = np.interp(s_start, s_rl_raw, n_rl_dot_raw, period=track_handler.s[-1])
        n_rl_ddot_start = np.interp(s_start, s_rl_raw, n_rl_ddot_raw, period=track_handler.s[-1])
        s_rl_dot_start = np.interp(s_start, s_rl_raw, s_rl_dot_raw, period=track_handler.s[-1])
        s_rl_ddot_start = np.interp(s_start, s_rl_raw, s_rl_ddot_raw, period=track_handler.s[-1])
        v_rl_start = np.interp(s_start, s_rl_raw, v_rl_raw, period=track_handler.s[-1])
        chi_rl_start = np.interp(s_start, s_rl_raw, chi_rl_raw, period=track_handler.s[-1])
        ax_rl_start = np.interp(s_start, s_rl_raw, ax_rl_raw, period=track_handler.s[-1])
        ay_rl_start = np.interp(s_start, s_rl_raw, ay_rl_raw, period=track_handler.s[-1])

        # insert data points at s_start into raceline
        rl_idx_start = np.searchsorted(t_rl_raw, t_rl_start)
        t_rl = np.insert(t_rl_raw, rl_idx_start, t_rl_start)
        n_rl = np.insert(n_rl_raw, rl_idx_start, n_rl_start)
        n_rl_dot = np.insert(n_rl_dot_raw, rl_idx_start, n_rl_dot_start)
        n_rl_ddot = np.insert(n_rl_ddot_raw, rl_idx_start, n_rl_ddot_start)
        s_rl = np.insert(s_rl_raw, rl_idx_start, s_start)
        s_rl_dot = np.insert(s_rl_dot_raw, rl_idx_start, s_rl_dot_start)
        s_rl_ddot = np.insert(s_rl_ddot_raw, rl_idx_start, s_rl_ddot_start)
        v_rl = np.insert(v_rl_raw, rl_idx_start, v_rl_start)
        chi_rl = np.insert(chi_rl_raw, rl_idx_start, chi_rl_start)
        ax_rl = np.insert(ax_rl_raw, rl_idx_start, ax_rl_start)
        ay_rl = np.insert(ay_rl_raw, rl_idx_start, ay_rl_start)

        # remove all points before data point at s_start
        t_rl = t_rl[rl_idx_start:]
        n_rl = n_rl[rl_idx_start:]
        n_rl_dot = n_rl_dot[rl_idx_start:]
        n_rl_ddot = n_rl_ddot[rl_idx_start:]
        s_rl = s_rl[rl_idx_start:]
        s_rl_dot = s_rl_dot[rl_idx_start:]
        s_rl_ddot = s_rl_ddot[rl_idx_start:]
        v_rl = v_rl[rl_idx_start:]
        chi_rl = chi_rl[rl_idx_start:]
        ax_rl = ax_rl[rl_idx_start:]
        ay_rl = ay_rl[rl_idx_start:]

        # shift data point at s_start to t=0
        t_rl = t_rl - t_rl[0]

        # ### HJ : wrap-attach — fix "sample tail clamped at lap end" near s ≈ L.
        # ---------------------------------------------------------------------------
        # Problem:
        #   The raceline CSV covers a single lap (s = 0 .. L). When s_start is close
        #   to L the slice above leaves only a few meters in the tail, so
        #   t_rl[-1] < horizon*1.5. Then inside generate_longitudinal_curves the
        #   call `np.interp(t, t_post, s_post)` clamps at the last value when
        #   t > t_rl[-1] — i.e. s_rl_eval freezes at s = L. The resulting polynomial
        #   candidates produce a trajectory whose tail stays glued to the lap-end
        #   position (visible as the "sample end stuck / going backward" artifact
        #   near the finish line).
        #
        # Fix:
        #   Concatenate the raceline HEAD (the first samples of the next lap) to the
        #   tail, offset in both s (+L) and t (+shifted tail end). Because the CSV
        #   is a closed loop (first point == last point in cartesian), this attach
        #   is physically continuous — it just labels the second-lap samples with
        #   s > L so the slice-based code downstream sees a monotonically growing
        #   reference covering the full planning horizon.
        #
        # Safety:
        #   The horizon*1.5 clip directly below this block removes any samples that
        #   go past the planning window, so over-attaching is harmless. K is chosen
        #   to just barely cover `remainder_t`, with a +2 sample margin.
        #
        # How the wrap round-trip stays consistent:
        #   generate_longitudinal_curves finally wraps output via `s = np.mod(s, L)`
        #   (line 582), so s values end up in [0, L) for storage. That `mod` IS the
        #   canonical statement of "s = 81 and s = 1 are the same on an 80 m track".
        #   The visual continuity across the wrap is restored in the ROS node's
        #   _publish_trajectory (ds < -L/2 unwrap + cummax + sn2cartesian(s%L, n)).
        L     = float(track_handler.s[-1])
        T_lap = float(t_rl_raw[-1] - t_rl_raw[0])
        # ### HJ : log attach decision in lap-boundary region.
        _near_b_pp = True   # ### HJ : full-lap capture (ungated)
        if _near_b_pp:
            _hj_dbg('  PP-ATTACH check: t_rl[-1]=%.4f horizon*1.5=%.4f T_lap=%.4f → %s'
                    % (float(t_rl[-1]), horizon * 1.5, T_lap,
                       'WILL ATTACH' if (t_rl[-1] < horizon * 1.5 and T_lap > 1e-6) else 'no attach'))
        # ### HJ : DIAGNOSTIC — disable wrap-attach to test if it's the lightning cause.
        if len(t_rl) > 0 and t_rl[-1] < horizon * 1.5 and T_lap > 1e-6:
            # Only cover the missing time window; +1e-3 s margin for boundary rounding.
            remainder_t = horizon * 1.5 - t_rl[-1] + 1e-3
            # Number of head samples required. +2 guards against searchsorted
            # underestimation when raw sample spacing is irregular.
            K = int(np.searchsorted(t_rl_raw - t_rl_raw[0], remainder_t)) + 2
            K = min(K, len(t_rl_raw))
            if _near_b_pp:
                _hj_dbg('  PP-ATTACH: remainder_t=%.4f K=%d  raw head s=[%.3f..%.3f]  raw head t=[%.3f..%.3f]'
                        % (remainder_t, K,
                           float(s_rl_raw[0]), float(s_rl_raw[K-1] if K>0 else s_rl_raw[0]),
                           float(t_rl_raw[0]), float(t_rl_raw[K-1] if K>0 else t_rl_raw[0])))
            if K > 1:
                # ### HJ : SKIP raw[0] to avoid duplicate-seam plateau.
                # Reason: raceline['s'][-1] is pinned to L - 1e-6 (for numpy periodic-interp
                # safety). Attach values start from `raw[0] + L = 0 + L_track = L`. So
                # s_post seam is [..., L - 1e-6, L, raw[1] + L, ...]. The two seam samples
                # are only 1e-6 m apart in s BUT 67 ms apart in t (the dt_first gap).
                # np.interp(t, t_post, s_post) then returns ~L for ANY t query inside that
                # 67 ms window — a plateau — which combined with the negative-slope
                # polynomial `s_sample` makes the total `s` run BACKWARD by 10–15 mm
                # near the horizon end → the "lightning" cartesian kink.
                # Fix: attach starting from raw[1] (skip the duplicate). Now seam s values
                # differ by ~raw spacing (~20 cm) with the same ~67 ms t gap, so the
                # interp through that window is physically-meaningful and smooth.
                t_offset = t_rl[-1] - t_rl_raw[0]    # align raw[1]'s t to tail_end + dt_first
                head_slice = slice(1, K + 1)          # raw[1], raw[2], ..., raw[K]
                t_attach      = t_rl_raw[head_slice]      + t_offset
                s_attach      = s_rl_raw[head_slice]      + L
                n_attach      = n_rl_raw[head_slice]
                n_dot_attach  = n_rl_dot_raw[head_slice]
                n_ddot_attach = n_rl_ddot_raw[head_slice]
                s_dot_attach  = s_rl_dot_raw[head_slice]
                s_ddot_attach = s_rl_ddot_raw[head_slice]
                v_attach      = v_rl_raw[head_slice]
                chi_attach    = chi_rl_raw[head_slice]
                ax_attach     = ax_rl_raw[head_slice]
                ay_attach     = ay_rl_raw[head_slice]

                t_rl       = np.concatenate([t_rl,       t_attach])
                s_rl       = np.concatenate([s_rl,       s_attach])
                n_rl       = np.concatenate([n_rl,       n_attach])
                n_rl_dot   = np.concatenate([n_rl_dot,   n_dot_attach])
                n_rl_ddot  = np.concatenate([n_rl_ddot,  n_ddot_attach])
                s_rl_dot   = np.concatenate([s_rl_dot,   s_dot_attach])
                s_rl_ddot  = np.concatenate([s_rl_ddot,  s_ddot_attach])
                v_rl       = np.concatenate([v_rl,       v_attach])
                chi_rl     = np.concatenate([chi_rl,     chi_attach])
                ax_rl      = np.concatenate([ax_rl,      ax_attach])
                ay_rl      = np.concatenate([ay_rl,      ay_attach])
                if _near_b_pp:
                    _ds_chk = np.diff(s_rl)
                    _dt_chk = np.diff(t_rl)
                    _hj_dbg('  PP-ATTACH done: now N=%d  s_rl=[%.3f..%.3f] (mono=%s)  t_rl=[%.3f..%.3f] (mono=%s)'
                            % (len(s_rl), float(s_rl[0]), float(s_rl[-1]),
                               'YES' if np.all(_ds_chk >= -1e-9) else 'NO',
                               float(t_rl[0]), float(t_rl[-1]),
                               'YES' if np.all(_dt_chk >= -1e-9) else 'NO'))
        ### HJ : wrap-attach end

        # remove all points greater planning horizon
        idxs = np.where(t_rl > horizon*1.5)
        t_rl = np.delete(t_rl, idxs[0][1:])
        n_rl = np.delete(n_rl, idxs[0][1:])
        n_rl_dot = np.delete(n_rl_dot, idxs[0][1:])
        n_rl_ddot = np.delete(n_rl_ddot, idxs[0][1:])
        s_rl = np.delete(s_rl, idxs[0][1:])
        s_rl_dot = np.delete(s_rl_dot, idxs[0][1:])
        s_rl_ddot = np.delete(s_rl_ddot, idxs[0][1:])
        v_rl = np.delete(v_rl, idxs[0][1:])
        chi_rl = np.delete(chi_rl, idxs[0][1:])
        ax_rl = np.delete(ax_rl, idxs[0][1:])
        ay_rl = np.delete(ay_rl, idxs[0][1:])

        # save postprocessed data into dictionary
        postprocessed_raceline['t_post'] = t_rl
        postprocessed_raceline['n_post'] = n_rl
        postprocessed_raceline['n_dot_post'] = n_rl_dot
        postprocessed_raceline['n_ddot_post'] = n_rl_ddot
        postprocessed_raceline['s_post'] = s_rl
        postprocessed_raceline['s_dot_post'] = s_rl_dot
        postprocessed_raceline['s_ddot_post'] = s_rl_ddot
        postprocessed_raceline['V_post'] = v_rl
        postprocessed_raceline['chi_post'] = chi_rl
        postprocessed_raceline['ax_post'] = ax_rl
        postprocessed_raceline['ay_post'] = ay_rl

        return postprocessed_raceline

    def generate_longitudinal_curves(
            self,
            track_handler: Track3D,
            s_start: float,
            s_dot_start: float,
            s_ddot_start: float,
            s_dot_min: float,
            t_array: np.ndarray,
            v_samples: int,
            n_samples: int,
            postprocessed_raceline: dict,
            horizon: float,
            raceline_tendency: bool,
    ):
        s_array = np.zeros_like(t_array)
        s_dot_array = np.zeros_like(t_array)
        s_ddot_array = np.zeros_like(t_array)

        # raceline end conditions
        s_dot_end_rl = np.interp(horizon, postprocessed_raceline['t_post'], postprocessed_raceline['s_dot_post'])
        s_ddot_end_rl = np.interp(horizon, postprocessed_raceline['t_post'], postprocessed_raceline['s_ddot_post'])

        # sampled s_dot end conditions
        s_dot_max = min(max(s_dot_start, s_dot_end_rl) * 1.2, self.gggv_handler.V_max)
        s_dot_end_values = np.concatenate((np.linspace(s_dot_min, s_dot_max, v_samples - 1), [s_dot_end_rl]))  # always sample raceline

        # end values of s and s_dot (needed for lateral curves)
        s_end_values = np.zeros_like(s_dot_end_values)

        # ### HJ : full-lap capture (no longer gated).
        L_dbg2 = float(track_handler.s[-1])
        near_boundary2 = True
        if near_boundary2:
            # Dump raceline data context — is s_post smooth or does it have
            # weird values around the horizon?
            _ps = postprocessed_raceline['s_post']
            _sd = postprocessed_raceline['s_dot_post']
            _sdd = postprocessed_raceline['s_ddot_post']
            _tp = postprocessed_raceline['t_post']
            _hj_dbg('  GLC-CTX: s_start=%.3f s_dot_start=%.3f s_ddot_start=%.3f'
                    % (s_start, s_dot_start, s_ddot_start))
            _hj_dbg('           s_dot_end_rl=%.3f s_ddot_end_rl=%.3f s_dot_max=%.3f'
                    % (s_dot_end_rl, s_ddot_end_rl, s_dot_max))
            _hj_dbg('           s_dot_end_values=%s'
                    % np.array2string(s_dot_end_values, precision=3, separator=','))
            _hj_dbg('  RL:  t_post=[%.3f..%.3f] (N=%d)'
                    % (_tp[0], _tp[-1], len(_tp)))
            _hj_dbg('       s_post[0..5]=%s'
                    % np.array2string(_ps[:6], precision=3))
            _hj_dbg('       s_post[-5:]=%s'
                    % np.array2string(_ps[-5:], precision=3))
            _hj_dbg('       s_dot_post[-5:]=%s'
                    % np.array2string(_sd[-5:], precision=3))
            _hj_dbg('       s_ddot_post[-5:]=%s'
                    % np.array2string(_sdd[-5:], precision=3))

        for i, (s_dot_end, t_end) in enumerate(zip(s_dot_end_values, t_array[:, -1])):

            # set end acceleration between 0 and raceline acceleration dependent on sampled velocity
            s_ddot_end_tmp = np.interp(s_dot_end, [0.0, s_dot_end_rl], [0.0, s_ddot_end_rl])
            # only adhere to end acceleration of raceline when start velocity is also near to raceline velocity
            s_ddot_end = np.interp(s_dot_start, [0.0, postprocessed_raceline['s_dot_post'][0]], [0.0, s_ddot_end_tmp])

            # formulate linear system of equations
            a = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 2, 0, 0],
                          [0, 1, 2 * t_end, 3 * t_end ** 2, 4 * t_end ** 3],
                          [0, 0, 2, 6 * t_end, 12 * t_end ** 2]])
            if raceline_tendency:  # sample curves relative to raceline
                b = np.array([s_start-postprocessed_raceline['s_post'][0], s_dot_start-postprocessed_raceline['s_dot_post'][0], s_ddot_start-postprocessed_raceline['s_ddot_post'][0], s_dot_end-s_dot_end_rl, s_ddot_end-s_ddot_end_rl])
            else:  # sample curves absolute
                b = np.array([s_start, s_dot_start, s_ddot_start, s_dot_end, s_ddot_end])

            # calculate coefficients of quartic polynomial
            c = np.linalg.solve(a=a, b=b)
            
            # sampled s curve
            s_sample = c[0] + c[1] * t_array[i] + c[2] * t_array[i] ** 2 + c[3] * t_array[i] ** 3 + c[4] * t_array[i] ** 4
            s_dot_sample = c[1] + 2 * c[2] * t_array[i] + 3 * c[3] * t_array[i] ** 2 + 4 * c[4] * t_array[i] ** 3
            s_ddot_sample = 2 * c[2] + 6 * c[3] * t_array[i] + 12 * c[4] * t_array[i] ** 2

            if raceline_tendency:
                # evaluate raceline s data at t_array points
                s_continuous = np.unwrap(postprocessed_raceline['s_post'], discont=track_handler.s[-1]/2, period=track_handler.s[-1])
                s_rl_eval = np.mod(np.interp(t_array[i], postprocessed_raceline['t_post'], s_continuous), track_handler.s[-1])
                s_dot_rl_eval = np.interp(t_array[i], postprocessed_raceline['t_post'], postprocessed_raceline['s_dot_post'])
                s_ddot_rl_eval = np.interp(t_array[i], postprocessed_raceline['t_post'], postprocessed_raceline['s_ddot_post'])

                # add raceline s data to sampled relative s curve
                s = s_sample + s_rl_eval
                s_dot = s_dot_sample + s_dot_rl_eval
                s_ddot = s_ddot_sample + s_ddot_rl_eval
            else:
                s = s_sample
                s_dot = s_dot_sample
                s_ddot = s_ddot_sample

            # ### HJ : log per-candidate polynomial BEFORE the mod-L wrap.
            if near_boundary2:
                ds_pre = np.diff(s)
                _max_step = float(np.max(np.abs(ds_pre))) if len(ds_pre) else 0.0
                # Check for tail oscillation: does s go backward in last 5 samples?
                tail_min_ds = float(np.min(ds_pre[-5:])) if len(ds_pre) >= 5 else 0.0
                _hj_dbg(('    cand i=%d s_end=%.2f s_ddot_end=%.3f  '
                         + 'b=[%.3f, %.3f, %.3f, %.3f, %.3f]  '
                         + 's_sample_tail5=%s  '
                         + 's_rl_eval_tail5=%s  '
                         + 's_tail_ds5=%s  tail_min_ds=%.4f')
                        % (i, float(s_dot_end), float(s_ddot_end),
                           float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]),
                           np.array2string(s_sample[-5:], precision=4) if raceline_tendency else 'N/A',
                           np.array2string(s_rl_eval[-5:], precision=4) if raceline_tendency else 'N/A',
                           np.array2string(ds_pre[-5:], precision=5),
                           tail_min_ds))

            # consider track length
            s = np.mod(s, track_handler.s[-1])

            # save last values
            s_end_values[i] = s[-1]

            s_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s, (n_samples, 1))
            s_dot_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s_dot, (n_samples, 1))
            s_ddot_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s_ddot, (n_samples, 1))

        return s_array, s_dot_array, s_ddot_array, s_end_values, s_dot_end_values

    def generate_lateral_curves(
            self,
            track_handler: Track3D,
            s_array: np.array,
            s_dot_array: np.array,
            s_ddot_array: np.array,
            s_end_values: np.array,
            s_dot_end_values: np.array,
            n_start: float,
            n_dot_start: float,
            n_ddot_start: float,
            t_array: np.ndarray,
            n_samples: int,
            postprocessed_raceline: dict,
            safety_distance: float,
            raceline_tendency: bool,
            endpoint_chi_raceline_only: bool = False,
            L_connector: float = 4.0,
    ):
        # ### HJ : two-step lateral curves (recovery in-place, d_i-independent α).
        #   Step 1 (ego-agnostic): n_sample_i(s) = n_rl(s) + d_i for all i.
        #   Step 2 (recovery, α-blend, same α for all candidates):
        #       n_ego(s)  = n_start + m0·(s − s0), m0 = n_dot_start/s_dot_start
        #       α(s): cubic Hermite on [s0, s0+L_connector], (1,0)→(0,0) in (α,α').
        #       n_final_i = (1−α)·n_sample_i + α·n_ego
        # Every candidate starts at ego (n_start, m0); tail is pure shifted raceline.
        # Walls remain FILTERING-ONLY (check_path_collision).
        # `raceline_tendency`/`endpoint_chi_raceline_only` kept for API compat.
        n_array = np.zeros_like(t_array)
        n_dot_array = np.zeros_like(t_array)
        n_ddot_array = np.zeros_like(t_array)

        # ### HJ : our wrap-attach in postprocess_raceline now makes s_post extend past L
        # (e.g. [..., 89.87, 89.93, 90.13, ...]). The original `period=L` interp requires
        # xp strictly within one period — otherwise numpy's `xp % period` wraps the
        # attached samples back to ~0, sorts xp, reorders fp, and produces garbage
        # (the real cause of the "n-direction lightning" at lap boundary).
        # Fix: unwrap s_array[i] (mod-L) to match s_post's continuous range, then use
        # regular non-periodic np.interp. s_post is guaranteed monotonic by postprocess.
        L_track_ln = track_handler.s[-1]
        s_post_cont = postprocessed_raceline['s_post']      # monotonic, may exceed L

        i = 0
        for s_end, s_dot_end in zip(s_end_values, s_dot_end_values):

            # unwrap candidate s (was mod-L by line 582) to match s_post continuous form
            s_q = s_array[i].copy()
            _ds = np.diff(s_q)
            _adj = np.cumsum(np.where(_ds < -L_track_ln / 2.0, L_track_ln, 0.0))
            s_q[1:] += _adj

            # evaluate raceline at specific s points (non-periodic, unwrapped query)
            s_dot_rl   = np.interp(s_q, s_post_cont, postprocessed_raceline['s_dot_post'])
            s_ddot_rl  = np.interp(s_q, s_post_cont, postprocessed_raceline['s_ddot_post'])
            n_rl       = np.interp(s_q, s_post_cont, postprocessed_raceline['n_post'])
            n_dot_rl   = np.interp(s_q, s_post_cont, postprocessed_raceline['n_dot_post'])
            n_ddot_rl  = np.interp(s_q, s_post_cont, postprocessed_raceline['n_ddot_post'])

            n_rl_eval = n_rl
            n_dot_rl_eval = n_dot_rl / s_dot_rl * s_dot_array[i]
            n_ddot_rl_eval = n_ddot_rl / (s_dot_rl ** 2) * (s_dot_array[i] ** 2) \
                             - n_dot_rl / (s_dot_rl ** 3) * s_ddot_rl * (s_dot_array[i] ** 2) \
                             + n_dot_rl / s_dot_rl * s_ddot_array[i]

            # ### HJ : Step 1 sampling + Step 2 in-place recovery (α-blend).
            n_rl_end = float(n_rl_eval[-1])

            # raceline-relative shift range — wall-independent.
            d_nominal = float(getattr(self, 'n_shift_nominal', 0.6))
            d_left_rl  = d_nominal
            d_right_rl = d_nominal

            # velocity-aware kinematic cap — smoothness gate (not wall-based).
            s_horizon_eff = max(float(s_end) - float(s_array[i, 0]), 0.1)
            v_end_clip = max(1.0, min(4.0, float(s_dot_end)))
            max_slope = 0.15 + (0.45 - 0.15) * (v_end_clip - 1.0) / (4.0 - 1.0)
            max_swing = max_slope * s_horizon_eff
            d_left_rl  = min(d_left_rl,  max_swing)
            d_right_rl = min(d_right_rl, max_swing)

            # prev-anchored + rate-cap preserved B-plan (wall-independent).
            prev_anchor = getattr(self, 'prev_chosen_n_end', None)
            rate_cap    = float(getattr(self, 'n_end_rate_cap', 0.12))
            if prev_anchor is not None:
                off_prev = float(np.clip(float(prev_anchor) - n_rl_end,
                                          -d_right_rl, d_left_rl))
                half_span = min(rate_cap * 3.0, max(d_left_rl, d_right_rl))
                lo = max(off_prev - half_span, -d_right_rl)
                hi = min(off_prev + half_span,  d_left_rl)
                n_end_offsets = np.linspace(lo, hi, n_samples - 1)
                n_end_values = n_rl_end + np.concatenate((n_end_offsets, [0.0]))
            else:
                N_right = n_samples // 2
                N_left  = n_samples - 1 - N_right
                right_side = np.linspace(-d_right_rl, 0.0, N_right, endpoint=False) if N_right > 0 else np.array([])
                left_side  = np.linspace(0.0, d_left_rl, N_left) if N_left > 0 else np.array([])
                n_end_values = n_rl_end + np.concatenate((right_side, left_side, [0.0]))

            # Precomputed terms for Step 2 (α-blend, d_i-independent).
            s_dot_rl_safe = np.maximum(s_dot_rl, 1e-6)
            dn_rl_ds      = n_dot_rl  / s_dot_rl_safe
            d2n_rl_ds2    = n_ddot_rl / (s_dot_rl_safe ** 2) \
                            - (n_dot_rl / (s_dot_rl_safe ** 3)) * s_ddot_rl

            s0_q    = float(s_q[0])
            s_end_q = float(s_q[-1])
            L_rec_eff = float(min(float(L_connector),
                                  max(s_end_q - s0_q, 0.1) * 0.95))
            L_safe    = max(L_rec_eff, 1e-6)

            sdot0 = float(s_dot_array[i, 0])
            m0    = float(n_dot_start) / sdot0 if sdot0 > 1e-6 else 0.0
            ds_q      = s_q - s0_q
            n_ego     = n_start + m0 * ds_q
            dn_ego_ds = np.full_like(s_q, m0)

            u = np.clip(ds_q / L_safe, 0.0, 1.0)
            inside      = ds_q < L_rec_eff
            # ### HJ : C2 smootherstep (Perlin) — zero 1st AND 2nd derivatives at
            # both endpoints, so n̈(s) is continuous across the blend boundary.
            # Previous C1 smoothstep 2u³-3u²+1 leaked a d²α/ds² jump of 6/L²,
            # which kicked κ (via n̈) and caused V to drop to near-zero locally.
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            u5 = u3 * u2
            alpha       = np.where(inside, 1.0 - 10.0*u3 + 15.0*u4 - 6.0*u5, 0.0)
            dalpha_ds   = np.where(inside, (-30.0*u2 + 60.0*u3 - 30.0*u4) / L_safe, 0.0)
            d2alpha_ds2 = np.where(inside, (-60.0*u + 180.0*u2 - 120.0*u3) / (L_safe**2), 0.0)
            one_minus_alpha = 1.0 - alpha

            sdot_cand  = s_dot_array[i, :]
            sddot_cand = s_ddot_array[i, :]

            for n_end in n_end_values:
                d_i          = float(n_end) - n_rl_end
                n_samp       = n_rl_eval + d_i
                dn_samp_ds   = dn_rl_ds
                d2n_samp_ds2 = d2n_rl_ds2

                n_final    = one_minus_alpha * n_samp       + alpha * n_ego
                dn_final   = one_minus_alpha * dn_samp_ds   + alpha * dn_ego_ds \
                             + dalpha_ds * (n_ego - n_samp)
                d2n_final  = one_minus_alpha * d2n_samp_ds2 \
                             + 2.0 * dalpha_ds * (dn_ego_ds - dn_samp_ds) \
                             + d2alpha_ds2 * (n_ego - n_samp)

                n_array[i, :]      = n_final
                n_dot_array[i, :]  = dn_final  * sdot_cand
                n_ddot_array[i, :] = d2n_final * (sdot_cand ** 2) + dn_final * sddot_cand
                i += 1

        return n_array, n_dot_array, n_ddot_array
