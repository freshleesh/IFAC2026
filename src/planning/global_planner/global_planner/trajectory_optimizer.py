#!/usr/bin/env python3
"""
Trajectory Optimizer Node for 2-stage pattern (IQP → SP).

Reads maps/{map_name}/centerline.csv produced by centerline_extractor and
computes two optimised racelines. Outputs global_waypoints.json consumed by
global_republisher.

Vehicle + algorithm parameters come from planner/config/racecar.ini.
ROS parameters:
  map_name          str   map folder name under IFAC2026_SH/maps/
  safety_width_iqp  float vehicle safety width for IQP [m]  (overrides ini width_opt)
  safety_width_sp   float vehicle safety width for SP  [m]  (overrides ini width_opt)
  enable_check_traj bool  run post-optimisation sanity checks
"""

import configparser
import csv
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from .tph import prep_track  # noqa: E402  (local flat copy — no pip/submodule needed)
from . import tph             # noqa: E402

# opt_mintime_traj is imported lazily inside _run_mintime().
# It lives in the git submodule (CasADi is heavy; kept separate from the flat copy).
_HERE         = os.path.dirname(os.path.realpath(__file__))
_PLANNER_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
_TUM_LIB_PATH = os.path.join(_PLANNER_ROOT, 'global_racetrajectory_optimization')
if _TUM_LIB_PATH not in sys.path:
    sys.path.insert(0, _TUM_LIB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Config paths — share dir works for both symlink and non-symlink installs
# because data_files are always copied to share/ during colcon build.
# ─────────────────────────────────────────────────────────────────────────────
_SHARE_DIR   = get_package_share_directory('global_planner')
_CONFIG_DIR  = os.path.join(_SHARE_DIR, 'config')
_INI_PATH    = os.path.join(_CONFIG_DIR, 'racecar.ini')
_VEH_DYN_DIR = os.path.join(_CONFIG_DIR, 'inputs', 'veh_dyn_info')

# ─────────────────────────────────────────────────────────────────────────────
# ROS parameters (vehicle/algo params come from ini)
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = {
    'map_name':             '',
    'safety_width_iqp':     0.6,
    'safety_width_sp':      0.4,
    'enable_check_traj':    True,
    'enable_mintime':       False,
    'safety_width_mintime': 0.70,
}

#map:=f
# 'safety_width_iqp':  0.8,
#  'safety_width_sp':   0.4,

#map:=icra
# 'safety_width_iqp':  0.7,
#  'safety_width_sp':   0.4,

class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        if not self.map_name:
            self.get_logger().error('map_name parameter is required!')
            return

        # 레포 루트의 maps/ 해석 — Ubuntu: 레포 == colcon ws (ws/maps),
        # Mac mini 실차: 레포가 ws/src/IFAC2026_SH 로 클론됨.
        _ws = os.path.normpath(os.path.join(next(
            p for p in os.environ.get('AMENT_PREFIX_PATH', '').split(':')
            if os.path.basename(p) == 'global_planner'), '..', '..'))
        _maps_root = next(
            (os.path.join(r, 'maps') for r in (_ws, os.path.join(_ws, 'src', 'IFAC2026_SH'))
             if os.path.isdir(os.path.join(r, 'maps'))),
            os.path.join(_ws, 'maps'))
        self.map_dir = os.path.join(_maps_root, self.map_name)
        self.get_logger().info(f'map_dir: {self.map_dir}')

        self.pars = self._load_pars()
        self.get_logger().info(
            f'safety_width_iqp={self.safety_width_iqp}  '
            f'safety_width_sp={self.safety_width_sp}')

        try:
            self.run()
        except Exception as exc:
            import traceback
            self.get_logger().error(f'Optimization failed: {exc}\n{traceback.format_exc()}')

    # ─────────────────────────────────────────────────────────────────────────
    # ini loading
    # ─────────────────────────────────────────────────────────────────────────
    def _load_pars(self) -> dict:
        parser = configparser.ConfigParser()
        if not parser.read(_INI_PATH):
            raise FileNotFoundError(f'racecar.ini not found: {_INI_PATH}')

        g = 'GENERAL_OPTIONS'
        o = 'OPTIMIZATION_OPTIONS'

        pars: dict = {}
        pars['ggv_file']             = json.loads(parser.get(g, 'ggv_file'))
        pars['ax_max_machines_file'] = json.loads(parser.get(g, 'ax_max_machines_file'))
        pars['stepsize_opts']        = json.loads(parser.get(g, 'stepsize_opts'))
        pars['reg_smooth_opts']      = json.loads(parser.get(g, 'reg_smooth_opts'))
        pars['veh_params']           = json.loads(parser.get(g, 'veh_params'))
        pars['vel_calc_opts']        = json.loads(parser.get(g, 'vel_calc_opts'))
        pars['curv_calc_opts']       = json.loads(parser.get(g, 'curv_calc_opts'))
        pars['imp_opts']             = json.loads(parser.get(g, 'imp_opts'))
        pars['optim_opts_mincurv']   = json.loads(parser.get(o, 'optim_opts_mincurv'))
        pars['optim_opts_sp']        = json.loads(parser.get(o, 'optim_opts_shortest_path'))
        pars['optim_opts_mintime']   = json.loads(parser.get(o, 'optim_opts_mintime'))
        pars['vehicle_params_mintime'] = json.loads(parser.get(o, 'vehicle_params_mintime'))
        pars['tire_params_mintime']  = json.loads(parser.get(o, 'tire_params_mintime'))
        pars['pwr_params_mintime']   = json.loads(parser.get(o, 'pwr_params_mintime'))

        # ROS params override width_opt from ini
        pars['optim_opts_mincurv']['width_opt']  = self.safety_width_iqp
        pars['optim_opts_sp']['width_opt']       = self.safety_width_sp
        pars['optim_opts_mintime']['width_opt']  = self.safety_width_mintime

        # min_track_width는 racecar.ini 값을 그대로 사용한다.
        # 이전에는 safety_width_iqp로 올렸으나, 실제 트랙이 좁은 구간에서
        # prep_track이 인위적으로 확장된 가상 트랙에서 IQP를 풀어
        # 실제 돌출 벽에 경로가 붙는 문제가 발생했다.
        # → min_track_width 강제 조정 제거. IQP가 실제 폭 기준으로 최적화.

        # wheelbase combined (required by opt_mintime)
        vp = pars['vehicle_params_mintime']
        vp['wheelbase'] = vp['wheelbase_front'] + vp['wheelbase_rear']

        # Load GGV + ax_max_machines
        pars['ggv'], pars['ax_max_machines'] = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=os.path.join(_VEH_DYN_DIR, pars['ggv_file']),
            ax_max_machines_import_path=os.path.join(_VEH_DYN_DIR, pars['ax_max_machines_file']),
        )

        return pars

    # ─────────────────────────────────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        # 1. Load centerline
        csv_path = os.path.join(self.map_dir, 'centerline.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'Centerline CSV not found: {csv_path}\n'
                'Run centerline_extractor first.')

        reftrack_imp = self._load_centerline(csv_path)
        self.get_logger().info(f'Centerline loaded: {len(reftrack_imp)} points')

        # 2. Load boundaries for check_traj (optional)
        bound_r, bound_l = self._load_boundaries()
        if bound_r is not None:
            self.get_logger().info(
                f'Boundaries: right={len(bound_r)}, left={len(bound_l)} points')

        # 3. Prep track ONCE — shared by IQP and SP
        self.get_logger().info('=== Preparing track ===')
        reftrack_interp, normvec_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
            tph.prep_track.prep_track(
                reftrack_imp=reftrack_imp,
                reg_smooth_opts=self.pars['reg_smooth_opts'],
                stepsize_opts=self.pars['stepsize_opts'],
                debug=False,
                min_width=self.pars['imp_opts']['min_track_width'],
            )
        self.get_logger().info(f'prep_track done: {len(reftrack_interp)} points')

        # Pre-compute spline kinematics required by iqp_handler
        spline_lengths_interp = tph.calc_spline_lengths.calc_spline_lengths(
            coeffs_x=coeffs_x_interp,
            coeffs_y=coeffs_y_interp,
        )
        psi_interp, kappa_interp, dkappa_interp = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_interp,
            coeffs_y=coeffs_y_interp,
            ind_spls=np.arange(coeffs_x_interp.shape[0]),
            t_spls=np.zeros(coeffs_x_interp.shape[0]),
            calc_curv=True,
            calc_dcurv=True,
        )

        # 4. IQP — minimum curvature racing line
        self.get_logger().info('=== Running mincurv_iqp ===')
        traj_iqp, lap_iqp, reftrack_iqp, normvec_iqp = self._run_iqp(
            reftrack_interp, normvec_interp, a_interp,
            spline_lengths_interp, psi_interp, kappa_interp, dkappa_interp)
        if self.enable_check_traj and bound_r is not None:
            self._run_check('IQP', traj_iqp, bound_r, bound_l, self.safety_width_iqp)

        # 5. SP — shortest path on IQP-refined track (UNICORN 방식)
        self.get_logger().info('=== Running shortest_path ===')
        traj_sp, lap_sp = self._run_sp(reftrack_iqp, normvec_iqp)

        self._save_json(traj_iqp, traj_sp, lap_iqp,
                        reftrack_interp, psi_interp, kappa_interp, spline_lengths_interp)

        if self.enable_check_traj and bound_r is not None:
            self._run_check('SP', traj_sp, bound_r, bound_l, self.safety_width_sp)

        # 6. opt_mintime — minimum lap time (optional, CasADi required)
        # Runs on IQP-refined track: better starting geometry, faster convergence.
        if self.enable_mintime:
            self.get_logger().info('=== Prepping IQP track for mintime ===')
            reftrack_mt, normvec_mt, a_mt, coeffs_x_mt, coeffs_y_mt = \
                tph.prep_track.prep_track(
                    reftrack_imp=reftrack_iqp,
                    reg_smooth_opts=self.pars['reg_smooth_opts'],
                    stepsize_opts=self.pars['stepsize_opts'],
                    debug=False,
                    min_width=self.pars['imp_opts']['min_track_width'],
                )
            self.get_logger().info('=== Running opt_mintime ===')
            traj_mt, lap_mt = self._run_mintime(
                reftrack_mt, normvec_mt, a_mt,
                coeffs_x_mt, coeffs_y_mt)
            if self.enable_check_traj and bound_r is not None:
                self._run_check('MinTime', traj_mt, bound_r, bound_l, self.safety_width_mintime)

        # 7. Summary
        self.get_logger().info('=== Summary ===')
        self._log_stats('IQP', traj_iqp, lap_iqp)
        self._log_stats('SP ', traj_sp,  lap_sp)
        if self.enable_mintime:
            self._log_stats('MinTime', traj_mt, lap_mt)
        self.get_logger().info('=== Done ===')

    # ─────────────────────────────────────────────────────────────────────────
    # IQP optimisation
    # ─────────────────────────────────────────────────────────────────────────
    def _run_iqp(self,
                 reftrack_interp: np.ndarray,
                 normvec_interp: np.ndarray,
                 a_interp: np.ndarray,
                 spline_lengths: np.ndarray,
                 psi: np.ndarray,
                 kappa: np.ndarray,
                 dkappa: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()

        alpha_opt, reftrack_iqp, normvec_iqp = tph.iqp_handler.iqp_handler(
            reftrack=reftrack_interp,
            normvectors=normvec_interp,
            A=a_interp,
            spline_len=spline_lengths,
            psi=psi,
            kappa=kappa,
            dkappa=dkappa,
            kappa_bound=self.pars['veh_params']['curvlim'],
            w_veh=self.safety_width_iqp,
            print_debug=False,
            plot_debug=False,
            stepsize_interp=self.pars['stepsize_opts']['stepsize_reg'],
            iters_min=self.pars['optim_opts_mincurv']['iqp_iters_min'],
            curv_error_allowed=self.pars['optim_opts_mincurv']['iqp_curverror_allowed'],
        )[0:3]

        traj, lap = self._build_trajectory(reftrack_iqp, normvec_iqp, alpha_opt)
        self.get_logger().info(
            f'[IQP] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap, reftrack_iqp, normvec_iqp

    # ─────────────────────────────────────────────────────────────────────────
    # SP optimisation
    # ─────────────────────────────────────────────────────────────────────────
    def _run_sp(self,
                reftrack_interp: np.ndarray,
                normvec_interp: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()

        alpha_opt = tph.opt_shortest_path.opt_shortest_path(
            reftrack=reftrack_interp,
            normvectors=normvec_interp,
            w_veh=self.safety_width_sp,
            print_debug=False,
        )

        traj, lap = self._build_trajectory(reftrack_interp, normvec_interp, alpha_opt)
        self.get_logger().info(
            f'[SP ] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap

    # ─────────────────────────────────────────────────────────────────────────
    # opt_mintime — minimum lap time (CasADi/IPOPT)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_mintime(self,
                     reftrack_interp: np.ndarray,
                     normvec_interp: np.ndarray,
                     a_interp: np.ndarray,
                     coeffs_x_interp: np.ndarray,
                     coeffs_y_interp: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()

        import opt_mintime_traj  # lazy: CasADi + sklearn only needed here

        export_path = os.path.join(self.map_dir, 'mintime_export')
        os.makedirs(export_path, exist_ok=True)

        # Build pars dict in the format expected by opt_mintime
        pars_mt = dict(self.pars)
        pars_mt['optim_opts'] = dict(self.pars['optim_opts_mintime'])
        pars_mt['optim_opts']['var_friction'] = None
        pars_mt['optim_opts']['warm_start']   = False

        # When reopt is enabled, widen width_opt for the first mintime pass
        # so the reopt IQP has room to smooth without wall violations — mirrors
        # TUM main_globaltraj.py lines 255-261.
        if pars_mt['optim_opts'].get('reopt_mintime_solution', False):
            opts = pars_mt['optim_opts']
            opts['width_opt'] = (opts['width_opt']
                                 + (opts['w_tr_reopt'] - opts['w_veh_reopt'])
                                 + opts['w_add_spl_regr'])

        alpha_opt, v_opt, reftrack_out, a_interp_out, normvec_out = \
            opt_mintime_traj.src.opt_mintime.opt_mintime(
                reftrack=reftrack_interp,
                coeffs_x=coeffs_x_interp,
                coeffs_y=coeffs_y_interp,
                normvectors=normvec_interp,
                pars=pars_mt,
                tpamap_path='',
                tpadata_path='',
                export_path=export_path,
                print_debug=True,
                plot_debug=False,
            )

        ref  = reftrack_out if reftrack_out is not None else reftrack_interp
        norm = normvec_out  if normvec_out  is not None else normvec_interp

        # ── Optional reopt: run mincurv IQP on the mintime path to smooth kappa ──
        # Mirrors TUM main_globaltraj.py "reopt_mintime_solution" block.
        if pars_mt['optim_opts'].get('reopt_mintime_solution', False):
            raceline_mt = ref[:, :2] + np.expand_dims(alpha_opt, 1) * norm
            w_tr_right_mt = ref[:, 2] - alpha_opt
            w_tr_left_mt  = ref[:, 3] + alpha_opt
            racetrack_mt  = np.column_stack((raceline_mt, w_tr_right_mt, w_tr_left_mt))

            ref_reopt, norm_reopt, a_reopt = \
                tph.prep_track.prep_track(
                    reftrack_imp=racetrack_mt,
                    reg_smooth_opts=self.pars['reg_smooth_opts'],
                    stepsize_opts=self.pars['stepsize_opts'],
                    debug=False,
                    min_width=self.pars['imp_opts']['min_track_width'],
                )[:3]

            w_tr_tmp = 0.5 * pars_mt['optim_opts']['w_tr_reopt'] * np.ones(ref_reopt.shape[0])
            racetrack_reopt = np.column_stack((ref_reopt[:, :2], w_tr_tmp, w_tr_tmp))

            alpha_opt = tph.opt_min_curv.opt_min_curv(
                reftrack=racetrack_reopt,
                normvectors=norm_reopt,
                A=a_reopt,
                kappa_bound=self.pars['veh_params']['curvlim'],
                w_veh=pars_mt['optim_opts']['w_veh_reopt'],
                print_debug=False,
                plot_debug=False,
            )[0]
            ref, norm = ref_reopt, norm_reopt

        # Build fine-grid raceline geometry from alpha_opt
        raceline_interp, _, coeffs_x_opt, coeffs_y_opt, \
            spline_inds_opt, t_vals_opt, s_points_opt, \
            spline_lengths_opt, el_lengths_opt = \
            tph.create_raceline.create_raceline(
                refline=ref[:, :2],
                normvectors=norm,
                alpha=alpha_opt,
                stepsize_interp=self.pars['stepsize_opts']['stepsize_interp_after_opt'],
            )

        psi_vel, kappa = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_opt,
            coeffs_y=coeffs_y_opt,
            ind_spls=spline_inds_opt,
            t_spls=t_vals_opt,
        )

        reopt = pars_mt['optim_opts'].get('reopt_mintime_solution', False)
        if reopt:
            # After reopt the path geometry differs from the NLP solution, so v_opt
            # no longer matches the new raceline — recalculate with GGV (same as
            # TUM recalc_vel_profile_by_tph=True).
            vx_profile = tph.calc_vel_profile.calc_vel_profile(
                ggv=self.pars['ggv'],
                ax_max_machines=self.pars['ax_max_machines'],
                v_max=self.pars['veh_params']['v_max'],
                kappa=kappa,
                el_lengths=el_lengths_opt,
                closed=True,
                filt_window=self.pars['vel_calc_opts']['vel_profile_conv_filt_window'],
                dyn_model_exp=self.pars['vel_calc_opts']['dyn_model_exp'],
                drag_coeff=self.pars['veh_params']['dragcoeff'],
                m_veh=self.pars['veh_params']['mass'],
            )
        else:
            # Interpolate v_opt onto fine grid — TUM reference approach
            # (main_globaltraj.py:395-397): use cumulative spline arc-lengths,
            # not evenly-spaced stepsize_reg.
            s_splines = np.cumsum(spline_lengths_opt)
            s_splines = np.insert(s_splines, 0, 0.0)
            vx_profile = np.interp(s_points_opt, s_splines[:-1], v_opt)
            vx_profile = np.minimum(vx_profile, self.pars['veh_params']['v_max'])
            # vx_profile *= 0.80  # speed scaling: uncomment if car can't track

        vx_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_cl, el_lengths=el_lengths_opt, eq_length_output=False)
        t_profile = tph.calc_t_profile.calc_t_profile(
            vx_profile=vx_profile, ax_profile=ax_profile, el_lengths=el_lengths_opt)

        traj = np.column_stack([
            s_points_opt,
            raceline_interp[:, 0],
            raceline_interp[:, 1],
            psi_vel,
            kappa,
            vx_profile,
            ax_profile,
        ])

        lap = float(t_profile[-1])
        self.get_logger().info(
            f'[MinTime] Done in {time.perf_counter()-t0:.2f}s, lap≈{lap:.2f}s')
        return traj, lap

    # ─────────────────────────────────────────────────────────────────────────
    # Shared post-optimisation pipeline (create_raceline → vel profile)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_trajectory(self,
                          reftrack: np.ndarray,
                          normvec: np.ndarray,
                          alpha_opt: np.ndarray) -> Tuple[np.ndarray, float]:
        pars = self.pars

        raceline_interp, _, coeffs_x_opt, coeffs_y_opt, \
            spline_inds_opt, t_vals_opt, s_points_opt, \
            _, el_lengths_opt = \
            tph.create_raceline.create_raceline(
                refline=reftrack[:, :2],
                normvectors=normvec,
                alpha=alpha_opt,
                stepsize_interp=pars['stepsize_opts']['stepsize_interp_after_opt'],
            )

        psi_vel, kappa = tph.calc_head_curv_an.calc_head_curv_an(
            coeffs_x=coeffs_x_opt,
            coeffs_y=coeffs_y_opt,
            ind_spls=spline_inds_opt,
            t_spls=t_vals_opt,
        )

        vx_profile = tph.calc_vel_profile.calc_vel_profile(
            ggv=pars['ggv'],
            ax_max_machines=pars['ax_max_machines'],
            v_max=pars['veh_params']['v_max'],
            kappa=kappa,
            el_lengths=el_lengths_opt,
            closed=True,
            filt_window=pars['vel_calc_opts']['vel_profile_conv_filt_window'],
            dyn_model_exp=pars['vel_calc_opts']['dyn_model_exp'],
            drag_coeff=pars['veh_params']['dragcoeff'],
            m_veh=pars['veh_params']['mass'],
        )

        vx_cl = np.append(vx_profile, vx_profile[0])
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_cl,
            el_lengths=el_lengths_opt,
            eq_length_output=False,
        )

        t_profile = tph.calc_t_profile.calc_t_profile(
            vx_profile=vx_profile,
            ax_profile=ax_profile,
            el_lengths=el_lengths_opt,
        )

        # columns: [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
        traj = np.column_stack([
            s_points_opt,
            raceline_interp[:, 0],
            raceline_interp[:, 1],
            psi_vel,
            kappa,
            vx_profile,
            ax_profile,
        ])

        return traj, float(t_profile[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # CSV I/O
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_centerline(csv_path: str) -> np.ndarray:
        """Load centerline.csv → (N, 4) reftrack_imp [x, y, w_right, w_left]."""
        rows = []
        with open(csv_path, 'r') as f:
            for r in csv.DictReader(f):
                rows.append([
                    float(r['x_m']),
                    float(r['y_m']),
                    float(r.get('w_tr_right_m', 1.0)),
                    float(r.get('w_tr_left_m', 1.0)),
                ])
        if len(rows) < 10:
            raise ValueError(f'Centerline too short: {len(rows)} points')
        return np.array(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # JSON output (global_waypoints.json — consumed by global_republisher)
    # ─────────────────────────────────────────────────────────────────────────
    def _save_json(self, traj_iqp: np.ndarray, traj_sp: np.ndarray, lap_iqp: float,
                   reftrack_interp: np.ndarray, psi_interp: np.ndarray,
                   kappa_interp: np.ndarray, spline_lengths: np.ndarray):
        import math
        import json as _json

        def _psi_ros(p):
            return (float(p) + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi

        def _header():
            return {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': 'map'}

        def _traj_to_wpnts(traj):
            return {
                'header': _header(),
                'wpnts': [
                    {
                        'id': i, 's_m': float(r[0]), 'd_m': 0.0,
                        'x_m': float(r[1]), 'y_m': float(r[2]),
                        'd_right': 0.0, 'd_left': 0.0,
                        'psi_rad': _psi_ros(r[3]),
                        'kappa_radpm': float(r[4]),
                        'vx_mps': float(r[5]), 'ax_mps2': float(r[6]),
                    }
                    for i, r in enumerate(traj)
                ],
            }

        def _vel_markers(traj, ns, col_r, col_g, col_b):
            markers = []
            for i, r in enumerate(traj):
                h = max(float(r[5]) * 0.1317, 0.01)
                markers.append({
                    'header': _header(), 'ns': ns, 'id': i, 'type': 3, 'action': 0,
                    'pose': {
                        'position': {'x': float(r[1]), 'y': float(r[2]), 'z': h / 2.0},
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                    },
                    'scale': {'x': 0.1, 'y': 0.1, 'z': h},
                    'color': {'r': col_r, 'g': col_g, 'b': col_b, 'a': 0.8},
                    'lifetime': {'sec': 0, 'nanosec': 0}, 'frame_locked': False,
                    'points': [], 'colors': [], 'text': '',
                    'mesh_resource': '', 'mesh_use_embedded_materials': False,
                })
            return {'markers': markers}

        def _sphere_markers(xys, ns, col_r, col_g, col_b, scale=0.1):
            return [
                {
                    'header': _header(), 'ns': ns, 'id': i, 'type': 2, 'action': 0,
                    'pose': {
                        'position': {'x': float(xy[0]), 'y': float(xy[1]), 'z': 0.0},
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                    },
                    'scale': {'x': scale, 'y': scale, 'z': scale},
                    'color': {'r': col_r, 'g': col_g, 'b': col_b, 'a': 0.8},
                    'lifetime': {'sec': 0, 'nanosec': 0}, 'frame_locked': False,
                    'points': [], 'colors': [], 'text': '',
                    'mesh_resource': '', 'mesh_use_embedded_materials': False,
                }
                for i, xy in enumerate(xys)
            ]

        # Centerline waypoints from prep_track reftrack (N points = N spline segments)
        # s_vals[i] = arc length to start of spline i
        s_vals = np.concatenate([[0.0], np.cumsum(spline_lengths)])
        cl_wpnts = [
            {
                'id': i, 's_m': float(s_vals[i]), 'd_m': 0.0,
                'x_m': float(row[0]), 'y_m': float(row[1]),
                'd_right': float(row[2]), 'd_left': float(row[3]),
                'psi_rad': _psi_ros(psi_interp[i]),
                'kappa_radpm': float(kappa_interp[i]),
                'vx_mps': 0.0, 'ax_mps2': 0.0,
            }
            for i, row in enumerate(reftrack_interp)
        ]

        # Trackbounds markers from boundary CSVs
        tb_markers = []
        mid = 0
        for side_path, (cr, cg, cb) in [
            (os.path.join(self.map_dir, 'boundary_right.csv'), (0.0, 1.0, 0.0)),
            (os.path.join(self.map_dir, 'boundary_left.csv'),  (1.0, 0.0, 1.0)),
        ]:
            if os.path.exists(side_path):
                pts = np.loadtxt(side_path, delimiter=',', skiprows=1)
                for pt in pts:
                    m = _sphere_markers([[pt[0], pt[1]]], 'trackbounds', cr, cg, cb)[0]
                    m['id'] = mid
                    tb_markers.append(m)
                    mid += 1

        data = {
            'map_info_str': {'data': self.map_name},
            'est_lap_time': {'data': float(lap_iqp)},
            'centerline_markers': {
                'markers': _sphere_markers(
                    [[w['x_m'], w['y_m']] for w in cl_wpnts],
                    'centerline', 0.0, 1.0, 1.0, 0.05)
            },
            'centerline_waypoints': {'header': _header(), 'wpnts': cl_wpnts},
            'global_traj_markers_iqp': _vel_markers(traj_iqp, 'iqp', 1.0, 0.0, 0.0),
            'global_traj_wpnts_iqp':   _traj_to_wpnts(traj_iqp),
            'global_traj_markers_sp':  _vel_markers(traj_sp, 'sp', 0.0, 0.0, 1.0),
            'global_traj_wpnts_sp':    _traj_to_wpnts(traj_sp),
            'trackbounds_markers':     {'markers': tb_markers},
        }

        json_path = os.path.join(self.map_dir, 'global_waypoints.json')
        with open(json_path, 'w') as f:
            _json.dump(data, f)
        self.get_logger().info(f'JSON saved: {json_path}')

    # ─────────────────────────────────────────────────────────────────────────
    # check_traj
    # ─────────────────────────────────────────────────────────────────────────
    def _load_boundaries(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        r_path = os.path.join(self.map_dir, 'boundary_right.csv')
        l_path = os.path.join(self.map_dir, 'boundary_left.csv')
        if not (os.path.exists(r_path) and os.path.exists(l_path)):
            self.get_logger().warn('[CheckTraj] boundary CSVs not found — validation disabled')
            return None, None
        return (np.loadtxt(r_path, delimiter=',', skiprows=1),
                np.loadtxt(l_path, delimiter=',', skiprows=1))

    def _run_check(self, label: str, traj: np.ndarray,
                   bound_r: np.ndarray, bound_l: np.ndarray, safety_width: float):
        raceline    = traj[:, 1:3]
        kappa       = traj[:, 4]
        vx          = traj[:, 5]

        veh_half    = self.pars['veh_params']['width'] / 2
        safety_half = safety_width / 2
        curvlim     = self.pars['veh_params']['curvlim']
        v_max       = self.pars['veh_params']['v_max']
        a_lat_max   = float(self.pars['ggv'][:, 2].min())

        errors: list = []
        warnings: list = []

        for side, bound in (('RIGHT', bound_r), ('LEFT', bound_l)):
            dist = np.array([np.min(np.linalg.norm(bound - pt, axis=1)) for pt in raceline])
            n_hit = int((dist < veh_half).sum())
            if n_hit:
                errors.append(
                    f'{side} wall hit: {n_hit} pts (min={dist.min():.3f}m < {veh_half:.2f}m)')
            n_close = int(((dist >= veh_half) & (dist < safety_half)).sum())
            if n_close:
                warnings.append(
                    f'Low margin to {side}: {n_close} pts < {safety_half:.2f}m')

        n_curv = int((np.abs(kappa) > curvlim).sum())
        if n_curv:
            warnings.append(
                f'Curvature limit exceeded: {n_curv} pts (max={np.abs(kappa).max():.3f})')

        n_vel = int((vx > v_max + 0.1).sum())
        if n_vel:
            warnings.append(f'Velocity limit exceeded: {n_vel} pts (max={vx.max():.2f})')

        a_lat = vx**2 * np.abs(kappa)
        n_alat = int((a_lat > a_lat_max * 1.05).sum())
        if n_alat:
            warnings.append(f'Lateral accel exceeded: {n_alat} pts (max={a_lat.max():.2f})')

        min_r = np.array([np.min(np.linalg.norm(bound_r - pt, axis=1)) for pt in raceline])
        min_l = np.array([np.min(np.linalg.norm(bound_l - pt, axis=1)) for pt in raceline])

        self.get_logger().info(
            f'[CheckTraj {label}] '
            f'min_r={min_r.min():.3f}m  min_l={min_l.min():.3f}m  '
            f'max_κ={np.abs(kappa).max():.3f}  '
            f'max_v={vx.max():.2f}  max_a_lat={a_lat.max():.2f}')

        for e in errors:
            self.get_logger().error(f'  [CheckTraj {label}] ERROR: {e}')
        for w in warnings:
            self.get_logger().warn(f'  [CheckTraj {label}] WARN: {w}')
        if not errors and not warnings:
            self.get_logger().info(f'  [CheckTraj {label}] OK')

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────
    def _log_stats(self, label: str, traj: np.ndarray, lap_time: float):
        vx = traj[:, 5]
        self.get_logger().info(
            f'[{label}] length={traj[-1, 0]:.2f}m  lap≈{lap_time:.2f}s  '
            f'v_max={vx.max():.2f}  v_min={vx.min():.2f}  v_avg={vx.mean():.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
