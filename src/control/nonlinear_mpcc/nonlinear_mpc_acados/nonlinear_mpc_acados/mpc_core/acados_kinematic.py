#!/usr/bin/env python3
"""acados port of EVO-MPCC base — 4-state Cartesian kinematic, EXTERNAL cost.

Identical model to Nonlinear_MPC.py (IPOPT) so the ROS node can swap
backends without re-tuning. Difference is the solver: SQP_RTI + HPIPM
runs in 1-5 ms per cycle vs IPOPT's 18-70 ms.

State (4):  [x, y, psi, s]
Input (3):  [v, delta, p]   (direct controls, NOT rates)

Per-stage parameters (model.p), set each cycle from the ROS node:
  [obs_dmin, obs_x, obs_y, side_pref,
   q_cte, q_lag, q_d_delta, R_safe, M_slack,
   left_x, left_y, right_x, right_y]

Track / centerline / boundaries supplied as CasADi spline interpolants
(self.center_lut_x(s) etc.) — same interface the IPOPT version uses, set
via set_track_data() before setup_MPC().
"""
import os
import math
import numpy as np
import casadi as ca
import scipy.linalg
from ._ros_compat import NullLogger, monotonic_now, yaw_to_quat

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class MPC:
    """acados backend mirroring the IPOPT EVO-MPCC interface."""

    def __init__(self, cost_type, system_model, logger=None):
        self._log = logger if logger is not None else NullLogger("[MPC-acados]")
        self.arch = "acados_evompcc"

        # Sizing — set in set_initial_params()
        self.N = None
        self.dT = None
        self.n_states = 4
        self.n_controls = 3
        # Wheelbase from yaml
        self.L = 0.307

        # ──────────────── Dynamic model (Phase 1: toggle + params) ────────
        # Default OFF: kinematic bicycle (current working configuration).
        # Switch via env var or class override; full dynamic adapter
        # comes online in Phase 2 (cost/output rewiring).
        self.use_dynamic = False
        # Vehicle dynamics — values match `stack_master/config/SIM/
        # SIM_pacejka.yaml` so the MPC's internal model is identical to
        # what the f110-simulator's std_kinematics::update_pacejka()
        # actually integrates. Identical params → MPC predictions match
        # simulator behaviour to numerical precision.
        self.dyn_m    = 3.54      # mass [kg]                  (sim: m)
        self.dyn_Iz   = 0.05797   # yaw inertia [kg·m²]        (sim: I_z)
        self.dyn_lf   = 0.162     # CG to front axle [m]        (sim: l_f)
        self.dyn_lr   = 0.145     # CG to rear axle [m]         (sim: l_r)
        self.dyn_h_cg = 0.014     # CG height [m] (load transfer) (sim: h_cg)
        self.dyn_mu   = 1.0       # friction coefficient        (sim: mu)
        # Linear tire stiffness — sim-matched (f110-simulator params.yaml).
        # F_y = μ · C_S · F_z · α (linear in slip angle).
        # F1Tenth realistic; dynamic effects are inherently small at
        # this scale + speed → looks kinematic-like. Pure visual
        # similarity to kinematic is EXPECTED for low-speed F1Tenth.
        self.dyn_Csf = 4.718
        self.dyn_Csr = 5.4562
        # Tire model selector. Three options:
        #   'linear'  : F_y = μ·C_S·F_z·α                — fastest, no saturation
        #   'tanh'    : F_y = μ·D·F_z·tanh(B·α)          — smooth saturation
        #   'pacejka' : F_y = μ·D·F_z·sin(C·atan(B·α−…)) — most accurate, unstable in RTI
        self.dyn_tire_model = 'linear'
        # Legacy flag kept for backward compat (read-only proxy).
        self.dyn_use_pacejka = (self.dyn_tire_model == 'pacejka')
        # STANDARD F1Tenth Pacejka (literature: TUM/AMZ/Liniger):
        #   B (stiffness): 10  — typical
        #   C (shape):     1.4 — magic number for tire curves
        #   D (peak):      1.0 — peak friction coefficient
        #   E (curvature): 0   — symmetric
        # Numerically stable for SQP_RTI single-iter (C ≈ 1.4 keeps
        # sin·atan derivative well-conditioned across operating range).
        self.dyn_Bf, self.dyn_Cf, self.dyn_Df, self.dyn_Ef = (
            10.0, 1.4, 1.0, 0.0)
        self.dyn_Br, self.dyn_Cr, self.dyn_Dr, self.dyn_Er = (
            10.0, 1.4, 1.0, 0.0)
        # Hybrid blend tuned for HIGH-SPEED DYNAMIC mode:
        # vx ≥ 0.5 m/s → essentially 100% Pacejka dynamic.
        # Below 0.5 (startup / stop only) blend toward kinematic so the
        # atan2-slip singularity is well-conditioned. ICRA / racing usage:
        # car spends ≪1% of time below 0.5 m/s, so this is effectively
        # always dynamic.
        #   vx=0.0: w_std = 0.07  (93% kinematic — startup only)
        #   vx=0.5: w_std = 0.50  (50/50 — passing through this is brief)
        #   vx=1.0: w_std = 0.95  (95% dynamic)
        #   vx=2.0: w_std = 1.00  (~100% dynamic)
        self.dyn_v_b   = 0.5       # blend centre [m/s] — was 3.0
        self.dyn_v_s   = 0.3       # blend spread [m/s] — was 1.0 (tighter)
        self.dyn_v_min = 0.2       # below: kinematic-dominated [m/s]
        self.dyn_a_max = 7.5       # max accel (matches sim max_accel)
        # Singularity epsilon for atan2 denominator. 1.0 m/s = robust for
        # SQP_RTI (proven stable on ICRA + F maps). Smaller (0.5) gives
        # better low-speed accuracy but causes IPM step collapse on
        # certain map start states. Stability over fidelity.
        self.dyn_v_eps = 1.0

        # Bounds
        self.v_max = 6.0
        self.v_min = 0.5
        self.theta_max = 0.4
        self.theta_min = -0.4
        self.s_min, self.s_max = 0.0, 1e3
        self.p_min, self.p_max = 0.0, 6.0

        self.is_ot = False
        self.vheid = {}
        self.param = {}

        # Track splines (set via set_track_data)
        self.center_lut_x = None
        self.center_lut_y = None
        self.center_lut_dx = None
        self.center_lut_dy = None
        self.right_lut_x = None
        self.right_lut_y = None
        self.left_lut_x = None
        self.left_lut_y = None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l = None
        self.path_length = None
        self.ref_v = None    # CasADi interpolant ref_v(s)

        # Live (rqt-tunable) weights — same names the IPOPT MPC uses.
        self.q_cte_live     = 8.0
        self.q_lag_live     = 200.0
        self.q_d_delta_live = 25.0
        # R_safe=0.3 (was 0.8 → 0.4 → 0.3): with corridor h-constraint now
        # also active (R_CAR=0.10 from each wall, ≈0.35 m usable each side
        # of centerline), the 0.4 m keepout combined with a 0.4 m D_DETOUR
        # forced the car to a position 0.535 m off-center to clear an
        # on-centerline obstacle — outside the corridor. Solver couldn't
        # satisfy both → braked to 2.9 m/s every lap to soften slack.
        # 0.3 m matches car_R 0.135 + obs_marker 0.135 + 0.03 m margin and
        # keeps the required offset (0.435 m) inside the corridor.
        self.R_safe_live    = 0.3
        self.a_lat_safe_live   = 6.0    # rqt: curvature-speed cap headroom
        self.D_detour_live     = 0.15   # rqt: side-cost detour offset
        self.D_apex_live       = 0.22   # rqt: apex-bias lateral target offset.
                                        # Pulls e_c toward inside of corner.
                                        # 0 = no apex bias (pure centerline).
                                        # 0.4 = aggressive racing line.
        self.R_car_live        = 0.0    # rqt: corridor + obs h-constraint margin
                                        # (0.0 = wall touchable for racing line)
        self.commit_dist_live  = 10.0   # rqt: obstacle commit trigger distance
        self.cost_spike_thr_live = 500.0  # rqt: fallback threshold
        self.alpha_steer_live  = 0.6    # rqt: steer EMA blend (Python only)
        # Baked codegen-time cost weights (must match q_*_def in
        # setup_MPC). Used to convert rqt-set absolute weights into the
        # scale multiplier pushed via p_sym slots. ALL major cost weights
        # are scale-wired so BO / rqt can sweep without acados rebuild.
        self.Q_CTE_BAKED    = 9.0
        self.Q_LAG_BAKED    = 45.0
        self.Q_PSI_BAKED    = 10.0
        self.Q_V_BAKED      = 8.0
        self.Q_DD_BAKED     = 5.0    # steer regularisation |δ|²
        self.Q_P_BAKED      = 4.0    # progress (p_v - v_max)²
        self.Q_DRATE_BAKED  = 80.0   # steer rate (Δδ)²
        self.q_cte_scale_live   = 1.0
        self.q_lag_scale_live   = 1.0
        self.q_psi_scale_live   = 1.0
        self.q_v_scale_live     = 1.0
        self.q_dd_scale_live    = 1.0
        self.q_p_scale_live     = 1.0
        self.q_drate_scale_live = 1.0
        self.M_slack_live   = 2.0e4
        # Static cost weights (set via param dict)
        self.q_v       = 15.0
        self.q_dv      = 15.0
        self.q_mu      = 0.05
        self.q_vp_proj = 60.0   # progress reward gamma

        # Solver storage
        self.solver = None
        self.X0 = None
        self.u0 = None
        self.WARM_START = False

        # Lap tracking — keep solver-internal s unbounded so the dynamics
        # constraint ṡ = p_v never sees a wrap discontinuity. Spline lookup
        # in cost uses s_periodic = s − L·floor(s/L) for [0, L) wrap.
        self.lap_count = 0
        self._last_sensor_s = None
        # Same persistent-offset trick for yaw (mirrors lap_count for s).
        # Without this, sensor yaw ∈ [−π, π] wraps every full revolution
        # and the cycle-by-cycle "shift initial_state[2] ± 2π" detection
        # fires repeatedly in the wrap zone, each time triggering multi-
        # iter SQP and disrupting the solve → cost spike (observed: 974
        # at the exact wrap moment + ~5 cycles of elevated cost).
        # Tracking yaw_offset as multiples of 2π keeps the solver-internal
        # ψ monotonically continuous, so no wrap event ever reaches the
        # solver.
        self._yaw_offset = 0.0
        self._last_sensor_yaw = None

        # Debug — same names IPOPT version uses, ROS node reads these
        self.dbg_n_obs_input = 0
        self.dbg_sel_dmin = float('inf')
        self.dbg_sel_x = float('inf')
        self.dbg_sel_y = float('inf')
        self.dbg_side_pref = 0.0
        self.dbg_solver_status = ""
        self.boundary_hook = None  # callable(shifted_points: list[(x,y)]) or None

    # ------------------------------------------------------------------
    # Setters (mirror IPOPT MPC interface so node code reuses)
    # ------------------------------------------------------------------
    def set_initial_params(self, param, vheid, is_ot):
        self.vheid = vheid
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        self.L = vheid.get('l_wb', param.get('L', 0.307))
        self.theta_max = param['theta_max']
        self.theta_min = -self.theta_max
        self.v_max = param['v_max']
        self.v_min = 0.5
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.is_ot = is_ot
        # Param overrides for static weights (LIVE ones come from defaults below)
        self.q_v       = param.get('mpc_v_track',     self.q_v)
        self.q_dv      = param.get('mpc_w_accel',     self.q_dv)
        self.q_mu      = param.get('q_mu',            self.q_mu)
        self.q_vp_proj = param.get('mpc_vp_project',  self.q_vp_proj)

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y,
                       element_arc_lengths, original_arc_length_total, ref_v):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total
        self.path_length = original_arc_length_total
        self.ref_v = ref_v

        # Precompute curvature κ(s) on a fine grid for runtime v-cap lookup.
        # κ = dθ/ds where θ = atan2(dy/ds, dx/ds). Sharp corners (hairpin)
        # have |κ| > 2 rad/m on f1tenth tracks; pure-Cartesian MPCC's
        # contour/lag cost geometry breaks there, so we cap v per stage by
        # v ≤ sqrt(a_lat_safe / |κ|) to keep the QP well-posed.
        self.kappa_ds = 0.05
        L = float(original_arc_length_total)
        s_grid = np.arange(0.0, L, self.kappa_ds)
        dx_g = np.array([float(c_dx(float(s))) for s in s_grid])
        dy_g = np.array([float(c_dy(float(s))) for s in s_grid])
        theta_g = np.unwrap(np.arctan2(dy_g, dx_g))
        self.kappa_grid = np.gradient(theta_g, self.kappa_ds)
        self._log.info("[MPC-acados] kappa grid: |k|_max=%.2f rad/m, |k|_p95=%.2f",
                      float(np.max(np.abs(self.kappa_grid))),
                      float(np.percentile(np.abs(self.kappa_grid), 95)))

        # Make |κ| available to CasADi as a linear interpolant — enables
        # per-stage curvature-aware speed cap inside the cost. Track is
        # closed-loop so append the s=0 value at s=L for clean wrap-around.
        # FORWARD-LOOKING: instead of |κ(s)|, use max(|κ(s')| for s' ∈ [s,
        # s+LOOKAHEAD]). At s=60 (just before the upper-left U-turn) the
        # raw |κ| is only 0.04 (v_kin=11 m/s, cap inactive), but at s=63
        # the peak |κ|=0.71 (v_kin=2.9 m/s). MPC's horizon (~2.5 m at
        # v=5) barely reaches the peak so per-stage cap fires too late
        # to brake. Forward-max over LOOKAHEAD = 4 m makes the cap kick
        # in 4 m before any sharp corner, giving MPC enough time to slow.
        abs_k_arr = np.abs(self.kappa_grid)
        # Gaussian-ish smoothing of |κ| profile before forward-max. Without
        # it, |κ| has high-freq variation from spline derivative noise →
        # ref_v_cap jitters cycle-to-cycle as car advances, manifesting as
        # speed surge after corner exit. 11-tap rolling avg ≈ 0.5 m smooth.
        smooth_win = 11
        kernel = np.ones(smooth_win) / float(smooth_win)
        abs_k_arr = np.convolve(abs_k_arr, kernel, mode='same')
        LOOKAHEAD_M = 6.0
        n_look = int(LOOKAHEAD_M / self.kappa_ds)
        n_grid = len(abs_k_arr)
        abs_k_fwd = np.empty(n_grid, dtype=float)
        for i in range(n_grid):
            j = min(i + n_look, n_grid)
            abs_k_fwd[i] = float(abs_k_arr[i:j].max())
        self._log.info("[MPC-acados] forward-max |κ| (lookahead=%.1f m): max=%.3f p95=%.3f",
                      LOOKAHEAD_M,
                      float(abs_k_fwd.max()),
                      float(np.percentile(abs_k_fwd, 95)))
        if self.use_dynamic:
            self._log.info("[MPC-acados] MODEL = DYNAMIC (%s tire) (n_states=8, "
                          "n_controls=3, LM=1.0, RTI 1-iter)",
                          self.dyn_tire_model)
        else:
            self._log.info("[MPC-acados] MODEL = KINEMATIC (n_states=5, n_controls=3, LM=0.2)")
        s_grid_ext = list(s_grid) + [L]
        abs_k_ext  = abs_k_fwd.tolist() + [abs_k_fwd[0]]
        self.abs_kappa_lut = ca.interpolant(
            'abs_kappa_lut', 'linear', [s_grid_ext], abs_k_ext
        )

        # Signed κ LUT for apex-biased lateral reference. Used by the cost
        # to shift the e_c target toward the INSIDE of the upcoming corner,
        # so centerline tracking produces racing-line behaviour. Sign:
        # κ_signed > 0 = left turn (CCW) → inside is LEFT.
        #
        # Profile design — emulate the global IQP raceline's behaviour
        # where straights aren't strictly tracked: the car drifts
        # diagonally toward the inside of the next corner during the
        # approach straight, hits apex, and decays back to centerline
        # gradually on the exit. Without this, centerline mode produces
        # straight-on-straight then sharp corner attack — slow.
        #
        # Construction:
        #  1. Gaussian kernel σ=0.8 m (3σ ≈ 2.4 m support) — covers a
        #     racing-line-like wide influence zone around each apex.
        #  2. Forward-shift output by 0.6 m (np.roll on closed loop).
        #     At s on the approach straight, the LUT returns the
        #     smoothed κ at s+0.6 m → bias kicks in BEFORE apex.
        # Net bias profile (peak-normalised) around an apex:
        #   apex-2.5 m: 0.10  (drift starts on approach straight)
        #   apex-1.5 m: 0.40  (clear diagonal on straight)
        #   apex-0.6 m: 1.00  (peak — slight before geometric apex)
        #   apex      : 0.75
        #   apex+0.5 m: 0.30  (sustain on exit)
        #   apex+1.5 m: 0.05  (almost back to centerline)
        sigma_m_apex   = 0.8
        forward_bias_m = 0.6
        half_idx = int(3 * sigma_m_apex / self.kappa_ds)
        kx = np.arange(-half_idx, half_idx + 1) * self.kappa_ds
        ker_apex = np.exp(-kx * kx / (2.0 * sigma_m_apex * sigma_m_apex))
        ker_apex /= ker_apex.sum()
        signed_k_smooth = np.convolve(self.kappa_grid, ker_apex, mode='same')
        forward_bias_idx = int(forward_bias_m / self.kappa_ds)
        signed_k_arr = np.roll(signed_k_smooth, -forward_bias_idx)
        signed_k_ext = signed_k_arr.tolist() + [signed_k_arr[0]]
        self.signed_kappa_lut = ca.interpolant(
            'signed_kappa_lut', 'linear', [s_grid_ext], signed_k_ext
        )

    # ------------------------------------------------------------------
    # Dynamic-bicycle model builder (Phase 1: helper, NOT YET WIRED)
    # ------------------------------------------------------------------
    def _build_dynamic_model(self):
        """Build the Pacejka single-track dynamic model expressions.

        Mirrors `f110-simulator/src/std_kinematics.cpp::update_pacejka`
        verbatim so MPC predictions match what the simulator integrates
        (assuming MPC and sim are configured with the same SIM_pacejka
        parameters — which `__init__` does by default).

        Returns a dict (intentionally NOT installed into `setup_MPC` yet).
        Phase 2 wiring will:
          - replace the kinematic `x` / `u` / `f_expl` with these
          - swap the cost residual `v − ref_v` → `vx − ref_v`
          - swap the h-constraint `a_lat = v² tan(δ)/L` → `a_lat = vx · r`
          - change `solve_step` initial state to include vx, vy, r
          - change `solve_step` output to use vx[1] as v_cmd
        """
        # ── States (8): x, y, ψ, vx, vy, r, s, δ_prev ─────────────────
        x_pos      = ca.SX.sym('x_pos')
        y_pos      = ca.SX.sym('y_pos')
        psi        = ca.SX.sym('psi')
        vx         = ca.SX.sym('vx')
        vy         = ca.SX.sym('vy')
        r_yaw      = ca.SX.sym('r_yaw')
        s          = ca.SX.sym('s')
        delta_prev = ca.SX.sym('delta_prev')
        x = ca.vertcat(x_pos, y_pos, psi, vx, vy, r_yaw, s, delta_prev)

        # ── Inputs (3): a_x, δ, p_v ───────────────────────────────────
        # Replaces kinematic's `v` with `a_x` (longitudinal accel).
        # F1Tenth's ackermann_drive accepts `drive.speed` (m/s); we'll
        # send `vx[1]` (next-stage predicted vx) as v_cmd in Phase 2.
        a_x   = ca.SX.sym('a_x')
        delta = ca.SX.sym('delta')
        p_v   = ca.SX.sym('p_v')
        u = ca.vertcat(a_x, delta, p_v)

        xdot = ca.SX.sym('xdot', 8)

        # ── Pacejka dynamic equations ─────────────────────────────────
        # Slip angles. Singularity at v_x → 0 in atan2 denominator: use
        # ca.fmax(vx, ε) so derivatives stay finite. Kinematic blend
        # below dyn_v_b further regularises (low-speed mode).
        vx_safe = ca.fmax(vx, self.dyn_v_eps)
        # sim convention: alpha_f = atan2(-vy - lf·r, vx) + delta
        alpha_f = ca.atan2(-vy - self.dyn_lf * r_yaw, vx_safe) + delta
        alpha_r = ca.atan2(-vy + self.dyn_lr * r_yaw, vx_safe)

        # Vertical loads with longitudinal load transfer
        # (sim: F_zf = m · (-a_x · h_cg + g · l_r) / L)
        g_const = 9.81
        L_wb = self.dyn_lf + self.dyn_lr
        F_zf = self.dyn_m * (-a_x * self.dyn_h_cg + g_const * self.dyn_lr) / L_wb
        F_zr = self.dyn_m * ( a_x * self.dyn_h_cg + g_const * self.dyn_lf) / L_wb

        # Tire force model selector — see __init__ self.dyn_tire_model.
        if self.dyn_tire_model == 'pacejka':
            # Full Pacejka Magic Formula. NOTE: SQP_RTI single iter
            # struggles with this — sin·atan derivative variation
            # between cycles → MINSTEP cascades. Use only with multi-
            # iter solve enabled (self.dyn_multi_iter).
            pacejka_arg_f = (self.dyn_Bf * alpha_f
                             - self.dyn_Ef * (self.dyn_Bf * alpha_f
                                              - ca.atan(self.dyn_Bf * alpha_f)))
            pacejka_arg_r = (self.dyn_Br * alpha_r
                             - self.dyn_Er * (self.dyn_Br * alpha_r
                                              - ca.atan(self.dyn_Br * alpha_r)))
            F_yf = (self.dyn_mu * self.dyn_Df * F_zf
                    * ca.sin(self.dyn_Cf * ca.atan(pacejka_arg_f)))
            F_yr = (self.dyn_mu * self.dyn_Dr * F_zr
                    * ca.sin(self.dyn_Cr * ca.atan(pacejka_arg_r)))
        elif self.dyn_tire_model == 'tanh':
            # tanh tire = SIMPLIFIED Pacejka. saturation built-in but
            # derivative is well-behaved everywhere (sech², bounded).
            # F_y = μ·D·F_z·tanh(B·α). Reaches saturation D·F_z at α≫0,
            # linear for small α with slope D·F_z·μ·B.
            # Used in: Hewing/Liniger ETH, Carrau et al, several
            # racing MPC papers. Sweet spot of accuracy + numerics.
            F_yf = self.dyn_mu * self.dyn_Df * F_zf * ca.tanh(self.dyn_Bf * alpha_f)
            F_yr = self.dyn_mu * self.dyn_Dr * F_zr * ca.tanh(self.dyn_Br * alpha_r)
        else:  # 'linear'
            # F_y = μ · C_S · F_z · α. No saturation.
            F_yf = self.dyn_mu * self.dyn_Csf * F_zf * alpha_f
            F_yr = self.dyn_mu * self.dyn_Csr * F_zr * alpha_r

        # Dynamic state derivatives (sim std_kinematics.cpp:82-88).
        # F_xf = 0 (RWD); F_xr is folded into a_x (sim's `accel` arg).
        f_dyn = ca.vertcat(
            vx * ca.cos(psi) - vy * ca.sin(psi),                         # ẋ
            vx * ca.sin(psi) + vy * ca.cos(psi),                         # ẏ
            r_yaw,                                                        # ψ̇
            a_x + (1.0 / self.dyn_m) * (-F_yf * ca.sin(delta)) + vy * r_yaw,
            (1.0 / self.dyn_m) * (F_yr + F_yf * ca.cos(delta)) - vx * r_yaw,
            (1.0 / self.dyn_Iz) * (F_yf * self.dyn_lf * ca.cos(delta)
                                    - F_yr * self.dyn_lr),
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # ── Kinematic equations (for low-vx blend) ────────────────────
        # When vx ≪ v_b, the dynamic atan2 and small lateral forces
        # produce noisy derivatives. Blend toward kinematic single-track
        # so prediction stays well-conditioned at start-up / near-stop.
        # Kinematic regime forces vy and r toward their kinematic-
        # consistent values via fast first-order decay (τ_kin small).
        beta_kin   = ca.atan(self.dyn_lr * ca.tan(delta) / L_wb)
        vy_target  = vx * ca.tan(beta_kin)
        r_target   = (vx / L_wb) * ca.tan(delta) * ca.cos(beta_kin)
        tau_kin    = 0.05  # rapid relaxation toward kinematic (50 ms)
        f_kin = ca.vertcat(
            vx * ca.cos(psi + beta_kin),                                  # ẋ
            vx * ca.sin(psi + beta_kin),                                  # ẏ
            r_target,                                                      # ψ̇ (set by kin geom)
            a_x,                                                           # v̇x
            (vy_target - vy) / tau_kin,                                    # v̇y → kin
            (r_target  - r_yaw) / tau_kin,                                 # ṙ   → kin
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # Smooth blend: w_std rises from 0 to 1 across vx ∈ [v_b−v_s, v_b+v_s]
        # (sim: w_std = 0.5·(1 + tanh((vx − v_b)/v_s)))
        w_std  = 0.5 * (1.0 + ca.tanh((vx - self.dyn_v_b) / self.dyn_v_s))
        f_expl = w_std * f_dyn + (1.0 - w_std) * f_kin

        # Real lateral acceleration (replaces v² tan(δ)/L kinematic form).
        a_lat_expr = vx * r_yaw

        return dict(
            x=x, u=u, xdot=xdot, f_expl=f_expl,
            x_pos=x_pos, y_pos=y_pos, psi=psi,
            vx=vx, vy=vy, r=r_yaw, s=s, delta_prev=delta_prev,
            a_x=a_x, delta=delta, p_v=p_v,
            F_yf=F_yf, F_yr=F_yr, F_zf=F_zf, F_zr=F_zr,
            alpha_f=alpha_f, alpha_r=alpha_r,
            f_dyn=f_dyn, f_kin=f_kin, w_std=w_std,
            a_lat_expr=a_lat_expr,
        )

    # ------------------------------------------------------------------
    # OCP setup
    # ------------------------------------------------------------------
    def setup_MPC(self):
        ocp = AcadosOcp()
        model_ac = AcadosModel()
        model_ac.name = 'mpcc_evompcc_acados'

        # ---- States: x, y, psi, s, delta_prev ----
        # delta_prev is augmented to enable a per-stage steer-rate cost
        # (residual = delta - delta_prev). NONLINEAR_LS y(x,u) can only
        # see one stage at a time, so we shadow the previous applied δ
        # as a state. Dynamics: δ_prev_dot = (δ − δ_prev)/dt → after one
        # Euler step, δ_prev_new = δ_old (the control just applied).
        # Without rate cost the solver picks bimodal δ patterns
        # ([+0.22, +0.08, +0.22, +0.08, ...] alternating) that satisfy
        # the same average heading change with similar |δ|² total but
        # produce a visibly wavy prediction line.
        # ── Branch on use_dynamic ────────────────────────────────────────
        # Dynamic mode: 8 states [x, y, ψ, vx, vy, r, s, δ_prev], 3 inputs
        #               [a_x, δ, p_v]. Pacejka tire forces with hybrid
        #               kinematic blend at low vx.
        # Kinematic mode (default): 5 states [x, y, ψ, s, δ_prev], 3 inputs
        #               [v, δ, p_v]. Standard kinematic bicycle.
        # `vx_for_cost` is the longitudinal-velocity symbol used by cost
        # residuals (vx state in dynamic, v input in kinematic). Same for
        # `a_lat_expr` (h-constraint) — vx·r vs v² tan(δ)/L.
        if self.use_dynamic:
            dyn = self._build_dynamic_model()
            x          = dyn['x']
            u          = dyn['u']
            f_expl     = dyn['f_expl']
            xdot       = dyn['xdot']
            x_         = dyn['x_pos']
            y_         = dyn['y_pos']
            psi        = dyn['psi']
            s          = dyn['s']
            delta_prev = dyn['delta_prev']
            delta      = dyn['delta']
            p_v        = dyn['p_v']
            vx_for_cost  = dyn['vx']
            a_lat_expr   = dyn['a_lat_expr']
            v_input_sym  = None              # no v input in dynamic
            a_x_input    = dyn['a_x']
            nx = 8
            nu = 3
        else:
            x_   = ca.SX.sym('x_')
            y_   = ca.SX.sym('y_')
            psi  = ca.SX.sym('psi')
            s    = ca.SX.sym('s')
            delta_prev = ca.SX.sym('delta_prev')
            x = ca.vertcat(x_, y_, psi, s, delta_prev)
            nx = x.size1()

            # ---- Direct controls: v, delta, p (path velocity) ----
            v     = ca.SX.sym('v')
            delta = ca.SX.sym('delta')
            p_v   = ca.SX.sym('p_v')
            u = ca.vertcat(v, delta, p_v)
            nu = u.size1()

            xdot = ca.SX.sym('xdot', nx)

            # ---- Kinematic bicycle (+ delta_prev tracker) ----
            f_expl = ca.vertcat(
                v * ca.cos(psi),
                v * ca.sin(psi),
                (v / self.L) * ca.tan(delta),
                p_v,
                (delta - delta_prev) / self.dT,   # δ_prev_dot tracks current δ
            )
            vx_for_cost  = v
            a_lat_expr   = v * v * ca.tan(delta) / self.L
            v_input_sym  = v
            a_x_input    = None
        self.n_states   = nx
        self.n_controls = nu

        # ---- Per-cycle parameters (constant across all stages) ----
        # idx: 0=obs_dmin, 1=obs_x, 2=obs_y, 3=side_pref,
        #      4=q_cte, 5=q_lag, 6=q_d_delta, 7=R_safe, 8=M_slack,
        #      9=e_c_obs (obstacle's lateral offset from centerline, Frenet)
        # 10 reserved (unused).
        # The half-plane uses e_c_obs (Frenet) instead of a tangent-anchored
        # Cartesian normal. The previous tangent-anchored form (obs_tan_sin/
        # cos) was stable per cycle but ignored track curvature — over a
        # curving section the prediction had to choose between following
        # the curve (e_c roughly tracks centerline) and respecting a
        # straight-line half-plane that rotated relative to the curve, so
        # prediction shot off-track. Frenet-frame e_c naturally follows
        # the curve, and a Cartesian-distance proximity gate disables the
        # constraint far from the obstacle.
        # Slot map (per-cycle constants):
        #   0  obs_dmin (debug)        9  e_c_obs (Frenet obstacle offset)
        #   1  obs_x                   10 a_lat_safe   (rqt)
        #   2  obs_y                   11 D_apex       (rqt)
        #   3  side_pref               12 q_psi_scale  (rqt) — NEW
        #   4  D_DETOUR    (rqt)       13 q_v_scale    (rqt) — NEW
        #   5  R_CAR       (rqt)       14 q_dd_scale   (rqt) — NEW
        #   6  q_cte_scale (rqt)       15 q_p_scale    (rqt) — NEW
        #   7  R_safe      (rqt)       16 q_drate_scale(rqt) — NEW
        #   8  q_lag_scale (rqt)
        # Per-stage (4):
        #   17 left_x  18 left_y  19 right_x  20 right_y
        n_p_const = 17
        n_p_stage = 4
        n_p_total = n_p_const + n_p_stage
        p_sym = ca.SX.sym('p_sym', n_p_total)
        obs_dmin = p_sym[0]; obs_x = p_sym[1]; obs_y = p_sym[2]
        side_pref = p_sym[3]
        D_detour_p     = p_sym[4]   # side-cost detour offset (rqt)
        R_car_p        = p_sym[5]   # corridor + obs h margin (rqt)
        q_cte_scale_p  = p_sym[6]   # × q_cte_def (rqt)
        R_safe_p       = p_sym[7]
        q_lag_scale_p  = p_sym[8]   # × q_lag_def (rqt)
        e_c_obs_p      = p_sym[9]
        a_lat_safe_p   = p_sym[10]  # curvature speed-cap headroom (rqt)
        D_apex_p       = p_sym[11]  # apex-bias offset (rqt)
        # NEW scale-multiplier slots so BO can sweep all cost weights live.
        q_psi_scale_p   = p_sym[12]  # × q_psi_def
        q_v_scale_p     = p_sym[13]  # × q_v_def
        q_dd_scale_p    = p_sym[14]  # × q_dd_def
        q_p_scale_p     = p_sym[15]  # × q_p_def
        q_drate_scale_p = p_sym[16]  # × q_d_rate_def
        left_x  = p_sym[17]; left_y  = p_sym[18]
        right_x = p_sym[19]; right_y = p_sym[20]

        # ---- Reference geometry ----
        # s_periodic = s − L·floor(s/L) ∈ [0, L). The solver-internal s
        # state grows unboundedly across laps (dynamics ṡ = p_v stays
        # smooth — no wrap discontinuity → no lap-rollover ACADOS_MINSTEP).
        # ca.floor's derivative is 0, so ∂s_periodic/∂s = 1 throughout,
        # which means the cost gradient w.r.t. s is correct everywhere
        # except exactly at multiples of L (a measure-zero set; for a
        # closed-loop track the spline values at s=L⁻ and s=0⁺ are nearly
        # equal anyway, so the cost-value jump is tiny).
        L_track = float(self.path_length)
        s_periodic = s - L_track * ca.floor(s / L_track)
        ref_x  = self.center_lut_x(s_periodic)
        ref_y  = self.center_lut_y(s_periodic)
        dxt    = self.center_lut_dx(s_periodic)
        dyt    = self.center_lut_dy(s_periodic)
        # Compute sin/cos of the centerline tangent DIRECTLY from dxt, dyt
        # — never form t_angle = atan2(dyt, dxt) as an intermediate
        # variable. atan2 has a branch cut at the −x axis: when the
        # centerline tangent crosses (dxt<0, dyt=0±ε) — which happens on
        # any leftward straight or curve in trackf at s≈56.7 — atan2
        # jumps from +π to −π even though the geometric direction is
        # smooth. That value-jump in t_angle, even with downstream
        # atan2(sin·,cos·) wrap-safety in yaw_err, creates a ~43k cost
        # spike at exactly that s every lap (observed on trackf at
        # (6-7, 18) wrap location). Forming sin/cos directly via
        # normalization is smooth across the jump.
        norm_t = ca.sqrt(dxt * dxt + dyt * dyt + 1e-12)
        sin_t = dyt / norm_t
        cos_t = dxt / norm_t

        # Contour (lateral) and lag (along-track) errors
        e_c = sin_t * (x_ - ref_x) - cos_t * (y_ - ref_y)
        e_l = -cos_t * (x_ - ref_x) - sin_t * (y_ - ref_y)
        # Reference velocity at s, capped by kinematic curvature limit
        # v ≤ √(a_lat_safe / |κ(s)|). Without this cap the centerline
        # ref_v=6 m/s applies even at the hairpins (κ_max≈0.84) where
        # the kinematically feasible v is sqrt(8/0.84) ≈ 3.08 m/s. MPC
        # was forced to choose between (a) tracking ref_v=6 → saturated
        # steer + a_lat slack absorbing huge violation → wall-stuck, or
        # (b) braking only on the q_v term, which competes with q_lag and
        # never wins enough. With the cap, ref_v becomes ~3 m/s in tight
        # corners → MPC naturally slows for the apex; the rest of the
        # cost machinery just tracks the new (kinematically-safe) target.
        # A_LAT_SAFE = 6.0 < a_lat_max = 8.0 leaves ~25% headroom so the
        # actual a_lat constraint is rarely binding (no slack spike).
        # Smooth fmin via 0.5·(a+b−sqrt((a−b)²+ε)) — differentiable
        # everywhere, kink rounded over ~0.03 m/s.
        ref_v_raw  = self.ref_v(s_periodic)
        kappa_at_s = self.abs_kappa_lut(s_periodic)
        # A_LAT_SAFE comes from p_sym[10] — rqt-tunable via "a_lat_safe"
        # in MPCTune.cfg. Lower = more conservative cornering, higher =
        # faster but closer to a_lat_max=8 limit.
        v_kin_max  = ca.sqrt(a_lat_safe_p / (kappa_at_s + 1e-3))
        EPS_VM     = 1e-3
        diff_v     = ref_v_raw - v_kin_max
        ref_v_expr = 0.5 * (ref_v_raw + v_kin_max
                            - ca.sqrt(diff_v * diff_v + EPS_VM))

        # ---- Distance² to obstacle (used by h constraint only) ----
        d2 = (x_ - obs_x) ** 2 + (y_ - obs_y) ** 2

        # Heading error — wrap to [-π, π] via atan2(sin, cos) of the
        # difference. Use the angle-subtraction identities so the inputs
        # to atan2 are computed from sin_t, cos_t (smooth) rather than
        # going through t_angle (would have the same branch-cut spike):
        #   sin(ψ − t) = sin(ψ)·cos(t) − cos(ψ)·sin(t)
        #   cos(ψ − t) = cos(ψ)·cos(t) + sin(ψ)·sin(t)
        sin_psi = ca.sin(psi); cos_psi = ca.cos(psi)
        sin_diff = sin_psi * cos_t - cos_psi * sin_t
        cos_diff = cos_psi * cos_t + sin_psi * sin_t
        yaw_err = ca.atan2(sin_diff, cos_diff)

        # ---- NONLINEAR_LS cost form ----
        # y(x,u) = [e_c, e_l, yaw_err, v − ref_v, δ, p − p_max, side_term]
        # The side_term re-introduces obstacle-side preference but in a
        # PSD form (multiplied by sqrt(proximity)·|side_pref|). When no
        # obstacle (sentinel) → proximity≈0 → cost contribution≈0.
        # σ=1.3 (was 0.6 → 1.0 → 1.3; 1.5 caused cost=3337 IPM blowup with
        # the old q_side=25, but with q_side=12 and D_DETOUR_SIDE=0.25
        # there's headroom). Earlier engagement so prediction starts to
        # bend laterally before the obstacle is in the dynamics horizon —
        # avoids the "pred line stuck against obstacle, brake at the last
        # moment" pattern (pred_endpoint within R_safe of obs in 14% of
        # cycles in the σ=1.0 run). Proximity table:
        #   d=2  m: σ=1.0→0.14   σ=1.3→0.31
        #   d=3  m: σ=1.0→0.01   σ=1.3→0.07
        #   d=4  m: σ=1.0→3e-4   σ=1.3→0.008
        sigma_side = 0.5  # 1.3 → 1.0 → 0.7 → 0.5. Tighter side cost
                          # window so MPC barely detours unless very close
                          # to obstacle. Proximity at d=1m: 0.14, at
                          # d=0.7m: 0.37 — engagement starts at ~1 m.
        proximity_side = ca.exp(-d2 / (2.0 * sigma_side * sigma_side))
        # 0.25 m (was 0.4 → 0.25 to pair with R_safe 0.3 and corridor
        # h-constraint). With the soft corridor active, a 0.4 m detour put
        # the cost minimum at the corridor edge → lateral push fighting the
        # corridor push, and MPC braked to release the tension. 0.25 m
        # places the detour line at e_c=±0.25 well inside the corridor
        # (±0.35 usable), so side cost and corridor cost don't compete.
        # Required clearance from obstacle still met: detour 0.25 + R_safe
        # 0.3 (with car edge buffer) = 0.55 m closest approach center-to-center.
        # D_DETOUR_SIDE now read from p_sym[4] (rqt-tunable). Default value
        # in self.D_detour_live = 0.15. Live tunable per-cycle.
        # |side_pref| ∈ {0, 1}; smooth approx avoids non-differentiable |·|
        # 1e-3 (was 1e-9): keeps sqrt's gradient bounded near 0 — with
        # 1e-9, ∂sqrt/∂x at x=0 is ~16k, which makes the GN Hessian
        # spike when proximity → 0 and overshoots IPM step.
        abs_side_smooth = ca.sqrt(side_pref * side_pref + 1e-3)
        side_term = (abs_side_smooth
                     * ca.sqrt(proximity_side + 1e-3)
                     * (e_c - side_pref * D_detour_p))
        # Adaptive lane-tracking attenuation (mirrors IPOPT MPCC.py technique).
        # Multiplies the e_c/e_l cost by 1 - 0.95·exp(-d²/2σ²): centerline
        # tracking is normal far away, fades to ~5% at obstacle center.
        # Without this, q_cte (pull to centerline) and q_side (push to detour
        # line) fight each other near the obstacle → MPC brakes hard to
        # release the tension and outputs jerky control. With attenuation,
        # the centerline pull naturally relaxes when needed, so detour is
        # "free" near obstacle and tracking resumes once past. NONLINEAR_LS
        # form: cost = q · residual², so multiplying residual by sqrt(att)
        # multiplies cost by att.
        # σ_atten = 1.0 (Gaussian width):
        #   d=∞    → att=1.00 (normal)
        #   d=2 m  → att≈0.87
        #   d=1 m  → att≈0.42 (cost halved)
        #   d=0.5  → att≈0.16
        #   d=0    → att≈0.05 (tracking off)
        sigma_atten = 1.0
        attenuation = 1.0 - 0.95 * ca.exp(-d2 / (2.0 * sigma_atten * sigma_atten))
        # κ-based attenuation (deadband-shaped, quartic). Centerline mode:
        # fade q_cte at high κ so MPC drifts toward racing line at corner
        # apex while preserving full tracking on straights. Forward-max
        # κ LUT (6 m lookahead) makes this fade kick in on the APPROACH
        # to a corner, not at the apex itself.
        #   κ=0    att=1.00      κ=0.4  att=0.57
        #   κ=0.2  att=0.95      κ=0.6  att=0.20
        #   κ=0.3  att=0.81      κ=0.8  att=0.08
        kappa_sq = kappa_at_s * kappa_at_s
        att_kappa = 1.0 / (1.0 + 30.0 * kappa_sq * kappa_sq)
        sqrt_att  = ca.sqrt(attenuation * att_kappa + 1e-6)
        # 8th residual: δ − δ_prev → steer-rate cost. Penalises stage-to-
        # stage δ change directly, which kills within-prediction zigzag.
        # Cost weights q_cte_def, q_lag_def in W are baked at codegen,
        # but multiplied by q_cte_scale_p / q_lag_scale_p at the residual
        # level so the rqt slider can change the EFFECTIVE weight live:
        #   cost = W_baked · (sqrt(scale_p) · sqrt(att) · e_c)²
        #        = W_baked · scale_p · att · e_c²
        # default scale=1.0 → effective = baked.
        sqrt_q_cte_scale   = ca.sqrt(q_cte_scale_p   + 1e-9)
        sqrt_q_lag_scale   = ca.sqrt(q_lag_scale_p   + 1e-9)
        sqrt_q_psi_scale   = ca.sqrt(q_psi_scale_p   + 1e-9)
        sqrt_q_v_scale     = ca.sqrt(q_v_scale_p     + 1e-9)
        sqrt_q_dd_scale    = ca.sqrt(q_dd_scale_p    + 1e-9)
        sqrt_q_p_scale     = ca.sqrt(q_p_scale_p     + 1e-9)
        sqrt_q_drate_scale = ca.sqrt(q_drate_scale_p + 1e-9)

        # Apex-biased lateral reference. Shifts the e_c cost minimum toward
        # the INSIDE of the upcoming corner so centerline mode produces
        # racing-line cornering (wide entry implicit via κ-att fade,
        # inside apex via this bias, sustained inside on EXIT via the
        # 21-tap smoothing of signed κ — bias decays slowly so the car
        # holds the inside line into the next straight). Standard
        # technique in MPCC literature (Liniger et al., TUM AR).
        # e_c convention here: +e_c = RIGHT of path-forward; inside of a
        # LEFT turn (κ_signed > 0) is LEFT → e_c_ref < 0. Hence the −sign.
        # tanh saturates fast (K=8): κ=0.05→ref=−0.13, κ=0.15→−0.20,
        # κ≥0.3→±D_apex (saturated). D_apex=0.22 → 22 cm bias at apex.
        # Bias dies on straights (κ≈0 → ref≈0) but the 21-tap kernel
        # keeps κ_smoothed > 0 for ≈1 m past the geometric apex →
        # inside line persists through corner→straight transition.
        # Combined with κ-att deadband: at apex, weight is faded but
        # ref pulls inside; on approach, weight is faded already
        # (forward-max κ) but ref kicks in once stage reaches high-κ.
        # D_apex is now LIVE-tunable via p_sym[11] (rqt slider). K_apex
        # stays baked (controls only the κ → bias-magnitude saturation
        # curve shape, not the magnitude itself).
        K_apex = 8.0
        signed_kappa_at_s = self.signed_kappa_lut(s_periodic)
        e_c_ref = -D_apex_p * ca.tanh(K_apex * signed_kappa_at_s)

        y_expr   = ca.vertcat(sqrt_q_cte_scale   * sqrt_att * (e_c - e_c_ref),
                              sqrt_q_lag_scale   * sqrt_att * e_l,
                              sqrt_q_psi_scale   * yaw_err,
                              sqrt_q_v_scale     * (vx_for_cost - ref_v_expr),
                              sqrt_q_dd_scale    * delta,
                              sqrt_q_p_scale     * (p_v - self.v_max),
                              side_term,
                              sqrt_q_drate_scale * (delta - delta_prev))
        y_expr_e = ca.vertcat(e_c, e_l, yaw_err)

        # ---- Constraints (h) — minimal set for stable SQP_RTI ----
        # 1) obstacle half-plane (replaces 2026-05-06 the non-convex annulus
        #    d²−R²≥0). The annulus form is bistable: slack absorbs the
        #    "diving through obstacle" path as a local optimum, so the
        #    predicted trajectory ends *inside* the keepout (observed:
        #    pred_endpoint within R_safe of obs in 14% of obstacle-active
        #    cycles → "경로 라인이 장애물에 막힘" symptom + last-moment
        #    brake/burst). Half-plane: project (car − obs) onto the
        #    right-perpendicular of the centerline tangent at the predicted
        #    stage, then require side_pref · projection ≥ R_safe + R_CAR.
        #    Convex, one-sided, prediction physically cannot cross to the
        #    other side. side_pref ∈ {-1, 0, +1} from decide_side_pref;
        #    when side_pref ≈ 0 (no obstacle / sentinel), a |side_pref|-
        #    gated big-M term keeps the constraint trivially satisfied.
        # 2) corridor lateral bound — added 2026-05-06 to prevent sim-wall
        #    stuck. Project corridor boundary points onto the same right-
        #    perpendicular as e_c, then bound e_c. Smooth max/min so it
        #    works for CW/CCW orientation and avoids the kink that caused
        #    HPIPM S_MINSTEP at narrow stages.
        # 3) lateral accel limit (kinematic).
        # R_CAR now read from p_sym[5] (rqt-tunable). Default value in
        # self.R_car_live = 0.02. Live tunable per-cycle. Used by both
        # h_obs (obstacle margin) and h_corridor (wall margin).
        R_CAR = R_car_p
        # ---- (1) obstacle half-plane (Frenet + proximity-gated) ----
        # Frenet form: side_pref · (e_c − e_c_obs) ≥ R + R_CAR. Naturally
        # curve-aware because e_c at each predicted stage is measured
        # against the centerline at THAT stage's s. No tangent rotation
        # issue.
        # Proximity gate: gate = |side_pref| · exp(−d²/2σ²). Disables the
        # constraint smoothly when far from the obstacle (d>3 m → gate≈0)
        # or when no obstacle is selected (side_pref≈0). Big-M on the
        # complementary side keeps the constraint trivially satisfied
        # outside the active zone.
        SIGMA_OBS = 1.0   # 2.0 → 1.4 → 1.0. Half-plane gate fires in
                          # a tighter window (~0.7 m vs ~1 m). Less
                          # detour anticipation, smaller swerve.
        prox_obs  = ca.exp(-d2 / (2.0 * SIGMA_OBS * SIGMA_OBS))
        abs_side  = ca.sqrt(side_pref * side_pref + 1e-3)
        gate      = abs_side * prox_obs           # ≈ 1 only when active
        BIG_OBS   = 50.0
        h_obs = (gate * (side_pref * (e_c - e_c_obs_p) - (R_safe_p + R_CAR))
                 + (1.0 - gate) * BIG_OBS)
        # ---- (2) corridor ----
        w_left  = sin_t * (left_x - ref_x) - cos_t * (left_y - ref_y)
        w_right = sin_t * (right_x - ref_x) - cos_t * (right_y - ref_y)
        EPS_MM    = 1e-4
        diff_lr   = w_left - w_right
        sqrt_diff = ca.sqrt(diff_lr * diff_lr + EPS_MM)
        upper_lat = 0.5 * (w_left + w_right + sqrt_diff) - R_CAR
        lower_lat = 0.5 * (w_left + w_right - sqrt_diff) + R_CAR
        h_corridor_top = upper_lat - e_c
        h_corridor_bot = e_c - lower_lat
        # ---- (3) a_lat ----
        # Kinematic: a_lat = v² tan(δ)/L (geometric centripetal acc).
        # Dynamic: a_lat = vx · r (true lateral acceleration). Both
        # are computed in the model branch above into `a_lat_expr`.
        a_lat = a_lat_expr
        model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat)

        # ---- Compose model ----
        model_ac.f_impl_expr = xdot - f_expl
        model_ac.f_expl_expr = f_expl
        model_ac.x = x
        model_ac.xdot = xdot
        model_ac.u = u
        model_ac.z = ca.vertcat([])
        model_ac.p = p_sym
        ocp.model = model_ac
        ocp.dims.np = n_p_total

        # ---- Cost: NONLINEAR_LS ----
        # W is fixed at codegen; LIVE tuning of q_cte/q_lag/q_d_delta done by
        # multiplying inside y_expr if needed later. For now, use defaults.
        # Tuned for kinematic limits: too large q_cte/q_psi forces tighter
        # tracking than dynamics can follow at corners → car stalls or
        # bounces off the line.
        q_cte_def     = 9.0     # was 30 → 12 → 6 → 9. q_cte=6 was too weak:
                                # prediction drifted off centerline even on
                                # STRAIGHT sections (visible angled green
                                # line in RViz) because the lateral pull
                                # couldn't dominate other near-zero cost
                                # terms. 9 is a balance — keeps centerline
                                # tracking on straights but with attenuation
                                # still allows free detour near obstacles.
        q_lag_def     = 45.0    # was 200 → 60 → 30 → 45. Same reasoning —
                                # along-track tracking too weak at 30.
        q_psi_def     = 10.0    # heading enforcement (was 1 → 5 → 10).
                                # Higher q_psi forces ψ to track the
                                # centerline tangent so δ is uniquely
                                # determined by path geometry rather than
                                # solver-arbitrary picks among flat-cost
                                # near-optima.
        q_v_def       = 8.0     # softer ref_v tracking → natural slowdown
        q_dd_def      = 5.0     # steer reg on |δ|² (was 2 → 15 → 50 → 5).
                                # × 20 stages now totals ~48 vs q_lag 60,
                                # so solver feels significant cost for
                                # large |δ|. Without this the solver was
                                # cycle-to-cycle picking +0.4 / −0.1
                                # alternating (65% sign-flip rate observed)
                                # because both gave similar predicted cost
                                # — flat surface. Higher q_d_delta forces
                                # the solver to commit to moderate δ
                                # spread across the horizon, killing the
                                # prediction-line trembling.
        q_p_def       = 4.0     # progress pull
        q_side_def    = 3.0     # very soft hint (was 12 → 3, mirrors
                                # IPOPT MPCC.py W_SIDE=3). With the new
                                # attenuation that fades q_cte/q_lag near
                                # the obstacle, side cost no longer needs
                                # to fight q_cte for the detour — it's
                                # just a gentle nudge. q_side=12 was too
                                # much: it produced cost spikes when the
                                # car was forced to choose between the
                                # detour line and the half-plane edge.
        q_d_rate_def  = 80.0    # steer-rate cost on (δ − δ_prev)². Bumped
                                # 30 → 50 → 80. Strong rate cost minimises
                                # corner→straight transition wobble.
                                # Penalises stage-to-stage δ change so
                                # the solver can't pick zigzag patterns.
                                # 30 makes a 0.14-rad jump cost 30·0.0196
                                # = 0.59 per stage; over 19 transitions
                                # ≈ 11. Constant δ has zero rate cost,
                                # so optimizer prefers smooth profile.
        ny   = 8   # 8 residuals — must match y_expr length (added Δδ)
        ny_e = 3
        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # Stage 0 mirrors stage cost; required in newer acados-template
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.cost.W = np.diag([q_cte_def, q_lag_def, q_psi_def,
                              q_v_def, q_dd_def, q_p_def, q_side_def,
                              q_d_rate_def])
        ocp.cost.W_0 = ocp.cost.W
        ocp.cost.W_e = np.diag([q_cte_def * 5.0, q_lag_def * 5.0, q_psi_def * 4.0])
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_0 = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.model.cost_y_expr   = y_expr
        ocp.model.cost_y_expr_0 = y_expr
        ocp.model.cost_y_expr_e = y_expr_e

        # ---- Dimensions / horizon ----
        ocp.solver_options.N_horizon = self.N
        Tf = self.N * self.dT
        ocp.solver_options.tf = Tf

        # ---- Initial state placeholder ----
        ocp.constraints.x0 = np.zeros(nx)

        # ---- Input bounds ----
        # Kinematic: u = [v, δ, p_v] — bounds on v, δ, p_v.
        # Dynamic:   u = [a_x, δ, p_v] — bounds on accel/decel from sim
        #            (max_accel=7.51, max_decel=8.26). v_max enforced as
        #            a STATE bound on vx instead.
        ocp.constraints.idxbu = np.array([0, 1, 2])
        if self.use_dynamic:
            a_min_dyn = -8.26
            a_max_dyn =  7.51
            ocp.constraints.lbu = np.array([a_min_dyn, self.theta_min, 0.0])
            ocp.constraints.ubu = np.array([a_max_dyn, self.theta_max, self.p_max])
            # State bounds on vx, vy, r — WIDE so they rarely bind.
            # Tight bounds caused MINSTEP cascades when transient slip
            # values briefly exceeded the limit. Pacejka physics keeps
            # states naturally bounded; these are just safety net.
            ocp.constraints.idxbx = np.array([3, 4, 5])
            ocp.constraints.lbx   = np.array([-1.0, -10.0, -20.0])
            ocp.constraints.ubx   = np.array([self.v_max + 2.0, 10.0, 20.0])
        else:
            ocp.constraints.lbu = np.array([0.0, self.theta_min, 0.0])
            ocp.constraints.ubu = np.array([self.v_max, self.theta_max, self.p_max])

        # ---- h bounds (only h_obs + a_lat now) ----
        # uh[0] HUGE: obstacle absence is encoded by setting obs_x/y to a
        # sentinel ~1e6 m, which makes h_obs = d² ≈ 1e12. With BIG=1e3
        # that's a 10⁹-magnitude constraint violation every cycle, slack
        # cost explodes, and HPIPM produces NaN step directions →
        # ACADOS_MINSTEP. With uh=1e15 the trivial case is well within
        # bounds.
        # Reverted to 8 (briefly tried 12 → caused hairpin trembling:
        # cost surface too flat at high κ, IPM picked different
        # near-optima cycle-to-cycle and prediction wobbled). Trade-off
        # accepted: obstacle-on-curve will struggle but hairpin steady.
        a_lat_max = 8.0
        # h order: [h_obs, h_corridor_top, h_corridor_bot, a_lat]
        ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max])
        ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max])
        # Slack on all four — corridor and obstacle and a_lat can be
        # transiently violated. Slack absorbs without triggering cascade.
        ocp.constraints.idxsh = np.array([0, 1, 2, 3])
        ns = 4
        ocp.constraints.lsh = np.zeros(ns)
        ocp.constraints.ush = np.zeros(ns)
        # Per-constraint slack tuning — second reduction. Quadratic Zl was
        # the dominant source of cost spikes (slack=2m × Zl=500 = 2000
        # per stage); reducing 500→80 capped spikes at ~6k. Reducing
        # further to 30 caps them at ~2-3k while keeping a bounded but
        # still-meaningful push back to the constraint via the linear zl
        # term. Linear zl is bumped slightly so the gradient at small
        # violations stays informative (push = zl + 2·Zl·s; with smaller
        # Zl, the linear term is what the optimizer feels for s < 0.5m).
        #   idx 0 (h_obs):       zl=40,  Zl=30   (was 30/80)
        #   idx 1,2 (corridor):  zl=20,  Zl=15   (was 15/30)
        #   idx 3 (a_lat):       zl=50,  Zl=15   (was 50/30)
        ocp.cost.zl = np.array([40.0, 20.0, 20.0, 50.0])
        ocp.cost.zu = np.array([40.0, 20.0, 20.0, 50.0])
        ocp.cost.Zl = np.array([30.0, 15.0, 15.0, 15.0])
        ocp.cost.Zu = np.array([30.0, 15.0, 15.0, 15.0])

        # ---- Initial parameter values (overridden every cycle) ----
        ocp.parameter_values = np.zeros(n_p_total)
        ocp.parameter_values[4]  = self.D_detour_live       # D_DETOUR
        ocp.parameter_values[5]  = self.R_car_live          # R_CAR
        ocp.parameter_values[6]  = self.q_cte_scale_live    # q_cte scale
        ocp.parameter_values[7]  = self.R_safe_live         # R_safe
        ocp.parameter_values[8]  = self.q_lag_scale_live    # q_lag scale
        ocp.parameter_values[10] = self.a_lat_safe_live     # A_LAT_SAFE
        ocp.parameter_values[11] = self.D_apex_live         # D_apex
        ocp.parameter_values[12] = self.q_psi_scale_live    # q_psi scale
        ocp.parameter_values[13] = self.q_v_scale_live      # q_v scale
        ocp.parameter_values[14] = self.q_dd_scale_live     # q_dd scale (steer reg)
        ocp.parameter_values[15] = self.q_p_scale_live      # q_p scale (progress)
        ocp.parameter_values[16] = self.q_drate_scale_live  # q_d_rate scale

        # ---- Solver options ----
        # NONLINEAR_LS form makes the Gauss-Newton Hessian = J^T·W·J
        # automatically PSD. SQP_RTI works again. PROJECT regularize is
        # an extra safety net (acados forum recommendation).
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # SQP_RTI single-iter for BOTH kinematic and dynamic.
        # Multi-iter SQP was diverging at startup (~15 iters too much
        # freedom when warm-start far from optimum → IPM step grows
        # unbounded). RTI = 1 SQP iter per cycle, naturally stable.
        # Pacejka's higher-order derivatives are tamed by strong LM (3.0)
        # below — same approach as kinematic, just more regularization.
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # PROJECT (CONVEXIFY caused hairpin trembling — Hessian rebuilt
        # slightly differently each cycle, IPM step direction wobbled
        # across multiple near-equal optima in tight corners).
        # PROJECT just clips negative eigenvalues to small positive,
        # leaving block structure stable cycle-to-cycle → deterministic
        # step direction.
        ocp.solver_options.regularize_method = 'PROJECT'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0
        # Back to 100 — 200 was an exploration knob that combined with
        # CONVEXIFY/loose a_lat caused hairpin instability (more iters
        # = more chance to wander between near-equal optima). 100 is
        # sufficient when cost surface is well-conditioned.
        ocp.solver_options.qp_solver_iter_max = 100
        # LM for dynamic linear tire = 1.0. Balanced: strong enough for
        # IPM stability with slip derivatives, loose enough that cost
        # weights can actually steer the solution (not over-damped).
        ocp.solver_options.levenberg_marquardt = 1.0 if self.use_dynamic else 0.2

        # Codegen + build
        ocp.code_export_directory = '/tmp/acados_codegen_evompcc'
        json_path = '/tmp/acados_ocp_evompcc.json'
        self._log.info("[MPC-acados] generating solver (~30 s first time)...")
        self.solver = AcadosOcpSolver(ocp, json_file=json_path)
        self._log.info("[MPC-acados] solver ready")

        # Stash dim info for solve()
        self._n_p_const = n_p_const
        self._n_p_total = n_p_total

        # Storage
        self.X0 = np.zeros((self.N + 1, self.n_states))
        self.u0 = np.zeros((self.N, self.n_controls))

    # ------------------------------------------------------------------
    # Bound construction helpers (mirror IPOPT solve())
    # ------------------------------------------------------------------
    def _obstacle_frenet(self, ox, oy):
        """Obstacle's Frenet coords (s_obs, e_c_obs) at its nearest centerline
        point. Returns (0.0, 0.0) for sentinel. Computed ONCE when an
        obstacle is committed (see solve()'s commit-once logic), not every
        cycle, so the brute-force search is one-shot per obstacle pass.
        """
        if ox > 1e5 or oy > 1e5 or self.kappa_grid is None:
            return 0.0, 0.0
        n = len(self.kappa_grid)
        best_d2 = float("inf"); best_s = 0.0
        for i in range(n):
            s = i * self.kappa_ds
            cxi = float(self.center_lut_x(s))
            cyi = float(self.center_lut_y(s))
            d2 = (cxi - ox) ** 2 + (cyi - oy) ** 2
            if d2 < best_d2:
                best_d2 = d2; best_s = s
        rxc = float(self.center_lut_x(best_s))
        ryc = float(self.center_lut_y(best_s))
        dxt = float(self.center_lut_dx(best_s))
        dyt = float(self.center_lut_dy(best_s))
        nrm = math.sqrt(dxt * dxt + dyt * dyt) + 1e-9
        sin_t_obs = dyt / nrm
        cos_t_obs = dxt / nrm
        e_c_obs = sin_t_obs * (ox - rxc) - cos_t_obs * (oy - ryc)
        return best_s, e_c_obs

    def get_path_constraints_points(self, prev_soln):
        """Sample left/right boundary at each predicted stage's s."""
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        for k in range(1, self.N + 1):
            sk = float(prev_soln[k, 3]) % self.path_length
            right_points[k - 1, :] = np.array([self.right_lut_x(sk),
                                               self.right_lut_y(sk)],
                                              dtype=object).squeeze()
            left_points[k - 1, :] = np.array([self.left_lut_x(sk),
                                              self.left_lut_y(sk)],
                                             dtype=object).squeeze()
        return right_points, left_points

    def _kappa_at(self, s):
        """O(1) curvature lookup at arclength s (modulo path_length).
        Kept for diagnostics — not used to tighten constraints anymore."""
        if self.kappa_grid is None:
            return 0.0
        sw = float(s) % self.path_length
        idx = int(sw / self.kappa_ds)
        if idx < 0:
            idx = 0
        elif idx >= len(self.kappa_grid):
            idx = len(self.kappa_grid) - 1
        return abs(float(self.kappa_grid[idx]))

    def select_front_obstacle(self, curr_x, curr_y, curr_yaw,
                              obstacles, D_max=20.0, D_min_trig=10.0,
                              ang_max=2.0 * math.pi / 3.0):
        """Pick the closest obstacle in front of the car, or sentinel
        [1e6, 1e6, 1e6] when none qualify.

        D_max bumped 12→20 so curving tracks where obstacle is on next
        segment (Euclidean farther than path-distance) still trigger.
        ang_max bumped π/2→2π/3 (90°→120°) so obstacles slightly to the
        side on tight curves aren't filtered out before the car has
        rotated to face them.
        """
        if obstacles is None or len(obstacles) == 0:
            return [1e6, 1e6, 1e6]
        best = [1e6, 1e6, 1e6]
        cx, cy = curr_x + math.cos(curr_yaw) * 0.05, curr_y + math.sin(curr_yaw) * 0.05
        for ob in obstacles:
            if hasattr(ob, '__len__') and len(ob) >= 2:
                ox, oy = float(ob[0]), float(ob[1])
            else:
                continue
            dx, dy = ox - cx, oy - cy
            d = math.hypot(dx, dy)
            if d > D_max:
                continue
            # in front (with looser cone)?
            ang = math.atan2(dy, dx) - curr_yaw
            while ang > math.pi: ang -= 2 * math.pi
            while ang < -math.pi: ang += 2 * math.pi
            if abs(ang) > ang_max:
                continue
            if d < best[0]:
                best = [d, ox, oy]
        return best

    def decide_side_pref(self, obstacle_pos, left_points, right_points, margin=0.1):
        """Same rules as the IPOPT version (user-defined)."""
        x_o, y_o = obstacle_pos[0], obstacle_pos[1]
        if x_o > 1e5 and y_o > 1e5:
            return 0
        dist_L = np.sqrt((left_points[:, 0] - x_o) ** 2 + (left_points[:, 1] - y_o) ** 2)
        dist_R = np.sqrt((right_points[:, 0] - x_o) ** 2 + (right_points[:, 1] - y_o) ** 2)
        top2_L = np.sort(dist_L)[:2] if len(dist_L) >= 2 else dist_L
        top2_R = np.sort(dist_R)[:2] if len(dist_R) >= 2 else dist_R
        mean_L = float(np.mean(top2_L))
        mean_R = float(np.mean(top2_R))
        W_CAR_SAFE = 0.21
        left_blocked  = mean_L < W_CAR_SAFE
        right_blocked = mean_R < W_CAR_SAFE
        if left_blocked and not right_blocked:
            return -1
        if right_blocked and not left_blocked:
            return +1
        if left_blocked and right_blocked:
            return -1
        diff = mean_L - mean_R
        if abs(diff) <= margin:
            return -1   # tie → right
        return +1 if diff > 0 else -1

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, initial_state, obstacles):
        """initial_state: numpy length 4 from ROS node = [x, y, psi, s].
        We extend internally to 5: [x, y, psi, s, delta_prev] where
        delta_prev = the steer command we sent last cycle (initialised to 0).
        """
        L = float(self.path_length)

        # Augment state with the previous applied δ for steer-rate cost.
        # Kinematic: node sends 4 → expand to 5 (append δ_prev).
        # Dynamic:   node sends 7 → expand to 8 (append δ_prev).
        # Either way, target = self.n_states; append δ_prev to fill.
        if len(initial_state) == self.n_states - 1:
            last_delta = float(getattr(self, '_last_delta_applied', 0.0))
            initial_state = np.append(np.asarray(initial_state, dtype=float),
                                       last_delta)

        # State-vector index helpers — shared between kinematic (5) and
        # dynamic (8) layouts. x_pos/y_pos/psi at slots 0/1/2 in both.
        # s and δ_prev shift in dynamic since vx/vy/r occupy 3/4/5.
        if self.use_dynamic:
            IDX_S = 6
            IDX_DELTA_PREV = 7
        else:
            IDX_S = 3
            IDX_DELTA_PREV = 4

        # ---- Unwrap sensor s into solver-internal monotonic coordinate ----
        # Node passes current_s ∈ [0, L). We track lap_count internally and
        # produce solver_s = sensor_s + lap_count·L which grows unboundedly
        # across laps. Dynamics ṡ = p_v then stays smooth — no wrap
        # discontinuity ever reaches the QP, so no lap-rollover MINSTEP and
        # no cost spike at lap boundaries.
        sensor_s = float(initial_state[IDX_S])
        if self._last_sensor_s is not None:
            d = sensor_s - self._last_sensor_s
            if d < -L / 2.0:
                self.lap_count += 1   # forward wrap (typical: 86 → 0)
            elif d > L / 2.0:
                self.lap_count -= 1   # reverse (rare: backward sensor jump)
        self._last_sensor_s = sensor_s
        initial_state[IDX_S] = sensor_s + self.lap_count * L
        s0 = float(initial_state[IDX_S])

        # ---- ψ unwrap via persistent offset (same trick as lap_count for s) ----
        # Sensor yaw ∈ [−π, π]; we accumulate ±2π each time the sensor
        # wraps and pass `sensor_yaw + yaw_offset` to the solver. The
        # solver's internal ψ then grows monotonically — no wrap event
        # ever reaches the QP, no multi-iter SQP fixup needed, no
        # repeated wrap-detection at every cycle in the wrap zone (which
        # was the underlying cause of the 974-cost spike at s≈56 every
        # lap). Cost (yaw_err via sin/cos identities) and dynamics (cos/
        # sin of ψ) are 2π-periodic, so unbounded ψ is harmless.
        sensor_yaw = float(initial_state[2])
        if self._last_sensor_yaw is not None:
            d_yaw = sensor_yaw - self._last_sensor_yaw
            if d_yaw < -math.pi:
                self._yaw_offset += 2.0 * math.pi
            elif d_yaw > math.pi:
                self._yaw_offset -= 2.0 * math.pi
        self._last_sensor_yaw = sensor_yaw
        initial_state[2] = sensor_yaw + self._yaw_offset
        psi_unwrapped = False  # legacy flag — nothing to do, kept for the
                               # multi-iter trigger below (always False now)

        # First-cycle warm start — dynamics-feasible forward rollout from
        # the actual car state. Previously we placed X0[1..N] on the
        # centerline regardless of where the car was, so the X0[0]→X0[1]
        # jump violated dynamics by up to ~0.5 m laterally. acados then
        # could not make any progress and reported ACADOS_MINSTEP at QP
        # iter 1. Rolling forward with the same controls used in u0 keeps
        # X0[k+1] = X0[k] + dt·f(X0[k], u_k), satisfying dynamics by
        # construction.
        if not self.WARM_START:
            # Cold-start path fires after stuck/spike fallback or obstacle
            # commit/release. solver.reset() here is critical for the
            # stuck case: stuck override sets u_seq=zeros, the solver's
            # dual variables then carry a bias "u=0 minimizes the QP",
            # and the next cycle's 1-iter SQP_RTI can't escape — the car
            # stays at v=0 forever even though cold-rollout pushes a
            # reasonable u0=seed_v warm-start. Reset zeros the duals so
            # the rollout actually drives the solution. Note: ψ-wrap
            # doesn't go through this path (it sets X0 directly without
            # WARM_START=False), so the Plan-B "no reset on wrap" stays.
            try:
                self.solver.reset()
            except Exception:
                pass
            try:
                seed_v = max(float(self.ref_v(s0 % L)) * 0.5, 1.0)
            except Exception:
                seed_v = max(self.v_max * 0.4, 1.0)
            self.u0 = np.zeros((self.N, self.n_controls))
            if self.use_dynamic:
                # Dynamic warm-start: u = [a_x, δ, p_v]. Hold a_x = 0
                # (no acceleration during warm-start) so vx stays at the
                # car's current velocity. Skip slip dynamics — vy, r
                # propagated from initial state via simple kinematic
                # surrogate so X0 stays well-conditioned.
                self.u0[:, 0] = 0.0      # a_x
                # Dynamic warm-start: keep it MINIMAL. a_x=0 means vx
                # stays at initial. No ramp = no input bound risk.
                # Solver's first iter then finds the correct a_x. p_v
                # set to seed_v so s grows during warm-start.
                self.u0[:, 0] = 0.0      # a_x (no acceleration in seed)
                self.u0[:, 1] = 0.0      # delta
                self.u0[:, 2] = seed_v   # p_v (input, can be set free)
                self.X0[0, :] = initial_state
                for k in range(self.N):
                    xk = self.X0[k, :]
                    uk = self.u0[k, :]
                    delta_, p_ = uk[1], uk[2]
                    psi_ = xk[2]
                    vx_  = xk[3]            # vx stays constant in seed
                    delta_prev_ = xk[7]
                    dx_dt   = vx_ * np.cos(psi_)
                    dy_dt   = vx_ * np.sin(psi_)
                    dpsi_dt = (vx_ / self.L) * np.tan(delta_)
                    ds_dt   = p_
                    ddprev  = (delta_ - delta_prev_) / self.dT
                    self.X0[k + 1, :] = xk + self.dT * np.array(
                        [dx_dt, dy_dt, dpsi_dt, 0.0, 0.0, 0.0,
                         ds_dt, ddprev])
            else:
                self.u0[:, 0] = seed_v   # v
                self.u0[:, 1] = 0.0      # delta
                self.u0[:, 2] = seed_v   # p
                self.X0[0, :] = initial_state
                for k in range(self.N):
                    xk = self.X0[k, :]
                    uk = self.u0[k, :]
                    v_, delta_, p_ = uk[0], uk[1], uk[2]
                    psi_ = xk[2]
                    delta_prev_ = xk[4]
                    # Euler 1-step (matches integrator's coarse warm-start need)
                    dx_dt   = v_ * np.cos(psi_)
                    dy_dt   = v_ * np.sin(psi_)
                    dpsi_dt = (v_ / self.L) * np.tan(delta_)
                    ds_dt   = p_
                    ddprev  = (delta_ - delta_prev_) / self.dT
                    self.X0[k + 1, :] = xk + self.dT * np.array(
                        [dx_dt, dy_dt, dpsi_dt, ds_dt, ddprev])
            for k in range(self.N + 1):
                self.solver.set(k, "x", self.X0[k, :])
            for k in range(self.N):
                self.solver.set(k, "u", self.u0[k, :])
            self.WARM_START = True
            # Flag for multi-iter SQP — solver.reset() above zeroed dual
            # variables; without this flag the main solve below runs only
            # 1 SQP_RTI iteration which can't rebuild the dual active set
            # → returns near-zero (or otherwise corrupt) controls →
            # stuck-at-zero loop after every stuck recovery / fallback.
            self._just_cold_started = True

        # (No s-wrap fixup needed — solver-internal s is monotonic.)

        # ---- Per-cycle parameters ----
        sel = self.select_front_obstacle(initial_state[0], initial_state[1],
                                         initial_state[2], obstacles)
        self.dbg_n_obs_input = len(obstacles) if obstacles is not None else 0
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # ---- Commit-once obstacle / side decision ----
        # Once we engage an obstacle, FREEZE the chosen avoidance side and
        # the obstacle Frenet coords until the car has driven past it
        # longitudinally. Without this, decide_side_pref / select_front_obs
        # may flip mid-pass (e.g., as the car turns into the obstacle the
        # corridor sample distances change), and the half-plane direction
        # would suddenly reverse → prediction "bursts" the other way.
        # Stored: (ox, oy, s_obs, side_pref, e_c_obs).
        right_pts, left_pts = self.get_path_constraints_points(self.X0)
        sel_is_real = sel[1] < 1e5
        if not hasattr(self, '_committed_obs'):
            self._committed_obs = None

        # Release current commitment if (a) car drove past it in s, (b) it
        # is no longer in the obstacle list, or (c) we drifted very far.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            s_car = float(initial_state[IDX_S]) % L
            delta_s = (s_car - cs_obs) % L
            car_to_obs = math.hypot(initial_state[0] - cox,
                                     initial_state[1] - coy)
            obs_still_present = False
            if obstacles is not None and len(obstacles) > 0:
                obs_still_present = any(
                    (op[0] - cox) ** 2 + (op[1] - coy) ** 2 < 0.09  # 0.3 m
                    for op in obstacles)
            if 1.5 < delta_s < 0.5 * L:
                self._log.info("[MPC] passed committed obs (Δs=%.2f m) — release", delta_s)
                self._committed_obs = None
                self.WARM_START = False
            elif not obs_still_present:
                self._log.info("[MPC] committed obs removed — release")
                self._committed_obs = None
                self.WARM_START = False
            elif car_to_obs > 12.0:
                # Drifted way past; safety release.
                self._committed_obs = None

        # Track NEW commit this cycle so we can:
        #   (a) push a stronger detour prior into X0 (option 2)
        #   (b) run extra SQP iters before the main solve (option 3)
        just_committed = False

        # Side-decision cache (hysteresis). Maps obstacle position →
        # last committed side. Re-using the same side for the same
        # obstacle prevents the "sometimes goes up, sometimes down" flip
        # symptom — once a side is chosen for an obstacle, stick with it
        # for all subsequent passes.
        if not hasattr(self, '_side_history'):
            self._side_history = {}

        # Commit on first engagement. Trigger distance is rqt-tunable
        # via self.commit_dist_live (default 10 m). Larger = engage
        # detour earlier (smoother but more conservative); smaller =
        # late commit (snappy but riskier).
        if (self._committed_obs is None and sel_is_real
                and float(sel[0]) < self.commit_dist_live):
            s_obs_new, e_c_obs_new = self._obstacle_frenet(sel[1], sel[2])
            s_car_now = float(initial_state[IDX_S]) % L
            delta_s_now = (s_car_now - s_obs_new) % L
            # In-front zone (see prior comment block).
            obs_is_ahead = (delta_s_now < 1.5) or (delta_s_now > 0.5 * L)
            if obs_is_ahead:
                # Cache key: obstacle position to 0.1 m precision so
                # near-identical obstacles map to same side.
                key = (round(float(sel[1]), 1), round(float(sel[2]), 1))
                if key in self._side_history:
                    sp_new = self._side_history[key]
                    self._log.info("[MPC] reusing cached side=%+d for obs at %s", sp_new, key)
                else:
                    sp_new = self.decide_side_pref(
                        [sel[1], sel[2]], left_pts, right_pts)
                    if sp_new != 0:
                        self._side_history[key] = sp_new
                if sp_new != 0:
                    self._committed_obs = (
                        float(sel[1]), float(sel[2]),
                        s_obs_new, int(sp_new), e_c_obs_new)
                    self._log.info(
                        "[MPC] committed obs=(%.2f,%.2f) s_obs=%.2f e_c_obs=%+.2f side=%+d",
                        sel[1], sel[2], s_obs_new, e_c_obs_new, sp_new)
                    self.WARM_START = False  # rebuild with frozen side
                    just_committed = True

        # Use committed values; if no commitment, fall back to sentinel.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            sel = [math.hypot(initial_state[0] - cox,
                              initial_state[1] - coy), cox, coy]
            side_pref = cside
            e_c_obs_val = ce_c
        else:
            sel = [1e6, 1e6, 1e6]
            side_pref = 0
            e_c_obs_val = 0.0
        self.dbg_side_pref = float(side_pref)
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # Avoidance prior on warm-start X0 — Option 2: at the just-committed
        # cycle, push HARDER and FARTHER. Without this strong push, X0 is
        # the previous (no-obstacle) prediction; the half-plane constraint
        # is suddenly active and slack absorbs a 1-2 m violation → cost
        # spike. With the strong commit-time push, X0 already curves around
        # the obstacle, so the half-plane is near-satisfied from cycle one.
        # After commit (subsequent cycles), the regular gentler push
        # maintains the detour shape as the car approaches.
        if side_pref != 0:
            if just_committed:
                D_PRIOR     = 0.40   # bigger lateral kick at commit
                SIGMA_SQ    = 4.0    # σ=2.0 — wider Gaussian, more stages
                TRIGGER_D   = 8.0    # always push at commit
            else:
                D_PRIOR     = 0.30
                SIGMA_SQ    = 1.0
                TRIGGER_D   = 4.0
            if float(sel[0]) < TRIGGER_D:
                ox, oy = sel[1], sel[2]
                for k in range(1, self.N + 1):
                    xk, yk, psik = self.X0[k, 0], self.X0[k, 1], self.X0[k, 2]
                    d2k = (xk - ox) ** 2 + (yk - oy) ** 2
                    w = math.exp(-d2k / (2.0 * SIGMA_SQ))
                    nx_, ny_ = -math.sin(psik), math.cos(psik)
                    self.X0[k, 0] += side_pref * D_PRIOR * w * nx_
                    self.X0[k, 1] += side_pref * D_PRIOR * w * ny_
                for k in range(self.N + 1):
                    self.solver.set(k, "x", self.X0[k, :])

        # Per-stage parameter array
        # X0[k, IDX_S] is unbounded (solver-internal monotonic s), so always
        # wrap with % L for spline lookups (corridor / track boundary).
        for k in range(self.N + 1):
            sk = float(self.X0[k, IDX_S]) % L
            try:
                lx = float(self.left_lut_x(sk));  ly = float(self.left_lut_y(sk))
                rx = float(self.right_lut_x(sk)); ry = float(self.right_lut_y(sk))
            except Exception:
                lx = ly = rx = ry = 0.0
            p_arr = np.array([
                float(sel[0]),                  # 0: dmin (debug)
                float(sel[1]),                  # 1: obs_x
                float(sel[2]),                  # 2: obs_y
                float(side_pref),               # 3: side_pref
                float(self.D_detour_live),      # 4: D_DETOUR (rqt)
                float(self.R_car_live),         # 5: R_CAR (rqt)
                float(self.q_cte_scale_live),   # 6: q_cte scale (rqt)
                float(self.R_safe_live),        # 7: R_safe (rqt)
                float(self.q_lag_scale_live),   # 8: q_lag scale (rqt)
                e_c_obs_val,                    # 9: e_c_obs (Frenet)
                float(self.a_lat_safe_live),    # 10: a_lat_safe (rqt)
                float(self.D_apex_live),        # 11: D_apex (rqt)
                float(self.q_psi_scale_live),   # 12: q_psi scale (rqt)
                float(self.q_v_scale_live),     # 13: q_v scale (rqt)
                float(self.q_dd_scale_live),    # 14: q_dd scale (rqt)
                float(self.q_p_scale_live),     # 15: q_p scale (rqt)
                float(self.q_drate_scale_live), # 16: q_d_rate scale (rqt)
                lx, ly, rx, ry,                 # 17..20 (corridor)
            ], dtype=float)
            self.solver.set(k, "p", p_arr)

        # ---- Stage 0 init state via tightened bounds ----
        self.solver.set(0, "lbx", initial_state)
        self.solver.set(0, "ubx", initial_state)

        # Option 3: at "transient" cycles (new commit, ψ-wrap, just-cold-
        # started after a stuck/spike fallback), the cost surface and/or
        # dual active set change abruptly. SQP_RTI's single iter cannot
        # converge duals from cold (or near-cold) state on those cycles
        # → corrupt control output (near-zero, saturated, or oscillating).
        # Run 2 extra solve() calls before the main solve so SQP has
        # converged. Pure cycles (~99%) skip and do single-iter as before.
        # Cost: ~6 ms extra on transient cycles, well within 25 ms budget.
        # Critical for breaking the stuck-at-zero loop: stuck override
        # produces u=0; next cycle's cold-start reset() zeros duals; if
        # we don't multi-iter here, 1-iter from cold dual returns ~0
        # again → infinite stuck.
        cold_started = getattr(self, '_just_cold_started', False)
        # Multi-iter pre-solve ONLY at transient events (commit, ψ-wrap,
        # cold-start). Per-cycle multi-iter caused oscillating predictions
        # because the second iter found a slightly different optimum
        # than the first → trajectory zigzag visible in RViz.
        if just_committed or psi_unwrapped or cold_started:
            for _ in range(2):
                try:
                    self.solver.solve()
                except Exception:
                    break
        self._just_cold_started = False

        # ---- Solve ----
        try:
            status = self.solver.solve()
            self.dbg_solver_status = self._status_to_string(status)
            traj = np.array([self.solver.get(k, "x") for k in range(self.N + 1)])
            u_seq = np.array([self.solver.get(k, "u") for k in range(self.N)])
            # status: 0=OK, 1=NaN/Failure, 2=MaxIter, 3=MinStep, 4=QP_Failure.
            # 3/4 at the hairpin produces a corrupt warm-start that locks
            # subsequent QPs into a 36-iter limit cycle (observed s≈48m
            # cascade). Forcing WARM_START=False rebuilds X0/u0 from the
            # current car state via Euler rollout next cycle.
            bad_status = status in (1, 3, 4)
            has_nan = np.isnan(traj).any() or np.isnan(u_seq).any()
            if bad_status or has_nan:
                self._log.warn_throttle(1.0,
                    "[MPC-acados] status=%s nan=%s — reset warm-start",
                    self.dbg_solver_status, has_nan)
                self.WARM_START = False
                # Safe fallback control: gentle slow-down, hold heading.
                # Avoids feeding the ROS node a corrupt v_cmd while next
                # cycle re-seeds.
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.3, 0.5)
                except Exception:
                    seed_v = 1.0
                traj = np.tile(initial_state, (self.N + 1, 1))
                u_seq = np.zeros((self.N, self.n_controls))
                if self.use_dynamic:
                    # u[0] = a_x in dynamic — set to small positive accel
                    # to keep the car coasting forward. p_v = seed_v.
                    u_seq[:, 0] = 0.5
                    # Also seed predicted vx so the output speed is sensible.
                    traj[:, 3] = seed_v
                else:
                    u_seq[:, 0] = seed_v   # v (kinematic input)
                u_seq[:, 2] = seed_v
        except Exception as e:
            self._log.warn_throttle(2.0, "[MPC-acados] solver exception %s — reset", str(e))
            self.WARM_START = False
            traj = np.tile(initial_state, (self.N + 1, 1))
            u_seq = np.zeros((self.N, self.n_controls))
            self.dbg_solver_status = "exception"

        try:
            opti_value = float(self.solver.get_cost())
        except Exception:
            opti_value = float('nan')

        # Cost-spike fallback — when the QP "succeeded" but cost is far
        # above the steady-state level, the solution is untrustworthy
        # (typically saturated v_max + steer=±0.4 from a corrupt warm-
        # start at obstacle commit / ψ-wrap). Output gentle ref_v·0.3
        # straight forward + warm-start rebuild.
        cost_spike = ((not np.isnan(opti_value))
                      and opti_value > self.cost_spike_thr_live)

        # Stuck detection (option D) — cost-spike fallback alone misses
        # the case where the QP returns saturated controls with cost
        # below threshold (observed: cost ≈ 270, vcmd=6, steer=+0.4 at
        # s=60 lap 4, car wedged into wall, fallback never engages).
        # Estimate v_actual from sensed position diff; if it stays below
        # 0.1 m/s while we keep commanding > 2 m/s for several cycles,
        # the car is physically stuck even though the optimizer thinks
        # everything is fine. Override with v=0/steer=0 to release the
        # wall contact, then rebuild warm-start next cycle.
        now_t = monotonic_now()
        v_est = 0.0
        if hasattr(self, '_pos_for_v') and self._pos_for_v is not None:
            px, py, pt = self._pos_for_v
            dtm = max(now_t - pt, 1e-3)
            v_est = math.hypot(initial_state[0] - px,
                                initial_state[1] - py) / dtm
        self._pos_for_v = (float(initial_state[0]),
                            float(initial_state[1]), now_t)
        v_cmd_prev = getattr(self, '_v_cmd_for_stuck', 0.0)
        if v_est < 0.1 and v_cmd_prev > 2.0:
            self._stuck_count = getattr(self, '_stuck_count', 0) + 1
        else:
            self._stuck_count = 0
        is_stuck = self._stuck_count > 10  # ~0.25 s @ 40 Hz

        if cost_spike or is_stuck:
            if is_stuck:
                self._log.warn_throttle(1.0,
                    "[MPC] STUCK (v_est=%.2f, last_vcmd=%.2f, n=%d) — release",
                    v_est, v_cmd_prev, self._stuck_count)
                u_seq = np.zeros((self.N, self.n_controls))
                # zero v + zero steer to break wall contact.
                # Dynamic: also force the trajectory's vx to 0 so the
                # output speed (read from traj[1, 3]) is actually 0.
                if self.use_dynamic:
                    traj[:, 3] = 0.0
            else:
                self._log.warn_throttle(1.0,
                    "[MPC] cost %.0f > %.0f — safe fallback",
                    opti_value, self.cost_spike_thr_live)
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.3, 0.5)
                except Exception:
                    seed_v = 1.0
                u_seq = np.zeros((self.N, self.n_controls))
                if self.use_dynamic:
                    u_seq[:, 0] = 0.5      # a_x (gentle accel)
                    traj[:, 3] = seed_v    # set predicted vx for output
                else:
                    u_seq[:, 0] = seed_v   # v (kinematic input)
                u_seq[:, 1] = 0.0          # delta
                u_seq[:, 2] = seed_v       # p_v
            self.WARM_START = False
        # Stash this cycle's commanded v for next-cycle stuck-check.
        # Kinematic: u[0] = v (input velocity). Dynamic: u[0] = a_x — meaningless
        # for "is the car moving" check, so fall back to the predicted vx at
        # stage 1 (the speed actually sent to the actuator).
        if self.use_dynamic:
            self._v_cmd_for_stuck = float(traj[1, 3]) if traj.shape[0] > 1 else 0.0
        else:
            self._v_cmd_for_stuck = float(u_seq[0, 0])

        # Steer-output EMA filter — corrects cycle-to-cycle steer trembling
        # observed at curves/hairpin (65% of cycles had steer sign flips:
        # +0.40 → −0.10 → +0.40 oscillating). Cost surface is ~flat over
        # multiple δ values that all achieve a similar predicted ψ change,
        # so the IPM picks different corners each cycle. Smoothing the
        # output via EMA absorbs the cycle-to-cycle jitter without
        # changing what the solver sees (warm-start is unaffected). Only
        # δ is filtered — v is left alone so braking and accel stay snappy.
        # α = 0.6 — moderate smoothing (60% new sample, 40% previous);
        # racing-friendly response while killing the +/− oscillation.
        alpha = self.alpha_steer_live
        new_steer = float(u_seq[0, 1])
        prev_filt = getattr(self, '_steer_filt', new_steer)
        filt_steer = alpha * new_steer + (1.0 - alpha) * prev_filt
        self._steer_filt = filt_steer
        u_seq[0, 1] = filt_steer

        # Track applied δ for next cycle's state-augmentation (delta_prev).
        # We use the FILTERED δ (what actually goes to the actuator) so the
        # rate-cost residual is consistent with the physical command stream.
        self._last_delta_applied = filt_steer

        self.X0 = traj
        self.u0 = u_seq

        # Visualize boundary along predicted s
        try:
            self._publish_boundary(traj)
        except Exception:
            pass

        # Return shapes that match IPOPT MPC: (first_control DM, traj, u_seq, opti)
        return ca.DM(u_seq[0, :]), traj, u_seq, opti_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _status_to_string(status):
        return {
            0: 'Solve_Succeeded',
            1: 'Failure',
            2: 'Maximum_Iterations_Exceeded',
            3: 'Minimum_Step_Size_Reached',
            4: 'QP_Failure',
        }.get(status, f'status_{status}')

    def _publish_boundary(self, traj):
        """Visualize the EFFECTIVE drivable corridor (track boundary minus
        R_CAR margin), so the user sees what region the MPC actually
        permits the car to occupy. Lowering R_CAR via rqt → dots move
        outward (closer to wall); raising → dots move inward.
        """
        if self.boundary_hook is None:
            return
        right_pts, left_pts = self.get_path_constraints_points(traj)
        # Shift each boundary point toward the centerline by R_CAR. We
        # approximate the inward normal at boundary_i as (center_i - boundary_i)
        # normalized, where center_i is the centerline point at the same s.
        L = float(self.path_length)
        margin = float(self.R_car_live)
        IDX_S_LOCAL = 6 if self.use_dynamic else 3
        shifted = []
        for k in range(traj.shape[0] - 1):
            sk = float(traj[k + 1, IDX_S_LOCAL]) % L
            try:
                cx_s = float(self.center_lut_x(sk))
                cy_s = float(self.center_lut_y(sk))
            except Exception:
                cx_s = 0.0; cy_s = 0.0
            for px, py in (right_pts[k], left_pts[k]):
                vx = cx_s - px; vy = cy_s - py
                d  = math.hypot(vx, vy) + 1e-9
                shifted.append((px + margin * vx / d, py + margin * vy / d))
        self.boundary_hook(shifted)

    def heading(self, yaw):
        """yaw → (x, y, z, w) tuple. Caller wraps into geometry_msgs/Quaternion."""
        return yaw_to_quat(yaw)

    def init_mpc_start_conditions(self):
        self.X0 = np.zeros((self.N + 1, self.n_states))
        self.u0 = np.zeros((self.N, self.n_controls))
