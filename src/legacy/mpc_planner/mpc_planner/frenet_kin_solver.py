#!/usr/bin/env python3
"""### HJ : Frenet kinematic-bicycle MPC for overtaking (3D-safe).

2026-04-21 redesign v3 — Liniger-style with C^1 curvature + solution continuity.

v3 changes vs v2c (why v2c failed):
  v2c put steering delta directly as an input and penalised only
  (Δδ)^2. That made δ piece-wise constant between knots → κ discontinuous
  at every knot AND tick-to-tick jumps were unbounded in the n-profile.
  User feedback: "직선으로 꺾이는 궤적", "번개 치듯 discrete 하게 바뀜".

  v3 moves δ into the STATE. The control is δ̇ (steering rate). This:
    (a) Makes δ a continuous function of k → κ = tan(δ)/L is C^1 → no
        kinked trajectory at knot boundaries.
    (b) Allows a "jerk-like" penalty r_dd_rate × Σ(Δδ̇)^2 → curvature
        rate is smooth.
    (c) Lets us warm-start δ from the previous solve's δ[1] (hard
        constraint on δ[0]) so the physical steer angle is carried
        across ticks → tick-to-tick action smoothness.

  Additional v3 cost terms:
    + w_cont × Σ (n_k - n_prev_shifted_k)^2   (solution continuity)
    + q_n_term × n_N^2 + q_v_term × (v_N - v_ref_N)^2   (terminal)
    Stage ramp on contour/lag REMOVED (it was making cost k-stepwise
    which drove the "벽에 붙다가 끝만 복귀" shape).

  Wall cushion weight raised (150 → 2500) so it wins against obstacle
  bubble without bound glue — obstacle is strong (180) but wall is
  stronger closer to the edge (quadratic 2500×(0.30)² = 225 per step).

State   x = [n, μ, v, δ]      (frenet lateral, heading vs tangent, speed, steer)
Control u = [a, δ̇]            (accel, steer rate)
Dynamics (discrete, dT = 0.05s, explicit-Euler of the frenet kinematic bicycle):
    n_{k+1}  = n_k  + v_k * sin(μ_k)                           * dT
    μ_{k+1}  = μ_k  + (v_k / L * tan(δ_k)
                       - κ_ref_k * v_k * cos(μ_k))             * dT
    v_{k+1}  = v_k  + a_k                                      * dT
    δ_{k+1}  = δ_k  + δ̇_k                                      * dT

Cost (all C^1+):
  stationary running:
    q_n       * Σ n_k^2                                   (contour, no ramp)
    r_reg     * Σ δ_k^2                                   (steer zero-bias)
    r_dd      * Σ δ̇_k^2                                   (steer-rate, κ̇ analogue)
    r_dd_rate * Σ (δ̇_{k+1} - δ̇_k)^2                       (steer-rate-rate, jerk-like)
    r_a       * Σ (a_{k+1} - a_k)^2                       (accel smoothness)
    -gamma    * Σ v_k*cos(μ_k)*dT                         (progress pull)
    w_obs     * Σ_o Σ_k prox_sk * exp(-(dn/σ_n)^2)        (obstacle bubble)
    w_bias    * Σ_o Σ_k prox_sk * hinge^2                 (side bias, one-sided)
    w_wall_buf* Σ_k (hinge_up^2 + hinge_dn^2)             (soft wall cushion)
    w_cont    * Σ_k (n_k - n_prev_k)^2 * cont_active      (tick-to-tick continuity)
    w_slack   * Σ_k slk_k^2                               (corridor slack)
  terminal:
    q_n_term  * n_N^2
    q_v_term  * (v_N - v_ref_N)^2

Hard constraints:
    n_k ∈ [n_lb_k - slk_k, n_ub_k + slk_k]       (wall corridor, slackable)
    slk_k ≥ 0
    v_k ∈ [v_min, v_max_k]                        (v_max_k lowers in TRAIL)
    a_k ∈ [a_min, a_max]
    δ_k ∈ [-δ_max, δ_max]
    δ̇_k ∈ [-δ̇_max, δ̇_max]
    μ_k ∈ [-μ_max, μ_max]
    δ[0] == δ_prev1   (carry prev solution's δ[1] → no knot jump at boundary)
    n[0] == n0, μ[0] == μ0, v[0] == v0

TRAIL behaviour:
    v_max_k capped by obstacle's s-speed × 0.95 → solver naturally holds
    station behind obstacle. Combined with w_cont + terminal cost pulling
    n → 0 (raceline), this produces a smooth "swing back to centerline +
    decelerate" morph over ~5 ticks.

Outputs:
    traj  = np.ndarray (N+1, 4) [s, n, μ, v]
    speed0 = float (v[0] + a[0]*dT) for the controller
    steer0 = float δ[0]
    success = bool
"""

import os
import ctypes
import warnings
import numpy as np
import casadi as ca


# ### HJ : v3c+ — linear_solver auto-fallback. HSL (ma27/ma57/ma77/ma86/ma97)
# requires libhsl.so dlopen-able by IPOPT. libhsl.so is built by
# planner/3d_gb_optimizer/fast_ggv_gen/solver/setup_hsl.sh. For users
# without HSL, we silently fall back to MUMPS (bundled with CasADi).
_HSL_NAMES = ('ma27', 'ma57', 'ma77', 'ma86', 'ma97')


def _hsl_available():
    """Probe whether libhsl.so is loadable. Checks the CasADi-local symlink
    first (where setup_hsl.sh installs it), then ld-cache via ctypes."""
    try:
        ca_dir = os.path.dirname(ca.__file__)
        cand = os.path.join(ca_dir, 'libhsl.so')
        if os.path.exists(cand):
            return True
    except Exception:
        pass
    for name in ('libhsl.so', 'libcoinhsl.so'):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _resolve_linear_solver(requested):
    req = (requested or 'mumps').strip().lower()
    if req == 'mumps':
        return 'mumps'
    if req in _HSL_NAMES:
        if _hsl_available():
            return req
        warnings.warn(
            "[frenet_kin_solver] linear_solver=%r requested but libhsl.so "
            "not loadable; falling back to 'mumps'. To enable HSL, run "
            "planner/3d_gb_optimizer/fast_ggv_gen/solver/setup_hsl.sh "
            "inside the Docker container." % req)
        return 'mumps'
    warnings.warn(
        "[frenet_kin_solver] unknown linear_solver=%r; using 'mumps'." % req)
    return 'mumps'


SIDE_CLEAR = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2
SIDE_TRAIL = 3


class FrenetKinMPC:
    FAR = 1e3  # sentinel station for inactive obstacle slots

    def __init__(self, **params):
        # horizon
        self.N = int(params.get('N', 20))
        self.dT = float(params.get('dT', 0.05))
        self.L = float(params.get('vehicle_L', 0.33))

        # cost weights (running cost)
        self.q_n = float(params.get('q_n', 3.0))
        # ### HJ : 2026-04-24 — stage n k-ramp (recovery convergence).
        # Default 0.0 = legacy behaviour (constant q_n). YAML / rqt set
        # a positive value to enable per-step ramp up to k=N.
        self.q_n_ramp = float(params.get('q_n_ramp', 0.0))
        self.gamma = float(params.get('gamma_progress', 10.0))
        self.r_a = float(params.get('r_a', 0.5))
        self.r_reg = float(params.get('r_steer_reg', 0.1))
        # v3: δ̇ (rate) and δ̈ (rate-of-rate / jerk-like) penalties
        self.r_dd = float(params.get('r_dd', 5.0))
        self.r_dd_rate = float(params.get('r_dd_rate', 1.0))
        # v3: terminal stability cost
        self.q_n_term = float(params.get('q_n_term', 10.0))
        self.q_v_term = float(params.get('q_v_term', 0.5))
        # v3: tick-to-tick solution continuity
        self.w_cont = float(params.get('w_cont', 20.0))

        self.w_slack = float(params.get('w_slack', 5000.0))

        # obstacle bubble
        self.w_obs = float(params.get('w_obs', 180.0))
        self.sigma_s_obs = float(params.get('sigma_s_obs', 0.7))
        self.sigma_n_obs = float(params.get('sigma_n_obs', 0.18))

        # side bias
        self.w_side_bias = float(params.get('w_side_bias', 25.0))

        # wall cushion (quadratic hinge). v3 strengthens x6+
        self.w_wall_buf = float(params.get('w_wall_buf', 2500.0))
        self.wall_buf = float(params.get('wall_buf', 0.30))

        # limits
        self.v_min = float(params.get('v_min', 0.5))
        self.v_max = float(params.get('v_max', 8.0))
        self.a_min = float(params.get('a_min', -4.0))
        self.a_max = float(params.get('a_max', 3.0))
        # ### HJ : v3c — softer deceleration envelope used to ramp TRAIL
        # vmax[k] from v0 down to the obstacle cap. Below a_min's hard
        # limit (-4) so the solver has strictly-feasible slack; conservative
        # enough that the ramp doesn't induce aggressive braking in the
        # common case.
        self.a_dec_ramp = float(params.get('a_dec_ramp', 3.0))
        self.delta_max = float(params.get('delta_max', 0.6))
        # v3: steer-rate limit (rad/s). Default 3.0 rad/s is conservative
        # for a 1:10 R/C servo; reasoned with Liniger's 2015 setup.
        self.delta_rate_max = float(params.get('delta_rate_max', 3.0))
        self.mu_max = float(params.get('mu_max', 0.9))

        # geometry / safety
        self.inflation = float(params.get('inflation', 0.05))
        self.wall_safe = float(params.get('wall_safe', 0.10))
        self.gap_lat = float(params.get('gap_lat', 0.25))
        self.gap_long = float(params.get('gap_long', 0.8))
        # ### HJ : v3b — vehicle half-width. Previously only the SideDecider
        # knew about ego body size; the solver built the hard corridor and
        # the wall cushion from the *centroid* alone, so "waypoint 15 cm
        # off the wall" meant body-edge = 0 cm (scraping). Now the solver
        # inflates both bounds by ego_half → centroid ≥ ego_half+wall_safe
        # +inflation from each wall → body edge ≥ wall_safe+inflation.
        self.ego_half = float(params.get('ego_half_width', 0.15))

        # ipopt
        self.ipopt_max_iter = int(params.get('ipopt_max_iter', 1000))
        self.ipopt_print_level = int(params.get('ipopt_print_level', 0))
        # ### HJ : v3c+ — HSL ma27 2× faster than MUMPS on our sparsity.
        # libhsl.so provided via solver/setup_hsl.sh (fast_ggv_gen). If HSL
        # not installed, auto-fall back to mumps (probe below).
        requested_solver = str(params.get('linear_solver', 'ma27'))
        self.linear_solver = _resolve_linear_solver(requested_solver)
        # ### HJ : v3c+ — CasADi JIT. Compiles obj/constraint/Jacobian/Hessian
        # to native C at construct time (one-shot ~10-20s cold start). With
        # HSL already absorbing most linear-solve cost, JIT targets the
        # remaining function-eval cost. Set False to skip compile entirely.
        self.ipopt_jit = bool(params.get('ipopt_jit', True))
        self.ipopt_jit_flags = list(params.get(
            'ipopt_jit_flags', ['-O3', '-march=native', '-ffast-math']))

        # runtime state
        self.n_obs_max = int(params.get('n_obs_max', 2))
        self.ready = False
        self.warm = False
        self._warm_X = None  # (N+1, 4) [n, μ, v, δ]
        self._warm_U = None  # (N, 2)   [a, δ̇]
        # prev solution for continuity cost parameter
        self._prev_n_profile = None   # (N+1,) n-profile from last successful solve
        # prev δ[1] for hard initial-δ constraint
        self._prev_de1 = 0.0
        self._have_prev = False

        # last-solve metadata
        self.last_u_sol = None
        self.last_return_status = '-'
        self.last_iter_count = -1
        self.last_slack_max = 0.0
        self.last_cost_breakdown = {}
        # ### HJ : v3b diagnostics — populated every solve(). `last_input` is
        # the exact numeric state/params the NLP was fed (so failure can be
        # replayed offline). `last_infeas_info` identifies which hard bound
        # was violated on failure (max violation per constraint family).
        self.last_input = {}
        self.last_infeas_info = {}
        # Retry-ladder bookkeeping (see solve()). 1=decider side, 2=TRAIL
        # retry, 3=obs-off + wider slack, 0=never attempted.
        self.last_pass = 0
        self.last_pass_history = []

        self._opti = None
        self._vars = None
        self._pars = None

    # ------------------------------------------------------------------ setup
    def setup(self):
        N = self.N
        dT = self.dT
        L = self.L
        n_obs = self.n_obs_max

        opti = ca.Opti()

        # decision vars
        n_ = opti.variable(N + 1)
        mu = opti.variable(N + 1)
        v_ = opti.variable(N + 1)
        de = opti.variable(N + 1)   # δ is now STATE (v3)
        a_ = opti.variable(N)
        dd = opti.variable(N)        # δ̇ is now CONTROL (v3)
        slk = opti.variable(N + 1)

        # parameters
        P_n0 = opti.parameter()
        P_mu0 = opti.parameter()
        P_v0 = opti.parameter()
        P_de0 = opti.parameter()     # v3: prev-solve δ[1] → hard initial δ
        P_de0_active = opti.parameter()  # 0 on first solve, 1 afterward
        P_kappa = opti.parameter(N + 1)
        P_vmax = opti.parameter(N + 1)
        P_nlb = opti.parameter(N + 1)
        P_nub = opti.parameter(N + 1)
        P_dL = opti.parameter(N + 1)
        P_dR = opti.parameter(N + 1)
        P_ref_s = opti.parameter(N + 1)
        P_ref_v = opti.parameter(N + 1)   # v3: used by terminal cost
        P_obs_s = opti.parameter(n_obs, N + 1)
        P_obs_n = opti.parameter(n_obs, N + 1)
        P_obs_active = opti.parameter(n_obs)
        P_bias_L = opti.parameter()
        P_bias_R = opti.parameter()
        # v3: previous-solution n-profile for tick-to-tick continuity cost
        P_n_prev = opti.parameter(N + 1)
        P_cont_active = opti.parameter()

        ### HJ : 2026-04-26 (A4-b) — per-step wall_buf weight ramp.
        ###      When ego starts inside / outside the cushion zone, the FIXED
        ###      cost J_wall(k=0) = w_wall_buf · viol(ego_n)² can dominate
        ###      and force a kinematically-impossible escape rate at k=1.
        ###      With this parameter we ramp wall_buf weight from 0 (or small)
        ###      at k=0..K_entry to 1.0 from K_entry onward. Solver gets a
        ###      "soft entry": initial steps allow wall violation, mid-late
        ###      steps enforce. This produces smooth escape paths.
        ###      Same ramp gates J_slack so slack at k=0 doesn't dominate.
        P_wall_ramp = opti.parameter(N + 1)
        ### HJ : end

        # Live-tunable cost parameters (promoted to opti.parameter so they
        # can be updated per-solve from rqt_reconfigure without NLP rebuild).
        # Default values are seeded from self.* in _solve_single_pass; any
        # change to self.q_n etc. is picked up on the next tick.
        P_q_n = opti.parameter()
        # ### HJ : 2026-04-24 — stage n k-ramp for recovery convergence.
        # Effective q_n at step k becomes `q_n + q_n_ramp · (k/N)` so early
        # horizon stays soft (solver free to deviate around obstacles) and
        # late horizon progressively enforces n→0 (converges to raceline
        # before terminal). Works with both WITH_OBS (gentle ramp) and
        # NO_OBS (aggressive ramp — recovery mode).
        P_q_n_ramp = opti.parameter()
        P_q_n_term = opti.parameter()
        P_q_v_term = opti.parameter()
        P_gamma = opti.parameter()
        P_r_a = opti.parameter()
        P_r_reg = opti.parameter()
        P_r_dd = opti.parameter()
        P_r_dd_rate = opti.parameter()
        P_w_obs = opti.parameter()
        P_w_cont = opti.parameter()
        P_w_wall_buf = opti.parameter()
        P_w_slack = opti.parameter()
        P_sigma_s_obs = opti.parameter()
        P_sigma_n_obs = opti.parameter()
        P_wall_buf = opti.parameter()
        P_gap_lat = opti.parameter()

        # ---- dynamics ----
        for k in range(N):
            opti.subject_to(n_[k + 1] == n_[k]
                            + v_[k] * ca.sin(mu[k]) * dT)
            opti.subject_to(mu[k + 1] == mu[k]
                            + (v_[k] / L * ca.tan(de[k])
                               - P_kappa[k] * v_[k] * ca.cos(mu[k])) * dT)
            opti.subject_to(v_[k + 1] == v_[k] + a_[k] * dT)
            opti.subject_to(de[k + 1] == de[k] + dd[k] * dT)

        # ---- initial conditions ----
        opti.subject_to(n_[0] == P_n0)
        opti.subject_to(mu[0] == P_mu0)
        opti.subject_to(v_[0] == P_v0)
        # v3: δ[0] soft-hard constrained to prev δ[1] (hard when active=1).
        # "soft-hard" = we multiply the residual by P_de0_active so the first
        # solve (active=0) leaves δ[0] free and later solves lock it in.
        opti.subject_to(P_de0_active * (de[0] - P_de0) == 0)

        # ---- input / state bounds ----
        opti.subject_to(opti.bounded(self.a_min, a_, self.a_max))
        opti.subject_to(opti.bounded(-self.delta_rate_max,
                                     dd, self.delta_rate_max))
        opti.subject_to(opti.bounded(-self.delta_max, de, self.delta_max))
        opti.subject_to(opti.bounded(-self.mu_max, mu, self.mu_max))
        for k in range(N + 1):
            # ### HJ : v3c — vmax hard bound skipped at k=0. v_[0] is already
            # pinned by the equality v_[0] == P_v0 (line above), so adding
            # v_[0] <= P_vmax[0] creates a structural conflict whenever the
            # TRAIL cap drops below the live ego speed (e.g. sudden side-flip
            # to TRAIL when v_obs < v_ego). IPOPT then returns
            # Infeasible_Problem_Detected even though the physical MPC problem
            # is solvable — the solver only needs one step to decelerate.
            # Slack bound also skipped at k=0 for the same reason (n_[0] is
            # pinned to P_n0 and always inside corridor after clamp).
            if k > 0:
                opti.subject_to(v_[k] <= P_vmax[k])
            opti.subject_to(v_[k] >= self.v_min)
            opti.subject_to(n_[k] >= P_nlb[k] - slk[k])
            opti.subject_to(n_[k] <= P_nub[k] + slk[k])
            opti.subject_to(slk[k] >= 0.0)
            ### HJ : 2026-04-27 v4 — slack cap = 0 for k>=1 (truly HARD
            ###      corridor). User pushed back on giving any "여유"
            ###      (relaxation) via slack — solver must satisfy the
            ###      pre-margin'd corridor or fail outright.
            ###      Effect:
            ###        n[k] ≤ nub[k]  with nub = d_L - (ego_half +
            ###          wall_safe + inflation) = d_L - 0.28
            ###        path_edge_max = nub + ego_half = d_L - 0.13
            ###        gap_to_wall = wall_safe + inflation (= 0.13m)
            ###      So car edge always ≥ 13cm from actual wall.
            ###      k=0 still lenient (slk allowed up to 1.0m) — ego_n
            ###      is forced from /odom_frenet, may legitimately be
            ###      past corridor at spawn. Tier-1 / tier-2 absorb
            ###      infeasibility on subsequent ticks.
            if k == 0:
                opti.subject_to(slk[k] <= 1.0)
            else:
                opti.subject_to(slk[k] <= 0.0)
            ### HJ : end

        # ---- cost ----
        # All cost weights reference opti.parameter() handles so live
        # rqt_reconfigure updates take effect on the next solve without
        # rebuilding the symbolic NLP (JIT stays warm).
        J_contour = 0
        _denom_N = float(max(N, 1))
        for k in range(N + 1):
            # q_n_k = P_q_n + P_q_n_ramp * (k / N)  (see P_q_n_ramp docstring)
            q_n_k = P_q_n + P_q_n_ramp * (float(k) / _denom_N)
            J_contour = J_contour + q_n_k * n_[k] ** 2

        J_reg = 0
        for k in range(N + 1):
            J_reg = J_reg + P_r_reg * de[k] ** 2

        # v3: δ̇ (steer-rate) penalty — directly penalises curvature rate
        # since κ = tan(δ)/L and δ̇ changes κ̇. This is the C^1 guarantee
        # replacement for the v2 (Δδ)^2 trick.
        J_dd = 0
        for k in range(N):
            J_dd = J_dd + P_r_dd * dd[k] ** 2

        # v3: δ̈ (steer-rate-rate) penalty — jerk analogue. Keeps κ̇
        # continuous → no kinked knots even under disturbance.
        J_dd_rate = 0
        for k in range(N - 1):
            J_dd_rate = J_dd_rate + P_r_dd_rate * (dd[k + 1] - dd[k]) ** 2

        J_smooth_a = 0
        for k in range(N - 1):
            J_smooth_a = J_smooth_a + P_r_a * (a_[k + 1] - a_[k]) ** 2

        prog = 0
        for k in range(N):
            prog = prog + v_[k] * ca.cos(mu[k]) * dT
        J_progress = -P_gamma * prog

        # Obstacle bubble + side bias (same structure as v2; weights tuned).
        # σ_s / σ_n / gap_lat are parameters so the bubble shape is
        # live-tunable too.
        J_obs = 0
        J_bias = 0
        for o in range(n_obs):
            for k in range(N + 1):
                dx = (P_ref_s[k] - P_obs_s[o, k]) / P_sigma_s_obs
                dy = (n_[k] - P_obs_n[o, k]) / P_sigma_n_obs
                prox_sk = P_obs_active[o] * ca.exp(-(dx * dx))
                J_obs = J_obs + (P_w_obs * prox_sk
                                 * ca.exp(-(dy * dy)))
                viol_L = ca.fmax(0.0,
                                 (P_obs_n[o, k] + P_gap_lat) - n_[k])
                viol_R = ca.fmax(0.0,
                                 n_[k] - (P_obs_n[o, k] - P_gap_lat))
                J_bias = J_bias + P_bias_L * prox_sk * viol_L ** 2
                J_bias = J_bias + P_bias_R * prox_sk * viol_R ** 2

        # Wall cushion (strong quadratic hinge inside wall_buf).
        # ### HJ : v3b — cushion fires when the CAR BODY gets within
        # wall_buf of the wall, i.e. centroid within (ego_half + wall_buf).
        # ego_half is a hardware constant so stays Python; wall_buf is
        # a parameter (cushion depth tunable live).
        J_wall = 0
        buf_c = self.ego_half + P_wall_buf
        for k in range(N + 1):
            viol_up = ca.fmax(0.0, n_[k] - (P_dL[k] - buf_c))
            viol_dn = ca.fmax(0.0, -n_[k] - (P_dR[k] - buf_c))
            J_wall = J_wall + P_wall_ramp[k] * P_w_wall_buf * (viol_up ** 2 + viol_dn ** 2)

        # v3: solution continuity — pulls new n[k] toward prev solution's
        # shifted n[k+1]. w_cont * Σ (n - n_prev)^2 with per-tick gate.
        J_cont = 0
        for k in range(N + 1):
            J_cont = J_cont + P_w_cont * P_cont_active \
                            * (n_[k] - P_n_prev[k]) ** 2

        # v3: terminal cost — raceline return + speed-match near horizon end.
        J_term = (P_q_n_term * n_[N] ** 2
                  + P_q_v_term * (v_[N] - P_ref_v[N]) ** 2)

        J_slack = 0
        for k in range(N + 1):
            ### HJ : 2026-04-27 — J_slack NOT ramped. Reverted from A4-b
            ###      where wall_ramp scaled both J_wall AND J_slack — that
            ###      removed the corridor-violation pressure entirely at
            ###      k=0..K_entry, allowing the solver to plan paths that
            ###      went FURTHER outside the corridor (punching through
            ###      the wall in xy). Now only J_wall (cushion) is relaxed
            ###      at entry; J_slack stays full so ego is always pushed
            ###      back inside hard corridor as fast as kinematics allow.
            J_slack = J_slack + P_w_slack * slk[k] ** 2

        J = (J_contour + J_reg + J_dd + J_dd_rate + J_smooth_a
             + J_progress + J_obs + J_bias + J_wall
             + J_cont + J_term + J_slack)

        opti.minimize(J)
        solver_opts = {
            'ipopt.max_iter':       self.ipopt_max_iter,
            'ipopt.print_level':    self.ipopt_print_level,
            'ipopt.linear_solver':  self.linear_solver,
            'print_time':           0,
            'ipopt.sb':             'yes',
        }
        # ### HJ : v3c+ — attach JIT (compile symbolic fns to native C).
        if self.ipopt_jit:
            solver_opts['jit'] = True
            solver_opts['compiler'] = 'shell'
            solver_opts['jit_options'] = {
                'flags':   self.ipopt_jit_flags,
                'verbose': False,
            }
        opti.solver('ipopt', solver_opts)

        self._opti = opti
        self._vars = dict(n=n_, mu=mu, v=v_, de=de,
                          a=a_, dd=dd, slk=slk)
        self._pars = dict(n0=P_n0, mu0=P_mu0, v0=P_v0,
                          de0=P_de0, de0_active=P_de0_active,
                          kappa=P_kappa, vmax=P_vmax,
                          nlb=P_nlb, nub=P_nub,
                          dL=P_dL, dR=P_dR,
                          ref_s=P_ref_s, ref_v=P_ref_v,
                          obs_s=P_obs_s, obs_n=P_obs_n,
                          obs_active=P_obs_active,
                          bias_L=P_bias_L, bias_R=P_bias_R,
                          n_prev=P_n_prev, cont_active=P_cont_active,
                          ### HJ : per-step wall ramp (A4-b)
                          wall_ramp=P_wall_ramp,
                          # Live-tunable cost parameters
                          q_n=P_q_n, q_n_ramp=P_q_n_ramp,
                          q_n_term=P_q_n_term, q_v_term=P_q_v_term,
                          gamma=P_gamma,
                          r_a=P_r_a, r_reg=P_r_reg,
                          r_dd=P_r_dd, r_dd_rate=P_r_dd_rate,
                          w_obs=P_w_obs, w_cont=P_w_cont,
                          w_wall_buf=P_w_wall_buf, w_slack=P_w_slack,
                          sigma_s_obs=P_sigma_s_obs,
                          sigma_n_obs=P_sigma_n_obs,
                          wall_buf=P_wall_buf, gap_lat=P_gap_lat)
        self._cost_exprs = dict(
            contour=J_contour, reg=J_reg, dd=J_dd, dd_rate=J_dd_rate,
            smooth_a=J_smooth_a, progress=J_progress,
            obs=J_obs, bias=J_bias, wall=J_wall,
            cont=J_cont, term=J_term, slack=J_slack)
        self.ready = True

    # ----------------------------------------------------------------- helpers
    ### HJ : 2026-04-26 (A4-b) — push per-step wall_buf weight ramp.
    ###      arr length must equal N+1, values typically 0..1.
    ###      None → reset to all-ones (default behaviour).
    def set_wall_ramp(self, arr):
        if arr is None:
            self._wall_ramp_arr = None
            return
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.N + 1:
            raise ValueError(
                f'set_wall_ramp: expected len={self.N+1}, got {a.shape[0]}')
        self._wall_ramp_arr = a
    ### HJ : end

    def reset_warm_start(self):
        self._warm_X = None
        self._warm_U = None
        self._prev_n_profile = None
        self._have_prev = False
        self._prev_de1 = 0.0
        self.warm = False

    # Live-tunable weight attribute names. Every attribute listed here is
    # bound to a matching opti.parameter() in setup() and pushed to the
    # solver each solve() via set_value, so changing the attribute takes
    # effect on the NEXT tick without any NLP rebuild.
    LIVE_TUNABLE_WEIGHTS = (
        'q_n', 'q_n_ramp',
        'q_n_term', 'q_v_term', 'gamma',
        'r_a', 'r_reg', 'r_dd', 'r_dd_rate',
        'w_obs', 'w_side_bias', 'w_cont', 'w_wall_buf', 'w_slack',
        'sigma_s_obs', 'sigma_n_obs',
        'wall_buf', 'gap_lat',
    )

    def update_weights(self, **kwargs):
        """Hot-swap cost weights / bubble shape without rebuilding the NLP.

        Accepts any subset of LIVE_TUNABLE_WEIGHTS plus the alias
        'gamma_progress' (mapped to 'gamma'). Unknown keys are ignored
        with a one-shot warning so stale YAML / rqt fields do not crash.

        Returns a dict of {name: (old, new)} for every attribute actually
        changed, for logging.
        """
        changed = {}
        for key, raw in kwargs.items():
            attr = 'gamma' if key == 'gamma_progress' else key
            if attr not in self.LIVE_TUNABLE_WEIGHTS:
                warnings.warn(
                    "[frenet_kin_solver.update_weights] unknown key %r — "
                    "ignored" % key, stacklevel=2)
                continue
            try:
                new = float(raw)
            except (TypeError, ValueError):
                warnings.warn(
                    "[frenet_kin_solver.update_weights] non-numeric value "
                    "for %r: %r — ignored" % (key, raw), stacklevel=2)
                continue
            old = float(getattr(self, attr))
            if abs(old - new) > 1e-12:
                setattr(self, attr, new)
                changed[attr] = (old, new)
        return changed

    def get_weights(self):
        """Current snapshot of every live-tunable weight. Used by the node
        to seed rqt sliders without duplicating attribute names."""
        return {k: float(getattr(self, k)) for k in self.LIVE_TUNABLE_WEIGHTS}

    def _build_wall_bounds(self, d_left_arr, d_right_arr):
        dL = np.asarray(d_left_arr, dtype=float)
        dR = np.asarray(d_right_arr, dtype=float)
        # ### HJ : v3b — include ego_half so CENTROID bound guarantees
        # body-edge ≥ wall_safe + inflation from the wall.
        margin = self.ego_half + self.inflation + self.wall_safe
        nub = np.maximum(dL - margin, 1e-3)
        nlb = -np.maximum(dR - margin, 1e-3)
        return nlb, nub

    def _build_vmax(self, ref_v, obs_arr, side, v0=None):
        """Compute per-step vmax[k] for the horizon.

        ### HJ : 2026-04-26 (A1) — universal v0-protect ramp.
        ###      Previously only TRAIL applied a ramp from v0; non-TRAIL paths
        ###      were left with vmax[k] = min(ref_v[k], v_max). When ref_v
        ###      itself drops below v0 (entering a slow corner), the same
        ###      v_[1] <= vmax[1] vs v_[0] = v0 conflict happens — IPOPT
        ###      returns Infeasible_Problem_Detected even though the physical
        ###      problem is solvable (just need a few ticks of deceleration).
        ###      Fix: regardless of side, lift vmax[k] up to v0_ramp_floor[k]:
        ###          vmax[k] = max(cap[k], v0 - a_dec_ramp · k · dT)
        ###      cap[k] = base cap (ref_v + v_max + TRAIL obs cap if active).
        ###      vmax[0] is therefore always ≥ v0 → v_[0]==v0 satisfies bound.
        ###      As k grows, ramp drops; once below cap, cap takes over (true
        ###      enforcement). a_dec_ramp ≤ |a_min| guarantees feasibility.
        """
        # ---- base cap (ref_v + v_max global) ----
        cap = np.minimum(np.asarray(ref_v, dtype=float), self.v_max)

        # ---- TRAIL obstacle cap ----
        if side == SIDE_TRAIL and obs_arr is not None:
            obs_cap = self.v_max
            for o in range(obs_arr.shape[0]):
                if float(np.max(obs_arr[o, :, 2])) <= 0.0:
                    continue
                s0 = float(obs_arr[o, 0, 0])
                sN = float(obs_arr[o, -1, 0])
                v_obs_s = max((sN - s0)
                              / max(self.N * self.dT, 1e-3), 0.0)
                obs_cap = min(obs_cap, max(v_obs_s * 0.95, self.v_min))
            cap = np.minimum(cap, obs_cap)

        # ---- universal v0-protect ramp ----
        if v0 is not None:
            a_dec_ramp = float(getattr(self, 'a_dec_ramp', 3.0))
            N1 = self.N + 1
            v0f = float(v0)
            v0_ramp = np.array([
                max(self.v_min + 0.1, v0f - a_dec_ramp * self.dT * k)
                for k in range(N1)
            ], dtype=float)
            # vmax[k] = max(cap[k], v0_ramp[k])
            #   - early k: v0_ramp ≥ v0, cap may be < v0 → vmax follows ramp
            #   - late k: v0_ramp drops below cap → vmax follows cap (enforce)
            vmax = np.maximum(cap, v0_ramp)
        else:
            vmax = cap

        vmax = np.maximum(vmax, self.v_min + 0.1)
        return vmax

    def _build_obs_params(self, obs_arr):
        N1 = self.N + 1
        s_mat = np.full((self.n_obs_max, N1), self.FAR, dtype=float)
        n_mat = np.full((self.n_obs_max, N1), self.FAR, dtype=float)
        active = np.zeros(self.n_obs_max, dtype=float)
        if obs_arr is None:
            return s_mat, n_mat, active
        for o in range(min(obs_arr.shape[0], self.n_obs_max)):
            w_ts = obs_arr[o, :, 2]
            if float(np.max(w_ts)) <= 0.0:
                continue
            s_mat[o, :] = obs_arr[o, :, 0]
            n_mat[o, :] = obs_arr[o, :, 1]
            active[o] = 1.0
        return s_mat, n_mat, active

    def _seed_warm_start(self, n0, mu0, v0, de0, nlb, nub, vmax):
        N = self.N
        X = np.zeros((N + 1, 4))
        U = np.zeros((N, 2))
        for k in range(N + 1):
            X[k, 0] = float(np.clip(n0, nlb[k], nub[k]))
            X[k, 1] = mu0 * (1.0 - k / max(N, 1))
            X[k, 2] = float(np.clip(v0, self.v_min + 0.1, vmax[k]))
            X[k, 3] = de0
        return X, U

    # ------------------------------------------------------------------- solve
    def solve(self, initial_state, ref_slice,
              obstacles=None, side=SIDE_CLEAR, bias_scale=1.0):
        """Two-pass retry ladder. See class docstring for motivation.

        pass 1: decider's side, full obstacles active.
        pass 2 (on fail): force SIDE_TRAIL → vmax caps (ramped from v0)
                          to v_obs×0.95, giving the solver a feasible
                          "hold behind" plan.

        ### HJ : v3c — former Pass 3 (obs-off + CLEAR) removed. It masked
        genuine infeasibility with a trajectory that drove straight through
        the obstacle at ref_v, starving the node-level fallback ladder of
        the failure signal it needs. Now, if both passes fail, solver
        returns success=False and the node engages HOLD_LAST → geometric
        quintic → convergence quintic → raceline-slice (recovery chain
        the user explicitly asked to use).
        """
        if not self.ready:
            raise RuntimeError('FrenetKinMPC.setup() must be called first')

        self.last_pass_history = []

        # pass 1 — caller's side, obstacles active
        ok, out = self._solve_single_pass(
            initial_state, ref_slice, obstacles=obstacles,
            side=side, bias_scale=bias_scale, pass_idx=1)
        if ok:
            self.last_pass = 1
            return out

        # pass 2 — force TRAIL (v_max cap ramped from v0) but keep obstacles
        ok, out = self._solve_single_pass(
            initial_state, ref_slice, obstacles=obstacles,
            side=SIDE_TRAIL, bias_scale=bias_scale, pass_idx=2)
        if ok:
            self.last_pass = 2
            return out

        # Both passes failed. Report failure to node so the tier1/2/3
        # recovery ladder engages instead of publishing a dangerous
        # "obs-off straight-line" plan.
        self.last_pass = 0
        return out

    def _solve_single_pass(self, initial_state, ref_slice,
                           obstacles=None, side=SIDE_CLEAR, bias_scale=1.0,
                           pass_idx=1):
        """Single NLP solve. Returns (ok_bool, (speed0, steer0, traj, success)).

        Also populates `self.last_input`, `self.last_infeas_info`,
        `self.last_return_status`, `self.last_pass_history` so the caller
        (node.tick_json) can explain WHY the pass failed.
        """
        n0, mu0, v0 = (float(initial_state[0]),
                       float(initial_state[1]),
                       float(initial_state[2]))
        v0 = min(max(v0, self.v_min + 1e-3), self.v_max)

        kappa = np.asarray(ref_slice['kappa_ref'], dtype=float)
        dL = np.asarray(ref_slice['d_left_arr'], dtype=float)
        dR = np.asarray(ref_slice['d_right_arr'], dtype=float)
        rv = np.asarray(ref_slice['ref_v'], dtype=float)
        rs = np.asarray(ref_slice['ref_s'], dtype=float)

        nlb, nub = self._build_wall_bounds(dL, dR)
        # ### HJ : v3c — pass v0 so TRAIL vmax can be ramped from v0 down to
        # the obstacle cap via physical deceleration (a_dec_ramp ≈ 3 m/s²)
        # rather than snapping to the cap at k=0.
        vmax = self._build_vmax(rv, obstacles, side, v0=v0)
        ### HJ : expose computed vmax for tick_json diagnostics.
        self._last_vmax = vmax
        ### HJ : end

        ### HJ : 2026-04-26 (A4-a) — n0 clamp 폐기.
        ###      Previously clamped ego_n into [nlb[0], nub[0]] for solver
        ###      stability. But this HID the wall-violation cost from the
        ###      solver: it thought ego was at corridor edge, so J_wall(k=0)
        ###      ≈ 0, slack[0] = 0, and the solver had no urgency to escape.
        ###      Result: when ego started near-wall, solver kept "drifting
        ###      along wall" instead of converging to GB.
        ###      Now we pass real n0. Hard initial n[0]==P_n0 with corridor
        ###      slk[k] absorbs the violation; solver naturally generates an
        ###      escape path. JIT-safe (no graph change). The wall_ramp added
        ###      in A4-b/c relaxes wall_buf cost at k=0,1 so the slacked
        ###      escape stays kinematically smooth.
        n0_clamped = float(n0)
        ### HJ : end

        obs_s_mat, obs_n_mat, obs_active = self._build_obs_params(obstacles)

        _ = bias_scale
        w_bias = self.w_side_bias
        if side == SIDE_LEFT:
            bL, bR = w_bias, 0.0
        elif side == SIDE_RIGHT:
            bL, bR = 0.0, w_bias
        else:
            bL, bR = 0.0, 0.0

        if self._have_prev and self._prev_n_profile is not None:
            n_prev = self._prev_n_profile.copy()
            cont_active = 1.0
        else:
            n_prev = np.zeros(self.N + 1)
            cont_active = 0.0

        de0_val = float(self._prev_de1)
        de0_active_val = 1.0 if self._have_prev else 0.0

        opti = self._opti
        V = self._vars
        P = self._pars

        opti.set_value(P['n0'], n0_clamped)
        opti.set_value(P['mu0'], mu0)
        opti.set_value(P['v0'], v0)
        opti.set_value(P['de0'], de0_val)
        opti.set_value(P['de0_active'], de0_active_val)
        opti.set_value(P['kappa'], kappa)
        opti.set_value(P['vmax'], vmax)
        opti.set_value(P['nlb'], nlb)
        opti.set_value(P['nub'], nub)
        opti.set_value(P['dL'], dL)
        opti.set_value(P['dR'], dR)
        opti.set_value(P['ref_s'], rs)
        opti.set_value(P['ref_v'], rv)
        opti.set_value(P['obs_s'], obs_s_mat)
        opti.set_value(P['obs_n'], obs_n_mat)
        opti.set_value(P['obs_active'], obs_active)
        opti.set_value(P['bias_L'], bL)
        opti.set_value(P['bias_R'], bR)
        opti.set_value(P['n_prev'], n_prev)
        opti.set_value(P['cont_active'], cont_active)
        ### HJ : push per-step wall ramp. Default = ones (no-op, identical to
        ###      pre-A4-b behaviour). Node sets it via set_wall_ramp() to
        ###      relax wall_buf cost at early k when ego is near/past wall.
        wall_ramp_arr = getattr(self, '_wall_ramp_arr', None)
        if wall_ramp_arr is None or wall_ramp_arr.shape[0] != self.N + 1:
            wall_ramp_arr = np.ones(self.N + 1, dtype=np.float64)
        opti.set_value(P['wall_ramp'], wall_ramp_arr)
        ### HJ : end

        # Push live-tunable cost weights every tick so rqt_reconfigure
        # changes take effect on the next solve.
        opti.set_value(P['q_n'],         self.q_n)
        opti.set_value(P['q_n_ramp'],    self.q_n_ramp)
        opti.set_value(P['q_n_term'],    self.q_n_term)
        opti.set_value(P['q_v_term'],    self.q_v_term)
        opti.set_value(P['gamma'],       self.gamma)
        opti.set_value(P['r_a'],         self.r_a)
        opti.set_value(P['r_reg'],       self.r_reg)
        opti.set_value(P['r_dd'],        self.r_dd)
        opti.set_value(P['r_dd_rate'],   self.r_dd_rate)
        opti.set_value(P['w_obs'],       self.w_obs)
        opti.set_value(P['w_cont'],      self.w_cont)
        opti.set_value(P['w_wall_buf'],  self.w_wall_buf)
        opti.set_value(P['w_slack'],     self.w_slack)
        opti.set_value(P['sigma_s_obs'], max(self.sigma_s_obs, 1e-3))
        opti.set_value(P['sigma_n_obs'], max(self.sigma_n_obs, 1e-3))
        opti.set_value(P['wall_buf'],    self.wall_buf)
        opti.set_value(P['gap_lat'],     self.gap_lat)

        # ---- warm start ----
        # Pass 2+ always re-seed (previous pass's debug values may be in
        # an inconsistent region).
        if (pass_idx == 1 and self._warm_X is not None
                and self._warm_X.shape == (self.N + 1, 4)):
            Xw, Uw = self._warm_X, self._warm_U
        else:
            Xw, Uw = self._seed_warm_start(n0_clamped, mu0, v0,
                                           de0_val, nlb, nub, vmax)
        opti.set_initial(V['n'],  Xw[:, 0])
        opti.set_initial(V['mu'], Xw[:, 1])
        opti.set_initial(V['v'],  Xw[:, 2])
        opti.set_initial(V['de'], Xw[:, 3])
        opti.set_initial(V['a'],  Uw[:, 0])
        opti.set_initial(V['dd'], Uw[:, 1])
        ### HJ : 2026-04-27 — explicit slk seed = 0 so ipopt doesn't keep
        ###      a stale 99999 sentinel from previous solver state. Slack
        ###      should be 0 (or small forced value at k=0 only). Combined
        ###      with the slk[k] <= 1.0 upper bound, this keeps slack
        ###      reporting numerically sane.
        opti.set_initial(V['slk'], np.zeros(self.N + 1, dtype=np.float64))
        ### HJ : end

        # Capture input snapshot for every pass (overwritten; only the
        # LAST pass's input survives on failure — the most revealing one).
        self.last_input = {
            'pass': int(pass_idx),
            'side_effective': int(side),
            'n0': float(n0), 'n0_clamped': float(n0_clamped),
            'mu0': float(mu0), 'v0': float(v0),
            'de0_locked': float(de0_val),
            'de0_active': float(de0_active_val),
            'nlb_min': float(np.min(nlb)),
            'nub_max': float(np.max(nub)),
            'nlb_0': float(nlb[0]),
            'nub_0': float(nub[0]),
            'nlb_min_k': int(np.argmin(np.abs(nlb - nub))),  # tightest slot
            'corridor_min_width': float(np.min(nub - nlb)),
            'vmax_min': float(np.min(vmax)),
            'vmax_max': float(np.max(vmax)),
            'kappa_abs_max': float(np.max(np.abs(kappa))),
            'dL_min': float(np.min(dL)), 'dL_max': float(np.max(dL)),
            'dR_min': float(np.min(dR)), 'dR_max': float(np.max(dR)),
            'n_obs_active': int(np.sum(obs_active > 0)),
        }

        success = False
        try:
            sol = opti.solve()
            n_sol = np.asarray(sol.value(V['n'])).ravel()
            mu_sol = np.asarray(sol.value(V['mu'])).ravel()
            v_sol = np.asarray(sol.value(V['v'])).ravel()
            de_sol = np.asarray(sol.value(V['de'])).ravel()
            a_sol = np.asarray(sol.value(V['a'])).ravel()
            dd_sol = np.asarray(sol.value(V['dd'])).ravel()
            slk_sol = np.asarray(sol.value(V['slk'])).ravel()
            self.last_return_status = sol.stats().get('return_status', 'OK')
            self.last_iter_count = int(sol.stats().get('iter_count', -1))
            try:
                self.last_cost_breakdown = {
                    k: float(sol.value(vv))
                    for k, vv in self._cost_exprs.items()}
            except Exception:
                self.last_cost_breakdown = {}
            self.last_infeas_info = {}
            self.last_pass_history.append({
                'pass': int(pass_idx),
                'status': self.last_return_status,
                'ok': True,
            })
            success = True
        except RuntimeError:
            self.last_return_status = opti.stats().get('return_status', 'FAIL')
            self.last_iter_count = int(opti.stats().get('iter_count', -1))
            try:
                n_sol = np.asarray(opti.debug.value(V['n'])).ravel()
                mu_sol = np.asarray(opti.debug.value(V['mu'])).ravel()
                v_sol = np.asarray(opti.debug.value(V['v'])).ravel()
                de_sol = np.asarray(opti.debug.value(V['de'])).ravel()
                a_sol = np.asarray(opti.debug.value(V['a'])).ravel()
                dd_sol = np.asarray(opti.debug.value(V['dd'])).ravel()
                slk_sol = np.asarray(opti.debug.value(V['slk'])).ravel()
                self.last_cost_breakdown = {}
                # Probe which hard constraint family is violated most.
                self.last_infeas_info = self._probe_infeasibility(
                    n_sol, mu_sol, v_sol, de_sol, dd_sol, slk_sol,
                    nlb, nub, vmax)
            except Exception as exc:
                self.last_infeas_info = {'probe_error': str(exc)}
                self.last_pass_history.append({
                    'pass': int(pass_idx),
                    'status': self.last_return_status,
                    'ok': False,
                    'err': 'debug_value_failed',
                })
                return False, (0.0, 0.0, None, False)
            self.last_pass_history.append({
                'pass': int(pass_idx),
                'status': self.last_return_status,
                'ok': False,
                'worst': self.last_infeas_info.get('worst'),
            })
            return False, (0.0, 0.0, None, False)

        # --- shift warm-start / continuity anchors one step ---
        N = self.N
        X_new = np.column_stack([n_sol, mu_sol, v_sol, de_sol])
        U_new = np.column_stack([a_sol, dd_sol])
        X_shift = np.empty_like(X_new)
        X_shift[:-1] = X_new[1:]
        X_shift[-1] = X_new[-1]
        U_shift = np.empty_like(U_new)
        U_shift[:-1] = U_new[1:]
        U_shift[-1] = U_new[-1]
        self._warm_X = X_shift
        self._warm_U = U_shift
        self._prev_n_profile = X_shift[:, 0].copy()
        self._prev_de1 = (float(de_sol[1]) if de_sol.shape[0] > 1
                          else float(de_sol[0]))
        self._have_prev = True
        self.warm = True

        self.last_u_sol = np.column_stack([v_sol[:N], de_sol[:N]])
        self.last_slack_max = float(np.max(np.abs(slk_sol)))

        traj = np.column_stack([rs[:N + 1], n_sol, mu_sol, v_sol])
        speed0 = float(v_sol[0] + a_sol[0] * self.dT)
        steer0 = float(de_sol[0])
        return True, (speed0, steer0, traj, success)

    def _probe_infeasibility(self, n_sol, mu_sol, v_sol, de_sol, dd_sol,
                             slk_sol, nlb, nub, vmax):
        """Identify the most violated hard constraint family. Uses the
        debug trajectory (IPOPT's last iterate) since the true optimum is
        infeasible. Returns dict mapping family → (max_violation_m_or_rad,
        k_of_max) plus 'worst' key naming the family with highest violation.
        """
        viol = {}
        # corridor upper:  n[k] - nub[k] ≤ slack  →  viol = max(0, n - nub - slk)
        corr_up = (n_sol - nub) - slk_sol
        corr_dn = (nlb - n_sol) - slk_sol
        viol['corridor_upper'] = (float(np.max(np.maximum(corr_up, 0.0))),
                                  int(np.argmax(corr_up)))
        viol['corridor_lower'] = (float(np.max(np.maximum(corr_dn, 0.0))),
                                  int(np.argmax(corr_dn)))
        # speed bounds
        v_over = v_sol - vmax
        v_under = self.v_min - v_sol
        viol['v_over_max'] = (float(np.max(np.maximum(v_over, 0.0))),
                              int(np.argmax(v_over)))
        viol['v_under_min'] = (float(np.max(np.maximum(v_under, 0.0))),
                               int(np.argmax(v_under)))
        # heading bound
        mu_abs = np.abs(mu_sol) - self.mu_max
        viol['mu_abs'] = (float(np.max(np.maximum(mu_abs, 0.0))),
                          int(np.argmax(mu_abs)))
        # steering bound
        de_abs = np.abs(de_sol) - self.delta_max
        viol['delta_abs'] = (float(np.max(np.maximum(de_abs, 0.0))),
                             int(np.argmax(de_abs)))
        # steering-rate bound
        dd_abs = np.abs(dd_sol) - self.delta_rate_max
        viol['delta_rate_abs'] = (float(np.max(np.maximum(dd_abs, 0.0))),
                                  int(np.argmax(dd_abs)))
        # slack negativity (slk >= 0)
        slk_neg = -slk_sol
        viol['slack_neg'] = (float(np.max(np.maximum(slk_neg, 0.0))),
                             int(np.argmax(slk_neg)))

        # Identify worst family
        worst_name = max(viol.keys(), key=lambda k: viol[k][0])
        worst_val, worst_k = viol[worst_name]
        return {
            'worst': worst_name,
            'worst_val': round(worst_val, 4),
            'worst_k': worst_k,
            'slack_peak': float(np.max(slk_sol)),
            **{k: round(v[0], 4) for k, v in viol.items()},
        }
