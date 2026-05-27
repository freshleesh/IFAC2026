"""
Kinematic MPC Solver for planner use.

Simplified from the original EVO-MPCC port (see mpcc_solver_original.py).
Reference is pre-sliced by the caller, so s/p/LUTs are removed.

States:   [x, y, psi]   - position and heading
Controls: [v, delta]    - forward speed and steering angle

Cost:
  - contouring error (e_c / 0.5)^2
  - lag error (e_l / 0.5)^2
  - velocity tracking: w_velocity * ((v - ref_v) / v_bias_max)^2
  - control smoothness: w_dv, w_dsteering
  - ### HJ : Phase 3.5 — per-step per-obstacle soft repulsive cost:
      sum_o  w_obs_o * exp(-((x_k - ox)^2 + (y_k - oy)^2) / (2 sigma^2))
    Obstacles are passed in as (n_obs_max, N, 3) = [ox, oy, w_obs].
    Unused slots set w_obs = 0 AND coord = FAR_XY so they contribute
    numerically zero.
  - ### HJ : Phase 4.1 — tiny w_steer_reg * delta^2 regularizer so that
    at v ≈ 0 the control delta still has a well-defined minimizer.

Constraints:
  - Dynamics: kinematic bicycle (Euler) with v_eff = v + kinematic_v_eps
    so yaw-rate v/L * tan(delta) does not vanish identically at v=0.
  - Track boundary: single half-plane per step, soft (slack).
  - Box: state/control/slack bounds.

Run-time mutators (no NLP rebuild):
  - update_box_bounds(v_min, v_max, theta_max)  → lbx/ubx swap
  - update_inflation(inflation)                 → caller-side ref slice only

Run-time mutators (NLP rebuild):
  - rebuild_nlp(**kwargs)  → any cost weight, ipopt_max_iter, obstacle_sigma.
"""

import numpy as np
from casadi import (MX, atan2, cos, sin, tan, exp, vertcat, reshape, nlpsol)


class MPCCSolver:

    # ### HJ : Phase 3.5 — parameter-vector slot layout.
    # Per step: [ref_x, ref_y, ref_dx, ref_dy, ref_v, bound_a, bound_b,
    #            ox_0, oy_0, w_0,  ox_1, oy_1, w_1,  ...]
    REF_BLOCK = 7      # reference (x, y, dx, dy, v, bound_a, bound_b)
    OBS_BLOCK = 3      # per obstacle (ox, oy, w_obs)
    FAR_XY = 1.0e4     # "obstacle at infinity" sentinel for unused slots

    def __init__(self, params):
        # Horizon
        self.N = params['N']
        self.dT = params['dT']
        self.L = params['vehicle_L']

        # Limits
        self.v_max = params['max_speed']
        self.v_min = params.get('min_speed', 0.0)
        self.theta_max = params['max_steering']

        # Cost weights
        self.w_contour = params['w_contour']
        self.w_lag = params['w_lag']
        self.w_velocity = params['w_velocity']
        self.v_bias_max = params.get('v_bias_max', 10.0)
        self.w_dv = params['w_dv']
        self.w_dsteering = params['w_dsteering']

        # ### HJ : stage-wise contour/lag ramp. Value = scale at k=0; ramps
        # linearly to 1.0 at k=N-1. Default 1.0 = flat (legacy behavior).
        # Use <1.0 to let near-car stages drift freely and only pull back to
        # raceline at horizon end — removes the "snap-back overshoot" seen in
        # observe when the car sits 2-3cm off the raceline.
        self.contour_ramp_start = float(params.get('contour_ramp_start', 1.0))
        self.lag_ramp_start = float(params.get('lag_ramp_start', 1.0))

        # Boundary
        self.inflation = params.get('boundary_inflation', 0.1)
        self.w_slack = params.get('w_slack', 1000.0)

        # ### HJ : Phase 3.5 — obstacle cost
        self.n_obs_max = int(params.get('n_obs_max', 2))
        self.obstacle_sigma = float(params.get('obstacle_sigma', 0.35))

        # ### HJ : Phase 4.1 — v→0 degeneracy guards
        self.kinematic_v_eps = float(params.get('kinematic_v_eps', 0.05))
        self.w_steer_reg = float(params.get('w_steer_reg', 1.0e-3))

        # IPOPT
        self.ipopt_max_iter = params.get('ipopt_max_iter', 200)
        self.ipopt_print_level = params.get('ipopt_print_level', 0)

        # Dimensions
        self.n_states = 3    # x, y, psi
        self.n_controls = 2  # v, delta

        # Per-step parameter block size (ref + obstacles)
        self.STEP_BLOCK = self.REF_BLOCK + self.OBS_BLOCK * self.n_obs_max

        # State
        self.solver_nlp = None
        self.ready = False

        # Warm start
        self.X0 = None
        self.u0 = None
        self.warm = False

        # Last-solve diagnostics
        self.last_return_status = 'unset'
        self.last_iter_count = -1
        self.last_slack_max = 0.0
        self.last_u_sol = None

    def setup(self):
        """Build the NLP. Call once after construction."""
        self._build_nlp()
        self.ready = True

    # ----------------------------------------------------------------- rebuild
    def rebuild_nlp(self, **kwargs):
        """### HJ : Phase 5 — rebuild the NLP with updated weights/iter/sigma.

        Called by dyn_reconfigure for knobs that bake into the symbolic NLP
        (weights, sigma) or the nlpsol instance (ipopt_max_iter). Box bound
        changes should prefer update_box_bounds(); this path is for true
        structure/coef changes.
        """
        for key in ('w_contour', 'w_lag', 'w_velocity', 'v_bias_max',
                    'w_dv', 'w_dsteering', 'w_slack',
                    'obstacle_sigma', 'ipopt_max_iter',
                    'kinematic_v_eps', 'w_steer_reg',
                    'contour_ramp_start', 'lag_ramp_start'):
            if key in kwargs and kwargs[key] is not None:
                setattr(self, key, type(getattr(self, key))(kwargs[key]))
        # Invalidate warm start — symbolic coefficients changed, stale X0/u0
        # can confuse IPOPT in the first restart.
        self.warm = False
        self._build_nlp()
        self.ready = True

    def update_box_bounds(self, v_min=None, v_max=None, theta_max=None):
        """### HJ : Phase 5 — hot swap of v/δ box bounds (lbx/ubx only)."""
        if v_min is not None:
            self.v_min = float(v_min)
        if v_max is not None:
            self.v_max = float(v_max)
        if theta_max is not None:
            self.theta_max = float(theta_max)
        if not self.ready:
            return
        ns = self.n_states
        nc = self.n_controls
        N = self.N
        state_count = ns * (N + 1)
        for k in range(N):
            self.lbx[state_count:state_count + nc, 0] = [self.v_min, -self.theta_max]
            self.ubx[state_count:state_count + nc, 0] = [self.v_max, self.theta_max]
            state_count += nc

    def update_inflation(self, inflation):
        """Corridor inflation lives on the node side (used when slicing ref).
        Solver just stores the current value so node can query via
        solver.inflation."""
        self.inflation = float(inflation)

    # ----------------------------------------------------------------- build
    def _build_nlp(self):
        N = self.N
        ns = self.n_states
        nc = self.n_controls

        X = MX.sym('X', ns, N + 1)
        U = MX.sym('U', nc, N)
        SL = MX.sym('SL', N, 1)

        STEP_BLOCK = self.STEP_BLOCK
        n_params = ns + STEP_BLOCK * N
        P = MX.sym('P', n_params)

        inv_two_sigma2 = 1.0 / (2.0 * self.obstacle_sigma * self.obstacle_sigma)

        obj = 0
        g = []

        # Initial-state constraint
        g.append(X[:, 0] - P[0:ns])

        for k in range(N):
            st = X[:, k]
            st_next = X[:, k + 1]
            con = U[:, k]

            ref_idx = ns + STEP_BLOCK * k
            ref_x = P[ref_idx]
            ref_y = P[ref_idx + 1]
            t_dx = P[ref_idx + 2]
            t_dy = P[ref_idx + 3]
            ref_v = P[ref_idx + 4]
            bound_a = P[ref_idx + 5]
            bound_b = P[ref_idx + 6]

            t_angle = atan2(t_dy, t_dx)

            # Contouring/lag (normalized by half-width 0.5)
            e_c = (sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)) / 0.5
            e_l = (-cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)) / 0.5

            # ### HJ : stage-wise ramp. k=0 uses ramp_start, k=N-1 uses 1.0.
            if N > 1:
                t_ramp = k / float(N - 1)
            else:
                t_ramp = 1.0
            alpha_c = self.contour_ramp_start + (1.0 - self.contour_ramp_start) * t_ramp
            alpha_l = self.lag_ramp_start + (1.0 - self.lag_ramp_start) * t_ramp
            obj += self.w_contour * alpha_c * (e_c ** 2)
            obj += self.w_lag * alpha_l * (e_l ** 2)

            # Velocity tracking
            obj += self.w_velocity * ((con[0] - ref_v) / self.v_bias_max) ** 2

            # ### HJ : Phase 4.1 — steering regularizer (tiny; keeps v=0 sane)
            obj += self.w_steer_reg * (con[1] ** 2)

            # Control smoothness
            if k < N - 1:
                con_next = U[:, k + 1]
                obj += self.w_dv * (con_next[0] - con[0]) ** 2
                obj += self.w_dsteering * (con_next[1] - con[1]) ** 2

            # Slack penalty (boundary softening)
            obj += self.w_slack * SL[k] ** 2

            # ### HJ : Phase 3.5 — soft repulsive obstacle cost, evaluated at
            # the *next* state so k=0 term already acts on the first predicted
            # pose rather than the fixed initial state.
            obs_base = ref_idx + self.REF_BLOCK
            for o in range(self.n_obs_max):
                ob = obs_base + self.OBS_BLOCK * o
                ox = P[ob]
                oy = P[ob + 1]
                w_o = P[ob + 2]
                dx = st_next[0] - ox
                dy = st_next[1] - oy
                obj += w_o * exp(-(dx * dx + dy * dy) * inv_two_sigma2)

            # Dynamics (Euler kinematic bicycle with v_eff)
            # ### HJ : Phase 4.1 — v_eff = v + eps so yaw-rate term stays
            # alive when v → 0 (recovery / min_speed=0).
            v_eff = con[0] + self.kinematic_v_eps
            st_next_euler = st + self.dT * vertcat(
                con[0] * cos(st[2]),
                con[0] * sin(st[2]),
                (v_eff / self.L) * tan(con[1]),
            )
            g.append(st_next - st_next_euler)

            # Soft boundary: lo - slack <= bound_a*x - bound_b*y <= hi + slack
            c_expr = bound_a * st_next[0] - bound_b * st_next[1]
            g.append(c_expr - SL[k])   # upper side (ubg = hi)
            g.append(c_expr + SL[k])   # lower side (lbg = lo)

        g = vertcat(*g)

        OPT_variables = vertcat(
            reshape(X, ns * (N + 1), 1),
            reshape(U, nc * N, 1),
            SL,
        )

        # Constraint vector layout:
        #   [initial(ns)] + N * [dynamics(ns) + boundary_upper(1) + boundary_lower(1)]
        n_per_step = ns + 2
        n_constraints = ns + N * n_per_step
        self.lbg = np.zeros((n_constraints, 1))
        self.ubg = np.zeros((n_constraints, 1))

        # Box constraints
        n_slack = N
        n_vars = ns * (N + 1) + nc * N + n_slack
        self.lbx = np.zeros((n_vars, 1))
        self.ubx = np.zeros((n_vars, 1))

        for k in range(N + 1):
            self.lbx[ns * k:ns * (k + 1), 0] = [-200, -200, -1000]
            self.ubx[ns * k:ns * (k + 1), 0] = [200, 200, 1000]

        state_count = ns * (N + 1)
        for k in range(N):
            self.lbx[state_count:state_count + nc, 0] = [self.v_min, -self.theta_max]
            self.ubx[state_count:state_count + nc, 0] = [self.v_max, self.theta_max]
            state_count += nc

        # ### HJ : Phase 4.2 R6 — slack cap (2 m) so degenerate slack cannot
        # blow up gradient; penalty still drives it to ~0 in healthy solves.
        for k in range(N):
            self.lbx[state_count, 0] = 0.0
            self.ubx[state_count, 0] = 2.0
            state_count += 1

        nlp = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
        opts = {
            'ipopt': {
                'max_iter': self.ipopt_max_iter,
                'print_level': self.ipopt_print_level,
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-3,
                'fixed_variable_treatment': 'make_parameter',
            },
            'print_time': 0,
        }
        self.solver_nlp = nlpsol('solver', 'ipopt', nlp, opts)
        self.n_params = n_params

    # ----------------------------------------------------------------- solve
    @staticmethod
    def _coerce_obstacles(obstacles, n_obs_max, N):
        """### HJ : Phase 3.5 — normalize obstacles arg to (n_obs_max, N, 3).

        None / shape-mismatch → all-disabled slots (far + w=0).
        """
        arr = np.zeros((n_obs_max, N, 3), dtype=np.float64)
        arr[:, :, 0] = MPCCSolver.FAR_XY
        arr[:, :, 1] = MPCCSolver.FAR_XY
        if obstacles is None:
            return arr
        obstacles = np.asarray(obstacles, dtype=np.float64)
        if obstacles.shape != (n_obs_max, N, 3):
            return arr
        return obstacles

    def solve(self, initial_state, ref_data, obstacles=None):
        """
        Args:
            initial_state: [x, y, psi]
            ref_data: dict with
                center_points: (>=N, 2)   reference centers
                left_points:   (>=N, 2)   left boundary points
                right_points:  (>=N, 2)   right boundary points
                ref_v:         (>=N,)     reference speeds
                ref_dx, ref_dy: (>=N,)    reference tangent unit vector
            obstacles: optional (n_obs_max, N, 3) ndarray with per-step
                [ox, oy, w_obs]. Unused slots pass w_obs=0 and far coords.

        Returns:
            speed (u_0[0]), steering (u_0[1]), trajectory (N+1, 3), success
        """
        if not self.ready:
            return 0.0, 0.0, None, False

        N = self.N
        ns = self.n_states
        nc = self.n_controls

        # ### HJ : Phase 4.2 R2 (fixed) — anchor the unwrap against the
        # raceline reference tangent (always in [-pi, pi] from arctan2),
        # NOT against the warm-start X0. Anchoring on X0 caused runaway:
        # X0 accumulated +2π each lap, pulling initial_state with it, and
        # the solver emitted spiral-shaped trajectories to connect the
        # unwrapped initial heading to the wrapped ref tangent (seen in
        # debug_log as steer-saturated loops with psi_init ≈ 11, 17, 36 rad).
        # The NLP dynamics use cos/sin of psi → invariant to 2π shifts,
        # so the only role of unwrap is warm-start continuity against ref.
        ref_dx_arr = ref_data['ref_dx']
        ref_dy_arr = ref_data['ref_dy']
        ref_yaw_anchor = float(np.arctan2(ref_dy_arr[0], ref_dx_arr[0]))
        delta_yaw = ref_yaw_anchor - initial_state[2]
        if abs(delta_yaw) >= np.pi:
            k_shift = round(delta_yaw / (2 * np.pi))
            initial_state[2] = initial_state[2] + k_shift * (2 * np.pi)

        # ### HJ : Align warm-start X0 psi column with the (now ref-anchored)
        # initial_state. X0 carries the previous solve's psi, which can have
        # accumulated 2π drift over laps while initial_state stays bounded
        # near the ref tangent. Without this shift, X0[0] and initial_state
        # disagree by 2πk, and IPOPT warm-starts from a trajectory offset by
        # one or more full revolutions from the actual pose.
        if self.X0 is not None:
            x0_shift = round(
                (initial_state[2] - self.X0[0, 2]) / (2 * np.pi)
            ) * (2 * np.pi)
            if x0_shift != 0.0:
                self.X0[:, 2] = self.X0[:, 2] + x0_shift

        obs = self._coerce_obstacles(obstacles, self.n_obs_max, N)

        # Parameter vector
        p = np.zeros(self.n_params)
        p[0:ns] = initial_state

        center = ref_data['center_points']
        left = ref_data['left_points']
        right = ref_data['right_points']
        ref_v_arr = ref_data['ref_v']
        ref_dx = ref_data['ref_dx']
        ref_dy = ref_data['ref_dy']

        BIG = 1e6

        for k in range(N):
            ref_idx = ns + self.STEP_BLOCK * k
            idx = min(k, len(center) - 1)

            p[ref_idx] = center[idx, 0]
            p[ref_idx + 1] = center[idx, 1]
            p[ref_idx + 2] = ref_dx[idx]
            p[ref_idx + 3] = ref_dy[idx]
            p[ref_idx + 4] = ref_v_arr[idx]

            delta_bx = right[idx, 0] - left[idx, 0]
            delta_by = right[idx, 1] - left[idx, 1]
            p[ref_idx + 5] = -delta_bx
            p[ref_idx + 6] = delta_by

            val_r = -delta_bx * right[idx, 0] - delta_by * right[idx, 1]
            val_l = -delta_bx * left[idx, 0] - delta_by * left[idx, 1]
            lo = min(val_r, val_l)
            hi = max(val_r, val_l)

            base = ns + k * (ns + 2)
            self.lbg[base + ns, 0] = -BIG
            self.ubg[base + ns, 0] = hi
            self.lbg[base + ns + 1, 0] = lo
            self.ubg[base + ns + 1, 0] = BIG

            # ### HJ : Phase 3.5 — pack obstacle slots for step k.
            obs_base = ref_idx + self.REF_BLOCK
            for o in range(self.n_obs_max):
                ob = obs_base + self.OBS_BLOCK * o
                p[ob] = obs[o, k, 0]
                p[ob + 1] = obs[o, k, 1]
                p[ob + 2] = obs[o, k, 2]

        # Warm start
        if not self.warm:
            self._construct_warm_start(initial_state, ref_data)

        x_init = vertcat(
            reshape(self.X0.T, ns * (N + 1), 1),
            reshape(self.u0.T, nc * N, 1),
            np.zeros((N, 1)),
        )

        # ### HJ : capture solver ref locally so a concurrent rebuild_nlp()
        # (dyn_reconfigure callback thread) mid-solve doesn't swap the
        # attribute between the evaluate and stats() calls. Without this,
        # stats() hits the newly-built solver (not yet numerically evaluated)
        # and raises "No stats available: Function 'solver' not set up".
        solver_nlp = self.solver_nlp
        sol = solver_nlp(x0=x_init, lbx=self.lbx, ubx=self.ubx,
                         lbg=self.lbg, ubg=self.ubg, p=p)

        stats = solver_nlp.stats()
        success = stats['success']
        self.last_return_status = stats.get('return_status', 'unknown')
        self.last_iter_count = stats.get('iter_count', -1)

        state_end = ns * (N + 1)
        ctrl_end = state_end + nc * N
        x_sol = reshape(sol['x'][:state_end], ns, N + 1).T
        u_sol = reshape(sol['x'][state_end:ctrl_end], nc, N).T
        slack_sol = np.array(sol['x'][ctrl_end:ctrl_end + N].full()).flatten()
        self.last_slack_max = float(np.max(slack_sol)) if slack_sol.size else 0.0

        speed = float(u_sol[0, 0])
        steering = float(u_sol[0, 1])
        trajectory = np.array(x_sol.full())
        self.last_u_sol = np.array(u_sol.full())

        self.X0 = np.array(vertcat(x_sol[1:, :], x_sol[-1, :]).full())
        self.u0 = np.array(vertcat(u_sol[1:, :], u_sol[-1, :]).full())

        if not self.warm:
            self.warm = True

        return speed, steering, trajectory, success

    def reset_warm_start(self):
        """Drop stored warm-start so next solve uses a fresh propagation."""
        self.warm = False

    def _construct_warm_start(self, initial_state, ref_data):
        """
        Seed X0, u0 using the sliced reference directly — each step anchors
        to the k-th center point with heading from the ref tangent.
        """
        N = self.N
        self.X0 = np.zeros((N + 1, self.n_states))
        self.u0 = np.zeros((N, self.n_controls))

        center = ref_data['center_points']
        ref_dx = ref_data['ref_dx']
        ref_dy = ref_data['ref_dy']
        ref_v_arr = ref_data['ref_v']
        n_ref = len(center)

        self.X0[0, :] = initial_state

        for k in range(N):
            idx = min(k + 1, n_ref - 1)
            x_next = float(center[idx, 0])
            y_next = float(center[idx, 1])
            psi_next = float(np.arctan2(ref_dy[idx], ref_dx[idx]))

            init_speed = max(float(ref_v_arr[idx]) * 0.5, 1.0)

            dpsi = np.arctan2(
                np.sin(psi_next - self.X0[k, 2]),
                np.cos(psi_next - self.X0[k, 2])
            )
            steer = np.clip(
                np.arctan(dpsi * self.L / max(init_speed * self.dT, 0.01)),
                -self.theta_max * 0.5, self.theta_max * 0.5
            )

            self.X0[k + 1, :] = [x_next, y_next, psi_next]
            self.u0[k, :] = [init_speed, steer]
