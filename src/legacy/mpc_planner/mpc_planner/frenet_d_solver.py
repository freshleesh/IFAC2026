"""
### HJ : Frenet-d MPC Solver — shape-first lateral planner (Gaussian obstacle).

Decision: n_k (k=0..N) perturbation from raceline on a fixed s-grid.
Obstacle cost is the **Gaussian soft-repulsive** form (same family as the
xy solver's Phase-3.5 cost, re-expressed in Frenet coords). Hard corridor
box guarantees the trajectory never leaves d_left/d_right by construction.

Earlier attempts used a non-convex `max(0, 1-d²)²` hinge — it caused
warm-start local-min lockup where the solver would stick to "hug wall and
track the obstacle's n" basins. Gaussian has a single convex peak per
obstacle, smooth gradient everywhere, and reliably pulls n away.

Cost:
  - contour:    w_contour · Σ α_c(k) · n_k²
  - smoothness: w_dv · Σ (n_{k+1} - n_k)²                (slope)
                w_dsteering · Σ (n_{k+2} - 2 n_{k+1} + n_k)²   (curvature)
  - obstacle:   w_obs · Σ_k Σ_o exp(-((Δs)² + (Δn)²) / (2σ²))
                Δs = ref_s[k] - s_o[k], Δn = n_k - n_o[k]
                (flat-Frenet distance; σ = obstacle_sigma)
  - wall buf:   w_wall · Σ [ max(0, n_k - n_hi)² + max(0, n_lo - n_k)² ]
                n_hi = d_left  - inflation - wall_safe
                n_lo = -(d_right - inflation - wall_safe)

Hard constraints (no slack — corridor never violated):
  - init:   n_0 = n_ego
  - corridor: -(d_right - inflation) ≤ n_k ≤ (d_left - inflation)

Run-time mutators:
  - rebuild_nlp(**kwargs) for cost weights / sigma / wall params / iter.

API parity with MPCCSolver: solve() takes ref_data + obstacles (Frenet,
same (n_obs_max, N+1, 3) = [s_o, n_o, w_obs] layout).
"""

import numpy as np
from casadi import MX, vertcat, exp, fmax, nlpsol


class FrenetDSolver:

    # Per-step parameter layout:
    # [s_k, x_r, y_r, nx, ny, d_left, d_right,
    #  s_o_0, n_o_0, w_o_0, s_o_1, n_o_1, w_o_1, ...]
    REF_BLOCK = 7
    OBS_BLOCK = 3
    FAR_SN = 1.0e4

    def __init__(self, params):
        self.N = int(params['N'])
        self.dT = float(params['dT'])
        self.L = float(params['vehicle_L'])

        # Box bounds (kept for API parity; velocity/steering are derived,
        # not decision vars, so these mostly don't affect the solve itself).
        self.v_max = float(params['max_speed'])
        self.v_min = float(params.get('min_speed', 0.0))
        self.theta_max = float(params['max_steering'])

        # Cost weights
        self.w_contour = float(params.get('w_contour', 2.0))
        self.w_lag = float(params.get('w_lag', 0.0))           # unused in frenet
        self.w_velocity = float(params.get('w_velocity', 0.0))  # unused
        self.v_bias_max = float(params.get('v_bias_max', 10.0))
        # Repurposed: w_dv -> w_d1 (slope penalty), w_dsteering -> w_d2
        # (curvature penalty). Naming kept so YAML/dynparam unchanged.
        self.w_dv = float(params.get('w_dv', 5.0))
        self.w_dsteering = float(params.get('w_dsteering', 20.0))

        self.contour_ramp_start = float(params.get('contour_ramp_start', 1.0))
        self.lag_ramp_start = float(params.get('lag_ramp_start', 1.0))  # unused

        self.inflation = float(params.get('boundary_inflation', 0.05))
        self.w_slack = float(params.get('w_slack', 1000.0))  # unused (hard box)

        self.n_obs_max = int(params.get('n_obs_max', 2))
        # ### HJ : Gaussian soft-repulsive cost in (Δs, Δn) Frenet space.
        # Same formula family as the xy solver but with Frenet coords. σ
        # controls both longitudinal and lateral falloff — larger σ means
        # the solver starts bending n_k earlier. For 1v1 racing a σ ~ 0.5m
        # is a reasonable starting point (vehicle-length scale).
        self.obstacle_sigma = float(params.get('obstacle_sigma', 0.5))

        # ### HJ : wall buffer soft cost — pushes n_k away from corridor
        # edges. Quadratic hinge (one-sided, so still convex):
        #   w_wall · max(0, n_k - (d_left  - infl - wall_safe))²
        # + w_wall · max(0, -(d_right - infl - wall_safe) - n_k)²
        # Zero inside the "safe band", quadratic growth close to the wall.
        # This complements the Gaussian (which only pushes away from the
        # obstacle, not away from the wall) and prevents the "hug the
        # corridor edge" behavior we saw before.
        self.w_wall = float(params.get('w_wall', 0.0))
        self.wall_safe = float(params.get('wall_safe', 0.15))

        self.kinematic_v_eps = float(params.get('kinematic_v_eps', 0.05))
        self.w_steer_reg = float(params.get('w_steer_reg', 1.0e-3))

        self.ipopt_max_iter = int(params.get('ipopt_max_iter', 200))
        self.ipopt_print_level = int(params.get('ipopt_print_level', 0))

        # Dimensions — internal. Trajectory output mimics xy solver (3 states).
        self.n_states = 3
        self.n_controls = 2

        self.STEP_BLOCK = self.REF_BLOCK + self.OBS_BLOCK * self.n_obs_max

        self.solver_nlp = None
        self.ready = False

        # Warm start (N+1 vector of n)
        self.n_warm = None
        self.warm = False

        self.last_return_status = 'unset'
        self.last_iter_count = -1
        self.last_slack_max = 0.0
        self.last_u_sol = None

    def setup(self):
        self._build_nlp()
        self.ready = True

    # -------------------------------------------------- rebuild / hot update
    def rebuild_nlp(self, **kwargs):
        for key in ('w_contour', 'w_lag', 'w_velocity', 'v_bias_max',
                    'w_dv', 'w_dsteering', 'w_slack',
                    'obstacle_sigma', 'w_wall', 'wall_safe',
                    'ipopt_max_iter',
                    'kinematic_v_eps', 'w_steer_reg',
                    'contour_ramp_start', 'lag_ramp_start'):
            if key in kwargs and kwargs[key] is not None:
                setattr(self, key, type(getattr(self, key))(kwargs[key]))
        self.warm = False
        self._build_nlp()
        self.ready = True

    def update_box_bounds(self, v_min=None, v_max=None, theta_max=None):
        # API parity only; no effect on the frenet NLP.
        if v_min is not None:
            self.v_min = float(v_min)
        if v_max is not None:
            self.v_max = float(v_max)
        if theta_max is not None:
            self.theta_max = float(theta_max)

    def update_inflation(self, inflation):
        self.inflation = float(inflation)

    def reset_warm_start(self):
        self.warm = False
        self.n_warm = None

    # ----------------------------------------------------------- build NLP
    def _build_nlp(self):
        N = self.N
        # Decision variable: n_0 .. n_N (N+1)
        n_var = MX.sym('n', N + 1)

        # Parameters:
        # p[0] = n_ego (initial offset)
        # For each k in 0..N: [s_k, x_r, y_r, nx, ny, d_left, d_right]
        # Per-step per-obstacle: [s_o, n_o, w_o]
        n_params = 1 + (N + 1) * self.STEP_BLOCK
        P = MX.sym('P', n_params)

        n_ego = P[0]

        # ### HJ : obstacle cost — Gaussian in Frenet (Δs, Δn).
        #   cost = Σ_o w_o · exp(-(Δs² + Δn²) / (2σ²))
        # Single smooth peak per obstacle, convex locally, gradient smooth
        # everywhere → no warm-start local-min lockup.
        sigma = max(self.obstacle_sigma, 1.0e-3)
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

        # Wall buffer "safe band" bounds, per step (derived from d_left/d_right
        # parameters — kept symbolic so dyn_reconfigure of wall_safe / infl
        # takes effect without rebuilding NLP … no, `inflation` and
        # `wall_safe` are Python floats bound into the graph at build time, so
        # rebuild_nlp triggers a rebuild when they change).
        w_wall = self.w_wall
        wall_safe = self.wall_safe

        obj = 0
        g_list = []

        # Initial constraint: n_0 = n_ego
        g_list.append(n_var[0] - n_ego)

        # Per-step cost + corridor constraints
        for k in range(N + 1):
            base = 1 + k * self.STEP_BLOCK
            s_k = P[base + 0]
            d_left_k = P[base + 5]
            d_right_k = P[base + 6]

            nk = n_var[k]

            # Contour (pull to raceline). Ramp as in xy solver.
            if N > 1:
                t_ramp = k / float(N)
            else:
                t_ramp = 1.0
            alpha_c = self.contour_ramp_start + (1.0 - self.contour_ramp_start) * t_ramp
            obj += self.w_contour * alpha_c * (nk * nk)

            # Corridor hard box (unchanged).
            g_list.append(nk - (d_left_k - self.inflation))
            g_list.append(-nk - (d_right_k - self.inflation))

            # Wall buffer — one-sided quadratic hinge (convex).
            if w_wall > 0.0:
                n_hi = d_left_k - self.inflation - wall_safe
                n_lo_neg = d_right_k - self.inflation - wall_safe  # |n_lo|
                over_hi = fmax(0.0, nk - n_hi)
                over_lo = fmax(0.0, -nk - n_lo_neg)
                obj += w_wall * (over_hi * over_hi + over_lo * over_lo)

            # Obstacle cost — Gaussian in Frenet (Δs, Δn).
            obs_base = base + self.REF_BLOCK
            for o in range(self.n_obs_max):
                ob = obs_base + self.OBS_BLOCK * o
                s_o = P[ob + 0]
                n_o = P[ob + 1]
                w_o = P[ob + 2]
                ds = s_k - s_o
                dn = nk - n_o
                obj += w_o * exp(-(ds * ds + dn * dn) * inv_two_sigma2)

        # Smoothness: first difference (w_dv) and second difference (w_dsteering)
        for k in range(N):
            d1 = n_var[k + 1] - n_var[k]
            obj += self.w_dv * d1 * d1
        for k in range(N - 1):
            d2 = n_var[k + 2] - 2 * n_var[k + 1] + n_var[k]
            obj += self.w_dsteering * d2 * d2

        g = vertcat(*g_list)

        # Constraint bounds
        # - init: equality → lbg=ubg=0
        # - 2 corridor inequalities per step, both ≤ 0 → lbg=-inf, ubg=0
        n_g = 1 + 2 * (N + 1)
        self.lbg = np.zeros((n_g, 1))
        self.ubg = np.zeros((n_g, 1))
        # init equality row 0: already zeros.
        # rows 1..end: inequalities ≤ 0
        self.lbg[1:, 0] = -np.inf
        self.ubg[1:, 0] = 0.0

        # Decision bounds on n — keep a large box; corridor enforced via g.
        n_vars = N + 1
        self.lbx = -10.0 * np.ones((n_vars, 1))
        self.ubx = +10.0 * np.ones((n_vars, 1))

        nlp = {'f': obj, 'x': n_var, 'g': g, 'p': P}
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

    # --------------------------------------------------------- obstacle util
    @staticmethod
    def _coerce_obstacles(obstacles, n_obs_max, N):
        """Return (n_obs_max, N, 3) ndarray with [s_o, n_o, w_obs]. Unused
        slots get far coords and w=0."""
        if obstacles is None:
            out = np.zeros((n_obs_max, N, 3), dtype=np.float64)
            out[:, :, 0] = FrenetDSolver.FAR_SN
            return out
        arr = np.asarray(obstacles, dtype=np.float64)
        if arr.shape != (n_obs_max, N, 3):
            raise ValueError(
                'obstacles shape mismatch: got %s, expected (%d, %d, 3)' %
                (arr.shape, n_obs_max, N)
            )
        return arr

    # ------------------------------------------------------------------ solve
    def solve(self, initial_state, ref_data, obstacles=None):
        """Solve for n_0..n_N and return (speed, steering, trajectory, success).

        `initial_state` is (x, y, psi) but we use only the Frenet projection
        of (x, y) via `ref_data['n_ego']`, which the caller computes.

        Trajectory output is (N+1, 3) of (x, y, psi) reconstructed from
        raceline + n_k for caller's `fill_wpnt`.
        """
        if not self.ready:
            return 0.0, 0.0, None, False

        N = self.N

        center = ref_data['center_points']       # (N+1, 2)
        ref_dx = ref_data['ref_dx']              # (N+1,)
        ref_dy = ref_data['ref_dy']              # (N+1,)
        ref_s = ref_data['ref_s']                # (N+1,)
        d_left_arr = ref_data['d_left_arr']      # (N+1,) after inflation applied by caller? We subtract below.
        d_right_arr = ref_data['d_right_arr']    # (N+1,)
        ref_v_arr = ref_data['ref_v']            # (N+1,)
        n_ego = float(ref_data['n_ego'])

        obs = self._coerce_obstacles(obstacles, self.n_obs_max, N + 1)

        # Parameter vector
        p = np.zeros(self.n_params)
        p[0] = n_ego
        for k in range(N + 1):
            base = 1 + k * self.STEP_BLOCK
            p[base + 0] = ref_s[k]
            p[base + 1] = center[k, 0]
            p[base + 2] = center[k, 1]
            # Left-normal from tangent
            p[base + 3] = -ref_dy[k]
            p[base + 4] = +ref_dx[k]
            p[base + 5] = d_left_arr[k]
            p[base + 6] = d_right_arr[k]
            obs_base = base + self.REF_BLOCK
            for o in range(self.n_obs_max):
                ob = obs_base + self.OBS_BLOCK * o
                p[ob + 0] = obs[o, k, 0]
                p[ob + 1] = obs[o, k, 1]
                p[ob + 2] = obs[o, k, 2]

        # Warm start
        if not self.warm or self.n_warm is None:
            x_init = np.zeros(N + 1)
            x_init[0] = n_ego
        else:
            # Shift previous solution one step forward
            x_init = np.concatenate([self.n_warm[1:], self.n_warm[-1:]])
            x_init[0] = n_ego

        solver_nlp = self.solver_nlp
        sol = solver_nlp(x0=x_init.reshape(-1, 1),
                         lbx=self.lbx, ubx=self.ubx,
                         lbg=self.lbg, ubg=self.ubg, p=p)

        stats = solver_nlp.stats()
        success = bool(stats['success'])
        self.last_return_status = stats.get('return_status', 'unknown')
        self.last_iter_count = stats.get('iter_count', -1)

        n_sol = np.array(sol['x'].full()).flatten()

        # Reconstruct (x, y, psi) trajectory for fill_wpnt compatibility.
        traj = np.zeros((N + 1, 3), dtype=np.float64)
        traj[:, 0] = center[:, 0] + n_sol * (-ref_dy)
        traj[:, 1] = center[:, 1] + n_sol * ref_dx
        # psi_k ≈ raceline tangent + arctan(dn/ds). Use finite-difference.
        for k in range(N + 1):
            if k < N:
                ds = max(ref_s[k + 1] - ref_s[k], 1e-3)
                dn = n_sol[k + 1] - n_sol[k]
                dpsi = np.arctan(dn / ds)
            else:
                dpsi = 0.0
            tangent_psi = float(np.arctan2(ref_dy[k], ref_dx[k]))
            traj[k, 2] = tangent_psi + dpsi

        # Speed / steering output — speed = ref_v (vel planner overrides),
        # steering synthesized so the node's Wpnt ax_mps2 calc is sane.
        speed = float(ref_v_arr[0])
        # Pseudo-steering from κ ≈ d²n/ds² (small-angle). MPC doesn't command
        # this; it's only used for diagnostics and ax estimation.
        if N >= 2:
            ds = max(ref_s[1] - ref_s[0], 1e-3)
            d1 = n_sol[1] - n_sol[0]
            d2 = n_sol[2] - 2 * n_sol[1] + n_sol[0]
            kappa = d2 / (ds * ds)
            steering = float(np.arctan(self.L * kappa))
        else:
            steering = 0.0

        # Mimic u_sol for node's ax_mps2 pipeline
        self.last_u_sol = np.column_stack([
            ref_v_arr[:N], np.zeros(N)
        ])
        self.last_slack_max = 0.0  # hard corridor → no slack
        self.n_warm = n_sol
        self.warm = True

        return speed, steering, traj, success

    def _construct_warm_start(self, *_args, **_kw):
        # API parity — warm start handled inside solve().
        self.warm = False
