#!/usr/bin/env python3
"""
### HJ : Phase 2 3D lift — raceline-base, no Track3D.

MPC solves a 2D kinematic bicycle in Cartesian and emits (x, y, psi) per
horizon step. State-machine / controller / SQP all consume /global_waypoints
whose `s_m` axis is the *raceline* arc length (see
planner/3d_gb_optimizer/global_line/global_racing_line/export_global_waypoints.py:12
— "s_m = 레이싱라인 실제 arc length, d_m = 0"), so MPC output must be
raceline-base too.

This lifter intentionally does NOT use Track3D:
  - Track3D.s is centerline-base (loaded from *_3d_smoothed.csv; see
    planner/3d_sampling_based_planner/src/track3D.py:42). Its `s`-axis does
    NOT align with /global_waypoints.
  - Track3D gives sn2cartesian but no cartesian2sn, so reverse projection
    would require a separate centerline frenet converter.
  - The sampling planner publishes centerline-base `w.s_m`
    (sampling_planner_state_node.py:894) — we deliberately diverge and
    stay raceline-base for state-machine / controller consistency.

What this lifter provides: arc-length linear interpolation on
/global_waypoints for every Wpnt field. The 3D attributes (z_m, psi_rad,
kappa_radpm, mu_rad, d_left, d_right) are the raceline-sampled values from
global_planner's smoothed_opt export — raceline-base by construction.

Lateral-offset precision note: because /global_waypoints only carries the
raceline (d=0) samples, attributes at ±n lateral offsets (overtake line)
lose banking detail. For Phase 2 this is acceptable (formal z correction
is a Phase 2.1 option once measurements demand it — can fall back to a
centerline Track3D + frenet_converter(centerline) dual lookup). Phase 2
keeps it simple: raceline-base interpolation only.
"""

import numpy as np


class MPCRacelineLifter:
    """Linear-interpolate /global_waypoints (raceline-base) fields along s."""

    def __init__(self, g_s, g_x, g_y, g_z, g_psi, g_kappa, g_mu,
                 g_dleft, g_dright, g_vx, track_length):
        """Build the lifter from cached /global_waypoints arrays.

        Parameters
        ----------
        g_*        : np.ndarray, shape (M,), values at raceline s-axis samples.
        track_length : float, closed-loop length (last raceline s_m).

        Note: caller passes the already-cached arrays from the node's
        _global_wpnts_cb. The lifter keeps references (no copy) because
        /global_waypoints is latched-once in this stack.
        """
        self.g_s = np.asarray(g_s, dtype=np.float64)
        self.g_x = np.asarray(g_x, dtype=np.float64)
        self.g_y = np.asarray(g_y, dtype=np.float64)
        self.g_z = np.asarray(g_z, dtype=np.float64)
        self.g_psi = np.asarray(g_psi, dtype=np.float64)
        self.g_kappa = np.asarray(g_kappa, dtype=np.float64)
        self.g_mu = np.asarray(g_mu, dtype=np.float64)
        self.g_dleft = np.asarray(g_dleft, dtype=np.float64)
        self.g_dright = np.asarray(g_dright, dtype=np.float64)
        self.g_vx = np.asarray(g_vx, dtype=np.float64)
        self.L = float(track_length)

        # Unwrap psi so np.interp does not cross ±pi (we re-wrap at output).
        self._g_psi_unwrapped = np.unwrap(self.g_psi)

    # ---------------- raceline projection ----------------
    def project_xy_to_sn(self, x, y, idx_hint=None):
        """2D Cartesian (x, y) → raceline-base (s, n).

        Uses the nearest raceline segment, then projects onto that segment
        (orthogonal lateral distance). `idx_hint` (segment anchor index from
        a previous search) bounds the local-window argmin; skipping it is
        fine on first call.
        """
        if idx_hint is None:
            idx = self._nearest_idx(x, y)
        else:
            idx = int(idx_hint)

        # Use the two segments centered on idx — pick whichever foot of
        # perpendicular falls inside [0, 1] with smallest |n|.
        s_a, s_b, n_signed = self._segment_project(x, y, idx)
        return float(s_a + (s_b - s_a) * self._last_t), float(n_signed)

    def _nearest_idx(self, x, y):
        d2 = (self.g_x - x) ** 2 + (self.g_y - y) ** 2
        return int(np.argmin(d2))

    def _segment_project(self, x, y, idx):
        """Project (x, y) onto the better of the two segments around idx.

        Returns (s_a, s_b, n_signed); stores t∈[0,1] in self._last_t for
        project_xy_to_sn to reconstruct s.
        """
        M = len(self.g_s)
        candidates = []
        for j_prev, j_next in ((idx - 1, idx), (idx, idx + 1)):
            j_prev %= M
            j_next %= M
            ax, ay = float(self.g_x[j_prev]), float(self.g_y[j_prev])
            bx, by = float(self.g_x[j_next]), float(self.g_y[j_next])
            ex, ey = bx - ax, by - ay
            L2 = ex * ex + ey * ey
            if L2 < 1e-12:
                continue
            t = ((x - ax) * ex + (y - ay) * ey) / L2
            t_clamped = max(0.0, min(1.0, t))
            fx = ax + ex * t_clamped
            fy = ay + ey * t_clamped
            dist = float(np.hypot(x - fx, y - fy))
            # Signed n: left of tangent = +, right = -.
            # tangent = (ex, ey) normalized; normal (left) = (-ey, ex).
            Lseg = float(np.sqrt(L2))
            nx, ny = -ey / Lseg, ex / Lseg
            n_signed = (x - fx) * nx + (y - fy) * ny
            s_a = float(self.g_s[j_prev])
            s_b = float(self.g_s[j_next])
            # Handle lap wrap (s_b < s_a at the seam).
            if s_b < s_a - 0.5 * self.L:
                s_b += self.L
            candidates.append((dist, s_a, s_b, t_clamped, n_signed))

        if not candidates:
            # degenerate — fall back to nearest sample
            self._last_t = 0.0
            return float(self.g_s[idx]), float(self.g_s[idx]), 0.0

        candidates.sort(key=lambda c: c[0])
        _, s_a, s_b, t, n_signed = candidates[0]
        self._last_t = t
        return s_a, s_b, n_signed

    # ---------------- field interpolation ----------------
    def _interp(self, s, arr):
        """np.interp with lap-wrap. s is raceline arc length."""
        s_mod = float(s) % self.L
        return float(np.interp(s_mod, self.g_s, arr))

    def _interp_psi(self, s):
        # Use unwrapped, then wrap back to (-pi, pi].
        p = self._interp(s, self._g_psi_unwrapped)
        return float(np.arctan2(np.sin(p), np.cos(p)))

    # ---------------- inverse: (s, n) → (x, y) ----------------
    def sn_to_xy(self, s, n):
        """### HJ : Phase 3.5 — raceline-base Frenet → Cartesian.

        Uses raceline point + left-normal at that s to place (s, n) in the
        world frame. Left-normal sign matches project_xy_to_sn (n > 0 = left
        of raceline). No banking/3D — planar approximation, sufficient for
        feeding MPC obstacle cost where only (x, y) matters.
        """
        x_r = self._interp(s, self.g_x)
        y_r = self._interp(s, self.g_y)
        psi = self._interp_psi(s)
        # Left-normal = rotate tangent (cos ψ, sin ψ) by +90°.
        nx = -np.sin(psi)
        ny = np.cos(psi)
        return float(x_r + n * nx), float(y_r + n * ny)

    # ---------------- public API ----------------
    def fill_wpnt_from_s(self, s_ref, n_ref, x, y, psi_mpc,
                         vx_mpc, ax_mpc, psi_blend=0.5):
        """### HJ : v3 — s-direct variant (no xy round-trip).

        The 2D project_xy_to_sn used by fill_wpnt() fails at overpass
        crossings in 3D tracks: both floors share the same (x, y), so
        nearest-xy returns the wrong floor's s and z flips between the
        ground and the bridge every tick. The MPC solver already holds
        s on the correct floor (frenet state), so we pass s directly
        and skip the inverse projection entirely.

        Used by _publish_outputs when the solver is frenet_kin / frenet_d.
        """
        s = float(s_ref) % self.L
        n = float(n_ref)

        z = self._interp(s, self.g_z)
        kappa = self._interp(s, self.g_kappa)
        mu = self._interp(s, self.g_mu)
        dleft = self._interp(s, self.g_dleft)
        dright = self._interp(s, self.g_dright)
        psi_track = self._interp_psi(s)

        dpsi = np.arctan2(np.sin(psi_mpc - psi_track),
                          np.cos(psi_mpc - psi_track))
        psi_out = psi_track + psi_blend * dpsi

        return {
            's_m': float(s),
            'd_m': float(n),
            'x_m': float(x),
            'y_m': float(y),
            'z_m': float(z),
            'psi_rad': float(np.arctan2(np.sin(psi_out),
                                        np.cos(psi_out))),
            'kappa_radpm': float(kappa),
            'vx_mps': float(vx_mpc),
            'ax_mps2': float(ax_mpc),
            'mu_rad': float(mu),
            'd_left': float(dleft),
            'd_right': float(dright),
        }

    def fill_wpnt(self, x, y, psi_mpc, vx_mpc, ax_mpc, psi_blend=0.5, idx_hint=None):
        """Return a dict of Wpnt fields at (x, y).

        Parameters
        ----------
        x, y         : Cartesian MPC step.
        psi_mpc      : heading predicted by the MPC solver.
        vx_mpc       : speed command from the solver u_sol[k, 0].
        ax_mpc       : acceleration estimate (u_sol[k+1, 0] - u_sol[k, 0]) / dT.
        psi_blend    : 0..1 weight for MPC heading vs raceline tangent. 0.5
                       is the Phase 2 default — favor the solver but stay
                       anchored to the tangent so small solver-side wiggles
                       don't distort the controller's lookahead.
        idx_hint     : optional prior nearest index for locality.

        Returns dict ready to splat into f110_msgs/Wpnt.
        """
        s, n = self.project_xy_to_sn(x, y, idx_hint=idx_hint)

        z = self._interp(s, self.g_z)
        kappa = self._interp(s, self.g_kappa)
        mu = self._interp(s, self.g_mu)
        dleft = self._interp(s, self.g_dleft)
        dright = self._interp(s, self.g_dright)
        psi_track = self._interp_psi(s)

        dpsi = np.arctan2(np.sin(psi_mpc - psi_track), np.cos(psi_mpc - psi_track))
        psi_out = psi_track + psi_blend * dpsi

        return {
            's_m': float(s % self.L),
            'd_m': float(n),
            'x_m': float(x),
            'y_m': float(y),
            'z_m': float(z),
            'psi_rad': float(np.arctan2(np.sin(psi_out), np.cos(psi_out))),
            'kappa_radpm': float(kappa),
            'vx_mps': float(vx_mpc),
            'ax_mps2': float(ax_mpc),
            'mu_rad': float(mu),
            'd_left': float(dleft),
            'd_right': float(dright),
        }
