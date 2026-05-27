### HJ : Track3D_raceline_based — scaffold for the RACELINE-framed Track3D,
###      to be consumed by local_raceline_mux_node_HJ_raceline_based.py.
###
###      Intended difference from track3D_local (centerline-framed):
###        * Source of truth: global_waypoints.json raceline wpnts (not
###          smoothed.csv centerline). Read {x_m, y_m, z_m, psi_rad,
###          kappa_radpm, d_left, d_right, mu_rad} from JSON.
###        * self.s        ← raceline arc length (computed from xyz)
###        * self.theta    ← psi_rad (raceline heading)
###        * self.Omega_z  ← kappa_radpm (raceline curvature)
###        * self.w_tr_left/right ← d_left / d_right (wall distances from
###          raceline, already signed consistently in JSON)
###        * self.mu/phi   ← mu_rad from JSON; phi_rad interpolated from
###          centerline CSV at matching x,y,z (or zero if flat).
###        * Omega_x/y, dOmega_x/y/z: either interpolated from centerline
###          CSV at matching arc-length or numerically differentiated
###          from raceline geometry. TBD during refactor.
###
###      Current state: this file is a byte-identical copy of track3D_local.
###      The rebuild-from-raceline-JSON refactor is WIP — nothing is wired
###      up yet. Importers should not expect the raceline frame until the
###      refactor lands.
###
### Keeps track3D_lite and track3D_local untouched.

import json
import os

import numpy as np
import casadi as ca

g_earth = 9.81


class Track3D:
    """### HJ : raceline-framed Track3D.

    Constructor accepts two independent paths:
      * raceline_json_path — REQUIRED. global_waypoints.json; we pull the
        iqp raceline wpnts. Supplies (x, y, z, s, psi, kappa, d_left,
        d_right, mu) on the raceline arc-length grid.
      * smoothed_csv_path  — OPTIONAL. smoothed centerline CSV; we sample
        mu/phi/Omega_x/Omega_y at each raceline point by nearest-neighbor
        lookup on cartesian (x, y, z). Needed for proper 3D banking terms
        in calc_apparent_accelerations. If absent, phi/Omega_x/Omega_y are
        set to zero (flat-track approximation).

    Backward-compat: we keep the single-arg __init__(path=...) signature
    that track3D_local used — if 'path' looks like a .json file, we treat
    it as raceline_json_path; otherwise we fall back to the legacy CSV
    loader (identical to track3D_local's behaviour) so this module can
    still be used as a drop-in replacement for the centerline variant if
    the caller passes a smoothed csv.
    """

    def __init__(self, path=None,
                 raceline_json_path=None,
                 smoothed_csv_path=None):
        # resolve which mode: raceline-json or legacy-csv
        if raceline_json_path is None and path is not None and path.endswith('.json'):
            raceline_json_path = path
            path = None
        if path is not None and smoothed_csv_path is None and path.endswith('.csv'):
            smoothed_csv_path = path

        self.__raceline_json_path = raceline_json_path
        self.__smoothed_csv_path = smoothed_csv_path

        if self.__raceline_json_path is not None:
            self.__mode = 'raceline'
            self.__load_raceline_json()
        elif self.__smoothed_csv_path is not None:
            # legacy CSV mode — byte-identical to track3D_local
            self.__mode = 'csv'
            raw = np.genfromtxt(self.__smoothed_csv_path, delimiter=',', names=True)
            self.__column_names = raw.dtype.names
            self.__data = {name: raw[name] for name in self.__column_names}
        else:
            self.__mode = None
            self.__data = None
            self.track_locked = False
            return

        self.track_locked = True

    def __load_raceline_json(self):
        with open(self.__raceline_json_path, 'r') as f:
            data = json.load(f)
        wpnts = data['global_traj_wpnts_iqp']['wpnts']
        rl_x = np.array([w['x_m'] for w in wpnts], dtype=np.float64)
        rl_y = np.array([w['y_m'] for w in wpnts], dtype=np.float64)
        rl_z = np.array([w['z_m'] for w in wpnts], dtype=np.float64)
        rl_s = np.array([w['s_m'] for w in wpnts], dtype=np.float64)
        rl_psi = np.array([w['psi_rad'] for w in wpnts], dtype=np.float64)
        rl_kappa = np.array([w['kappa_radpm'] for w in wpnts], dtype=np.float64)
        rl_dleft = np.array([w['d_left'] for w in wpnts], dtype=np.float64)
        rl_dright = np.array([w['d_right'] for w in wpnts], dtype=np.float64)
        rl_mu = np.array([w.get('mu_rad', 0.0) for w in wpnts], dtype=np.float64)
        self.__rl = dict(
            x=rl_x, y=rl_y, z=rl_z, s=rl_s, psi=rl_psi, kappa=rl_kappa,
            d_left=rl_dleft, d_right=rl_dright, mu=rl_mu,
        )

    def __enrich_from_smoothed(self, rl_x, rl_y, rl_z):
        """Use smoothed centerline CSV to sample mu, phi, Omega_x, Omega_y
        at each raceline point via nearest-neighbour cartesian lookup.
        Returns four arrays of length len(rl_x) or None if no CSV."""
        if self.__smoothed_csv_path is None or not os.path.exists(self.__smoothed_csv_path):
            return None
        raw = np.genfromtxt(self.__smoothed_csv_path, delimiter=',', names=True)
        cl_x = raw['x_m']
        cl_y = raw['y_m']
        cl_z = raw['z_m']
        cl_mu = raw['mu_rad']
        cl_phi = raw['phi_rad']
        cl_Ox = raw['omega_x_radpm']
        cl_Oy = raw['omega_y_radpm']
        N = len(rl_x)
        mu = np.zeros(N)
        phi = np.zeros(N)
        Ox = np.zeros(N)
        Oy = np.zeros(N)
        for i in range(N):
            d2 = (cl_x - rl_x[i]) ** 2 + (cl_y - rl_y[i]) ** 2 + (cl_z - rl_z[i]) ** 2
            j = int(np.argmin(d2))
            mu[i] = cl_mu[j]
            phi[i] = cl_phi[j]
            Ox[i] = cl_Ox[j]
            Oy[i] = cl_Oy[j]
        return mu, phi, Ox, Oy

    @property
    def track_locked(self):
        return self.__track_locked

    @track_locked.setter
    def track_locked(self, value):
        if value:
            self.lock_track_data()
        self.__track_locked = value

    def lock_track_data(self):
        if self.__mode == 'raceline':
            self.__lock_from_raceline()
        else:
            self.__lock_from_csv()

    def __lock_from_raceline(self):
        rl = self.__rl
        self.s = rl['s']
        self.ds = float(np.mean(np.diff(self.s)))
        self.x = rl['x']
        self.y = rl['y']
        self.z = rl['z']
        self.theta = rl['psi']
        self.Omega_z = rl['kappa']
        ### HJ : JSON stores d_left/d_right as POSITIVE magnitudes; Track3D's
        #        convention is signed (right = negative n). Convert here.
        self.w_tr_left = rl['d_left']
        self.w_tr_right = -rl['d_right']

        # 3D banking terms — sample from centerline CSV if available,
        # otherwise fall back to flat track.
        enriched = self.__enrich_from_smoothed(rl['x'], rl['y'], rl['z'])
        if enriched is not None:
            mu, phi, Ox, Oy = enriched
            self.mu = mu
            self.phi = phi
            self.Omega_x = Ox
            self.Omega_y = Oy
        else:
            self.mu = rl['mu']
            self.phi = np.zeros_like(self.s)
            self.Omega_x = np.zeros_like(self.s)
            self.Omega_y = np.zeros_like(self.s)

        self.__build_interpolators()

    def __lock_from_csv(self):
        # legacy path — byte-identical to track3D_local.lock_track_data()
        self.s = self.__data['s_m']
        self.ds = float(np.mean(np.diff(self.s)))
        self.x = self.__data['x_m']
        self.y = self.__data['y_m']
        self.z = self.__data['z_m']
        self.theta = self.__data['theta_rad']
        self.mu = self.__data['mu_rad']
        self.phi = self.__data['phi_rad']
        self.w_tr_right = self.__data['w_tr_right_m']
        self.w_tr_left = self.__data['w_tr_left_m']
        self.Omega_x = self.__data['omega_x_radpm']
        self.Omega_y = self.__data['omega_y_radpm']
        self.Omega_z = self.__data['omega_z_radpm']
        self.__build_interpolators()

    def __build_interpolators(self):
        """Shared: compute Ω derivatives, unwrap angle series, build all
        CasADi interpolators. Called by both __lock_from_raceline and
        __lock_from_csv after self.* fields are set.
        """
        # derivatives of omega with finite differencing
        self.dOmega_x = np.diff(self.Omega_x) / self.ds
        self.dOmega_x = np.append(self.dOmega_x, self.dOmega_x[0])
        self.dOmega_y = np.diff(self.Omega_y) / self.ds
        self.dOmega_y = np.append(self.dOmega_y, self.dOmega_y[0])
        self.dOmega_z = np.diff(self.Omega_z) / self.ds
        self.dOmega_z = np.append(self.dOmega_z, self.dOmega_z[0])

        # track spine interpolator
        def concatenate_arr(arr):
            return np.concatenate((arr, arr[1:], arr[1:]))  # 2 track lengths
        s_augmented = np.concatenate((self.s, self.s[-1] + self.s[1:], 2 * self.s[-1] + self.s[1:]))

        ### HJ : unwrap angle-series to remove ±π discontinuity at lap end.
        self.theta = np.unwrap(self.theta)
        self.mu = np.unwrap(self.mu)
        self.phi = np.unwrap(self.phi)

        ### HJ : per-lap delta for extending to the 2nd and 3rd lap without
        #        re-wrapping. For theta, d_theta_lap ≈ ±2π. For mu/phi ≈ 0.
        d_theta_lap = float(self.theta[-1] - self.theta[0])
        d_mu_lap = float(self.mu[-1] - self.mu[0])
        d_phi_lap = float(self.phi[-1] - self.phi[0])

        def concat_angle(arr, d_lap):
            # lap1 = arr, lap2 = arr[1:] + d_lap, lap3 = arr[1:] + 2*d_lap
            return np.concatenate((arr, arr[1:] + d_lap, arr[1:] + 2.0 * d_lap))

        # casadi interpolator instances
        self.x_interpolator = ca.interpolant('x', 'linear', [s_augmented], concatenate_arr(self.x))
        self.y_interpolator = ca.interpolant('y', 'linear', [s_augmented], concatenate_arr(self.y))
        self.z_interpolator = ca.interpolant('z', 'linear', [s_augmented], concatenate_arr(self.z))
        self.theta_interpolator = ca.interpolant('theta', 'linear', [s_augmented], concat_angle(self.theta, d_theta_lap))
        self.mu_interpolator = ca.interpolant('mu', 'linear', [s_augmented], concat_angle(self.mu, d_mu_lap))
        self.phi_interpolator = ca.interpolant('phi', 'linear', [s_augmented], concat_angle(self.phi, d_phi_lap))
        self.w_tr_right_interpolator = ca.interpolant('w_tr_right', 'linear', [s_augmented], concatenate_arr(self.w_tr_right))
        self.w_tr_left_interpolator = ca.interpolant('w_tr_left', 'linear', [s_augmented], concatenate_arr(self.w_tr_left))
        self.Omega_x_interpolator = ca.interpolant('omega_x', 'linear', [s_augmented], concatenate_arr(self.Omega_x))
        self.Omega_y_interpolator = ca.interpolant('omega_y', 'linear', [s_augmented], concatenate_arr(self.Omega_y))
        self.Omega_z_interpolator = ca.interpolant('omega_z', 'linear', [s_augmented], concatenate_arr(self.Omega_z))
        self.dOmega_x_interpolator = ca.interpolant('domega_x', 'linear', [s_augmented], concatenate_arr(self.dOmega_x))
        self.dOmega_y_interpolator = ca.interpolant('domega_y', 'linear', [s_augmented], concatenate_arr(self.dOmega_y))
        self.dOmega_z_interpolator = ca.interpolant('domega_z', 'linear', [s_augmented], concatenate_arr(self.dOmega_z))

    def sn2cartesian(self, s, n, normal_vector_factor: float = 1.0):
        if not self.track_locked:
            raise RuntimeError('Cannot transform. Track is not locked.')
        ### IY : convert casadi DM to numpy before stacking (DM arrays break np.array)
        euler_p = np.array([
            np.array(self.theta_interpolator(s)).flatten(),
            np.array(self.mu_interpolator(s)).flatten(),
            np.array(self.phi_interpolator(s)).flatten(),
        ])
        ref_p = np.array([
            np.array(self.x_interpolator(s)).flatten(),
            np.array(self.y_interpolator(s)).flatten(),
            np.array(self.z_interpolator(s)).flatten(),
        ]).transpose()

        return ref_p + (self.get_normal_vector_numpy(*euler_p) * normal_vector_factor * n).transpose()

    def calc_apparent_accelerations(
            self, V, n, chi, ax, ay, s, h,
            neglect_w_omega_y: bool = True, neglect_w_omega_x: bool = True, neglect_euler: bool = True,
            neglect_centrifugal: bool = True, neglect_w_dot: bool = False, neglect_V_omega: bool = False,
    ):
        if not self.track_locked:
            raise RuntimeError('Cannot calculate apparent accelerations. Track is not locked.')

        mu = self.mu_interpolator(s)
        phi = self.phi_interpolator(s)
        Omega_x = self.Omega_x_interpolator(s)
        dOmega_x = self.dOmega_x_interpolator(s)
        Omega_y = self.Omega_y_interpolator(s)
        dOmega_y = self.dOmega_y_interpolator(s)
        Omega_z = self.Omega_z_interpolator(s)
        dOmega_z = self.dOmega_z_interpolator(s)

        s_dot = (V * ca.cos(chi)) / (1.0 - n * Omega_z)
        w = n * Omega_x * s_dot

        V_dot = ax
        if not neglect_w_omega_y:
            V_dot += w * (Omega_x * ca.sin(chi) - Omega_y * ca.cos(chi)) * s_dot

        n_dot = V * ca.sin(chi)

        chi_dot = ay / V - Omega_z * s_dot
        if not neglect_w_omega_x:
            chi_dot += w * (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot / V

        s_ddot = ((V_dot * ca.cos(chi) - V * ca.sin(chi) * chi_dot) * (1.0 - n * Omega_z) - (V * ca.cos(chi)) * (- n_dot * Omega_z - n * dOmega_z * s_dot)) / (1.0 + 2.0 * n * Omega_z + n ** 2 * Omega_z ** 2)

        omega_x_dot = 0.0
        omega_y_dot = 0.0
        if not neglect_euler:
            omega_x_dot = (dOmega_x * s_dot * ca.cos(chi) - Omega_x * ca.sin(chi) * chi_dot + dOmega_y * s_dot * ca.sin(chi) + Omega_y * ca.cos(chi) * chi_dot) * s_dot + (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_ddot
            omega_y_dot = (-dOmega_x * s_dot * ca.sin(chi) - Omega_x * ca.cos(chi) * chi_dot + dOmega_y * s_dot * ca.cos(chi) - Omega_y * ca.sin(chi) * chi_dot) * s_dot + (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_ddot

        omega_x = 0.0
        omega_y = 0.0
        omega_z = 0.0
        if not neglect_centrifugal:
            omega_x = (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot
            omega_y = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot
            omega_z = Omega_z * s_dot + chi_dot

        w_dot = 0.0
        if not neglect_w_dot:
            w_dot = n_dot * Omega_x * s_dot + n * dOmega_x * s_dot ** 2 + n * Omega_x * s_ddot

        V_omega = 0.0
        if not neglect_V_omega:
            V_omega = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot * V

        ax_tilde = ax + omega_y_dot * h - omega_z * omega_x * h + g_earth * (- ca.sin(mu) * ca.cos(chi) + ca.cos(mu) * ca.sin(phi) * ca.sin(chi))
        ay_tilde = ay + omega_x_dot * h + omega_z * omega_y * h + g_earth * (ca.sin(mu) * ca.sin(chi) + ca.cos(mu) * ca.sin(phi) * ca.cos(chi))
        g_tilde = ca.fmax(w_dot - V_omega + (omega_x ** 2 - omega_y ** 2) * h + g_earth * ca.cos(mu) * ca.cos(phi), 0.0)

        return ax_tilde, ay_tilde, g_tilde

    def get_track_bounds(self, margin=0.0):
        normal_vector = self.get_normal_vector_numpy(self.theta, self.mu, self.phi)
        left = np.array([self.x + normal_vector[0] * (self.w_tr_left + margin),
                         self.y + normal_vector[1] * (self.w_tr_left + margin),
                         self.z + normal_vector[2] * (self.w_tr_left + margin)])
        right = np.array([self.x + normal_vector[0] * (self.w_tr_right - margin),
                          self.y + normal_vector[1] * (self.w_tr_right - margin),
                          self.z + normal_vector[2] * (self.w_tr_right - margin)])
        return left, right

    @staticmethod
    def get_rotation_matrix_numpy(theta, mu, phi):
        return np.array([
            [np.cos(theta) * np.cos(mu), np.cos(theta) * np.sin(mu) * np.sin(phi) - np.sin(theta) * np.cos(phi), np.cos(theta) * np.sin(mu) * np.cos(phi) + np.sin(theta) * np.sin(phi)],
            [np.sin(theta) * np.cos(mu), np.sin(theta) * np.sin(mu) * np.sin(phi) + np.cos(theta) * np.cos(phi), np.sin(theta) * np.sin(mu) * np.cos(phi) - np.cos(theta) * np.sin(phi)],
            [- np.sin(mu), np.cos(mu) * np.sin(phi), np.cos(mu) * np.cos(phi)]
        ]).squeeze()

    @staticmethod
    def get_normal_vector_numpy(theta, mu, phi):
        return Track3D.get_rotation_matrix_numpy(theta, mu, phi)[:, 1]

    @staticmethod
    def get_normal_vector_casadi(theta, mu, phi):
        return ca.vertcat(
            ca.cos(theta) * ca.sin(mu) * ca.sin(phi) - ca.sin(theta) * ca.cos(phi),
            ca.sin(theta) * ca.sin(mu) * ca.sin(phi) + ca.cos(theta) * ca.cos(phi),
            ca.cos(mu) * ca.sin(phi)
        )

    @staticmethod
    def get_jacobian_J(mu, phi):
        return np.array([
            [1, 0, -np.sin(mu)],
            [0, np.cos(phi), np.cos(mu) * np.sin(phi)],
            [0, -np.sin(phi), np.cos(mu) * np.cos(phi)]
        ])

# EOF
