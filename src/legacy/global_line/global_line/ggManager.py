import numpy as np
import os.path
import casadi as ca

g_earth = 9.81

class GGManager():
    def __init__(
            self,
            gg_path: str,
            gg_margin: float = 0.05,
    ):
        self.gg_margin = gg_margin
        self.slope_aware = False  ## IY : 3D slope flag

        self.__load_gggv_data(gg_path)

        self.rho_interpolator_no_margin = self.__get_rho_interpolator(margin=False)
        self.rho_interpolator = self.__get_rho_interpolator(margin=True)

        self.gg_exponent_interpolator, self.ax_max_interpolator, self.ax_min_interpolator, self.ay_max_interpolator, self.acc_interpolator = \
            self.__get_diamond_interpolators()

    def __load_gggv_data(self, gggv_path):
        self.V_list = np.load(os.path.join(gggv_path, 'v_list.npy'))
        self.V_list = np.insert(self.V_list, 0, 0.0)
        self.V_max = self.V_list.max()
        self.g_list = np.load(os.path.join(gggv_path, 'g_list.npy'))
        self.g_list = np.insert(self.g_list, 0, 0.0)
        self.g_max = self.g_list.max()
        # polar coordinates
        self.alpha_list = np.load(os.path.join(gggv_path, 'alpha_list.npy'))
        self.rho_list = np.load(os.path.join(gggv_path, 'rho.npy'))
        self.rho_list = np.insert(self.rho_list, 0, self.rho_list[0], axis=0)
        self.rho_list = np.insert(self.rho_list, 0, 1e-3*np.ones_like(self.rho_list[0, 1]), axis=1)

        # diamond approximation — 2D (standard)
        self.gg_exponent_list = np.load(os.path.join(gggv_path, 'gg_exponent.npy'))
        self.gg_exponent_list = np.insert(self.gg_exponent_list, 0, self.gg_exponent_list[0], axis=0)
        self.gg_exponent_list = np.insert(self.gg_exponent_list, 0, self.gg_exponent_list[0, 0], axis=1)
        self.ax_min_list = np.load(os.path.join(gggv_path, 'ax_min.npy'))
        self.ax_min_list = np.insert(self.ax_min_list, 0, self.ax_min_list[0], axis=0)
        self.ax_min_list = np.insert(self.ax_min_list, 0, 1e-3*np.ones_like(self.ax_min_list[0, 1]), axis=1)
        self.ax_max_list = np.load(os.path.join(gggv_path, 'ax_max.npy'))
        self.ax_max_list = np.insert(self.ax_max_list, 0, self.ax_max_list[0], axis=0)
        self.ax_max_list = np.insert(self.ax_max_list, 0, 1e-3*np.ones_like(self.ax_max_list[0, 1]), axis=1)
        self.ay_max_list = np.load(os.path.join(gggv_path, 'ay_max.npy'))
        self.ay_max_list = np.insert(self.ay_max_list, 0, self.ay_max_list[0], axis=0)
        self.ay_max_list = np.insert(self.ay_max_list, 0, 1e-3*np.ones_like(self.ay_max_list[0, 1]), axis=1)

        ## IY : load 3D slope diamond if available
        slope_path = os.path.join(gggv_path, 'slope_list.npy')
        ax_max_3d_path = os.path.join(gggv_path, 'ax_max_3d.npy')
        if os.path.exists(slope_path) and os.path.exists(ax_max_3d_path):
            self.slope_aware = True
            self.slope_list = np.load(slope_path)
            self.gg_exponent_3d = np.load(os.path.join(gggv_path, 'gg_exponent_3d.npy'))
            self.ax_min_3d = np.load(os.path.join(gggv_path, 'ax_min_3d.npy'))
            self.ax_max_3d = np.load(os.path.join(gggv_path, 'ax_max_3d.npy'))
            self.ay_max_3d = np.load(os.path.join(gggv_path, 'ay_max_3d.npy'))
            # Pad V=0 and g=0 boundaries (same logic as 2D)
            for attr in ['gg_exponent_3d', 'ax_min_3d', 'ax_max_3d', 'ay_max_3d']:
                arr = getattr(self, attr)
                arr = np.insert(arr, 0, arr[0], axis=0)  # V=0
                if attr == 'gg_exponent_3d':
                    arr = np.insert(arr, 0, arr[:, 0:1, :], axis=1)  # g=0
                else:
                    arr = np.insert(arr, 0, 1e-3 * np.ones_like(arr[:, 0:1, :]), axis=1)
                setattr(self, attr, arr)
            print(f'[GGManager] 3D slope-aware GGV loaded: '
                  f'slope_N={len(self.slope_list)}, '
                  f'range=[{np.degrees(self.slope_list[0]):.1f}°, '
                  f'{np.degrees(self.slope_list[-1]):.1f}°]')
        ## IY : end

    def __get_diamond_interpolators(self):
        gg_exponent_interpolator = ca.interpolant(
            'gg_exponent_interpolator', 'linear', [self.V_list, self.g_list], self.gg_exponent_list.ravel(order='F')
        )
        ax_max_interpolator = ca.interpolant(
            'ax_max_interpolator', 'linear', [self.V_list, self.g_list], self.ax_max_list.ravel(order='F') * (1.0 - self.gg_margin)
        )
        ax_min_interpolator = ca.interpolant(
            'ax_min_interpolator', 'linear', [self.V_list, self.g_list], self.ax_min_list.ravel(order='F') * (1.0 - self.gg_margin)
        )
        ay_max_interpolator = ca.interpolant(
            'ay_max_interpolator', 'linear', [self.V_list, self.g_list], self.ay_max_list.ravel(order='F') * (1.0 - self.gg_margin)
        )

        acc_interpolator = ca.interpolant(
            'acc_interpolator', 'linear', [self.V_list, self.g_list],
            np.array([self.gg_exponent_list, self.ax_min_list * (1.0 - self.gg_margin), self.ax_max_list * (1.0 - self.gg_margin), self.ay_max_list * (1.0 - self.gg_margin)]).ravel(order='F')
        )

        ## IY : 3D slope-aware interpolator
        if self.slope_aware:
            m = 1.0 - self.gg_margin
            self.acc_interpolator_3d = ca.interpolant(
                'acc_interpolator_3d', 'linear',
                [self.V_list, self.g_list, self.slope_list],
                np.array([
                    self.gg_exponent_3d,
                    self.ax_min_3d * m,
                    self.ax_max_3d * m,
                    self.ay_max_3d * m,
                ]).ravel(order='F')
            )
        ## IY : end

        return gg_exponent_interpolator, ax_max_interpolator, ax_min_interpolator, ay_max_interpolator, acc_interpolator

    def __get_rho_interpolator(self, margin: bool):
        # create interpolator
        factor = (1.0 - self.gg_margin) if margin else 1.0
        rho_interpolator = ca.interpolant(
            'rho_interpolator',
            'linear',
            [self.V_list, self.g_list, self.alpha_list],
            self.rho_list.ravel(order='F') * factor
        )
        return rho_interpolator