#!/usr/bin/env python3
### HJ : FBGA-based 3D velocity planner ROS node
### global_waypoints 토픽에서 waypoint를 받아 FBGA로 속도 재계산 후 publish

import os
import subprocess
import tempfile
import numpy as np
import json
import yaml
import struct

from f110_msgs.msg import WpntArray, Wpnt
import trajectory_planning_helpers as tph


class FBGAVelocityPlanner:

    def __init__(self):
        self.get_logger().info("[FBGA] Initializing...")

        # === 경로 설정 (컨테이너 내부 절대경로) ===
        race_stack = '/home/unicorn/catkin_ws/src/race_stack'
        self.fbga_bin = self._get_param_or_default(
            "~fbga_bin",
            os.path.join(race_stack, 'f110_utils', 'libs', 'FBGA', 'bin', 'GIGI_test_unicorn.exe'))

        ### HJ : rc_car_10th — g_list [3.92, 19.62]
        gg_bin_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'gg_diagrams', 'rc_car_10th', 'velocity_frame', 'gg.bin')
        self.gg_bin = self._get_param_or_default("~gg_bin", gg_bin_default)

        params_yml_default = os.path.join(
            race_stack, 'planner', '3d_gb_optimizer', 'global_line', 'data',
            'vehicle_params', 'params_rc_car_10th.yml')
        params_yml = self._get_param_or_default("~params_yml", params_yml_default)

        # 경로 검증 (gg.bin 없으면 npy에서 자동 생성)
        for name, path in [('fbga_bin', self.fbga_bin), ('params_yml', params_yml)]:
            if not os.path.exists(path):
                self.get_logger().error(f"[FBGA] File not found: {name}={path}")
                raise FileNotFoundError(path)

        if not os.path.exists(self.gg_bin):
            self._generate_gg_bin(self.gg_bin)

        # === params.txt 생성 (tmp) ===
        self.params_txt = os.path.join(tempfile.gettempdir(), 'fbga_params.txt')
        self._convert_params_yml(params_yml)

        # === FBGA 설정 ===
        self.n_laps = self._get_param_or_default("~n_laps", 3)
        self.max_iter = self._get_param_or_default("~max_iter", 50)
        self.tol = self._get_param_or_default("~tol", 0.05)        # m/s
        self.alpha = self._get_param_or_default("~alpha", 1.0)      # under-relaxation
        self.v0 = self._get_param_or_default("~v0", 1.0)
        ### HJ : mu 보정 on/off (True: ax_tilde + Vmax mu 보정, False: g_tilde만)
        self.enable_mu = self._get_param_or_default("~enable_mu", True)

        # GGV g_list 범위 (clamp용)
        self.g_min, self.g_max = self._read_g_range()

        # === Pub/Sub ===
        self.processed = False  ### HJ : 한번만 처리 (자기 출력 재수신 방지)
        self.pub = self.create_publisher(WpntArray, '/global_waypoints', 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.wpnts_callback, 10)

        self.get_logger().info(f"[FBGA] Ready. bin={self.fbga_bin}")
        self.get_logger().info(f"[FBGA] gg.bin={self.gg_bin}")
        self.get_logger().info(f"[FBGA] n_laps={self.n_laps}, max_iter={self.max_iter}, tol={self.tol}")

    def _generate_gg_bin(self, bin_path):
        """npy 파일에서 gg.bin 자동 생성"""
        npy_dir = os.path.dirname(bin_path)
        self.get_logger().info(f"[FBGA] gg.bin not found, generating from {npy_dir}")

        v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
        g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
        ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
        ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
        ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
        gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)

        nv, ng = len(v_list), len(g_list)
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('II', nv, ng))
            for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                arr.tofile(f)

        self.get_logger().info(f"[FBGA] gg.bin generated: nv={nv}, ng={ng}, size={os.path.getsize(bin_path)} bytes")

    def _convert_params_yml(self, yml_path):
        """params YAML → params.txt (FBGA C++ runner 입력)"""
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        vp = cfg['vehicle_params']
        tp = cfg['tire_params']
        with open(self.params_txt, 'w') as f:
            f.write(f"m={vp['m']}\n")
            f.write(f"P_max={vp['P_max']}\n")
            f.write(f"mu_x={tp['p_Dx_1']}\n")
            f.write(f"mu_y={tp['p_Dy_1']}\n")
            f.write(f"v_max={vp['v_max']}\n")
        self.get_logger().info(f"[FBGA] params.txt saved: m={vp['m']}, P_max={vp['P_max']}, v_max={vp['v_max']}")

    def _read_g_range(self):
        """gg.bin에서 g_list 범위 읽기"""
        with open(self.gg_bin, 'rb') as f:
            nv, ng = struct.unpack('II', f.read(8))
            v_list = np.frombuffer(f.read(nv * 8), dtype=np.float64)
            g_list = np.frombuffer(f.read(ng * 8), dtype=np.float64)
        self.get_logger().info(f"[FBGA] GGV range: v=[{v_list.min():.1f},{v_list.max():.1f}], "
                      f"g=[{g_list.min():.2f},{g_list.max():.2f}]")
        return float(g_list.min()), float(g_list.max())

    def _compute_g_tilde(self, mu, v, dmu_ds):
        """g_tilde = 9.81*cos(mu) - v^2 * dmu/ds, clamped to GGV range"""
        gt = 9.81 * np.cos(mu) - v**2 * dmu_ds
        return np.clip(gt, self.g_min, self.g_max)

    def _initial_speed_estimate(self, kappa, mu, dmu_ds):
        """XY곡률 + 수직곡률 결합 초기 속도 추정"""
        ay_max = 4.5  # TODO: GGV에서 읽기
        v_max = 12.0

        # XY 곡률 한계
        radius = np.where(np.abs(kappa) > 1e-4, 1.0 / np.abs(kappa), 1e4)
        v_lat = np.clip(np.sqrt(ay_max * radius), 0, v_max)

        # 수직 곡률 한계 (crest에서 g_tilde > 0 조건)
        v_vert = np.full_like(kappa, v_max)
        crest = dmu_ds > 1e-4
        v_vert[crest] = np.clip(
            np.sqrt(9.81 * np.cos(mu[crest]) / dmu_ds[crest]), 0.5, v_max)

        return np.minimum(v_lat, v_vert)

    def _stack_laps(self, s, kappa, g_tilde, mu, dmu_ds):
        """N-laps 이어붙이기 (closed loop 보완)"""
        n_pts = len(s)
        ### HJ : 첫 두 점 간격 사용
        ds = s[1] - s[0] if n_pts > 1 else 0.1
        lap_length = s[-1] - s[0] + ds

        s_stack = np.concatenate([s + i * lap_length for i in range(self.n_laps)])
        k_stack = np.tile(kappa, self.n_laps)
        g_stack = np.tile(g_tilde, self.n_laps)
        mu_stack = np.tile(mu, self.n_laps)
        dmu_stack = np.tile(dmu_ds, self.n_laps)  ### HJ : dmu_ds도 같이 stack
        return s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts

    def _run_fbga(self, s, kappa, g_tilde, mu, dmu_ds, v0):
        """tmp CSV로 FBGA 실행, 결과 반환"""
        input_csv = os.path.join(tempfile.gettempdir(), 'fbga_input.csv')
        output_csv = os.path.join(tempfile.gettempdir(), 'fbga_output.csv')

        ### HJ : enable_mu flag로 mu 보정 여부 제어
        with open(input_csv, 'w') as f:
            if self.enable_mu:
                f.write('s,kappa,g_tilde,mu,dmu_ds\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f},{mu[i]:.8f},{dmu_ds[i]:.8f}\n')
            else:
                f.write('s,kappa,g_tilde\n')
                for i in range(len(s)):
                    f.write(f'{s[i]:.6f},{kappa[i]:.8f},{g_tilde[i]:.6f}\n')

        # FBGA 실행
        cmd = [
            self.fbga_bin,
            '--model', 'lookup',
            '--input', input_csv,
            '--params', self.params_txt,
            '--gg', self.gg_bin,
            '--output', output_csv,
            '--v0', f'{v0:.4f}',
        ]

        try:
            self.get_logger().info(f"[FBGA] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            self.get_logger().info(f"[FBGA] exe stdout: {result.stdout[-200:]}")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"[FBGA] exe failed (rc={e.returncode}): {e.stderr[:500]}")
            return None
        except subprocess.TimeoutExpired:
            self.get_logger().error("[FBGA] exe timeout")
            return None
        except Exception as e:
            self.get_logger().error(f"[FBGA] exe error: {e}")
            return None

        # 결과 읽기
        v_out = []
        ax_out = []
        with open(output_csv) as f:
            for line in f:
                if line.startswith('#') or line.startswith('s,'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    v_out.append(float(parts[1]))
                    ax_out.append(float(parts[2]))

        ### HJ : tmp 파일 정리 (디버깅용 비활성화)
        try:
            os.remove(input_csv)
            os.remove(output_csv)
        except OSError:
            pass

        return np.array(v_out), np.array(ax_out)

    def _extract_middle_lap(self, v_full, ax_full, n_pts_per_lap):
        """N-laps 결과에서 중간 바퀴 추출"""
        middle = self.n_laps // 2
        start = middle * n_pts_per_lap
        end = start + n_pts_per_lap
        return v_full[start:end], ax_full[start:end]

    def wpnts_callback(self, msg):
        if self.processed:
            return
        self.processed = True

        wpnts = msg.wpnts
        n = len(wpnts)

        # waypoint 데이터 추출
        s = np.array([wp.s_m for wp in wpnts])
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        mu = np.array([wp.mu_rad for wp in wpnts])
        v_existing = np.array([wp.vx_mps for wp in wpnts])

        ### HJ : periodic central difference (run_fwbw.py와 동일)
        ### HJ : ds를 첫 두 점 간격으로 계산
        ds_grid = s[1] - s[0] if n > 1 else 0.1
        mu_wrap = np.concatenate([[mu[-1]], mu, [mu[0]]])
        dmu_ds = (mu_wrap[2:] - mu_wrap[:-2]) / (2.0 * ds_grid)

        # 초기 속도: waypoint에 이미 있으면 사용, 없으면 추정
        if np.any(v_existing > 0.1):
            v_prev = v_existing.copy()
            self.get_logger().info("[FBGA] Using existing waypoint speeds as initial estimate")
        else:
            v_prev = self._initial_speed_estimate(kappa, mu, dmu_ds)
            self.get_logger().info("[FBGA] Using curvature+slope initial speed estimate")

        # === Fixed-point iteration ===
        for it in range(self.max_iter):
            # g_tilde 계산
            g_tilde = self._compute_g_tilde(mu, v_prev, dmu_ds)

            # N-laps stack (mu, dmu_ds도 같이)
            s_stack, k_stack, g_stack, mu_stack, dmu_stack, lap_length, n_pts = self._stack_laps(s, kappa, g_tilde, mu, dmu_ds)

            # FBGA 실행 — v0은 첫 점 속도, 최소 1.0 (3-laps trick으로 v0 영향은 중간 바퀴에서 소멸)
            v0 = max(float(v_prev[0]), 1.0)
            result = self._run_fbga(s_stack, k_stack, g_stack, mu_stack, dmu_stack, v0)
            if result is None:
                self.get_logger().warning("[FBGA] Failed, keeping existing speeds")
                return

            v_full, ax_full = result

            # 중간 바퀴 추출
            v_new, ax_new = self._extract_middle_lap(v_full, ax_full, n_pts)

            # NaN 처리
            nan_mask = np.isnan(v_new)
            if nan_mask.any():
                n_nan = nan_mask.sum()
                if n_nan / n > 0.05:
                    self.get_logger().warning(f"[FBGA] Too many NaNs: {n_nan}/{n}")
                    return
                valid = np.where(~nan_mask)[0]
                v_new[nan_mask] = np.interp(np.where(nan_mask)[0], valid, v_new[valid])

            # 수렴 체크
            delta = float(np.max(np.abs(v_new - v_prev)))
            self.get_logger().info(f"[FBGA] iter {it}: max|dv|={delta:.4f} m/s, "
                          f"g_tilde=[{g_tilde.min():.2f},{g_tilde.max():.2f}]")

            if delta < self.tol:
                self.get_logger().info(f"[FBGA] Converged at iter {it}")
                break

            v_prev = self.alpha * v_new + (1.0 - self.alpha) * v_prev

        # === waypoint 업데이트 (v, ax 모두 FBGA 결과 사용) ===
        ### HJ : FBGA ax 출력을 직접 사용
        ax_nan_mask = np.isnan(ax_new)
        if ax_nan_mask.any():
            valid_ax = np.where(~ax_nan_mask)[0]
            ax_new[ax_nan_mask] = np.interp(np.where(ax_nan_mask)[0], valid_ax, ax_new[valid_ax])

        for i in range(n):
            wpnts[i].vx_mps = float(v_new[i])
            wpnts[i].ax_mps2 = float(ax_new[i])

        msg.wpnts = wpnts
        self.get_logger().info(f"[FBGA] Publishing: v=[{v_new.min():.2f},{v_new.max():.2f}] m/s")
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("fbga_velocity_planner")
    node = FBGAVelocityPlanner()
    rospy.spin()
