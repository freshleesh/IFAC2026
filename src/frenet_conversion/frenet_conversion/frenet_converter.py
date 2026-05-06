"""
3D Frenet ↔ Cartesian 변환 라이브러리.

원본 ROS1 (HJ): f110_utils/libs/frenet_conversion/src/frenet_converter/frenet_converter.py.
이번 ROS2 포팅에서 변경:
- ROS 의존 제거: 원본의 `_load_track_bounds` (`rospy.wait_for_message`) 제거.
  외부 호출자가 set_track_bounds(left, right) 또는 set_track_bounds_from_markers(markers)
  를 직접 호출하도록 책임 이전 (Server Node 가 담당).
- get_approx_s_3d_with_idx — service idx 응답 위해 인덱스도 함께 반환하는 헬퍼 추가.

알고리즘 (build_raceline, get_frenet, get_cartesian, height filter, boundary raycast,
rotational search, perpendicular projection iter) 은 원본 그대로 유지.
"""
from __future__ import annotations

from typing import Union, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


class FrenetConverter:
    def __init__(
        self,
        waypoints_x: np.ndarray,
        waypoints_y: np.ndarray,
        waypoints_z: np.ndarray | None = None,
    ):
        self.waypoints_x = np.asarray(waypoints_x, dtype=float)
        self.waypoints_y = np.asarray(waypoints_y, dtype=float)
        self.waypoints_z = (
            np.asarray(waypoints_z, dtype=float)
            if waypoints_z is not None
            else np.zeros_like(self.waypoints_x)
        )
        self.waypoints_s: np.ndarray | None = None
        self.spline_x: CubicSpline | None = None
        self.spline_y: CubicSpline | None = None
        self.spline_z: CubicSpline | None = None
        self.raceline_length: float | None = None
        self.waypoints_distance_m: float | None = None
        self.iter_max = 3
        self.has_z = waypoints_z is not None

        # 트랙 boundary (벽 교차 검사용). 외부에서 set_track_bounds 호출 전엔 비활성.
        self.left_bounds: np.ndarray | None = None
        self.right_bounds: np.ndarray | None = None
        self.has_track_bounds = False
        self.height_filter_threshold = 0.10  # [m] (C++ 와 동일)
        self.z_boundary_margin = 0.10        # [m]

        # waypoint 단위 psi/mu 사전계산 (height offset 용)
        self.waypoints_psi: np.ndarray | None = None
        self.waypoints_mu: np.ndarray | None = None

        self.build_raceline()

    # ---------- 초기화 ----------

    def build_raceline(self) -> None:
        """3D 거리 누적으로 s 계산 후 spline 빌드 + waypoint 단위 psi/mu 사전계산."""
        s_list = [0.0]
        prev_x = self.waypoints_x[0]
        prev_y = self.waypoints_y[0]
        prev_z = self.waypoints_z[0]
        for wx, wy, wz in zip(self.waypoints_x[1:], self.waypoints_y[1:], self.waypoints_z[1:]):
            dist = np.linalg.norm([wx - prev_x, wy - prev_y, wz - prev_z])
            prev_x, prev_y, prev_z = wx, wy, wz
            s_list.append(s_list[-1] + dist)
        self.waypoints_s = np.array(s_list)
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.spline_z = CubicSpline(self.waypoints_s, self.waypoints_z)
        self.raceline_length = float(self.waypoints_s[-1])
        self.waypoints_distance_m = float(np.median(np.diff(self.waypoints_s)))

        dx = self.spline_x(self.waypoints_s, 1)
        dy = self.spline_y(self.waypoints_s, 1)
        dz = self.spline_z(self.waypoints_s, 1)
        self.waypoints_psi = np.arctan2(dy, dx)
        ds_xy = np.sqrt(dx ** 2 + dy ** 2)
        self.waypoints_mu = np.arctan2(dz, ds_xy)

    # ---------- Track boundary ----------

    def set_track_bounds(self, left_bounds, right_bounds) -> None:
        """
        벽 교차 검사용 트랙 경계 설정.

        left_bounds / right_bounds: Nx3 list/array — [[x, y, z], ...].
        """
        self.left_bounds = np.array(left_bounds)
        self.right_bounds = np.array(right_bounds)

        self.left_seg_start = self.left_bounds[:-1, :2]
        self.left_seg_end = self.left_bounds[1:, :2]
        self.left_seg_z_avg = (self.left_bounds[:-1, 2] + self.left_bounds[1:, 2]) * 0.5

        self.right_seg_start = self.right_bounds[:-1, :2]
        self.right_seg_end = self.right_bounds[1:, :2]
        self.right_seg_z_avg = (self.right_bounds[:-1, 2] + self.right_bounds[1:, 2]) * 0.5

        self.has_track_bounds = True

    def set_track_bounds_from_markers(self, markers) -> None:
        """MarkerArray (alternating left/right) 에서 set_track_bounds 호출."""
        left = []
        right = []
        for i, m in enumerate(markers):
            pos = m.pose.position
            pt = [pos.x, pos.y, pos.z]
            if i % 2 == 0:
                left.append(pt)
            else:
                right.append(pt)
        self.set_track_bounds(left, right)

    # ---------- 헬퍼 ----------

    def _calc_height_offset(self, x, y, z, wpt_idx) -> float:
        """track surface normal 방향 height offset (양수 = 위, 음수 = 아래)."""
        dx = x - self.waypoints_x[wpt_idx]
        dy = y - self.waypoints_y[wpt_idx]
        dz = z - self.waypoints_z[wpt_idx]
        psi = self.waypoints_psi[wpt_idx]
        mu = self.waypoints_mu[wpt_idx]
        sin_mu = np.sin(mu)
        cos_mu = np.cos(mu)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        return dx * cos_psi * sin_mu + dy * sin_psi * sin_mu + dz * cos_mu

    def _is_line_crossing_boundary(self, x1, y1, x2, y2, z_ref) -> bool:
        """벽 (left + right) 와 선분 [(x1,y1), (x2,y2)] 의 교차 검사."""
        if not self.has_track_bounds:
            return False

        for seg_start, seg_end, seg_z_avg in [
            (self.left_seg_start, self.left_seg_end, self.left_seg_z_avg),
            (self.right_seg_start, self.right_seg_end, self.right_seg_z_avg),
        ]:
            z_mask = np.abs(seg_z_avg - z_ref) <= self.z_boundary_margin
            if not np.any(z_mask):
                continue

            cx = seg_start[z_mask, 0]
            cy = seg_start[z_mask, 1]
            dx = seg_end[z_mask, 0]
            dy = seg_end[z_mask, 1]

            d1 = (dx - cx) * (y1 - cy) - (dy - cy) * (x1 - cx)
            d2 = (dx - cx) * (y2 - cy) - (dy - cy) * (x2 - cx)
            d3 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
            d4 = (x2 - x1) * (dy - y1) - (y2 - y1) * (dx - x1)

            intersects = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
            if np.any(intersects):
                return bool(True)

        return False

    # ---------- Frenet 변환 (2D 호환 유지 + 3D) ----------

    def get_frenet(self, x, y, s=None) -> np.ndarray:
        """2D Frenet 변환 (z=0 가정). 반환: [[s], [d]] shape (2, N)."""
        if s is None:
            s = self.get_approx_s(x, y)
        s, d = self.get_frenet_coord(x, y, s)
        return np.array([s, d])

    def get_approx_s(self, x, y) -> np.ndarray:
        """가장 가까운 waypoint 의 s 값 반환 (2D)."""
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        dist_2d = np.linalg.norm([dist_x.T, dist_y.T], axis=0)
        return self.waypoints_s[np.argmin(dist_2d, axis=1)]

    def get_approx_s_3d(self, x, y, z) -> np.ndarray:
        """3D 가장 가까운 waypoint 의 s 값. height filter + 벽 교차 회피 포함."""
        s, _ = self.get_approx_s_3d_with_idx(x, y, z)
        return s

    def get_approx_s_3d_with_idx(
        self, x, y, z
    ) -> Tuple[np.ndarray, np.ndarray]:
        """get_approx_s_3d 와 동일하되 (s, idx) 둘 다 반환 — service idx 응답용 (SH 추가)."""
        lenx = len(x)
        result_indices = np.zeros(lenx, dtype=int)

        for qi in range(lenx):
            qx, qy, qz = x[qi], y[qi], z[qi]

            dx_all = qx - self.waypoints_x
            dy_all = qy - self.waypoints_y
            dz_all = qz - self.waypoints_z
            d_height = (dx_all * np.cos(self.waypoints_psi) * np.sin(self.waypoints_mu)
                        + dy_all * np.sin(self.waypoints_psi) * np.sin(self.waypoints_mu)
                        + dz_all * np.cos(self.waypoints_mu))
            height_mask = np.abs(d_height) <= self.height_filter_threshold

            d_sq_all = dx_all ** 2 + dy_all ** 2 + dz_all ** 2
            d_sq = d_sq_all.copy()
            d_sq[~height_mask] = np.inf

            nearest_idx = int(np.argmin(d_sq))

            max_valid_dist_sq = 2.0 * 2.0  # 2m
            if d_sq[nearest_idx] == np.inf or d_sq[nearest_idx] > max_valid_dist_sq:
                result_indices[qi] = int(np.argmin(d_sq_all))
                continue

            if (not self.has_track_bounds
                    or not self._is_line_crossing_boundary(
                        qx, qy,
                        self.waypoints_x[nearest_idx],
                        self.waypoints_y[nearest_idx], qz)):
                result_indices[qi] = nearest_idx
                continue

            # 벽 충돌 — 90/180/270 도 회전 검색
            vec_x = self.waypoints_x[nearest_idx] - qx
            vec_y = self.waypoints_y[nearest_idx] - qy
            found = False

            for angle_deg in [90, 180, 270]:
                angle_rad = np.radians(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                target_x = qx + vec_x * cos_a - vec_y * sin_a
                target_y = qy + vec_x * sin_a + vec_y * cos_a

                d_sq_target = (self.waypoints_x - target_x) ** 2 + (self.waypoints_y - target_y) ** 2
                d_sq_target[~height_mask] = np.inf
                candidate_idx = int(np.argmin(d_sq_target))

                if d_sq_target[candidate_idx] == np.inf:
                    continue

                if self._is_line_crossing_boundary(
                        qx, qy,
                        self.waypoints_x[candidate_idx],
                        self.waypoints_y[candidate_idx], qz):
                    continue

                # candidate 주변 ± s 방향 closer 탐색
                best_idx = candidate_idx
                best_dist = d_sq_all[candidate_idx]
                n_wpts = len(self.waypoints_x)

                for s_off in range(1, n_wpts):
                    test_idx = (candidate_idx + s_off) % n_wpts
                    if not height_mask[test_idx]:
                        break
                    if self._is_line_crossing_boundary(
                            qx, qy,
                            self.waypoints_x[test_idx],
                            self.waypoints_y[test_idx], qz):
                        break
                    if d_sq_all[test_idx] < best_dist:
                        best_dist = d_sq_all[test_idx]
                        best_idx = test_idx

                for s_off in range(1, n_wpts):
                    test_idx = (candidate_idx - s_off) % n_wpts
                    if not height_mask[test_idx]:
                        break
                    if self._is_line_crossing_boundary(
                            qx, qy,
                            self.waypoints_x[test_idx],
                            self.waypoints_y[test_idx], qz):
                        break
                    if d_sq_all[test_idx] < best_dist:
                        best_dist = d_sq_all[test_idx]
                        best_idx = test_idx

                result_indices[qi] = best_idx
                found = True
                break

            if not found:
                result_indices[qi] = nearest_idx

        return self.waypoints_s[result_indices], result_indices

    def get_frenet_3d(self, x, y, z, s=None) -> np.ndarray:
        """3D Frenet 변환 — z 사용한 nearest search + height filter + boundary."""
        if s is None:
            s = self.get_approx_s_3d(x, y, z)
        s, d = self.get_frenet_coord(x, y, s)
        return np.array([s, d])

    def get_frenet_coord(self, x, y, s, eps_m=0.01):
        """뉴턴식 perpendicular 투영 반복 (iter_max=3)."""
        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for _i in range(self.iter_max):
            cand_s = (s + projection) % self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
            cand_projection = np.clip(
                cand_projection,
                -self.waypoints_distance_m / (2 * self.iter_max),
                self.waypoints_distance_m / (2 * self.iter_max),
            )
            updated_idxs = np.abs(cand_projection) <= np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]
        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        dx_ds, dy_ds = self.get_derivative(s)
        tangent = np.array([dx_ds, dy_ds])
        if np.any(np.isnan(s)):
            raise ValueError("FRENET CONVERTER: s is nan")
        tangent /= np.linalg.norm(tangent, axis=0)

        x_vec = x - self.spline_x(s)
        y_vec = y - self.spline_y(s)
        point_to_track = np.array([x_vec, y_vec])

        proj = np.einsum("ij,ij->j", tangent, point_to_track)
        perps = np.array([-tangent[1, :], tangent[0, :]])
        d = np.einsum("ij,ij->j", perps, point_to_track)

        return None, proj, d

    def get_derivative(self, s) -> np.ndarray:
        s = s % self.raceline_length
        return [self.spline_x(s, 1), self.spline_y(s, 1)]

    # ---------- Cartesian 변환 ----------

    def get_cartesian(self, s: float, d: float) -> np.ndarray:
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        return np.array([x, y])

    def get_cartesian_3d(self, s: float, d: float) -> np.ndarray:
        x = self.spline_x(s)
        y = self.spline_y(s)
        z = self.spline_z(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        return np.array([x, y, z])

    # ---------- 헤딩 오차 ----------

    def get_e_psi(self, x: float, y: float, yaw: float) -> float:
        s = self.get_approx_s(np.array([x]), np.array([y]))[0]
        psi = np.arctan2(*self.get_derivative(s)[::-1])
        e_psi = yaw - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi  # [-π, π]
        return e_psi
