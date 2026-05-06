#!/usr/bin/env python3
"""
Frenet Conversion Service Server (ROS2 Jazzy port).

원본 ROS1 (C++): f110_utils/nodes/frenet_conversion_server/src/frenet_conversion_server_node.cc.
이번 포팅에서는 Python 으로 다시 작성 (C++ lib 미포팅 결정 — Python lib 가 3D 기능
더 풍부하기 때문).

토픽 / 서비스:
- subscribe: /global_waypoints (f110_msgs/WpntArray) — 받으면 FrenetConverter 빌드
- subscribe: /trackbounds/markers (visualization_msgs/MarkerArray) — 받으면 set_track_bounds
  (원본 C++ 서버는 이 구독 안 함 — Python lib 만의 기능이므로 옵셔널)
- service: convert_glob2frenet_service (Glob2Frenet)
- service: convert_glob2frenetarr_service (Glob2FrenetArr)
- service: convert_frenet2glob_service (Frenet2Glob)
- service: convert_frenet2globarr_service (Frenet2GlobArr)
- 파라미터 PerceptionOnly=true 면 service 이름에 _perception 접미사
  (예: convert_glob2frenet_perception_service)
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from f110_msgs.msg import WpntArray
from visualization_msgs.msg import MarkerArray
from frenet_conversion_msgs.srv import (
    Frenet2Glob,
    Frenet2GlobArr,
    Glob2Frenet,
    Glob2FrenetArr,
)

from frenet_conversion.frenet_converter import FrenetConverter


class FrenetConverterServer(Node):
    def __init__(self) -> None:
        super().__init__("frenet_conversion_server")

        # PerceptionOnly: true 면 service 이름에 _perception 접미사 추가
        self.declare_parameter("PerceptionOnly", False)
        perception_only = self.get_parameter("PerceptionOnly").value
        suffix = "_perception" if perception_only else ""

        self._converter: FrenetConverter | None = None
        self._has_global_trajectory = False

        # Sub
        self.create_subscription(
            WpntArray, "/global_waypoints", self._on_global_traj, 10
        )
        self.create_subscription(
            MarkerArray, "/trackbounds/markers", self._on_trackbounds, 10
        )

        # Services (4개)
        self.create_service(
            Glob2Frenet,
            f"convert_glob2frenet{suffix}_service",
            self._on_glob2frenet,
        )
        self.create_service(
            Glob2FrenetArr,
            f"convert_glob2frenetarr{suffix}_service",
            self._on_glob2frenet_arr,
        )
        self.create_service(
            Frenet2Glob,
            f"convert_frenet2glob{suffix}_service",
            self._on_frenet2glob,
        )
        self.create_service(
            Frenet2GlobArr,
            f"convert_frenet2globarr{suffix}_service",
            self._on_frenet2glob_arr,
        )

        self.get_logger().info(
            f"[Frenet Conversion] ready. PerceptionOnly={perception_only}, "
            f"waiting for /global_waypoints..."
        )

    # ---------- 콜백 ----------

    def _on_global_traj(self, msg: WpntArray) -> None:
        """waypoints 로 FrenetConverter 빌드 (도착할 때마다 재빌드)."""
        if len(msg.wpnts) < 2:  # spline 빌드 최소 2 점
            return
        x = np.array([w.x_m for w in msg.wpnts])
        y = np.array([w.y_m for w in msg.wpnts])
        z = np.array([w.z_m for w in msg.wpnts])
        try:
            converter = FrenetConverter(x, y, z)
        except (ValueError, IndexError) as e:
            # build_raceline 의 CubicSpline 이 strictly-increasing 위반 등으로 실패할 수 있다
            # (예: ros2 topic pub default-empty wpnts 가 도착했을 때).
            self.get_logger().warn(
                f"[Frenet Conversion] FrenetConverter build failed: {e}; ignoring this msg"
            )
            return
        self._converter = converter
        if not self._has_global_trajectory:
            self.get_logger().info(
                f"[Frenet Conversion] Global waypoints received "
                f"({len(msg.wpnts)} wpnts, raceline_length="
                f"{self._converter.raceline_length:.2f}m)"
            )
            self._has_global_trajectory = True

    def _on_trackbounds(self, msg: MarkerArray) -> None:
        """트랙 경계 한 번만 설정 (원본 Python lib 의 _load_track_bounds 와 동일 책임)."""
        if self._converter is None:
            return
        if self._converter.has_track_bounds:
            return
        self._converter.set_track_bounds_from_markers(msg.markers)
        n_left = len(self._converter.left_bounds)
        n_right = len(self._converter.right_bounds)
        self.get_logger().info(
            f"[Frenet Conversion] Track bounds loaded: {n_left} left, {n_right} right"
        )

    # ---------- Service 콜백 ----------

    def _on_glob2frenet(self, request, response):
        if self._converter is None:
            self.get_logger().warn(
                "[Frenet Conversion] glob2frenet called before global_waypoints; returning zeros"
            )
            return response
        x_arr = np.array([request.x])
        y_arr = np.array([request.y])
        z_arr = np.array([request.z])
        s_arr, idx_arr = self._converter.get_approx_s_3d_with_idx(x_arr, y_arr, z_arr)
        s, d = self._converter.get_frenet_coord(x_arr, y_arr, s_arr)
        response.s = float(s[0])
        response.d = float(d[0])
        response.idx = int(idx_arr[0])
        return response

    def _on_glob2frenet_arr(self, request, response):
        if self._converter is None or not request.x:
            return response
        x_arr = np.array(request.x)
        y_arr = np.array(request.y)
        # z 누락 / 짧은 경우 0 으로 패딩 (C++ 서버 의 (i < req.z.size()) ? req.z[i] : 0.0 와 동일)
        z_in = list(request.z)
        if len(z_in) < len(request.x):
            z_in = z_in + [0.0] * (len(request.x) - len(z_in))
        z_arr = np.array(z_in)

        s_arr, idx_arr = self._converter.get_approx_s_3d_with_idx(x_arr, y_arr, z_arr)
        s, d = self._converter.get_frenet_coord(x_arr, y_arr, s_arr)
        response.s = s.tolist()
        response.d = d.tolist()
        response.idx = idx_arr.astype(int).tolist()
        return response

    def _on_frenet2glob(self, request, response):
        if self._converter is None:
            return response
        xyz = self._converter.get_cartesian_3d(
            np.array([request.s]), np.array([request.d])
        )
        response.x = float(xyz[0])
        response.y = float(xyz[1])
        response.z = float(xyz[2])
        return response

    def _on_frenet2glob_arr(self, request, response):
        if self._converter is None or not request.s:
            return response
        s_arr = np.array(request.s)
        d_arr = np.array(request.d)
        xyz = self._converter.get_cartesian_3d(s_arr, d_arr)
        response.x = xyz[0].tolist()
        response.y = xyz[1].tolist()
        response.z = xyz[2].tolist()
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrenetConverterServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
