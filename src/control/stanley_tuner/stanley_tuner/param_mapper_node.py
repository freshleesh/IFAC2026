#!/usr/bin/env python3
"""Stanley 파라미터 매퍼 노드 (경기 모드).

stanley_params.yaml을 로드해 /stanley_tuner/param_profile을 발행한다.
stanley 노드는 이 토픽을 구독해 per-waypoint 파라미터를 추종한다.
학습 없음 — 순수 룩업 + 발행만 수행.

구독:
  /global_waypoints  (WpntArray) — waypoint s 값 확보 (한 번만)

발행:
  /stanley_tuner/param_profile  (Float64MultiArray)
    flat [s0,k0,kff0,ld0, s1,k1,kff1,ld1, ...] per global waypoint

파라미터:
  map             — 맵 이름 (stanley_params.yaml 경로 결정)
  kappa_threshold — 섹터 자동 감지 임계값 (YAML 없을 때 사용)
  publish_rate_hz — 발행 주기 (기본 10 Hz)
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray

from stanley_tuner.sector_map import (
    Sector, SectorParams, detect_sectors, load_yaml,
    apply_yaml_override, find_sector,
)


class ParamMapperNode(Node):
    def __init__(self):
        super().__init__("stanley_param_mapper")

        self.declare_parameter("map",              "")
        self.declare_parameter("kappa_threshold",  0.15)
        self.declare_parameter("publish_rate_hz",  10.0)

        self._map_name     = self.get_parameter("map").value
        self._kappa_thresh = float(self.get_parameter("kappa_threshold").value)
        self._pub_rate     = float(self.get_parameter("publish_rate_hz").value)

        if not self._map_name:
            self.get_logger().error("[ParamMapper] map parameter required")
            return

        self._sectors:      List[Sector]         = []
        self._global_s:     Optional[np.ndarray] = None
        self._track_length: float                = 0.0
        self._defaults:     SectorParams         = SectorParams()
        self._profile_msg:  Optional[Float64MultiArray] = None

        yaml_path = self._resolve_yaml_path()
        self._yaml_sectors, self._defaults = load_yaml(yaml_path)

        if self._yaml_sectors:
            self.get_logger().info(
                f"[ParamMapper] Loaded {len(self._yaml_sectors)} sectors from {yaml_path}"
            )
        else:
            self.get_logger().warn(
                f"[ParamMapper] {yaml_path} not found — will use default params "
                f"(k={self._defaults.k}, k_ff={self._defaults.k_ff}, "
                f"lookahead_d={self._defaults.lookahead_d})"
            )

        cb = ReentrantCallbackGroup()
        self.create_subscription(WpntArray, "/global_waypoints",
                                 self._on_wpnts, 10, callback_group=cb)

        self._profile_pub = self.create_publisher(
            Float64MultiArray, "/stanley_tuner/param_profile", 10
        )

        self.create_timer(1.0 / self._pub_rate, self._publish)

    # ── 콜백 ──────────────────────────────────────────── #

    def _on_wpnts(self, msg: WpntArray) -> None:
        if self._global_s is not None:
            return
        if not msg.wpnts:
            return

        s     = np.array([w.s_m         for w in msg.wpnts])
        kappa = np.array([w.kappa_radpm  for w in msg.wpnts])
        self._global_s     = s
        self._track_length = float(s[-1])

        auto = detect_sectors(s, kappa, kappa_threshold=self._kappa_thresh)
        if self._yaml_sectors:
            self._sectors = apply_yaml_override(auto, self._yaml_sectors)
        else:
            self._sectors = auto

        self.get_logger().info(
            f"[ParamMapper] Ready — {len(self._sectors)} sectors, "
            f"track={self._track_length:.1f}m"
        )

        self._profile_msg = self._build_profile_msg()

    # ── Profile 빌드 & 발행 ────────────────────────────── #

    def _build_profile_msg(self) -> Optional[Float64MultiArray]:
        if self._global_s is None:
            return None

        N = len(self._global_s)
        profile = np.zeros((N, 4), dtype=float)
        profile[:, 0] = self._global_s

        for i, s_val in enumerate(self._global_s):
            sector = find_sector(self._sectors, s_val, self._track_length)
            if sector is not None:
                p = sector.params
            else:
                p = self._defaults
            profile[i, 1] = p.k
            profile[i, 2] = p.k_ff
            profile[i, 3] = p.lookahead_d

        msg = Float64MultiArray()
        msg.data = profile.flatten().tolist()
        return msg

    def _publish(self) -> None:
        if self._profile_msg is not None:
            self._profile_pub.publish(self._profile_msg)

    # ── 경로 해결 ─────────────────────────────────────── #

    def _resolve_yaml_path(self) -> str:
        # install/stanley_tuner → ../.. → workspace root (IFAC2026_SH/)
        ament_prefix = os.environ.get("AMENT_PREFIX_PATH", "")
        for prefix in ament_prefix.split(":"):
            if os.path.basename(prefix) == "stanley_tuner":
                ws = os.path.normpath(os.path.join(prefix, "..", ".."))
                candidate = os.path.join(ws, "src", "system", "stack_master",
                                         "maps", self._map_name, "stanley_params.yaml")
                if os.path.isdir(os.path.join(ws, "src")):
                    return candidate
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("stack_master")
            return os.path.join(share, "maps", self._map_name, "stanley_params.yaml")
        except Exception:
            return f"/tmp/stanley_params_{self._map_name}.yaml"


def main(args=None):
    rclpy.init(args=args)
    node = ParamMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
