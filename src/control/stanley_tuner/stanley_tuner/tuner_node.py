#!/usr/bin/env python3
"""Stanley 파라미터 튜닝 노드 (프랙티스 모드).

구독:
  /global_waypoints        (WpntArray)  — 트랙 형상 (섹터 감지용)
  /car_state/odom_frenet   (Odometry)   — 현재 s 위치
  /stanley/cte             (Float64)    — cross-track error
  /stanley/heading_error   (Float64)    — heading error
  /stanley/steer           (Float64)    — steering command

발행:
  /stanley_tuner/param_profile  (Float64MultiArray)
    flat [s0,k0,kff0,ld0, s1,k1,kff1,ld1, ...] per global waypoint

Safety watchdog:
  - |CTE| > cte_safety_m 이 연속 cte_danger_frames 회 이상이면
    해당 섹터를 default 파라미터로 즉시 리셋 + 경고 로그
  - 이렇게 리셋된 섹터는 confidence=0으로 초기화되어 재학습 시작
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Float64, Float64MultiArray
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray

from stanley_tuner.sector_map import (
    Sector, SectorParams, detect_sectors, load_yaml,
    apply_yaml_override, save_yaml, find_sector,
)
from stanley_tuner.metrics_collector import MetricsCollector, SectorStats
from stanley_tuner import rule_tuner
from stanley_tuner.rule_tuner import RuleConfig
from stanley_tuner.bayes_tuner import BayesBounds, BayesTunerManager


class TunerNode(Node):
    def __init__(self):
        super().__init__("stanley_tuner")

        self.declare_parameter("map",               "")
        self.declare_parameter("tuner_mode",        "rule_based")
        self.declare_parameter("kappa_threshold",   0.15)
        self.declare_parameter("min_sector_len",    1.0)
        self.declare_parameter("rate_hz",           50.0)
        self.declare_parameter("save_interval_s",   30.0)
        self.declare_parameter("min_laps",          2)
        self.declare_parameter("lambda1",           0.3)
        self.declare_parameter("lambda2",           0.2)
        # safety watchdog
        self.declare_parameter("cte_safety_m",      0.35)
        self.declare_parameter("cte_danger_frames",  15)
        # rule_tuner 하이퍼파라미터
        self.declare_parameter("k_alpha",       0.07)
        self.declare_parameter("k_beta",        0.12)
        self.declare_parameter("kff_alpha",     0.10)
        self.declare_parameter("cte_high",      0.08)
        self.declare_parameter("cte_low",       0.03)
        self.declare_parameter("osc_high",      0.04)
        self.declare_parameter("hdg_high",      0.08)
        self.declare_parameter("base_lookahead", 1.5)
        self.declare_parameter("kappa_scale",   1.8)
        self.declare_parameter("k_min",         0.5)
        self.declare_parameter("k_max",         10.0)
        self.declare_parameter("kff_min",       0.0)
        self.declare_parameter("kff_max",       0.15)
        self.declare_parameter("ld_min",        0.3)
        self.declare_parameter("ld_max",        5.0)

        self._map_name       = self.get_parameter("map").value
        self._mode           = self.get_parameter("tuner_mode").value
        self._kappa_thresh   = float(self.get_parameter("kappa_threshold").value)
        self._min_sec_len    = float(self.get_parameter("min_sector_len").value)
        self._rate_hz        = float(self.get_parameter("rate_hz").value)
        self._save_interval  = float(self.get_parameter("save_interval_s").value)
        self._min_laps       = int(self.get_parameter("min_laps").value)
        self._lambda1        = float(self.get_parameter("lambda1").value)
        self._lambda2        = float(self.get_parameter("lambda2").value)
        self._cte_safety     = float(self.get_parameter("cte_safety_m").value)
        self._danger_frames  = int(self.get_parameter("cte_danger_frames").value)
        self._bayes_bounds = BayesBounds(
            k_min=  float(self.get_parameter("k_min").value),
            k_max=  float(self.get_parameter("k_max").value),
            kff_min=float(self.get_parameter("kff_min").value),
            kff_max=float(self.get_parameter("kff_max").value),
            ld_min= float(self.get_parameter("ld_min").value),
            ld_max= float(self.get_parameter("ld_max").value),
        )
        self._rule_cfg = RuleConfig(
            k_alpha=       float(self.get_parameter("k_alpha").value),
            k_beta=        float(self.get_parameter("k_beta").value),
            kff_alpha=     float(self.get_parameter("kff_alpha").value),
            cte_high=      float(self.get_parameter("cte_high").value),
            cte_low=       float(self.get_parameter("cte_low").value),
            osc_high=      float(self.get_parameter("osc_high").value),
            hdg_high=      float(self.get_parameter("hdg_high").value),
            base_lookahead=float(self.get_parameter("base_lookahead").value),
            kappa_scale=   float(self.get_parameter("kappa_scale").value),
            k_min=         float(self.get_parameter("k_min").value),
            k_max=         float(self.get_parameter("k_max").value),
            kff_min=       float(self.get_parameter("kff_min").value),
            kff_max=       float(self.get_parameter("kff_max").value),
            ld_min=        float(self.get_parameter("ld_min").value),
            ld_max=        float(self.get_parameter("ld_max").value),
        )

        if not self._map_name:
            self.get_logger().error("[TunerNode] map parameter required")
            return

        self.get_logger().info(
            f"[TunerNode] map={self._map_name} mode={self._mode} "
            f"κ_thresh={self._kappa_thresh} "
            f"safety_cte={self._cte_safety}m/{self._danger_frames}frames"
        )

        # 상태
        self._sectors:      List[Sector]            = []
        self._global_s:     Optional[np.ndarray]    = None
        self._track_length: float                   = 0.0
        self._defaults:     SectorParams            = SectorParams()

        self._cur_sector:    Optional[Sector]       = None
        self._cur_params:    Dict[int, SectorParams] = {}
        self._s:             float                  = 0.0

        self._cte:   float = 0.0
        self._steer: float = 0.0
        self._hdg:   float = 0.0

        # safety watchdog 상태
        self._danger_count:  int  = 0   # 연속 고CTE 프레임 수
        self._stable_count:  int  = 0   # 연속 안전 프레임 수 (_in_danger 해제 조건)
        self._in_danger:     bool = False
        _STABLE_FRAMES = 30             # 0.6s 연속 안전해야 위험 상태 해제

        self._metrics = MetricsCollector(self._lambda1, self._lambda2)
        self._bayes:   Optional[BayesTunerManager] = None

        self._yaml_path = self._resolve_yaml_path()

        cb = ReentrantCallbackGroup()
        self.create_subscription(WpntArray, "/global_waypoints",
                                 self._on_global_wpnts, 10, callback_group=cb)
        self.create_subscription(Odometry, "/car_state/odom_frenet",
                                 self._on_frenet, 10, callback_group=cb)
        self.create_subscription(Float64, "/stanley/cte",
                                 lambda m: setattr(self, "_cte", m.data), 10)
        self.create_subscription(Float64, "/stanley/heading_error",
                                 lambda m: setattr(self, "_hdg", m.data), 10)
        self.create_subscription(Float64, "/stanley/steer",
                                 lambda m: setattr(self, "_steer", m.data), 10)

        self._profile_pub = self.create_publisher(
            Float64MultiArray, "/stanley_tuner/param_profile", 10
        )

        self.create_timer(1.0 / self._rate_hz, self._tick)
        self.create_timer(self._save_interval, self._save)

    # ── 콜백 ──────────────────────────────────────────── #

    def _on_global_wpnts(self, msg: WpntArray) -> None:
        if self._global_s is not None:
            return
        if not msg.wpnts:
            return

        s     = np.array([w.s_m         for w in msg.wpnts])
        kappa = np.array([w.kappa_radpm  for w in msg.wpnts])
        self._global_s     = s
        self._track_length = float(s[-1])

        auto_sectors = detect_sectors(
            s, kappa,
            kappa_threshold=self._kappa_thresh,
            min_sector_len=self._min_sec_len,
        )

        yaml_sectors, self._defaults = load_yaml(self._yaml_path)
        if yaml_sectors:
            auto_sectors = apply_yaml_override(auto_sectors, yaml_sectors)
            self.get_logger().info(
                f"[TunerNode] Loaded {len(yaml_sectors)} sector overrides from YAML"
            )
        else:
            self.get_logger().warn(
                f"[TunerNode] {self._yaml_path} not found — using default params"
            )

        self._sectors    = auto_sectors
        self._cur_params = {s.id: s.params for s in auto_sectors}

        if self._mode == "bayes":
            self._bayes = BayesTunerManager(auto_sectors, bounds=self._bayes_bounds)

        n_corners = sum(1 for s in auto_sectors if s.corner_type.value != "straight")
        self.get_logger().info(
            f"[TunerNode] {len(auto_sectors)} sectors "
            f"({n_corners} corners), track={self._track_length:.1f}m"
        )

        self._publish_profile()

    def _on_frenet(self, msg: Odometry) -> None:
        self._s = msg.pose.pose.position.x

    # ── 메인 루프 ──────────────────────────────────────── #

    def _tick(self) -> None:
        if not self._sectors or self._global_s is None:
            return

        new_sector = find_sector(self._sectors, self._s, self._track_length)

        # safety watchdog — 섹터 업데이트보다 먼저 체크
        self._check_safety(new_sector)

        # 섹터 이탈 시 통계 계산 → 파라미터 업데이트
        stats = self._metrics.update(new_sector, self._cte, self._steer, self._hdg)
        if stats is not None:
            self._on_sector_exit(stats)

        if new_sector != self._cur_sector and new_sector is not None:
            self._cur_sector = new_sector

        self._publish_profile()

    # ── Safety Watchdog ────────────────────────────────── #

    # _in_danger 해제에 필요한 연속 안전 프레임 수
    _STABLE_FRAMES = 30  # 0.6s @ 50Hz

    def _check_safety(self, sector: Optional[Sector]) -> None:
        """CTE가 임계값을 연속으로 초과하면 현재 섹터 파라미터를 default로 리셋."""
        if abs(self._cte) > self._cte_safety:
            self._danger_count  += 1
            self._stable_count   = 0
        else:
            self._danger_count   = 0
            if self._in_danger:
                self._stable_count += 1
                # 충분히 안전한 상태가 유지돼야 위험 플래그 해제
                if self._stable_count >= self._STABLE_FRAMES:
                    self._in_danger    = False
                    self._stable_count = 0
            return

        if self._danger_count >= self._danger_frames and not self._in_danger:
            self._in_danger    = True
            self._stable_count = 0
            if sector is not None:
                safe = SectorParams(
                    k=self._defaults.k,
                    k_ff=self._defaults.k_ff,
                    lookahead_d=self._defaults.lookahead_d,
                    confidence=0.0,
                )
                sector.params = safe
                self._cur_params[sector.id] = safe
                self._metrics.clear_history(sector.id)
                if self._bayes:
                    self._bayes.reset(sector.id)          # Bayes 오염 데이터도 폐기
                self.get_logger().warn(
                    f"[TunerNode] SAFETY RESET — Sector #{sector.id} "
                    f"({sector.corner_type.value}): CTE={self._cte:.3f}m "
                    f">{self._cte_safety}m for {self._danger_frames} frames. "
                    f"Reverted to defaults (k={safe.k}, k_ff={safe.k_ff}, ld={safe.lookahead_d})"
                )
                self._publish_profile()

    # ── 섹터 이탈 처리 ────────────────────────────────── #

    def _on_sector_exit(self, stats: SectorStats) -> None:
        sector = next((s for s in self._sectors if s.id == stats.sector_id), None)
        if sector is None:
            return

        history = self._metrics.history(stats.sector_id)

        if self._mode == "rule_based":
            new_params = rule_tuner.suggest(sector, history, self._min_laps, self._rule_cfg)
        else:
            if self._bayes:
                cur_p = self._cur_params.get(stats.sector_id, sector.params)
                self._bayes.observe(stats.sector_id, cur_p, stats)
                # suggest_with_limit 사용: 현재값 대비 최대 35% 변화만 허용
                new_params = self._bayes.suggest(stats.sector_id, cur_p) or sector.params
            else:
                new_params = sector.params

        if new_params is not sector.params:
            sector.params = new_params
            self._cur_params[sector.id] = new_params
            self.get_logger().info(
                f"[TunerNode] Sector #{sector.id} ({sector.corner_type.value}) "
                f"updated: k={new_params.k:.3f} k_ff={new_params.k_ff:.4f} "
                f"ld={new_params.lookahead_d:.2f} "
                f"[CTE={stats.cte_mean:.3f} osc={stats.steer_std:.3f} "
                f"hdg={stats.hdg_mean:.3f}]"
            )

    # ── Profile 빌드 & 발행 ────────────────────────────── #

    def _build_profile(self) -> Optional[np.ndarray]:
        if self._global_s is None or not self._sectors:
            return None

        N = len(self._global_s)
        profile = np.zeros((N, 4), dtype=float)
        profile[:, 0] = self._global_s

        for i, s_val in enumerate(self._global_s):
            sector = find_sector(self._sectors, s_val, self._track_length)
            p = sector.params if sector is not None else self._defaults
            profile[i, 1] = p.k
            profile[i, 2] = p.k_ff
            profile[i, 3] = p.lookahead_d

        return profile

    def _publish_profile(self) -> None:
        profile = self._build_profile()
        if profile is None:
            return
        msg = Float64MultiArray()
        msg.data = profile.flatten().tolist()
        self._profile_pub.publish(msg)

    # ── 저장 ──────────────────────────────────────────── #

    def _save(self) -> None:
        if not self._sectors:
            return
        save_yaml(self._yaml_path, self._sectors, self._defaults)
        self.get_logger().info(f"[TunerNode] Saved → {self._yaml_path}")

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
    node = TunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._save()
    finally:
        node.destroy_node()
        rclpy.shutdown()
