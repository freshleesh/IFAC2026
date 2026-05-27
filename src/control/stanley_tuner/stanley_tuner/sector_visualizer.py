#!/usr/bin/env python3
"""섹터 맵 시각화 — matplotlib 플롯.

사용법:
    ros2 run stanley_tuner sector_visualizer --ros-args -p map:=midterm

기능:
  - global_waypoints.json 로드 → 섹터 자동 감지
  - stanley_params.yaml 있으면 튜닝된 파라미터도 표시
  - 섹터 타입별 컬러 코딩:
      직선   → 초록
      얕은코너 → 노랑
      중간코너 → 주황
      깊은코너 → 빨강
  - 각 섹터에 ID / kappa_max / 파라미터 텍스트 표시
"""
from __future__ import annotations

import json
import os
import sys
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import rclpy
from rclpy.node import Node

from stanley_tuner.sector_map import (
    detect_sectors, load_yaml, apply_yaml_override,
    CornerType, SectorParams,
)

# ──────────────── 컬러 ──────────────── #
_COLOR = {
    CornerType.STRAIGHT: "#3cb371",   # 초록
    CornerType.SHALLOW:  "#ffd700",   # 노랑
    CornerType.MEDIUM:   "#ff8c00",   # 주황
    CornerType.DEEP:     "#dc143c",   # 빨강
}
_ALPHA_BAND  = 0.55
_ALPHA_TRACK = 0.25


def _resolve_map_path(map_name: str) -> str:
    """stack_master/maps/<map>/ 경로 찾기 (src 우선, install fallback)."""
    ament_prefix = os.environ.get("AMENT_PREFIX_PATH", "")
    for prefix in ament_prefix.split(":"):
        if os.path.basename(prefix) == "stanley_tuner":
            ws = os.path.normpath(os.path.join(prefix, "..", "..", ".."))
            # src 경로 (develop 모드)
            src = os.path.join(ws, "src", "system", "stack_master", "maps", map_name)
            if os.path.isdir(src):
                return src
    # install 경로 fallback
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("stack_master")
        return os.path.join(share, "maps", map_name)
    except Exception:
        return ""


def _load_waypoints(map_dir: str):
    """global_waypoints.json → (x, y, kappa, s) numpy arrays."""
    json_path = os.path.join(map_dir, "global_waypoints.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"global_waypoints.json not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    wpnts = data["global_traj_wpnts_iqp"]["wpnts"]
    x     = np.array([w["x_m"]       for w in wpnts])
    y     = np.array([w["y_m"]       for w in wpnts])
    kappa = np.array([w["kappa_radpm"] for w in wpnts])
    s     = np.array([w["s_m"]       for w in wpnts])
    psi   = np.array([w.get("psi_rad", 0.0) for w in wpnts])
    return x, y, kappa, s, psi


def _sector_xy(x, y, s, sec):
    """섹터에 속하는 waypoint의 x, y 반환."""
    mask = (s >= sec.start_s) & (s < sec.end_s)
    return x[mask], y[mask]


def _mid_xy(x, y, s, sec):
    """섹터 중간 지점의 x, y."""
    mid_s = (sec.start_s + sec.end_s) / 2
    idx = int(np.argmin(np.abs(s - mid_s)))
    return float(x[idx]), float(y[idx])


def plot_sectors(map_name: str, kappa_threshold: float = 0.15):
    map_dir = _resolve_map_path(map_name)
    if not map_dir:
        print(f"[sector_visualizer] ERROR: map '{map_name}' not found", file=sys.stderr)
        return

    x, y, kappa, s, psi = _load_waypoints(map_dir)
    wpnts_arr = np.column_stack([x, y, np.zeros_like(x), psi, s])  # (N,5)

    # 섹터 자동 감지
    sectors = detect_sectors(s, kappa, kappa_threshold=kappa_threshold)

    # YAML override (stanley_params.yaml 있으면 파라미터 덮어쓰기)
    yaml_path = os.path.join(map_dir, "stanley_params.yaml")
    yaml_sectors, defaults = load_yaml(yaml_path)
    if yaml_sectors:
        sectors = apply_yaml_override(sectors, yaml_sectors)
        print(f"[sector_visualizer] Loaded tuned params from {yaml_path}")
    else:
        print(f"[sector_visualizer] No stanley_params.yaml found — showing auto-detected sectors")

    # ────────────────────── 플롯 ────────────────────── #
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Stanley Sector Map — {map_name}  "
                 f"({len(sectors)} sectors, κ_thresh={kappa_threshold})",
                 fontsize=13, fontweight="bold")

    # ── 왼쪽: 트랙 + 섹터 컬러 ── #
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_title("Track — sector type", fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    # 트랙 전체 (회색 배경)
    ax.plot(x, y, color="#cccccc", lw=4, zorder=1)

    for sec in sectors:
        sx, sy = _sector_xy(x, y, s, sec)
        if len(sx) == 0:
            continue
        color = _COLOR[sec.corner_type]
        ax.plot(sx, sy, color=color, lw=5, alpha=_ALPHA_BAND, zorder=2)

        # 섹터 시작 수직 마커
        idx = int(np.argmin(np.abs(s - sec.start_s)))
        psi_i = float(psi[idx])
        dx = 0.5 * np.cos(psi_i + np.pi / 2)
        dy = 0.5 * np.sin(psi_i + np.pi / 2)
        ax.plot(
            [x[idx] - dx, x[idx] + dx],
            [y[idx] - dy, y[idx] + dy],
            color=color, lw=2.0, zorder=4,
        )

        # 섹터 ID + kappa 텍스트
        mx, my = _mid_xy(x, y, s, sec)
        ax.annotate(
            f"#{sec.id}\nκ={sec.kappa_max:.2f}",
            xy=(mx, my), xytext=(mx + 0.3, my + 0.3),
            fontsize=7, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.7, lw=0),
            zorder=5,
        )

    # 범례
    legend_patches = [
        mpatches.Patch(color=_COLOR[CornerType.STRAIGHT], label="straight"),
        mpatches.Patch(color=_COLOR[CornerType.SHALLOW],  label="shallow corner"),
        mpatches.Patch(color=_COLOR[CornerType.MEDIUM],   label="medium corner"),
        mpatches.Patch(color=_COLOR[CornerType.DEEP],     label="deep corner"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # ── 오른쪽: 파라미터 막대 그래프 ── #
    ax2 = axes[1]
    ax2.set_title("Tuned parameters per sector", fontsize=11)

    ids    = [s_.id for s_ in sectors]
    k_vals  = [s_.params.k          for s_ in sectors]
    kff_vals = [s_.params.k_ff * 100 for s_ in sectors]   # ×100 for visibility
    ld_vals  = [s_.params.lookahead_d if s_.params.lookahead_d > 0 else float("nan")
                for s_ in sectors]
    colors = [_COLOR[s_.corner_type] for s_ in sectors]

    bar_w = 0.25
    xs = np.arange(len(ids))

    ax2.bar(xs - bar_w, k_vals,   width=bar_w, label="k",           color=colors, alpha=0.8)
    ax2.bar(xs,          kff_vals, width=bar_w, label="k_ff ×100",   color=colors, alpha=0.5, hatch="//")
    ax2.bar(xs + bar_w,  ld_vals,  width=bar_w, label="lookahead_d", color=colors, alpha=0.5, hatch="xx")

    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"#{i}" for i in ids], fontsize=8)
    ax2.set_xlabel("Sector ID")
    ax2.set_ylabel("Value")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    # confidence 점선 오버레이
    for xi, sec in zip(xs, sectors):
        conf = sec.params.confidence
        ax2.text(xi, max(k_vals + [1.0]) * 1.05,
                 f"{conf:.0%}", ha="center", fontsize=6.5, color="gray")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────── #
# ROS2 Node wrapper (param으로 map 받기)   #
# ──────────────────────────────────────── #

class SectorVisualizerNode(Node):
    def __init__(self):
        super().__init__("sector_visualizer")
        self.declare_parameter("map", "")
        self.declare_parameter("kappa_threshold", 0.15)
        map_name = self.get_parameter("map").value
        kappa_threshold = float(self.get_parameter("kappa_threshold").value)
        if not map_name:
            self.get_logger().error("map parameter required. --ros-args -p map:=<mapname>")
            return
        self.get_logger().info(f"[SectorVisualizer] map={map_name}, κ_thresh={kappa_threshold}")
        plot_sectors(map_name, kappa_threshold)


def main(args=None):
    rclpy.init(args=args)
    node = SectorVisualizerNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # 직접 실행: python sector_visualizer.py midterm
    if len(sys.argv) < 2:
        print("Usage: python sector_visualizer.py <map_name> [kappa_threshold]")
        sys.exit(1)
    map_name = sys.argv[1]
    kt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    plot_sectors(map_name, kt)
