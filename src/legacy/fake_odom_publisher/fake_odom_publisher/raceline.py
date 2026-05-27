"""
Pure functions for raceline waypoint interpolation.

ROS2 Node 와 분리 (단위 테스트 가능). 입력: waypoints (list of dict 또는 native).
출력: 보간된 (x, y, z, psi, vx, vz) 튜플.

원본 ROS1 fake_odom_publisher.py 의 run() 루프 안 수치 로직만 추출.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass
class Waypoint:
    s_m: float
    x_m: float
    y_m: float
    z_m: float
    psi_rad: float
    vx_mps: float


@dataclass
class Pose3D:
    x: float
    y: float
    z: float
    psi: float
    vx: float
    vz: float


def find_segment_index(wpnts: Sequence[Waypoint], s_current: float) -> int:
    """주어진 s 가 들어가는 segment 의 시작 인덱스를 반환."""
    n = len(wpnts)
    for i in range(n - 1):
        if wpnts[i + 1].s_m > s_current:
            return i
    return n - 1


def interpolate_pose(
    wpnts: Sequence[Waypoint],
    s_current: float,
    speed_scale: float,
    total_s: float,
) -> Pose3D:
    """s_current 위치의 raceline pose 를 두 인접 wpnt 선형보간으로 계산."""
    n = len(wpnts)
    idx = find_segment_index(wpnts, s_current)
    w = wpnts[idx]
    w_next = wpnts[(idx + 1) % n]

    ds = w_next.s_m - w.s_m
    if ds <= 0:
        # wrap-around segment (마지막 wpnt → 첫 wpnt)
        ds = total_s - w.s_m + w_next.s_m
    t = (s_current - w.s_m) / ds if ds > 1e-6 else 0.0
    t = max(0.0, min(1.0, t))

    x = w.x_m + t * (w_next.x_m - w.x_m)
    y = w.y_m + t * (w_next.y_m - w.y_m)
    z = w.z_m + t * (w_next.z_m - w.z_m)
    psi = w.psi_rad
    vx = w.vx_mps * speed_scale

    # 3D 슬로프로 vz 계산: dz/ds_3d * vx
    dz = w_next.z_m - w.z_m
    dx = w_next.x_m - w.x_m
    dy = w_next.y_m - w.y_m
    ds_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    slope = dz / ds_3d if ds_3d > 1e-6 else 0.0
    vz = vx * slope

    return Pose3D(x=x, y=y, z=z, psi=psi, vx=vx, vz=vz)


def yaw_to_quaternion(psi: float) -> tuple[float, float, float, float]:
    """yaw 만 갖는 quaternion (x, y, z, w). roll=pitch=0 가정."""
    half = 0.5 * psi
    return (0.0, 0.0, math.sin(half), math.cos(half))


def waypoints_from_dicts(wpnt_dicts: List[dict]) -> List[Waypoint]:
    """global_waypoints.json 의 wpnts dict 리스트를 Waypoint dataclass 리스트로 변환."""
    return [
        Waypoint(
            s_m=w["s_m"],
            x_m=w["x_m"],
            y_m=w["y_m"],
            z_m=w["z_m"],
            psi_rad=w["psi_rad"],
            vx_mps=w["vx_mps"],
        )
        for w in wpnt_dicts
    ]
