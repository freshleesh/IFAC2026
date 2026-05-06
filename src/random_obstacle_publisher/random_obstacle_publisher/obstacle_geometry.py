"""
Pure obstacle-geometry helpers (no ROS dependency).

원본 ROS1 노드의 update_obstacles / generate_random_obstacle / get_closest_point_on_traj /
publish_obstacles 의 lookahead 필터를 함수로 추출.
입력: numpy.random.Generator + waypoint dataclass + 파라미터.
출력: ObstacleSpec dataclass 리스트 (ROS 메시지 변환은 호출자가 담당).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from numpy.random import Generator


@dataclass
class WaypointSpec:
    """원본 f110_msgs/Wpnt 의 필요 필드만 추출."""
    id: int
    s_m: float
    d_left: float
    d_right: float


@dataclass
class ObstacleSpec:
    """원본 f110_msgs/Obstacle 의 필요 필드만 추출 (frenet 영역만)."""
    id: int
    s_start: float
    s_end: float
    d_left: float
    d_right: float
    is_actually_a_gap: bool = False


def get_closest_point_on_traj(
    s: float, gb_wpnts: Sequence[WaypointSpec]
) -> int:
    """주어진 s 와 가장 가까운 wpnt 의 id (원본 알고리즘 그대로: brute force min |s_m − s|²)."""
    min_d = float("inf")
    chosen_id = 0
    for wpt in gb_wpnts:
        d = (wpt.s_m - s) ** 2
        if d < min_d:
            min_d = d
            chosen_id = wpt.id
    return chosen_id


def generate_random_obstacle(
    obs_id: int,
    s_start: float,
    s_end: float,
    gen: Generator,
    obstacle_length: float,
    obstacle_width: float,
    obstacle_max_d_from_traj: float,
    gb_wpnts: Sequence[WaypointSpec],
) -> ObstacleSpec:
    """sector [s_start, s_end] 안에서 s 위치 + d 위치를 두 번 균등 추출."""
    p1 = float(gen.random())
    s_st = s_start + (s_end - s_start) * p1
    s_en = s_st + obstacle_length

    wpt_id = get_closest_point_on_traj(s_st, gb_wpnts)
    # 가까운 wpnt 의 트랙 폭 ± clamp 으로 obstacle 의 d 범위 결정
    track_right = -min(gb_wpnts[wpt_id].d_right, obstacle_max_d_from_traj)  # 음수
    track_left = min(
        gb_wpnts[wpt_id].d_left - obstacle_width, obstacle_max_d_from_traj
    )

    p2 = float(gen.random())
    d_right = track_right + (track_left - track_right) * p2
    d_left = d_right + obstacle_width

    return ObstacleSpec(
        id=obs_id,
        s_start=s_st,
        s_end=s_en,
        d_left=d_left,
        d_right=d_right,
        is_actually_a_gap=False,
    )


def build_sector_obstacles(
    n_sectors: int,
    gb_wpnts: Sequence[WaypointSpec],
    gen: Generator,
    obstacle_length: float,
    obstacle_width: float,
    obstacle_max_d_from_traj: float,
) -> list[ObstacleSpec]:
    """final_s 를 n_sectors 로 나눠 sector 마다 obstacle 1 개씩 생성."""
    final_s = gb_wpnts[-1].s_m
    s_spacing = final_s / n_sectors
    margin = max(0.5, obstacle_length)
    obstacles: list[ObstacleSpec] = []
    for sec in range(n_sectors):
        s_start = sec * s_spacing
        s_end = s_start + s_spacing - margin
        ob = generate_random_obstacle(
            obs_id=sec,
            s_start=s_start,
            s_end=s_end,
            gen=gen,
            obstacle_length=obstacle_length,
            obstacle_width=obstacle_width,
            obstacle_max_d_from_traj=obstacle_max_d_from_traj,
            gb_wpnts=gb_wpnts,
        )
        obstacles.append(ob)
    return obstacles


def select_obstacles_in_lookahead(
    obstacle_array: Sequence[ObstacleSpec],
    current_s: float,
    lookahead_distance: float,
    final_s: float,
) -> list[ObstacleSpec]:
    """publish_at_lookahead=True 일 때, 현재 s 로부터 lookahead 안에 들어오는 obstacle 만 선택.

    원본 주석: "too lazy to handle wrapping". wrap-around 정확도는 원본 그대로 재현.
    """
    selected: list[ObstacleSpec] = []
    for ob in obstacle_array:
        dist = math.fmod(current_s + lookahead_distance, final_s) - ob.s_start
        if 0 < dist < (lookahead_distance + 1):
            selected.append(ob)
    return selected
