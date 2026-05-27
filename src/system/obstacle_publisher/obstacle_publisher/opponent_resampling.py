"""
opponent waypoints 를 ego s_array 위에 resample 하는 순수 numpy 헬퍼.

원본 ros_loop 에서 frenet service 두 번 호출 사이에 들어있던 numpy 로직만 추출:
1. opponent (s_sorted, d_sorted) 정렬
2. ego s_array 위에 d 재샘플 (np.interp)
3. 결과 (s_array, d_resampled) 가 frenet → cartesian 변환의 입력
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def sort_opponent_by_s(
    opponent_s: Sequence[float], opponent_d: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """opponent (s, d) 를 s 오름차순으로 정렬."""
    s = np.asarray(opponent_s, dtype=float)
    d = np.asarray(opponent_d, dtype=float)
    order = np.argsort(s)
    return s[order], d[order]


def resample_opponent_d_on_ego_s(
    ego_s: Sequence[float],
    opponent_s_sorted: Sequence[float],
    opponent_d_sorted: Sequence[float],
) -> np.ndarray:
    """ego s_array 위 각 s 에서 opponent d 를 선형보간."""
    return np.interp(ego_s, opponent_s_sorted, opponent_d_sorted)


def find_nearest_idx(arr: Sequence[float], target: float) -> int:
    """원본의 np.abs(opponent_s_array - s).argmin() 추출."""
    return int(np.argmin(np.abs(np.asarray(arr) - target)))


def advance_s_with_wrap(
    s_current: float, ds: float, max_s: float
) -> float:
    """s += ds 후 % max_s — 원본의 (s + speed*looptime) % max_s 패턴."""
    return (s_current + ds) % max_s
