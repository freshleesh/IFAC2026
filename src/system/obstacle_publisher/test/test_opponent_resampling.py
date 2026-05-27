"""opponent_resampling 순수 함수 단위 테스트 — ROS 의존 없음."""
import numpy as np
import pytest

from obstacle_publisher.opponent_resampling import (
    advance_s_with_wrap,
    find_nearest_idx,
    resample_opponent_d_on_ego_s,
    sort_opponent_by_s,
)


# ---------- sort_opponent_by_s ----------

def test_sort_unsorted():
    s, d = sort_opponent_by_s([3.0, 1.0, 2.0], [0.3, 0.1, 0.2])
    assert list(s) == [1.0, 2.0, 3.0]
    assert list(d) == [0.1, 0.2, 0.3]


def test_sort_already_sorted():
    s, d = sort_opponent_by_s([0.0, 1.0, 2.0], [0.1, 0.2, 0.3])
    assert list(s) == [0.0, 1.0, 2.0]


def test_sort_with_negative():
    s, d = sort_opponent_by_s([0.0, -1.0, 2.0], [0.0, -0.1, 0.2])
    assert list(s) == [-1.0, 0.0, 2.0]


# ---------- resample_opponent_d_on_ego_s ----------

def test_resample_exact_match():
    """ego_s 가 opponent_s_sorted 와 같으면 d 그대로."""
    ego_s = [0.0, 1.0, 2.0]
    d = resample_opponent_d_on_ego_s(ego_s, [0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    assert list(d) == [0.0, 0.5, 1.0]


def test_resample_linear_interp():
    """ego_s=[0.5, 1.5] 에서 선형보간 → d=[0.25, 0.75]."""
    d = resample_opponent_d_on_ego_s([0.5, 1.5], [0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    assert list(d) == pytest.approx([0.25, 0.75])


def test_resample_outside_range_uses_endpoint():
    """np.interp 의 default — 범위 밖은 endpoint 값 (clamp)."""
    d = resample_opponent_d_on_ego_s([-1.0, 5.0], [0.0, 1.0], [0.5, 0.7])
    assert d[0] == pytest.approx(0.5)
    assert d[1] == pytest.approx(0.7)


# ---------- find_nearest_idx ----------

def test_find_nearest_exact():
    assert find_nearest_idx([0.0, 1.0, 2.0, 3.0], 2.0) == 2


def test_find_nearest_in_between():
    # 1.4 → idx 1 (|1.4-1|=0.4 < |1.4-2|=0.6)
    assert find_nearest_idx([0.0, 1.0, 2.0, 3.0], 1.4) == 1
    # 1.6 → idx 2
    assert find_nearest_idx([0.0, 1.0, 2.0, 3.0], 1.6) == 2


def test_find_nearest_outside_range():
    assert find_nearest_idx([0.0, 1.0, 2.0], -5.0) == 0
    assert find_nearest_idx([0.0, 1.0, 2.0], 100.0) == 2


# ---------- advance_s_with_wrap ----------

def test_advance_no_wrap():
    assert advance_s_with_wrap(2.0, 0.5, 10.0) == pytest.approx(2.5)


def test_advance_with_wrap():
    """s=9.5, ds=1.0, max=10.0 → (10.5) % 10 = 0.5."""
    assert advance_s_with_wrap(9.5, 1.0, 10.0) == pytest.approx(0.5)


def test_advance_zero_ds_keeps_same():
    assert advance_s_with_wrap(3.7, 0.0, 10.0) == pytest.approx(3.7)
