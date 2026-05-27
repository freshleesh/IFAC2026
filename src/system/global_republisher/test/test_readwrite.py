"""
read_global_waypoints 단위 테스트.

ROS msg 타입은 import 하지만 rclpy 의존은 없음 — set_message_fields 만 사용.
"""
import json
import os
from pathlib import Path

import pytest

from global_republisher.readwrite_global_waypoints import (
    GlobalWaypointsData,
    read_global_waypoints,
)


# ---------- 픽스처 ----------

@pytest.fixture
def tiny_json_file(tmp_path):
    """최소 dict 로 valid JSON 만들어서 파일에 저장."""
    d = {
        "map_info_str": {"data": "test_map"},
        "est_lap_time": {"data": 12.5},
        "centerline_markers": {"markers": []},
        "centerline_waypoints": {
            "header": {"frame_id": "map"},
            "wpnts": [
                {"id": 0, "s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                 "d_left": 1.0, "d_right": 1.0},
                {"id": 1, "s_m": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 0.0,
                 "d_left": 1.0, "d_right": 1.0},
            ],
        },
        "global_traj_markers_iqp": {"markers": []},
        "global_traj_wpnts_iqp": {
            "header": {"frame_id": "map"},
            "wpnts": [
                {"id": 0, "s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                 "vx_mps": 5.0, "d_left": 1.0, "d_right": 1.0},
                {"id": 1, "s_m": 5.0, "x_m": 5.0, "y_m": 0.0, "z_m": 0.0,
                 "vx_mps": 6.0, "d_left": 1.0, "d_right": 1.0},
            ],
        },
        "global_traj_markers_sp": {"markers": []},
        "global_traj_wpnts_sp": {"wpnts": []},
        "trackbounds_markers": {"markers": []},
    }
    f = tmp_path / "global_waypoints.json"
    f.write_text(json.dumps(d))
    return str(f)


# ---------- read_global_waypoints ----------

def test_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_global_waypoints(str(tmp_path / "nope.json"))


def test_minimum_dict_load(tiny_json_file):
    data = read_global_waypoints(tiny_json_file)
    assert isinstance(data, GlobalWaypointsData)
    assert data.map_info_str.data == "test_map"
    assert data.est_lap_time.data == pytest.approx(12.5)


def test_centerline_wpnts_loaded(tiny_json_file):
    data = read_global_waypoints(tiny_json_file)
    assert len(data.centerline_waypoints.wpnts) == 2
    assert data.centerline_waypoints.wpnts[1].s_m == 1.0


def test_global_traj_iqp_track_length(tiny_json_file):
    """노드가 startup 에 사용하는 마지막 wpnt 의 s_m 값."""
    data = read_global_waypoints(tiny_json_file)
    assert data.global_traj_wpnts_iqp.wpnts[-1].s_m == 5.0
    assert data.global_traj_wpnts_iqp.wpnts[0].vx_mps == 5.0


def test_optional_vel_markers_missing_is_none(tiny_json_file):
    """global_traj_vel_markers_sp 가 JSON 에 없으면 None."""
    data = read_global_waypoints(tiny_json_file)
    assert data.global_traj_vel_markers_sp is None


def test_optional_vel_markers_present(tmp_path):
    """JSON 에 있으면 MarkerArray 로 deserialize."""
    base = {
        "map_info_str": {"data": ""},
        "est_lap_time": {"data": 0.0},
        "centerline_markers": {"markers": []},
        "centerline_waypoints": {"wpnts": []},
        "global_traj_markers_iqp": {"markers": []},
        "global_traj_wpnts_iqp": {"wpnts": []},
        "global_traj_markers_sp": {"markers": []},
        "global_traj_wpnts_sp": {"wpnts": []},
        "trackbounds_markers": {"markers": []},
        "global_traj_vel_markers_sp": {"markers": []},
    }
    f = tmp_path / "g.json"
    f.write_text(json.dumps(base))
    data = read_global_waypoints(str(f))
    assert data.global_traj_vel_markers_sp is not None
    assert len(data.global_traj_vel_markers_sp.markers) == 0


# ---------- ROS1 ws 의 실제 JSON (있으면) ----------

REAL_JSON = os.path.expanduser(
    "~/unicorn_ws/ICRA2026_HJ/stack_master/maps/gazebo_wall_2/global_waypoints.json"
)


@pytest.mark.skipif(not os.path.exists(REAL_JSON), reason="real ROS1 JSON 없음")
def test_real_gazebo_wall_2_loads():
    data = read_global_waypoints(REAL_JSON)
    assert len(data.global_traj_wpnts_iqp.wpnts) > 0
    track_length = data.global_traj_wpnts_iqp.wpnts[-1].s_m
    assert track_length > 0
