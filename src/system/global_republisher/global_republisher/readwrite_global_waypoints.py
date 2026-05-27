"""
global_waypoints.json 읽기 (ROS2 native dict↔msg 변환).

원본: planner/gb_optimizer/src/readwrite_global_waypoints.py (ROS1, rospy_message_converter 사용).
이번 포팅: ROS2 의 rosidl_runtime_py.set_message_fields 로 대체. write 는 미포팅
(매핑 phase 책임이라 검증 / state_machine 진입에는 불필요).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from rosidl_runtime_py import set_message_fields
from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray
from f110_msgs.msg import WpntArray


@dataclass
class GlobalWaypointsData:
    """global_waypoints.json 한 파일 분량의 모든 메시지를 묶은 컨테이너."""
    map_info_str: String
    est_lap_time: Float32
    centerline_markers: MarkerArray
    centerline_waypoints: WpntArray
    global_traj_markers_iqp: MarkerArray
    global_traj_wpnts_iqp: WpntArray
    global_traj_markers_sp: MarkerArray
    global_traj_wpnts_sp: WpntArray
    trackbounds_markers: MarkerArray
    global_traj_vel_markers_sp: Optional[MarkerArray] = None  # HJ 가 3D 에서 추가, optional


# ROS1 → ROS2 마이그레이션 시 제거되었거나 구조가 바뀐 필드들.
# JSON 안에 남아 있으면 set_message_fields 가 AttributeError 던짐.
_LEGACY_FIELDS_TO_STRIP = {
    "seq",  # std_msgs/Header — ROS2 에서 제거 (sequence counter)
}
# ROS1 → ROS2 필드 이름 변경.
_LEGACY_FIELDS_TO_RENAME = {
    "secs": "sec",       # builtin_interfaces/Time
    "nsecs": "nanosec",  # builtin_interfaces/Time
}


def _strip_legacy_fields(value):
    """dict / list 재귀 순회: STRIP 키 제거 + RENAME 키 새 이름으로 변환."""
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            if k in _LEGACY_FIELDS_TO_STRIP:
                continue
            new_key = _LEGACY_FIELDS_TO_RENAME.get(k, k)
            cleaned[new_key] = _strip_legacy_fields(v)
        return cleaned
    if isinstance(value, list):
        return [_strip_legacy_fields(v) for v in value]
    return value


def _dict_to_msg(MsgType, d: dict):
    """단순 dict → ROS2 msg 변환 (set_message_fields wrapper, ROS1 legacy 필드 제거)."""
    msg = MsgType()
    set_message_fields(msg, _strip_legacy_fields(d))
    return msg


def read_global_waypoints(path: str) -> GlobalWaypointsData:
    """
    JSON 파일을 읽어 GlobalWaypointsData 로 deserialize.

    원본은 stack_master/maps/<map_name>/global_waypoints.json 에서 읽음.
    이번 포팅은 stack_master 미포팅이라 path 직접 인자로 받음.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"global_waypoints.json not found at {path}")

    with open(path) as f:
        d = json.load(f)

    return GlobalWaypointsData(
        map_info_str=_dict_to_msg(String, d["map_info_str"]),
        est_lap_time=_dict_to_msg(Float32, d["est_lap_time"]),
        centerline_markers=_dict_to_msg(MarkerArray, d["centerline_markers"]),
        centerline_waypoints=_dict_to_msg(WpntArray, d["centerline_waypoints"]),
        global_traj_markers_iqp=_dict_to_msg(MarkerArray, d["global_traj_markers_iqp"]),
        global_traj_wpnts_iqp=_dict_to_msg(WpntArray, d["global_traj_wpnts_iqp"]),
        global_traj_markers_sp=_dict_to_msg(MarkerArray, d["global_traj_markers_sp"]),
        global_traj_wpnts_sp=_dict_to_msg(WpntArray, d["global_traj_wpnts_sp"]),
        trackbounds_markers=_dict_to_msg(MarkerArray, d["trackbounds_markers"]),
        global_traj_vel_markers_sp=(
            _dict_to_msg(MarkerArray, d["global_traj_vel_markers_sp"])
            if "global_traj_vel_markers_sp" in d
            else None
        ),
    )
