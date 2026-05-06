"""
Pure transform helpers (no ROS dependency).

원본 ROS1 노드는 tf::Quaternion + tf::Matrix3x3 의 getRPY 로 yaw 추출.
ROS2 에서는 tf2 가 별도 패키지 의존이라 직접 계산 (B-1 의 yaw_to_quaternion 역).
"""
from __future__ import annotations

import math


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Quaternion (x, y, z, w) 에서 yaw (z 축 회전, ZYX 오일러 순서) 추출.

    공식:
      yaw = atan2(2 (qw qz + qx qy), 1 - 2 (qy² + qz²))
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)
