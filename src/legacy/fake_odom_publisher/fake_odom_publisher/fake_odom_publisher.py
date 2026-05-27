#!/usr/bin/env python3
"""
3D fake odometry publisher (ROS2 Jazzy port).

global_waypoints.json 을 읽어 raceline 을 따라 주행하는 가짜 odom 을 발행.
실제 하드웨어 / GLIM 없이 frenet · state_machine 동작 검증용.

원본: ICRA2026_HJ stack_master/scripts/fake_odom_publisher.py (HJ).
Port: SH (ROS2 Jazzy).
"""
from __future__ import annotations

import json
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from fake_odom_publisher.raceline import (
    Pose3D,
    interpolate_pose,
    waypoints_from_dicts,
    yaw_to_quaternion,
)


class FakeOdomPublisher(Node):
    """
    파라미터:
      - map (string, default "gazebo_wall_2_3d_rc_car_10th_timeoptimal"):
        share/fake_odom_publisher/maps 또는 절대경로 아래의 global_waypoints.json 디렉터리명.
      - waypoints_path (string, optional): 직접 json 경로 지정 (map 보다 우선).
      - speed_scale (float, default 1.0): vx_mps 에 곱할 스케일.
      - rate (float, default 50.0): 발행 주기 [Hz].
      - frame_id (string, default "map"), child_frame_id (string, default "base_link").
      - topic (string, default "/glim_ros/base_odom"): 호환성 위해 원본 topic 유지.
    """

    def __init__(self) -> None:
        super().__init__("fake_odom_publisher")

        # 파라미터 선언 + 읽기
        self.declare_parameter("map", "gazebo_wall_2")
        self.declare_parameter("waypoints_path", "")
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("topic", "/glim_ros/base_odom")

        map_name = self.get_parameter("map").get_parameter_value().string_value
        wpnt_path = (
            self.get_parameter("waypoints_path").get_parameter_value().string_value
        )
        speed_scale = (
            self.get_parameter("speed_scale").get_parameter_value().double_value
        )
        rate_hz = self.get_parameter("rate").get_parameter_value().double_value
        self._frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self._child_frame_id = (
            self.get_parameter("child_frame_id").get_parameter_value().string_value
        )
        topic = self.get_parameter("topic").get_parameter_value().string_value

        # 웨이포인트 로드
        json_path = self._resolve_waypoints_path(wpnt_path, map_name)
        with open(json_path) as f:
            data = json.load(f)
        self._wpnts = waypoints_from_dicts(data["global_traj_wpnts_sp"]["wpnts"])
        self._total_s = self._wpnts[-1].s_m
        self._speed_scale = speed_scale
        self.get_logger().info(
            f"Loaded {len(self._wpnts)} waypoints from {json_path}"
        )

        # 발행자 + 타이머
        self._pub = self.create_publisher(Odometry, topic, 10)
        self._dt = 1.0 / rate_hz
        self._s_current = 0.0
        self._timer = self.create_timer(self._dt, self._tick)

    def _resolve_waypoints_path(self, explicit: str, map_name: str) -> str:
        """waypoints_path 가 비어있지 않으면 그대로, 아니면 share/<pkg>/maps/<map>/global_waypoints.json."""
        if explicit:
            return explicit
        # ament_python 설치 후 share 경로 사용. 개발 환경 fallback 으로 원본 ROS1 ws 도 시도.
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("fake_odom_publisher")
        candidate = os.path.join(share, "maps", map_name, "global_waypoints.json")
        if os.path.exists(candidate):
            return candidate
        tried = [candidate]
        env_dir = os.environ.get("IFAC_MAPS_DIR")
        if env_dir:
            p = os.path.join(os.path.expanduser(env_dir), map_name, "global_waypoints.json")
            tried.append(p)
            if os.path.exists(p):
                return p
        mac_default = os.path.expanduser(
            f"~/ros2_ws/src/IFAC2026_SH/maps/{map_name}/global_waypoints.json"
        )
        tried.append(mac_default)
        if os.path.exists(mac_default):
            return mac_default
        ubuntu_legacy = os.path.expanduser(
            f"~/unicorn_ws/ICRA2026_HJ/stack_master/maps/{map_name}/global_waypoints.json"
        )
        tried.append(ubuntu_legacy)
        if os.path.exists(ubuntu_legacy):
            return ubuntu_legacy
        raise FileNotFoundError(
            "global_waypoints.json not found. Tried:\n  " + "\n  ".join(tried)
        )

    def _tick(self) -> None:
        pose = interpolate_pose(
            self._wpnts, self._s_current, self._speed_scale, self._total_s
        )
        self._publish_odom(pose)
        # s 진행 (주기 dt 동안 vx 만큼)
        self._s_current += pose.vx * self._dt
        if self._s_current >= self._total_s:
            self._s_current -= self._total_s

    def _publish_odom(self, pose: Pose3D) -> None:
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.position.z = pose.z
        qx, qy, qz, qw = yaw_to_quaternion(pose.psi)
        odom.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        odom.twist.twist.linear.x = pose.vx  # body frame forward
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = pose.vz
        self._pub.publish(odom)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
