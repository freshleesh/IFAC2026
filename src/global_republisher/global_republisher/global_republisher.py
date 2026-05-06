#!/usr/bin/env python3
"""
Global trajectory republisher (ROS2 Jazzy port).

원본 ROS1: planner/gb_optimizer/src/global_trajectory_publisher.py (173L).
이번 포팅: SH ROS2 Jazzy.

책임:
1. 매핑 phase 에서 만든 global_waypoints.json 을 한 번 로드.
2. 0.5 Hz 로 트랙 관련 토픽 sticky 발행 (상태 유지) — state_machine /
   planner / frenet 노드들의 공통 의존.
3. 자기가 발행하는 토픽도 sub → 외부 노드 (vel_planner 등) 가 같은
   토픽으로 발행하면 그것을 대신 republish (그래서 republisher).
4. /global_waypoints 의 마지막 wpnt s_m 값을 ROS2 parameter
   `track_length` 에 set (state_machine 등이 시작 시 읽음).

발행 토픽 (모두 0.5 Hz):
  - /global_waypoints (WpntArray) + /markers (MarkerArray)
  - /global_waypoints/shortest_path + /markers
  - /centerline_waypoints + /markers
  - /trackbounds/markers
  - /map_infos (String), /estimated_lap_time (Float32)
  - /lattice_viz (MarkerArray, optional)
  - /global_waypoints/vel_markers (MarkerArray, optional)
  - /global_waypoints/vel_markers_tuned — 매 tick 새로 빌드 (vx_mps 기반)

파라미터:
  - map_path (string): global_waypoints.json 절대경로 (지정 시 우선)
  - map (string): map 이름 (fallback: ROS1 ws stack_master/maps/<map>/global_waypoints.json)
  - publish_rate_hz (float, default 0.5)
"""
from __future__ import annotations

import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray, Marker
from f110_msgs.msg import WpntArray

from global_republisher.readwrite_global_waypoints import (
    GlobalWaypointsData,
    read_global_waypoints,
)


# vel_markers_tuned 빌드 시 vx → marker height 스케일 (원본 그대로)
VEL_MARKER_SCALE_FACTOR = 0.1317


class GlobalRepublisher(Node):
    def __init__(self) -> None:
        super().__init__("global_republisher")

        self.declare_parameter("map_path", "")
        self.declare_parameter("map", "gazebo_wall_2")
        self.declare_parameter("publish_rate_hz", 0.5)

        map_path = self.get_parameter("map_path").value
        map_name = self.get_parameter("map").value
        publish_rate_hz = self.get_parameter("publish_rate_hz").value

        # JSON 로드 (한 번만)
        json_path = self._resolve_path(map_path, map_name)
        self.get_logger().info(f"[GlobalRepublisher] Reading {json_path}")
        try:
            self._data: GlobalWaypointsData = read_global_waypoints(json_path)
        except FileNotFoundError as e:
            self.get_logger().error(f"[GlobalRepublisher] {e}")
            raise

        # track_length parameter set — state_machine 이 startup 에서 읽음
        # (원본은 self.glb_wpnts_cb 에서 set 했지만, 우리는 JSON 로드 직후 한 번만 set)
        track_length = (
            self._data.global_traj_wpnts_iqp.wpnts[-1].s_m
            if self._data.global_traj_wpnts_iqp.wpnts
            else 0.0
        )
        self.declare_parameter("track_length", track_length)
        self.get_logger().info(
            f"[GlobalRepublisher] track_length={track_length:.2f}m, "
            f"publishing at {publish_rate_hz} Hz"
        )

        # 자기 발행 토픽 sub — 외부 노드가 같은 토픽 발행하면 그걸로 갱신
        self.create_subscription(
            WpntArray, "/global_waypoints", self._on_glb_wpnts, 10
        )
        self.create_subscription(
            MarkerArray, "/global_waypoints/markers", self._on_glb_markers, 10
        )
        self.create_subscription(
            MarkerArray, "/trackbounds/markers", self._on_track_bounds, 10
        )
        self.create_subscription(
            WpntArray, "/global_waypoints/shortest_path", self._on_sp_wpnts, 10
        )
        self.create_subscription(
            MarkerArray, "/global_waypoints/shortest_path/markers", self._on_sp_markers, 10
        )
        self.create_subscription(
            WpntArray, "/centerline_waypoints", self._on_centerline_wpnts, 10
        )
        self.create_subscription(
            MarkerArray, "/centerline_waypoints/markers", self._on_centerline_markers, 10
        )
        self.create_subscription(String, "/map_infos", self._on_map_infos, 10)
        self.create_subscription(
            Float32, "/estimated_lap_time", self._on_est_lap_time, 10
        )
        self.create_subscription(MarkerArray, "/lattice_viz", self._on_lattice, 10)
        self.create_subscription(
            MarkerArray,
            "/global_waypoints/vel_markers",
            self._on_vel_markers,
            10,
        )

        # Pub
        self._glb_wpnts_pub = self.create_publisher(WpntArray, "/global_waypoints", 10)
        self._glb_markers_pub = self.create_publisher(
            MarkerArray, "/global_waypoints/markers", 10
        )
        self._trackbounds_pub = self.create_publisher(
            MarkerArray, "/trackbounds/markers", 10
        )
        self._sp_wpnts_pub = self.create_publisher(
            WpntArray, "/global_waypoints/shortest_path", 10
        )
        self._sp_markers_pub = self.create_publisher(
            MarkerArray, "/global_waypoints/shortest_path/markers", 10
        )
        self._centerline_wpnts_pub = self.create_publisher(
            WpntArray, "/centerline_waypoints", 10
        )
        self._centerline_markers_pub = self.create_publisher(
            MarkerArray, "/centerline_waypoints/markers", 10
        )
        self._map_infos_pub = self.create_publisher(String, "/map_infos", 10)
        self._est_lap_time_pub = self.create_publisher(Float32, "/estimated_lap_time", 10)
        self._lattice_pub = self.create_publisher(MarkerArray, "/lattice_viz", 10)
        self._vel_markers_pub = self.create_publisher(
            MarkerArray, "/global_waypoints/vel_markers", 10
        )
        self._vel_markers_tuned_pub = self.create_publisher(
            MarkerArray, "/global_waypoints/vel_markers_tuned", 10
        )

        # tick
        self.create_timer(1.0 / publish_rate_hz, self._tick)

    # ---------- 경로 해결 ----------

    def _resolve_path(self, map_path: str, map_name: str) -> str:
        if map_path:
            return map_path
        # fallback: ROS1 ws 의 stack_master/maps/<map>/global_waypoints.json (검증 편의)
        fallback = os.path.expanduser(
            f"~/unicorn_ws/ICRA2026_HJ/stack_master/maps/{map_name}/global_waypoints.json"
        )
        return fallback

    # ---------- 외부 발행 캡처 콜백 (모두 self._data 의 필드 갱신) ----------

    def _on_glb_wpnts(self, msg: WpntArray) -> None:
        self._data.global_traj_wpnts_iqp = msg

    def _on_glb_markers(self, msg: MarkerArray) -> None:
        self._data.global_traj_markers_iqp = msg

    def _on_track_bounds(self, msg: MarkerArray) -> None:
        self._data.trackbounds_markers = msg

    def _on_sp_wpnts(self, msg: WpntArray) -> None:
        self._data.global_traj_wpnts_sp = msg

    def _on_sp_markers(self, msg: MarkerArray) -> None:
        self._data.global_traj_markers_sp = msg

    def _on_centerline_wpnts(self, msg: WpntArray) -> None:
        self._data.centerline_waypoints = msg

    def _on_centerline_markers(self, msg: MarkerArray) -> None:
        self._data.centerline_markers = msg

    def _on_map_infos(self, msg: String) -> None:
        self._data.map_info_str = msg

    def _on_est_lap_time(self, msg: Float32) -> None:
        self._data.est_lap_time = msg

    def _on_lattice(self, msg: MarkerArray) -> None:
        # lattice 는 별도 컨테이너 필드가 없으니 인스턴스 attr 로 보관
        self._lattice = msg

    def _on_vel_markers(self, msg: MarkerArray) -> None:
        self._data.global_traj_vel_markers_sp = msg

    # ---------- 메인 tick ----------

    def _tick(self) -> None:
        d = self._data
        if d.global_traj_wpnts_iqp.wpnts and d.global_traj_markers_iqp.markers:
            self._glb_wpnts_pub.publish(d.global_traj_wpnts_iqp)
            self._glb_markers_pub.publish(d.global_traj_markers_iqp)
        if d.global_traj_wpnts_sp.wpnts and d.global_traj_markers_sp.markers:
            self._sp_wpnts_pub.publish(d.global_traj_wpnts_sp)
            self._sp_markers_pub.publish(d.global_traj_markers_sp)
        if d.centerline_waypoints.wpnts and d.centerline_markers.markers:
            self._centerline_wpnts_pub.publish(d.centerline_waypoints)
            self._centerline_markers_pub.publish(d.centerline_markers)
        if d.trackbounds_markers.markers:
            self._trackbounds_pub.publish(d.trackbounds_markers)
        if d.map_info_str.data:
            self._map_infos_pub.publish(d.map_info_str)
        # est_lap_time 은 0 도 valid 값일 수 있으므로 None 체크 대신 항상 발행
        self._est_lap_time_pub.publish(d.est_lap_time)
        if hasattr(self, "_lattice") and self._lattice is not None:
            self._lattice_pub.publish(self._lattice)
        if d.global_traj_vel_markers_sp is not None:
            self._vel_markers_pub.publish(d.global_traj_vel_markers_sp)
        # tuned vel_markers — 매 tick 새로 빌드 (현재 wpnts 의 vx_mps 반영)
        if d.global_traj_wpnts_iqp.wpnts:
            self._vel_markers_tuned_pub.publish(
                _build_vel_markers_tuned(d.global_traj_wpnts_iqp)
            )


def _build_vel_markers_tuned(wpnts: WpntArray) -> MarkerArray:
    """원본 cc 의 vel_markers_tuned 빌드 로직 (cylinder height = vx_mps * scale)."""
    arr = MarkerArray()
    for i, wp in enumerate(wpnts.wpnts):
        m = Marker()
        m.header.frame_id = "map"
        m.type = Marker.CYLINDER
        height = wp.vx_mps * VEL_MARKER_SCALE_FACTOR
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = height
        m.color.a = 0.5
        m.color.b = 1.0
        m.id = i
        m.pose.position.x = wp.x_m
        m.pose.position.y = wp.y_m
        m.pose.position.z = height / 2.0
        m.pose.orientation.w = 1.0
        arr.markers.append(m)
    return arr


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = GlobalRepublisher()
    except FileNotFoundError:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
