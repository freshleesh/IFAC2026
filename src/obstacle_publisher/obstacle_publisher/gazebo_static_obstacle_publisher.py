#!/usr/bin/env python3
"""Gazebo static obstacle → /tracking/obstacles publisher (ROS2 포팅).

원본 ROS1: f110_utils/nodes/obstacle_publisher/src/gazebo_static_obstacle_publisher.py.
Gazebo 환경의 정적 장애물을 detect + multi_tracking 우회하고 직접
/tracking/obstacles (ObstacleArray) 으로 발행. 3D Frenet 변환 + 시각화.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray
from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter


class GazeboStaticObstaclePublisher(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_static_obstacle_publisher")

        self.declare_parameter("rate", 10.0)
        self.declare_parameter("in_poses_topic", "/gazebo/static_obstacles/poses")
        self.declare_parameter("in_radii_topic", "/gazebo/static_obstacles/radii")
        self.declare_parameter("out_obstacles_topic", "/tracking/obstacles")
        self.declare_parameter("out_markers_topic", "/gazebo_static_obstacle_markers")
        self.declare_parameter("track_length", 0.0)

        rate = float(self.get_parameter("rate").value)
        self.in_poses_topic = self.get_parameter("in_poses_topic").value
        self.in_radii_topic = self.get_parameter("in_radii_topic").value
        self.out_obstacles_topic = self.get_parameter("out_obstacles_topic").value
        self.out_markers_topic = self.get_parameter("out_markers_topic").value
        self.track_length = float(self.get_parameter("track_length").value)

        self.poses = None
        self.radii = None
        self.converter: FrenetConverter | None = None

        # /global_waypoints 한 번 받으면 FrenetConverter 빌드
        self.create_subscription(WpntArray, "/global_waypoints", self._wpnts_cb, 10)

        # publishers
        self.obstacle_pub = self.create_publisher(ObstacleArray, self.out_obstacles_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.out_markers_topic, 10)

        # subscribers
        self.create_subscription(PoseArray, self.in_poses_topic, self._poses_cb, 10)
        self.create_subscription(Float64MultiArray, self.in_radii_topic, self._radii_cb, 10)

        # tick
        self.create_timer(1.0 / max(rate, 1e-3), self._tick)

        self.get_logger().info(
            f"[GazeboStaticObsPub] init — waiting for /global_waypoints + obstacles; rate={rate}Hz"
        )

    # ---- callbacks ----

    def _wpnts_cb(self, msg: WpntArray) -> None:
        if self.converter is not None:
            return
        wpts = np.array([[w.x_m, w.y_m, w.z_m] for w in msg.wpnts])
        if len(wpts) < 2:
            return
        try:
            self.converter = FrenetConverter(wpts[:, 0], wpts[:, 1], wpts[:, 2])
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[GazeboStaticObsPub] FrenetConverter build failed: {e}")
            return
        if self.track_length <= 0:
            # /global_republisher/track_length 가 ROS2 에서는 별도 노드 파라미터 → fallback: raceline 길이
            self.track_length = float(self.converter.raceline_length)
        self.get_logger().warn(
            f"[GazeboStaticObsPub] FrenetConverter ready, track_length={self.track_length:.2f}m"
        )

    def _poses_cb(self, msg: PoseArray) -> None:
        self.poses = msg.poses

    def _radii_cb(self, msg: Float64MultiArray) -> None:
        self.radii = msg.data

    # ---- tick ----

    def _tick(self) -> None:
        if self.poses is None or self.radii is None or self.converter is None:
            return
        self._publish_obstacles()

    def _publish_obstacles(self) -> None:
        now = self.get_clock().now().to_msg()
        obstacle_msg = ObstacleArray()
        obstacle_msg.header.stamp = now
        obstacle_msg.header.frame_id = "map"

        marker_msg = MarkerArray()
        n_obs = min(len(self.poses), len(self.radii))

        for i in range(n_obs):
            pose = self.poses[i]
            radius = float(self.radii[i])
            x, y, z = pose.position.x, pose.position.y, pose.position.z

            try:
                s_d = self.converter.get_frenet_3d(
                    np.array([x]), np.array([y]), np.array([z])
                )
                s_center = float(s_d[0][0])
                d_center = float(s_d[1][0])
            except Exception as e:  # noqa: BLE001
                # ROS2 에는 logwarn_throttle 이 없음 — get_logger().warn (every tick) 또는 단순화
                self.get_logger().warn(f"[GazeboStaticObsPub] Frenet conv failed: {e}")
                continue

            obs = Obstacle()
            obs.id = i
            obs.x_m = float(x)
            obs.y_m = float(y)
            obs.z_m = float(z)
            obs.s_center = s_center
            obs.d_center = d_center
            obs.s_start = (s_center - radius) % self.track_length
            obs.s_end = (s_center + radius) % self.track_length
            obs.d_right = d_center - radius
            obs.d_left = d_center + radius
            obs.size = radius * 2
            obs.vs = 0.0
            obs.vd = 0.0
            obs.is_static = True
            obs.is_visible = True
            obs.is_actually_a_gap = False
            obs.sector_id = -1
            obs.in_static_obs_sector = False
            obstacle_msg.obstacles.append(obs)

            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = now
            mrk.ns = "gazebo_static_obs"
            mrk.id = i
            mrk.type = Marker.CYLINDER
            mrk.action = Marker.ADD
            mrk.pose.position.x = float(x)
            mrk.pose.position.y = float(y)
            mrk.pose.position.z = float(z)
            mrk.pose.orientation.w = 1.0
            mrk.scale.x = radius * 2
            mrk.scale.y = radius * 2
            mrk.scale.z = 0.3
            mrk.color.r = 1.0
            mrk.color.g = 0.3
            mrk.color.b = 0.0
            mrk.color.a = 0.8
            marker_msg.markers.append(mrk)

        self.obstacle_pub.publish(obstacle_msg)
        self.marker_pub.publish(marker_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboStaticObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # shutdown 시 빈 ObstacleArray 발행 (consumer 잔재 정리)
        node.obstacle_pub.publish(ObstacleArray())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
