#!/usr/bin/env python3
"""Opponent Collision Detector (ROS2 포팅).

원본 ROS1: f110_utils/nodes/obstacle_publisher/src/collision_detector.py.
- /tracking/obstacles_truth (ground truth — 2D 맵 sim 의 perception 우회)
- /car_state/odom_frenet
- /global_waypoints
→ 충돌 시 /opponent_collision (Bool), /opponent_dist (Float32),
  /collision_marker (MarkerArray) 발행. 50 Hz.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
import tf_transformations

from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, WpntArray
from visualization_msgs.msg import Marker, MarkerArray


class OdCollisionDetector(Node):
    def __init__(self) -> None:
        super().__init__("collision_detector")

        # Sub
        self.create_subscription(ObstacleArray, "/tracking/obstacles_truth", self._od_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self._odom_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self._glb_wpnts_cb, 10)

        # Pub
        self.col_pub = self.create_publisher(Bool, "/opponent_collision", 10)
        self.opp_dist_pub = self.create_publisher(Float32, "/opponent_dist", 10)
        self.coll_marker_pub = self.create_publisher(MarkerArray, "/collision_marker", 10)

        self.first_visualization = True
        self.x_viz = 0.0
        self.y_viz = 0.0
        self.viz_q = tf_transformations.quaternion_from_euler(0, 0, 0)
        self.viz_counter = 0

        self.obs_arr = ObstacleArray()
        self.car_odom = Odometry()
        self.glb_waypoints: list = []

        # 50 Hz tick
        self.rate_hz = 50.0
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info("[Dummy OD] Collision Detector ready")

    # ---- Sub callbacks ----

    def _od_cb(self, data: ObstacleArray) -> None:
        self.obs_arr = data

    def _odom_cb(self, data: Odometry) -> None:
        self.car_odom = data

    def _glb_wpnts_cb(self, data: WpntArray) -> None:
        self.glb_waypoints = data.wpnts

    # ---- Tick ----

    def _tick(self) -> None:
        if not self.glb_waypoints:
            return

        collision_bool, min_dist_s, min_dist_d = self._collision_check(self.obs_arr, self.car_odom)

        if self.viz_counter > 0:
            self.viz_counter -= 1
            if self.viz_counter == 0:
                self._viz_collision(clear=True)

        col_msg = Bool()
        if collision_bool:
            col_msg.data = True
            self.viz_counter = int(self.rate_hz) * 2  # 2 sec 표시
            self._viz_collision(dist_s=min_dist_s, dist_d=min_dist_d, clear=False)
        else:
            col_msg.data = False
        self.col_pub.publish(col_msg)

        dist_msg = Float32()
        dist_msg.data = float(np.sqrt(min_dist_s ** 2 + min_dist_d ** 2))
        self.opp_dist_pub.publish(dist_msg)

    # ---- Collision logic ----

    def _collision_check(self, obs_arr: ObstacleArray, car_odom: Odometry):
        track_len = self.glb_waypoints[-2].s_m
        car_s = car_odom.pose.pose.position.x
        car_d = car_odom.pose.pose.position.y
        for obs in obs_arr.obstacles:
            od_s = obs.s_center
            od_d = obs.d_center
            if (od_s - car_s) % track_len < 0.55 and abs(car_d - od_d) < 0.35:
                self.get_logger().info(
                    f"[Dummy OD] Front Collision detected! Frenet s: {car_s:.2f} m"
                )
                return True, (od_s - car_s) % track_len, abs(car_d - od_d)
            if (car_s - od_s) % track_len < 0.25 and abs(car_d - od_d) < 0.30:
                self.get_logger().info(
                    f"[Dummy OD] Back Collision detected! Frenet s: {car_s:.2f} m"
                )
                return True, (car_s - od_s) % track_len, abs(car_d - od_d)
        return False, 100.0, 100.0

    # ---- Visualization ----

    def _viz_collision(self, dist_s: float = 0.0, dist_d: float = 0.0, clear: bool = False) -> None:
        if self.first_visualization:
            self.first_visualization = False
            wpnts = self.glb_waypoints
            i = len(wpnts) // 4
            x0, y0 = wpnts[i].x_m, wpnts[i].y_m
            x1, y1 = wpnts[i + 1].x_m, wpnts[i + 1].y_m
            xy_norm = -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1]) * 1.75 * wpnts[i].d_left
            yaw = np.arctan2(xy_norm[1], xy_norm[0])
            self.viz_q = tf_transformations.quaternion_from_euler(0, 0, yaw)
            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        coll_mrk = MarkerArray()
        marker_text = Marker()
        marker_text.header.frame_id = "map"
        marker_text.header.stamp = self.get_clock().now().to_msg()
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.text = (
            "" if clear else f"COLLISION: dist_s :{dist_s:.1f}, dist_d :{dist_d:.1f}m"
        )
        marker_text.scale.z = 1.0
        marker_text.color.r = 1.0
        marker_text.color.g = 0.0
        marker_text.color.b = 0.0
        marker_text.color.a = 1.0
        marker_text.pose.orientation.x = float(self.viz_q[0])
        marker_text.pose.orientation.y = float(self.viz_q[1])
        marker_text.pose.orientation.z = float(self.viz_q[2])
        marker_text.pose.orientation.w = float(self.viz_q[3])
        marker_text.pose.position.x = float(self.x_viz)
        marker_text.pose.position.y = float(self.y_viz)
        marker_text.pose.position.z = 0.0
        marker_text.id = 0
        coll_mrk.markers.append(marker_text)
        self.coll_marker_pub.publish(coll_mrk)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdCollisionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
