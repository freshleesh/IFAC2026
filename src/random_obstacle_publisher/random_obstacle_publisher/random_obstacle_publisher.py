#!/usr/bin/env python3
"""
Random obstacle publisher (ROS2 Jazzy port).

트랙(s)을 균등 sector 로 나눠 sector 마다 frenet 영역 obstacle 을 1 개씩 랜덤 생성.
- /global_waypoints (WpntArray) 와 /car_state/odom_frenet (Odometry) 가 모두 도착하면
  obstacle 한 번 생성하고
- 25 Hz 로 /obstacles (ObstacleArray) 발행.
- publish_at_lookahead=True 면 현재 s 기준 lookahead 거리 안의 obstacle 만 발행.

원본: ICRA2026_HJ f110_utils/nodes/random_obstacle_publisher (109L).
Port: SH ROS2 Jazzy.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from numpy.random import default_rng
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, Obstacle, WpntArray

from random_obstacle_publisher.obstacle_geometry import (
    ObstacleSpec,
    WaypointSpec,
    build_sector_obstacles,
    select_obstacles_in_lookahead,
)


class RandomObstaclePublisher(Node):
    """
    파라미터 (원본 launch default 그대로):
      - n_obstacles (int, default 8): 트랙을 몇 sector 로 나눌지 (sector 당 obstacle 1 개).
      - publish_at_lookahead (bool, default false): true 면 lookahead 안 obstacle 만 발행.
      - lookahead_distance (float, default 5.0)
      - obstacle_width (float, default 0.2)
      - obstacle_length (float, default 0.3)
      - obstacle_max_d_from_traj (float, default 1.0)
      - rnd_seed (int, default 84)
      - rate_hz (float, default 25.0): 발행 주기 (원본 25 Hz hard-coded → 파라미터화).
    """

    def __init__(self) -> None:
        super().__init__("random_obstacle_publisher")

        # 파라미터 선언 — 원본 launch default 와 동일
        self.declare_parameter("n_obstacles", 8)
        self.declare_parameter("publish_at_lookahead", False)
        self.declare_parameter("lookahead_distance", 5.0)
        self.declare_parameter("obstacle_width", 0.2)
        self.declare_parameter("obstacle_length", 0.3)
        self.declare_parameter("obstacle_max_d_from_traj", 1.0)
        self.declare_parameter("rnd_seed", 84)
        self.declare_parameter("rate_hz", 25.0)

        # 원본 코드: self.n_sectors = n_obstacles + 1 (margin sector)
        self._n_sectors = self.get_parameter("n_obstacles").value + 1
        self._publish_at_lookahead = self.get_parameter("publish_at_lookahead").value
        self._lookahead_distance = self.get_parameter("lookahead_distance").value
        self._obstacle_width = self.get_parameter("obstacle_width").value
        self._obstacle_length = self.get_parameter("obstacle_length").value
        self._obstacle_max_d_from_traj = self.get_parameter("obstacle_max_d_from_traj").value
        self._gen = default_rng(self.get_parameter("rnd_seed").value)
        rate_hz = self.get_parameter("rate_hz").value

        # 상태
        self._gb_wpnts: list[WaypointSpec] = []
        self._final_s = 0.0
        self._current_s = 0.0
        self._has_traj = False
        self._has_odom = False
        self._initialized = False
        self._obstacle_array: list[ObstacleSpec] = []

        # I/O
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self._odom_cb, 10
        )
        self.create_subscription(
            WpntArray, "/global_waypoints", self._global_traj_cb, 10
        )
        self._obstacle_pub = self.create_publisher(ObstacleArray, "/obstacles", 10)

        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"RandomObstaclePublisher: {self._n_sectors} sectors, "
            f"{'lookahead' if self._publish_at_lookahead else 'all'} mode, "
            f"{rate_hz:.1f} Hz"
        )

    # ---------- 콜백 ----------

    def _global_traj_cb(self, msg: WpntArray) -> None:
        # 빈 / 미초기화 wpnts (예: ros2 topic pub default empty) 는 무시.
        # 한 번 valid 메시지를 받으면 그 후 _initialized 가 True 가 되어 다시 갱신 안 함.
        if not msg.wpnts:
            return
        wpnts = [
            WaypointSpec(id=w.id, s_m=w.s_m, d_left=w.d_left, d_right=w.d_right)
            for w in msg.wpnts
        ]
        if wpnts[-1].s_m <= 0.0:
            return
        self._gb_wpnts = wpnts
        self._final_s = wpnts[-1].s_m
        self._has_traj = True

    def _odom_cb(self, msg: Odometry) -> None:
        # 원본: self.s = msg.pose.pose.position.x
        # /car_state/odom_frenet 노드 (frenet_odom_republisher) 가 position.x 에 s 값을 채움.
        self._current_s = msg.pose.pose.position.x
        self._has_odom = True

    # ---------- 메인 tick ----------

    def _tick(self) -> None:
        # 두 토픽 모두 도착할 때까지 대기 (원본의 startup while 루프 대체)
        if not (self._has_traj and self._has_odom):
            return

        # obstacle 목록은 한 번만 생성 (원본도 update_obstacles 1회 호출)
        if not self._initialized:
            self._obstacle_array = build_sector_obstacles(
                n_sectors=self._n_sectors,
                gb_wpnts=self._gb_wpnts,
                gen=self._gen,
                obstacle_length=self._obstacle_length,
                obstacle_width=self._obstacle_width,
                obstacle_max_d_from_traj=self._obstacle_max_d_from_traj,
            )
            self._initialized = True
            self.get_logger().info(
                f"Generated {len(self._obstacle_array)} random obstacles "
                f"(final_s={self._final_s:.2f}m)"
            )

        self._publish_obstacles()

    def _publish_obstacles(self) -> None:
        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "frenet"

        if self._publish_at_lookahead:
            specs = select_obstacles_in_lookahead(
                self._obstacle_array,
                current_s=self._current_s,
                lookahead_distance=self._lookahead_distance,
                final_s=self._final_s,
            )
        else:
            specs = self._obstacle_array

        msg.obstacles = [self._spec_to_obstacle_msg(s) for s in specs]
        self._obstacle_pub.publish(msg)

    @staticmethod
    def _spec_to_obstacle_msg(spec: ObstacleSpec) -> Obstacle:
        ob = Obstacle()
        ob.id = spec.id
        ob.s_start = spec.s_start
        ob.s_end = spec.s_end
        ob.d_left = spec.d_left
        ob.d_right = spec.d_right
        ob.is_actually_a_gap = spec.is_actually_a_gap
        # spliner do_spline 이 정적 분기에서만 spline 생성 — random sim 장애물은 정적 처리
        ob.is_static = True
        ob.is_visible = True
        # s_center / d_center / size — spliner _more_space 가 사용
        ob.s_center = (spec.s_start + spec.s_end) * 0.5
        ob.d_center = (spec.d_left + spec.d_right) * 0.5
        ob.size = abs(spec.d_left - spec.d_right)
        return ob


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RandomObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
