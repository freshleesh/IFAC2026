#!/usr/bin/env python3
"""
Dynamic obstacle publisher (ROS2 Jazzy port).

선택한 트랙 (min_curv / shortest_path / centerline / updated) 을 따라 가짜 상대
차량 1 대를 raceline 속도 * speed_scaler 로 주행시키며 /tracking/obstacles 에
ObstacleArray 발행. 원본 ROS1: f110_utils/nodes/obstacle_publisher (252L).

원본의 ros_loop 흐름을 ROS2 timer 패턴으로 재구성:
  startup phase (한 번):
    - frenet 두 service wait + client 생성
    - /global_waypoints + opponent waypoints (선택된 trajectory) 도착 대기
    - opponent (x,y,z) 를 frenet 으로 변환 → ego s_array 위에 d 재샘플 →
      다시 cartesian (x,y,z) 으로 → opponent_wpnts 빌드 (한 번만)
  tick (50 Hz):
    - 현재 s 에서 가장 가까운 opponent_wpnts 인덱스로 속도/d 가져옴
    - s 진행 (s += vs * dt) → frenet → cartesian (per-tick)
    - ObstacleArray + MarkerArray 발행

미포팅 (의도):
- dynamic_reconfigure 'cfg/dyn_obs_publisher.cfg' + dynamic_obs_pub_server.py
  → 원본 launch 에서 주석 처리되어 사용 안 됨 (dead code).
- 일부 sin 수식 (`#+ 0.5*global_wpnts[i].vx_mps*self.speed_scaler * np.sin(...)`)
  도 원본에서 주석. 그대로 미포팅.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import (
    ObstacleArray,
    Obstacle,
    WpntArray,
    OpponentTrajectory,
    OppWpnt,
)
from frenet_conversion_msgs.srv import Glob2FrenetArr, Frenet2GlobArr

from obstacle_publisher.opponent_resampling import (
    advance_s_with_wrap,
    find_nearest_idx,
    resample_opponent_d_on_ego_s,
    sort_opponent_by_s,
)


# 원본 hardcoded 상수
LOOP_RATE_HZ = 50
OBJECT_LENGTH_M = 0.5
OBJECT_SIZE_M = 0.4  # d 방향 폭 (d_left/d_right 계산용)


class ObstaclePublisher(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_publisher")

        # 파라미터 (원본 launch default)
        self.declare_parameter("speed_scaler", 0.5)
        self.declare_parameter("constant_speed", False)
        self.declare_parameter("trajectory", "min_curv")
        self.declare_parameter("start_s", 0.0)

        self._speed_scaler = self.get_parameter("speed_scaler").value
        self._constant = self.get_parameter("constant_speed").value
        self._waypoints_type = self.get_parameter("trajectory").value
        self._starting_s = self.get_parameter("start_s").value

        # trajectory 종류 → opponent waypoints 토픽
        topic_map = {
            "min_curv": "/global_waypoints",
            "shortest_path": "/global_waypoints/shortest_path",
            "centerline": "/centerline_waypoints",
            "updated": "/global_waypoints_updated",
        }
        if self._waypoints_type == "min_time":
            raise NotImplementedError(
                "LTO Trajectory is not implemented. Choose another trajectory type."
            )
        if self._waypoints_type not in topic_map:
            raise ValueError(
                f"Waypoints of type {self._waypoints_type} are not supported."
            )
        self._opponent_topic = topic_map[self._waypoints_type]

        # 상태
        self._car_odom = Odometry()
        self._opponent_traj: OpponentTrajectory | None = None
        self._opponent_s_array: np.ndarray | None = None
        # 실행시간에 갱신되는 동적 obstacle 상태
        self._dynamic_obstacle = self._init_dynamic_obstacle()
        self._dyn_obstacle_speed = 0.0
        self._opponent_traj_pub_counter = 0

        # Pub (먼저 만들어 둠 — startup 끝나야 발행 시작)
        self._obstacle_pub = self.create_publisher(
            ObstacleArray, "/tracking/obstacles", 10
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/dummy_obstacle_markers", 10
        )
        self._opponent_traj_pub = self.create_publisher(
            OpponentTrajectory, "/opponent_waypoints", 10
        )

        # Service clients (frenet 변환)
        self._glob2frenet_client = self.create_client(
            Glob2FrenetArr, "convert_glob2frenetarr_service"
        )
        self._frenet2glob_client = self.create_client(
            Frenet2GlobArr, "convert_frenet2globarr_service"
        )
        self.get_logger().info("[ObstaclePublisher] Waiting for frenet services...")
        if not self._glob2frenet_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("convert_glob2frenetarr_service not available within 10s")
        if not self._frenet2glob_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("convert_frenet2globarr_service not available within 10s")

        # ---- startup: wpnts 토픽 도착 대기 + opponent trajectory 빌드 (spin 전) ----
        self.get_logger().info(
            f"[ObstaclePublisher] frenet services ready. "
            f"Waiting for /global_waypoints + {self._opponent_topic}..."
        )
        self._build_opponent_trajectory_blocking()

        # 일반 콜백 + tick (이제 spin 시작해도 OK)
        self.create_subscription(
            Odometry, "/car_state/odom_frenet", self._on_odom, 10
        )
        self._dt = 1.0 / LOOP_RATE_HZ
        self.create_timer(self._dt, self._tick)
        self.get_logger().info("[ObstaclePublisher] tick started (50 Hz)")

    # ---------- 헬퍼 ----------

    @staticmethod
    def _init_dynamic_obstacle() -> Obstacle:
        ob = Obstacle()
        ob.id = 1
        ob.d_right = -0.1
        ob.d_left = 0.1
        ob.is_actually_a_gap = False
        return ob

    # ---------- 콜백 ----------

    def _on_odom(self, msg: Odometry) -> None:
        self._car_odom = msg

    # ---------- startup — opponent trajectory build (spin 전 blocking) ----------

    def _build_opponent_trajectory_blocking(self) -> None:
        """
        spin 시작 전에 호출. wait_for_message 로 wpnts 받기 + 동기 service 호출
        두 번 → opponent_traj 빌드 + s_array 셋업. 실패 시 RuntimeError.
        """
        # 1) /global_waypoints 한 번 받기
        ok, gb_msg = wait_for_message(
            WpntArray, self, "/global_waypoints", time_to_wait=10.0
        )
        if not ok:
            raise RuntimeError("/global_waypoints 10s timeout")
        global_wpnts = gb_msg.wpnts[:-1]  # wrap 마지막 잘라냄

        # 2) opponent trajectory 토픽도 받기 (다른 토픽이면)
        if self._opponent_topic == "/global_waypoints":
            opp_wpnts_list = global_wpnts
        else:
            ok, opp_msg = wait_for_message(
                WpntArray, self, self._opponent_topic, time_to_wait=10.0
            )
            if not ok:
                raise RuntimeError(f"{self._opponent_topic} 10s timeout")
            opp_wpnts_list = opp_msg.wpnts[:-1]

        # 3) opponent (x,y,z) → frenet (s,d)  — call_async + spin_until_future_complete
        req1 = Glob2FrenetArr.Request()
        req1.x = [w.x_m for w in opp_wpnts_list]
        req1.y = [w.y_m for w in opp_wpnts_list]
        req1.z = [w.z_m for w in opp_wpnts_list]
        future1 = self._glob2frenet_client.call_async(req1)
        rclpy.spin_until_future_complete(self, future1, timeout_sec=5.0)
        if not future1.done():
            raise RuntimeError("glob2frenetarr 5s timeout")
        resp1 = future1.result()

        # 4) ego s_array 위에 d 재샘플
        s_sorted, d_sorted = sort_opponent_by_s(resp1.s, resp1.d)
        ego_s_array = np.array([w.s_m for w in global_wpnts])
        d_resampled = resample_opponent_d_on_ego_s(ego_s_array, s_sorted, d_sorted)
        ego_vx = [w.vx_mps for w in global_wpnts]

        # 5) (s, d) → cartesian
        req2 = Frenet2GlobArr.Request()
        req2.s = ego_s_array.tolist()
        req2.d = d_resampled.tolist()
        future2 = self._frenet2glob_client.call_async(req2)
        rclpy.spin_until_future_complete(self, future2, timeout_sec=5.0)
        if not future2.done():
            raise RuntimeError("frenet2globarr 5s timeout")
        resp2 = future2.result()

        # 6) OpponentTrajectory 빌드 (원본과 동일)
        traj = OpponentTrajectory()
        for i in range(len(ego_s_array)):
            wp = OppWpnt()
            wp.x_m = float(resp2.x[i])
            wp.y_m = float(resp2.y[i])
            wp.proj_vs_mps = (
                float(self._speed_scaler) if self._constant
                else float(ego_vx[i] * self._speed_scaler)
            )
            wp.s_m = float(ego_s_array[i])
            wp.d_m = float(d_resampled[i])
            traj.oppwpnts.append(wp)

        self._opponent_traj = traj
        self._opponent_s_array = np.array(
            [w.s_m for w in traj.oppwpnts], dtype=float
        )
        self._dynamic_obstacle.s_center = self._starting_s
        self.get_logger().info(
            f"[ObstaclePublisher] opponent trajectory built "
            f"({len(traj.oppwpnts)} oppwpnts), start_s={self._starting_s:.2f}"
        )

    # ---------- 메인 tick ----------

    def _tick(self) -> None:
        traj = self._opponent_traj
        max_s = float(self._opponent_s_array[-1])

        # 현재 s 에서 가장 가까운 opponent index → 속도, d
        s = float(self._dynamic_obstacle.s_center)
        approx_idx = find_nearest_idx(self._opponent_s_array, s)
        self._dyn_obstacle_speed = float(traj.oppwpnts[approx_idx].proj_vs_mps)

        # s 진행 + obstacle 박스 좌표
        new_s_center = advance_s_with_wrap(s, self._dyn_obstacle_speed * self._dt, max_s)
        self._dynamic_obstacle.s_center = new_s_center
        self._dynamic_obstacle.s_start = (new_s_center - OBJECT_LENGTH_M / 2) % max_s
        self._dynamic_obstacle.s_end = (new_s_center + OBJECT_LENGTH_M / 2) % max_s
        self._dynamic_obstacle.d_center = float(traj.oppwpnts[approx_idx].d_m)
        self._dynamic_obstacle.size = OBJECT_SIZE_M
        self._dynamic_obstacle.d_right = self._dynamic_obstacle.d_center - OBJECT_SIZE_M / 2
        self._dynamic_obstacle.d_left = self._dynamic_obstacle.d_center + OBJECT_SIZE_M / 2
        self._dynamic_obstacle.vs = self._dyn_obstacle_speed

        # 매 tick 한 점만 frenet → cartesian (단발 service 호출).
        # 원본은 동기 호출로 작성됐으나 ROS2 timer callback 에서는 spin_until_future_complete
        # 가 deadlock 위험 → call_async + done callback 패턴으로.
        req = Frenet2GlobArr.Request()
        req.s = [new_s_center]
        req.d = [self._dynamic_obstacle.d_center]
        future = self._frenet2glob_client.call_async(req)
        future.add_done_callback(self._on_frenet2glob_done)

    def _on_frenet2glob_done(self, future) -> None:
        """매 tick frenet→cartesian 응답 받으면 obstacle 위치 채우고 발행."""
        resp = future.result()
        if resp is None or not resp.x:
            return
        self._dynamic_obstacle.x_m = float(resp.x[0])
        self._dynamic_obstacle.y_m = float(resp.y[0])
        self._dynamic_obstacle.z_m = float(resp.z[0]) if resp.z else 0.0

        msg = ObstacleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "frenet"
        msg.obstacles.append(self._dynamic_obstacle)
        self._publish_obstacle_markers([self._dynamic_obstacle])
        self._obstacle_pub.publish(msg)

        # opponent_waypoints 25 tick 마다 (원본의 counter > 25 패턴)
        self._opponent_traj_pub_counter += 1
        if self._opponent_traj_pub_counter > 25:
            opp_msg = OpponentTrajectory()
            opp_msg.header.frame_id = "map"
            opp_msg.header.stamp = self.get_clock().now().to_msg()
            opp_msg.lap_count = 2.0  # 원본 hard-coded (ROS2 의 lap_count 는 float64 — strict cast)
            opp_msg.oppwpnts = self._opponent_traj.oppwpnts
            self._opponent_traj_pub.publish(opp_msg)
            self._opponent_traj_pub_counter = 0

    def _publish_obstacle_markers(self, obstacles) -> None:
        arr = MarkerArray()
        for obs in obstacles:
            m = Marker()
            m.header.frame_id = "map"
            m.id = obs.id
            m.type = Marker.SPHERE
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.5
            m.color.a = 0.5
            m.color.b = 0.5
            m.color.r = 0.5
            m.pose.position.x = obs.x_m
            m.pose.position.y = obs.y_m
            m.pose.position.z = obs.z_m
            m.pose.orientation.w = 1.0
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = ObstaclePublisher()
    except (RuntimeError, ValueError, NotImplementedError) as e:
        print(f"[ObstaclePublisher] startup error: {e}")
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
