#!/usr/bin/env python3
"""Add ONE static obstacle for avoidance testing by publishing a single
PointStamped to /clicked_point — exactly the path RViz "Publish Point" uses.
The static_obstacle_manager picks it up (track-boundary checked), becomes the
single source of truth, and re-publishes the authoritative list on
/external_obstacles for the MPCC controller.

Usage: python3 pub_static_obstacle.py --x 1.23 --y 4.56 [--frame map]
Default coordinate is a Task-0-verified on-racing-line point on 'final'
(s_obs≈29.5). To CLEAR, use the RViz "Clear Obstacles" button (which now
propagates to the controller via the manager's /external_obstacles publish).
"""
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PointStamped


class ClickOnce(Node):
    def __init__(self, x, y, frame):
        super().__init__('pub_static_obstacle')
        # transient_local + reliable so the single sample is delivered robustly
        # to the already-running manager (one publish → one obstacle).
        qos = QoSProfile(depth=1,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PointStamped, '/clicked_point', qos)
        self.x, self.y, self.frame = x, y, frame
        self._sent = False
        self.create_timer(0.2, self._tick)

    def _tick(self):
        if self._sent:
            return
        msg = PointStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = self.x
        msg.point.y = self.y
        self.pub.publish(msg)
        self._sent = True
        self.get_logger().info(f'clicked obstacle at ({self.x:.2f}, {self.y:.2f}) on {self.frame}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=-4.41)
    ap.add_argument('--y', type=float, default=1.04)
    ap.add_argument('--frame', type=str, default='map')
    a = ap.parse_args()
    rclpy.init()
    node = ClickOnce(a.x, a.y, a.frame)
    # spin ~2.5 s so the transient_local sample is matched + delivered, then exit
    end = node.get_clock().now().nanoseconds + int(2.5e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
