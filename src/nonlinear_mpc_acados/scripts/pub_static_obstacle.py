#!/usr/bin/env python3
"""Publish a single static obstacle to /external_obstacles (PoseArray) for
avoidance testing. mpc_node overwrites self._obstacles from this topic each
message, so the obstacle persists while this runs.

Usage: python3 pub_static_obstacle.py --x 1.23 --y 4.56 [--frame map]
Pick (x, y) from one RViz 'Publish Point' click (it is logged) or from the
global racing-line CSV. Default coordinate is a Task-0-verified on-racing-line
point on the 'final' map (s_obs=29.5).
"""
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class StaticObstaclePub(Node):
    def __init__(self, x, y, frame):
        super().__init__('static_obstacle_pub')
        self.x, self.y, self.frame = x, y, frame
        self.pub = self.create_publisher(PoseArray, '/external_obstacles', 1)
        self.create_timer(0.1, self._tick)  # 10 Hz
        self.get_logger().info(f'publishing obstacle at ({x:.2f}, {y:.2f}) on {frame}')

    def _tick(self):
        msg = PoseArray()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        p = Pose()
        p.position.x = self.x
        p.position.y = self.y
        msg.poses = [p]
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=-4.41)
    ap.add_argument('--y', type=float, default=1.04)
    ap.add_argument('--frame', type=str, default='map')
    a = ap.parse_args()
    rclpy.init()
    node = StaticObstaclePub(a.x, a.y, a.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
