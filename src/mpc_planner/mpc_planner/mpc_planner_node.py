#!/usr/bin/env python3
"""
MPCC Planner Node.

Ported from icra_unicorn_heedo/mpc/scripts/mpcc_node.py. Used here as a
trajectory planner (no control output). The solver is a 2D kinematic
bicycle MPCC; this node slices the local reference window directly from
/global_waypoints (no BehaviorStrategy dependency) and publishes the
predicted MPC trajectory as a Path + MarkerArray for RViz.

Subscribes:
  /global_waypoints  (f110_msgs/WpntArray)
  /car_state/pose    (geometry_msgs/PoseStamped)
  /car_state/odom    (nav_msgs/Odometry)

Publishes:
  /planner/mpc/trajectory  (nav_msgs/Path)
  /planner/mpc/markers     (visualization_msgs/MarkerArray)
"""

import os
import sys
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import WpntArray

# Import solver from sibling src/ directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.normpath(os.path.join(_this_dir, '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from mpcc_solver import MPCCSolver  # noqa: E402


class MPCPlannerNode:

    def __init__(self):
        rospy.init_node('mpc_planner', anonymous=False)

        # Solver parameters
        self.freq = self._get_param_or_default('~planner_freq', self._get_param_or_default('~controller_freq', 30))
        params = {
            'N':                 self._get_param_or_default('~N', 20),
            'dT':                self._get_param_or_default('~dT', 0.05),
            'vehicle_L':         self._get_param_or_default('~vehicle_L', 0.33),
            'max_speed':         self._get_param_or_default('~max_speed', 12.0),
            'min_speed':         self._get_param_or_default('~min_speed', 0.5),
            'max_steering':      self._get_param_or_default('~max_steering', 0.6),
            'w_contour':         self._get_param_or_default('~w_contour', 3.9),
            'w_lag':             self._get_param_or_default('~w_lag', 2.0),
            'w_velocity':        self._get_param_or_default('~w_velocity', 3.0),
            'v_bias_max':        self._get_param_or_default('~v_bias_max', 1.0),
            'w_dv':              self._get_param_or_default('~w_dv', 9.5),
            'w_dsteering':       self._get_param_or_default('~w_dsteering', 14.0),
            'boundary_inflation': self._get_param_or_default('~boundary_inflation', 0.1),
            'w_slack':           self._get_param_or_default('~w_slack', 1000.0),
            'ipopt_max_iter':    self._get_param_or_default('~ipopt_max_iter', 500),
            'ipopt_print_level': self._get_param_or_default('~ipopt_print_level', 0),
        }
        self.N = params['N']
        self.solver = MPCCSolver(params)

        # Ego state
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.car_vx = 0.0
        self.pose_received = False

        # Global waypoint cache (populated on first /global_waypoints msg)
        self.global_cached = False
        self.g_s = None
        self.g_x = None
        self.g_y = None
        self.g_z = None              # ### HJ : z_m for 3D marker lift
        self.g_psi = None
        self.g_dleft = None
        self.g_dright = None
        self.g_vx = None
        self.track_length = None

        self._timer = None

        # Publishers
        self.traj_pub = self.create_publisher(Path, '/planner/mpc/trajectory', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner/mpc/markers', 1)

        # Subscribers
        self.create_subscription(WpntArray, '/global_waypoints', self._global_wpnts_cb, 10)
        self.create_subscription(PoseStamped, '/car_state/pose', self._pose_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom', self._odom_cb, 10)

        self.get_logger().info("[MPC planner] Waiting for /global_waypoints...")

    def _global_wpnts_cb(self, msg):
        if self.global_cached:
            return
        wpnts = msg.wpnts
        n = len(wpnts)
        if n < 10:
            self.get_logger().warning("[MPC planner] Too few global waypoints: %d", n)
            return

        self.g_s = np.array([w.s_m for w in wpnts], dtype=float)
        self.g_x = np.array([w.x_m for w in wpnts], dtype=float)
        self.g_y = np.array([w.y_m for w in wpnts], dtype=float)
        # ### HJ : cache z_m for lifting 2D MPC output to 3D visualization
        self.g_z = np.array([getattr(w, 'z_m', 0.0) for w in wpnts], dtype=float)
        self.g_psi = np.array([w.psi_rad for w in wpnts], dtype=float)
        self.g_dleft = np.array([w.d_left for w in wpnts], dtype=float)
        self.g_dright = np.array([w.d_right for w in wpnts], dtype=float)
        self.g_vx = np.array([w.vx_mps for w in wpnts], dtype=float)
        self.track_length = float(self.g_s[-1])

        self.solver.setup()
        self.global_cached = True
        self.get_logger().info("[MPC planner] Solver ready: %d waypoints, track=%.1fm",
                      n, self.track_length)

        if self._timer is None:
            self._timer = rospy.Timer(rospy.Duration(1.0 / self.freq), self._plan_loop)

    def _pose_cb(self, msg):
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.car_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose_received = True

    def _odom_cb(self, msg):
        self.car_vx = msg.twist.twist.linear.x

    def _slice_local_ref(self, s_cur):
        """
        Build N+1 local reference from /global_waypoints around s_cur.
        Reference s advances using the raceline's local vx at each step so
        that the slice window matches the speed profile the MPC is expected
        to track (avoids reference compression when the car is slow).
        Returns dict: center_points, left_points, right_points,
                      ref_v, ref_s, ref_dx, ref_dy.
        """
        N = self.N
        dT = self.solver.dT
        tl = self.track_length

        inflation = self.solver.inflation

        center_pts, left_pts, right_pts = [], [], []
        ref_v, ref_s, ref_dx, ref_dy = [], [], [], []
        s_offset = 0.0
        prev_s = None

        target_s = s_cur
        for k in range(N + 1):
            s_wrap = target_s % tl if tl > 0 else target_s
            idx = int(np.argmin(np.abs(self.g_s - s_wrap)))

            x = self.g_x[idx]
            y = self.g_y[idx]
            psi = self.g_psi[idx]
            dl = self.g_dleft[idx]
            dr = self.g_dright[idx]

            normal = np.array([-np.sin(psi), np.cos(psi)])
            center = np.array([x, y])
            center_pts.append(center)
            left_pts.append(center + normal * max(dl - inflation, 0.0))
            right_pts.append(center - normal * max(dr - inflation, 0.0))
            ref_v.append(float(self.g_vx[idx]))
            ref_dx.append(np.cos(psi))
            ref_dy.append(np.sin(psi))

            s_val = float(self.g_s[idx])
            if prev_s is not None and s_val + s_offset < prev_s - 1.0:
                s_offset += max(tl or 100.0, 1.0)
            s_val += s_offset
            ref_s.append(s_val)
            prev_s = s_val

            # Advance the slicing cursor by the local raceline speed.
            # Uses raceline vx at this waypoint so the window spans the
            # distance the car will actually cover in one MPC step.
            local_v = max(float(self.g_vx[idx]), 1.0)
            target_s += local_v * dT

        return {
            'center_points': np.array(center_pts),
            'left_points': np.array(left_pts),
            'right_points': np.array(right_pts),
            'ref_v': np.array(ref_v),
            'ref_s': np.array(ref_s),
            'ref_dx': np.array(ref_dx),
            'ref_dy': np.array(ref_dy),
        }

    def _plan_loop(self, event):
        if not self.solver.ready or not self.pose_received:
            return

        # Find nearest global waypoint (s_cur used only for ref slicing here).
        dists = (self.g_x - self.car_x) ** 2 + (self.g_y - self.car_y) ** 2
        s_cur = float(self.g_s[int(np.argmin(dists))])

        ref_slice = self._slice_local_ref(s_cur)
        initial_state = np.array([self.car_x, self.car_y, self.car_yaw])

        t0 = time.time()
        speed, steering, trajectory, success = self.solver.solve(initial_state, ref_slice)
        solve_ms = (time.time() - t0) * 1000.0

        if success and trajectory is not None:
            self._publish_trajectory(trajectory)
            self._publish_markers(trajectory)
        else:
            # Drop stale warm-start so the next solve starts clean and does
            # not inherit an infeasible initial guess.
            self.solver.reset_warm_start()

            # Failure dump: one compact line per failure to diagnose the
            # exact track state, reference pacing, and solver status.
            nearest_idx = int(np.argmin(
                (self.g_x - self.car_x) ** 2 + (self.g_y - self.car_y) ** 2))
            near_d_xy = float(np.hypot(
                self.g_x[nearest_idx] - self.car_x,
                self.g_y[nearest_idx] - self.car_y,
            ))
            near_psi = float(self.g_psi[nearest_idx])
            dpsi = float(np.arctan2(
                np.sin(self.car_yaw - near_psi), np.cos(self.car_yaw - near_psi)))
            ref_v_first = float(ref_slice['ref_v'][0])
            ref_v_last = float(ref_slice['ref_v'][-1])
            status = getattr(self.solver, 'last_return_status', '?')
            it = getattr(self.solver, 'last_iter_count', -1)

            # Min corridor width over the horizon (nearest + next N steps along raceline).
            N_pts = len(self.g_s)
            look_idx = [(nearest_idx + j) % N_pts for j in range(self.N + 1)]
            win_dl = self.g_dleft[look_idx]
            win_dr = self.g_dright[look_idx]
            inflation = self.solver.inflation
            win_corr = np.maximum(win_dl - inflation, 0.0) + np.maximum(win_dr - inflation, 0.0)
            corr_min = float(np.min(win_corr))
            corr_min_s = float(self.g_s[look_idx[int(np.argmin(win_corr))]])

            self.get_logger().warning(
                "[MPC fail] s=%.1f xy_err=%.2fm dpsi=%+.2frad "
                "ref_v=[%.2f..%.2f] dl=%.2f dr=%.2f "
                "win_corr_min=%.2fm@s=%.1f iter=%d status=%s solve=%.1fms",
                s_cur, near_d_xy, dpsi,
                ref_v_first, ref_v_last,
                float(self.g_dleft[nearest_idx]), float(self.g_dright[nearest_idx]),
                corr_min, corr_min_s, it, status, solve_ms,
            )

        rospy.loginfo_throttle(
            1.0,
            "[MPC planner] solve=%.1fms v0=%.2f steer=%.3f ok=%s",
            solve_ms, speed, steering, success,
        )

    def _lift_z(self, x, y):
        """### HJ : approximate z for a 2D MPC point via nearest global waypoint."""
        if self.g_z is None:
            return 0.0
        dists = (self.g_x - x) ** 2 + (self.g_y - y) ** 2
        idx = int(np.argmin(dists))
        return float(self.g_z[idx])

    def _publish_trajectory(self, trajectory):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for i in range(trajectory.shape[0]):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = trajectory[i, 0]
            p.pose.position.y = trajectory[i, 1]
            p.pose.position.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            q = quaternion_from_euler(0, 0, trajectory[i, 2])
            p.pose.orientation.x = q[0]
            p.pose.orientation.y = q[1]
            p.pose.orientation.z = q[2]
            p.pose.orientation.w = q[3]
            path.poses.append(p)
        self.traj_pub.publish(path)

    def _publish_markers(self, trajectory):
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()

        # LINE_STRIP of predicted path
        line = Marker()
        line.header.stamp = now
        line.header.frame_id = 'map'
        line.ns = 'mpc_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9)
        line.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            line.points.append(pt)
        arr.markers.append(line)

        # SPHERE_LIST at each MPC step
        pts = Marker()
        pts.header.stamp = now
        pts.header.frame_id = 'map'
        pts.ns = 'mpc_steps'
        pts.id = 1
        pts.type = Marker.SPHERE_LIST
        pts.action = Marker.ADD
        pts.scale.x = pts.scale.y = pts.scale.z = 0.12
        pts.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.9)
        pts.pose.orientation.w = 1.0
        for i in range(trajectory.shape[0]):
            pt = Point()
            pt.x = float(trajectory[i, 0])
            pt.y = float(trajectory[i, 1])
            pt.z = self._lift_z(trajectory[i, 0], trajectory[i, 1])  # ### HJ
            pts.points.append(pt)
        arr.markers.append(pts)

        self.marker_pub.publish(arr)


if __name__ == '__main__':
    try:
        MPCPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
