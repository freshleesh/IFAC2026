#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
from typing import List, Any, Tuple

import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
import tf_transformations as tf_trans
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph

class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
    
    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
    """

    def _get_param_or_default(self, name, default=None):
        """rospy.get_param 호환 helper."""
        candidates = [name]
        if "/" in name:
            candidates.append(name.replace("/", "."))
            candidates.append(name.lstrip("/"))
            candidates.append(name.lstrip("/").replace("/", "."))
        for n in candidates:
            try:
                v = self.get_parameter(n).value
                if v is not None:
                    return v
            except Exception:
                continue
        if default is None:
            return None
        try:
            self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return v if v is not None else default
        except Exception:
            return default

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "recovery_spliner_node"
        super().__init__(self.name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # initialize the instance variable
        self.gb_wpnts = WpntArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = 0
        self.cur_d = 0
        self.cur_vs = 0
        self.min_candidates_lookahead_n = self._get_param_or_default("/recovery_planner_spline/min_candidates_lookahead_n", 20)
        self.num_kappas = self._get_param_or_default("/recovery_planner_spline/num_kappas", 20)
        self.spline_scale = self._get_param_or_default("/recovery_planner_spline/spline_scale", 0.8)
        self.kernel_size = self._get_param_or_default("/recovery_planner_spline/spline_bound_mindist", 5)
        self.smooth_len = self._get_param_or_default("/recovery_planner_spline/smooth_len", 1.0)
        self.n_loc_wpnts = self._get_param_or_default("/state_machine/n_loc_wpnts", 80)

        self.gb_scaled_wpnts = WpntArray()

        self.from_bag = self._get_param_or_default("/from_bag", False)
        self.measuring = self._get_param_or_default("/measure", False)
        self.inflection_points = None

        # ===== IY ADDED: Fixed path tracking =====
        self.use_fixed_path = False
        self.cur_s_fixed = 0
        self.cur_d_fixed = 0
        self.cur_vs_fixed = 0
        self.fixed_path_wpnts = OTWpntArray()
        self.has_fixed_path = False
        # ===== IY ADD : RECOVERY - Fixed path specific converter =====
        self.fixed_path_converter = None
        self.fixed_inflection_points = np.array([])
        self.fixed_path_initialized = False
        # ===== IY ADD END =====
        # ===== IY ADDED END =====

        # Subscribe to the topics
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)

        # ===== IY ADDED: Subscribe to fixed path frenet and smart_static_active =====
        self.create_subscription(Odometry, "/car_state/odom_frenet_fixed", self.state_frenet_fixed_cb, 10)
        self.create_subscription(Bool, "/planner/avoidance/smart_static_active", self.smart_static_active_cb, 10)
        self.create_subscription(OTWpntArray, "/planner/avoidance/smart_static_otwpnts", self.fixed_path_wpnts_cb, 10)
        # ===== IY ADDED END =====

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/recovery/markers", 10)
        self.recovery_wpnts_pub = self.create_publisher(WpntArray, "/planner/recovery/wpnts", 10)
        self.recovery_lookahead_pub = self.create_publisher(Marker, "/planner/recovery/lookahead_point", 10)

        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/recovery/latency", 10)
            self.checkpoints_pub = self.create_publisher(MarkerArray, "/planner/recovery/checkpoints", 10)

        self.converter = self.initialize_converter()
        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)

        if not self.from_bag:
            pass  # ROS2: dyn_reconfigure tuner 미포팅 — 비활성

        # self.rate = rospy.Rate(40)  # ROS2: timer 또는 spin_once + time.sleep 으로 대체  # Hz

    #############
    # CALLBACKS #
    #############
    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

    # ===== IY ADDED: Callback for fixed path frenet coordinates =====
    def state_frenet_fixed_cb(self, data: Odometry):
        self.cur_s_fixed = data.pose.pose.position.x
        self.cur_d_fixed = data.pose.pose.position.y
        self.cur_vs_fixed = data.twist.twist.linear.x

    def smart_static_active_cb(self, data: Bool):
        self.use_fixed_path = data.data

    def fixed_path_wpnts_cb(self, data: OTWpntArray):
        self.fixed_path_wpnts = data  # Always update (keep latest waypoints)

        # ===== HJ MODIFIED: Initialize only once to avoid repeated converter creation =====
        if not self.fixed_path_initialized and len(data.wpnts) > 1:
            # Initialize converter (only once)
            fixed_path_xy = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
            self.fixed_path_converter = FrenetConverter(fixed_path_xy[:, 0], fixed_path_xy[:, 1], fixed_path_xy[:, 2])

            # ===== HJ ADDED: Calculate inflection points (same logic as GB) =====
            kappas = np.array([wpnt.kappa_radpm for wpnt in data.wpnts])
            sign_changes = np.sign(kappas)
            self.fixed_inflection_points = np.where(np.diff(sign_changes) != 0)[0]
            # ===== HJ ADDED END =====

            # Set initialization complete flags
            self.fixed_path_initialized = True
            self.has_fixed_path = True

            self.get_logger().info(f"[{self.name}] Fixed path initialized: {len(data.wpnts)} waypoints, "
                         f"{len(self.fixed_inflection_points)} inflection points")
        # ===== HJ MODIFIED END =====
    # ===== IY ADDED END =====

    def state_cb(self, data: Odometry):
        self.cur_x = data.pose.pose.position.x
        self.cur_y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        euler = tf_trans.euler_from_quaternion(q)
        self.cur_yaw = euler[2] 

    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):
        
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m
            
        # psi_array = np.array([wpnt.psi_rad for wpnt in data.wpnts])
        kappas = np.array([wpnt.kappa_radpm for wpnt in data.wpnts])
        
        sign_changes = np.sign(kappas)

        self.inflection_points = np.where(np.diff(sign_changes) != 0)[0]

        # max_indices = argrelextrema(psi_array, np.greater)[0]
        # min_indices = argrelextrema(psi_array, np.less)[0]
        
        if self.measuring:
            mrks = MarkerArray()
            for idx in self.inflection_points:
                # print(idx)
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.3
                mrk.scale.y = 0.3
                mrk.scale.z = 0.3
                mrk.color.a = 1.0
                mrk.color.b = 0.75
                mrk.color.r = 0.75
                mrk.id = idx
                mrk.pose.position.x = data.wpnts[idx].x_m
                mrk.pose.position.y = data.wpnts[idx].y_m
                mrk.pose.position.z = 0.0
                mrk.pose.orientation.w = 1.0
                mrks.markers.append(mrk)
            self.checkpoints_pub.publish(mrks)

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: object):
        """
        Notices the change in the parameters and changes spline params
        """
        self.min_candidates_lookahead_n = self._get_param_or_default("/dyn_planner_recovery/min_candidates_lookahead_n", 20)
        self.num_kappas = self._get_param_or_default("/dyn_planner_recovery/num_kappas", 20)
        self.spline_scale = self._get_param_or_default("/dyn_planner_recovery/spline_scale", 0.8)
        self.kernel_size = self._get_param_or_default("/dyn_planner_recovery/kernel_size", 4)
        self.smooth_len = self._get_param_or_default("/dyn_planner_recovery/smooth_len", 1.0)

        self.map_filter.set_erosion_kernel_size(self.kernel_size)
        return

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        # Wait for critical Messages and services
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        # rospy.wait_for_message("/global_waypoints_scaled", WpntArray)  # ROS2: ready flag polling 으로 변환 필요
        # rospy.wait_for_message("/car_state/odom", Odometry)  # ROS2: ready flag polling 으로 변환 필요
        # rospy.wait_for_message("/car_state/odom_frenet", Odometry)  # ROS2: ready flag polling 으로 변환 필요
        
        # rospy.wait_for_message("/dyn_planner_recovery/parameter_updates", Config)  # ROS2: ready flag polling 으로 변환 필요
        self.get_logger().info(f"[{self.name}] Ready!")

        while not (not rclpy.ok()):
            if self.measuring:
                start = time.perf_counter()

            # ===== IY MODIFIED: Use appropriate waypoints based on use_fixed_path flag =====
            if self.use_fixed_path and self.has_fixed_path:
                # Use fixed path waypoints
                reference_wpnts = self.fixed_path_wpnts.wpnts
                rospy.loginfo_throttle(2.0, f"[{self.name}] Using FIXED PATH for recovery ({len(reference_wpnts)} waypoints)")
            else:
                # Use GB path waypoints
                reference_wpnts = self.gb_scaled_wpnts.wpnts
                rospy.loginfo_throttle(2.0, f"[{self.name}] Using GB PATH for recovery ({len(reference_wpnts)} waypoints)")
            # ===== IY MODIFIED END =====

            wpnts = WpntArray()
            mrks = MarkerArray()

            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.append(del_mrk)
            self.mrks_pub.publish(mrks)

            wpnts, mrks = self.do_spline(gb_wpnts=reference_wpnts)

            # Publish wpnts and markers
            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(1/(end - start))
            self.recovery_wpnts_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)
            # self.rate.sleep()  # ROS2: timer-based 또는 rclpy.spin_once + time.sleep 으로 대체
    
    #########
    # UTILS #
    #########
    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        # rospy.wait_for_message("/global_waypoints", WpntArray)  # ROS2: ready flag polling 으로 변환 필요

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")

        return converter

    def find_tangent_idx(self, xy_m, psi_rads):
        # Get current position
        cur_x, cur_y = self.cur_x, self.cur_y
        smooth = np.cos(self.cur_yaw), np.sin(self.cur_yaw) * self.smooth_len


        # Compute direction vectors from the current position to waypoints
        dx = xy_m[:, 0] - (cur_x + smooth[0])
        dy = xy_m[:, 1] - (cur_y + smooth[1])

        # Normalize vectors
        norm = np.sqrt(dx**2 + dy**2)  # Vector magnitude
        unit_vectors = np.vstack((dx / norm, dy / norm)).T  # (N, 2) unit vectors

        # Convert waypoint heading angles to unit vectors
        psi_unit_vectors = np.vstack((np.cos(psi_rads), np.sin(psi_rads))).T  # (N, 2)

        # Compute cosine similarity between vectors
        cos_theta = np.sum(unit_vectors * psi_unit_vectors, axis=1)  # Dot product
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors

        # Compute angular difference in radians
        angles = np.arccos(cos_theta)  # Range: 0 ~ π

        # Find the index with the smallest angle
        tangent_idx = np.argmin(angles)

        return tangent_idx

    def do_spline(self, gb_wpnts: WpntArray) -> Tuple[WpntArray, MarkerArray]:
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()
        wpnts = WpntArray()

        # ===== IY ADD : RECOVERY - Safety check for empty waypoints =====
        if len(gb_wpnts) < 2:
            rospy.logwarn_throttle(2.0, f"[{self.name}] Not enough waypoints for recovery ({len(gb_wpnts)} < 2)")
            return wpnts, mrks
        # ===== IY ADD END =====

        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        # ===== IY ADD : RECOVERY - Use actual length for fixed path mode =====
        if self.use_fixed_path:
            ref_max_idx = len(gb_wpnts)  # 배열 길이를 사용 (모듈로 연산용)
            rospy.loginfo_throttle(5.0, f"[{self.name}] Fixed path ref_max_idx: {ref_max_idx}")
        else:
            ref_max_idx = self.gb_max_idx

        # 안전 체크
        if ref_max_idx <= 0:
            rospy.logwarn_throttle(2.0, f"[{self.name}] Invalid ref_max_idx: {ref_max_idx}")
            return wpnts, mrks
        # ===== IY ADD END =====

        # ===== IY MODIFIED: Use appropriate frenet coordinates based on use_fixed_path flag =====
        if self.use_fixed_path:
            cur_s = self.cur_s_fixed
            cur_d = self.cur_d_fixed
        else:
            cur_s = self.cur_s
            cur_d = self.cur_d
        # ===== IY MODIFIED END =====
        cur_s_idx = int(cur_s / wpnt_dist)

        # ===== HJ MODIFIED: Use inflection points logic for both GB and Fixed path =====
        if self.use_fixed_path:
            # Fixed path: use fixed_inflection_points (same logic as GB)
            if self.fixed_path_initialized and len(self.fixed_inflection_points) != 0:
                infl_sector_idx = np.searchsorted(self.fixed_inflection_points, cur_s_idx)
                next_infl_sector_idx = self.fixed_inflection_points[(infl_sector_idx) % len(self.fixed_inflection_points)]

                candidate_len = (next_infl_sector_idx - cur_s_idx + ref_max_idx
                                if infl_sector_idx == len(self.fixed_inflection_points)
                                else next_infl_sector_idx - cur_s_idx)
                rospy.loginfo_throttle(5.0, f"[{self.name}] Fixed path using inflection points: candidate_len={candidate_len}")
            else:
                # Fallback (inflection points not yet generated)
                candidate_len = int(ref_max_idx / 2)
                rospy.loginfo_throttle(5.0, f"[{self.name}] Fixed path fallback: candidate_len={candidate_len}")
        else:
            # GB path: use inflection_points (existing logic)
            if len(self.inflection_points) != 0:
                infl_sector_idx = np.searchsorted(self.inflection_points, cur_s_idx)
                next_infl_sector_idx = self.inflection_points[(infl_sector_idx) % len(self.inflection_points)]

                candidate_len = (next_infl_sector_idx - cur_s_idx + ref_max_idx
                                if infl_sector_idx == len(self.inflection_points)
                                else next_infl_sector_idx - cur_s_idx)
            else:
                candidate_len = int(ref_max_idx / 2)
        # ===== HJ MODIFIED END =====

        candidate_len = max(candidate_len, self.min_candidates_lookahead_n)

        # ===== IY ADD : RECOVERY - Use ref_max_idx instead of gb_max_idx =====
        gb_idxs = [(cur_s_idx + i) % ref_max_idx for i in range(candidate_len)]

        # 안전 체크: 생성된 인덱스가 실제 배열 범위 내에 있는지 확인
        max_available_idx = len(gb_wpnts) - 1
        if any(idx > max_available_idx for idx in gb_idxs):
            rospy.logwarn_throttle(2.0, f"[{self.name}] Index out of bounds! max_idx={max_available_idx}, ref_max_idx={ref_max_idx}")
            # 인덱스를 안전한 범위로 제한
            gb_idxs = [min(idx, max_available_idx) for idx in gb_idxs]
        # ===== IY ADD END =====

        num_kappas_ = min(self.num_kappas, self.min_candidates_lookahead_n)
        kappas = np.array([gb_wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs[:num_kappas_]])
        
        outside = True if np.sum(kappas) * cur_d < 0 else False

        tangent_idx = 20
        # if outside: # TODO : need to consider outside and inside
        if True:
            xy_m = np.array([(gb_wpnts[gb_idx].x_m, gb_wpnts[gb_idx].y_m) for gb_idx in gb_idxs])
            psi_rads = np.array([gb_wpnts[gb_idx].psi_rad for gb_idx in gb_idxs])

            tangent_idx = self.find_tangent_idx(xy_m, psi_rads)

            if self.measuring:
                mrk = Marker()
                mrk.header.frame_id = "map"
                mrk.header.stamp = self.get_clock().now().to_msg()
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.5
                mrk.scale.y = 0.5
                mrk.scale.z = 0.5
                mrk.color.a = 1.0
                mrk.color.b = 1.0
                mrk.color.r = 0.0
                mrk.color.g = 0.65

                mrk.pose.position.x = xy_m[tangent_idx, 0]
                mrk.pose.position.y = xy_m[tangent_idx, 1]
                mrk.pose.position.z = 0.01
                mrk.pose.orientation.w = 1.0
                self.recovery_lookahead_pub.publish(mrk)

            if tangent_idx != 0:
                target_s = tangent_idx * wpnt_dist


        points=[]
        tangents=[]

        points.append([self.cur_x,self.cur_y])
        points.append([xy_m[tangent_idx, 0],xy_m[tangent_idx, 1]])

        tangents.append(np.array([np.cos(self.cur_yaw), np.sin(self.cur_yaw)]))
        tangents.append(np.array([np.cos(psi_rads[tangent_idx]), np.sin(psi_rads[tangent_idx])]))

        tangents = np.dot(tangents, self.spline_scale*np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # Parametrization parameter s.
        dp = np.diff(points, axis=0)                 # difference between points
        dp = np.linalg.norm(dp, axis=1)              # distance between points
        d = np.cumsum(dp)                            # cumsum along the segments
        d = np.hstack([[0],d])                       # add distance from first point
        l = d[-1]                                    # length of point sequence
        nSamples =  int(l/wpnt_dist)                 # number of samples 
        s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step

        # Bring points and (optional) tangent information into correct format.
        assert(len(points) == len(tangents))
        spline_result = np.empty([nPoints, dim], dtype=object)
        for i,ref in enumerate(points):
            t = tangents[i]
            # Either tangent is None or has the same
            # number of dimensions as the point ref.
            assert(t is None or len(t)==dim)
            fuse = list(zip(ref,t) if t is not None else zip(ref,))
            spline_result[i,:] = fuse

        # Compute splines per dimension separately.
        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = BPoly.from_derivatives(d, spline_result[:,i])
            samples[:,i] = poly(s)



        # if samples.shape[0] < self.n_loc_wpnts:
        # ===== IY ADD : RECOVERY - Use ref_max_idx instead of gb_max_idx =====
        n_additional = 80
        xy_additional = np.array([
            (
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % ref_max_idx].x_m,
                gb_wpnts[(tangent_idx + cur_s_idx + i + 1) % ref_max_idx].y_m
            )
            for i in range(n_additional)
        ])
        # ===== IY ADD END =====

        samples = np.vstack([samples, xy_additional])

        # ===== IY ADD : RECOVERY - Use fixed path converter when in fixed path mode =====
        if self.use_fixed_path and self.fixed_path_converter is not None:
            s_, d_ = self.fixed_path_converter.get_frenet(samples[:, 0], samples[:, 1])
            rospy.loginfo_throttle(5.0, f"[{self.name}] Using fixed_path_converter for Frenet transformation")
        else:
            s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])
        # ===== IY ADD END =====

        # # ===== HJ MODIFIED: ALWAYS use GB converter for Frenet coordinates =====
        # # Recovery waypoints should always have s, d in GB Frenet coordinates
        # # regardless of whether we're in Smart mode or GB mode
        # s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])
        # rospy.loginfo_throttle(5.0, f"[{self.name}] Using GB converter for Frenet (use_fixed_path={self.use_fixed_path})")
        # # ===== HJ MODIFIED END =====

        psi_, kappa_ = tph.calc_head_curv_num.\
            calc_head_curv_num(
                path=samples,
                el_lengths=0.1*np.ones(len(samples)-1),
                is_closed=False
            )


        danger_flag = False
        for i in range(samples.shape[0]):
            # ===== IY ADD : RECOVERY - Use ref_max_idx instead of gb_max_idx =====
            gb_wpnt_i = int((s_[i] / wpnt_dist) % ref_max_idx)
            # ===== IY ADD END =====

            inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
            if not inside:
                rospy.loginfo_throttle_identical(
                    2, f"[{self.name}]: Evasion trajectory too close to TRACKBOUNDS, aborting evasion"
                )            # if abs(evasion_d[i]) > abs(tb_dist) - self.spline_bound_mindist:
                danger_flag = True
                break
            # Get V from gb wpnts and go slower if we are going through the inside
            vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9 # TODO make speed scaling ros param
            wpnts.wpnts.append(
                self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_[i], d=d_[i], v=vi, psi=psi_[i] + np.pi/2 , kappa= kappa_[i], wpnts=wpnts)
            )
            mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

        # Fill the rest of OTWpnts
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        if danger_flag:
            wpnts.wpnts = []
            mrks.markers = []


        return wpnts, mrks

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x:float, y:float, v:float, mrks: MarkerArray) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = v / self.gb_vmax
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75
        if self.from_bag:
            mrk.color.g = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = v / self.gb_vmax / 2
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x: float, y: float, opponent=True) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0
        mrk.color.g = 0.65

        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s: float, d: float, x: float, y: float, v: float, psi: float, kappa: float ,wpnts: WpntArray) -> Wpnt:
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = x
        wpnt.y_m = y
        wpnt.s_m = s
        wpnt.d_m = d
        wpnt.vx_mps = v
        wpnt.psi_rad = psi
        wpnt.kappa_radpm = kappa
        

        return wpnt


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSpliner()
    try:
        node.loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
