#!/usr/bin/env python3
import time
import numpy as np
import os
from nav_msgs.msg import Odometry
from f110_msgs.msg import Wpnt, WpntArray, Obstacle, ObstacleArray, OTWpntArray, OpponentTrajectory, OppWpnt, BehaviorStrategy, PredictionArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Float32
from scipy.optimize import minimize
from frenet_conversion.frenet_converter import FrenetConverter
from std_msgs.msg import Bool
from copy import deepcopy

from ccma import CCMA
import trajectory_planning_helpers as tph
from scipy.interpolate import BPoly
from grid_filter.grid_filter import GridFilter
import matplotlib.pyplot as plt


class ChangeAvoidanceNode:
    def __init__(self):
        # Initialize node
        rospy.init_node('change_avoidance_node')
        self.rate = rospy.Rate(20)

        # Params
        self.local_wpnts = None
        self.lookahead = 15
        self.past_avoidance_d = []
        
        # Scaled waypoints params
        self.scaled_wpnts = None
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        
        self.center_wpnts_msg = WpntArray()
        self.outer_lane_wpnts_msg = WpntArray()
        self.inner_lane_wpnts_msg = WpntArray()
        
        # Updated waypoints params
        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None
        
        # Obstalces params
        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_prediction = ObstacleArray()
        self.obs_prediction_pred = PredictionArray()
        
        # Opponent waypoint params
        self.opponent_waypoints = OpponentTrajectory()
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None
        
        # OT params
        self.last_ot_side = ""
        self.ot_section_check = False
        
        # Solver params
        self.min_radius = 0.55  # wheelbase / np.tan(max_steering)
        self.max_kappa = 1/self.min_radius
        self.width_car = 0.30
        self.safety_margin = 0.1
        self.avoidance_resolution = 20
        self.back_to_raceline_before = 5
        self.back_to_raceline_after = 5
        self.obs_traj_tresh = 2

        # Dynamic sovler params
        self.down_sampled_delta_s = 0.1
        self.global_traj_kappas = None

        # ROS Parameters
        self.measure = self._get_param_or_default("/measure", False)

        # Dynamic reconf params
        self.avoid_static_obs = True

        self.converter = None
        self.global_waypoints = None

        # Smart mode params
        self.smart_mode_active = False
        self.smart_converter = None
        self.smart_wpnts = None
        self.smart_wpnts_msg = None
        self.smart_max_idx = None
        self.smart_max_s = None
        self.smart_delta_s = None

        # Smart mode odom params
        self.current_s_fixed = None
        self.current_d_fixed = None
        self.current_vs_fixed = None

        # CCMA init
        self.ccma = CCMA(w_ma=10, w_cc=5)
        
        # Publishers
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers_sqp", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.merger_pub = self.create_publisher(Float32MultiArray, "/planner/avoidance/merger", 10)
        if self.measure:
            self.measure_pub = self.create_publisher(Float32, "/planner/pspliner_sqp/latency", 10)
            
        self.spline_sample_pub = self.create_publisher(MarkerArray, "/spline_sample_points", 10)

        # Subscribers
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obs_perception_cb, 10)
        self.create_subscription(ObstacleArray, "/opponent_prediction/obstacles", self.obs_prediction_cb, 10)
        self.create_subscription(PredictionArray, "/opponent_prediction/obstacles_pred", self.obstacle_prediction_cb, 10)
        self.create_subscription(PredictionArray, "/mpc_controller/ego_prediction", self.ego_prediction_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet_fixed", self.state_frenet_fixed_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cartesian_cb, 10)
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(Config, "/dynamic_change_tuner_node/parameter_updates", self.dyn_param_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.scaled_wpnts_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_updated", self.updated_wpnts_cb, 10)
        self.create_subscription(WpntArray, "/centerline_waypoints", self.center_wpnts_cb, 10)
        self.create_subscription(OpponentTrajectory, "/opponent_trajectory", self.opponent_trajectory_cb, 10)
        self.create_subscription(Bool, "/ot_section_check", self.ot_sections_check_cb, 10)
        self.create_subscription(Marker, "/lookahead_point", self.lookahead_callback, 10)
        self.create_subscription(Bool, "/planner/avoidance/smart_static_active", self.smart_static_active_cb, 10)
        self.create_subscription(OTWpntArray, "/planner/avoidance/smart_static_otwpnts", self.smart_static_otwpnts_cb, 10)

        self.converter = self.initialize_converter()

        self.map_filter = GridFilter(node=self, map_topic="/map", debug=False)
        self.map_filter.set_erosion_kernel_size(7)
        self.map_filter_fixed = None  # Smart mode용 GridFilter (obstacles_only_nearby.png)
        
        rospy.wait_for_message("/centerline_waypoints", WpntArray)
        self.generate_lanes(center_wpnts=self.center_wpnts_msg)
        
        self.num_extra_points = 100
        self.num_extra_global_points = 20
        
        self.lookahead_point_x = None
        self.lookahead_point_y = None

        # Map directory for GridFilter
        rospack = RosPack()
        stack_master_path = rospack.get_path("stack_master")
        self.map_name = self._get_param_or_default("/map_name", "")
        self.map_dir = os.path.join(stack_master_path, "maps")
        self.kernel_size = 7


    ### Callbacks ###
    def obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs_perception.obstacles = [obs for obs in data.obstacles if obs.is_static == False]

    def obs_prediction_cb(self, data: ObstacleArray):
        self.obs_prediction = data # 무시해

    def obstacle_prediction_cb(self, data: PredictionArray):
        self.obs_prediction_pred = data # 무시해

    def ego_prediction_cb(self, data: PredictionArray):
        self.ego_prediction = data

    def state_frenet_cb(self, data: Odometry):
        self.current_s = data.pose.pose.position.x
        self.current_d = data.pose.pose.position.y
        self.current_vs = data.twist.twist.linear.x

    def state_frenet_fixed_cb(self, data: Odometry):
        self.current_s_fixed = data.pose.pose.position.x
        self.current_d_fixed = data.pose.pose.position.y
        self.current_vs_fixed = data.twist.twist.linear.x

    def state_cartesian_cb(self, data: Odometry):
        self.current_x = data.pose.pose.position.x
        self.current_y = data.pose.pose.position.y
        quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        self.current_roll, self.current_pitch, self.current_yaw = euler_from_quaternion(quaternion)
        self.current_vx = data.twist.twist.linear.x

    def gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.gb_max_idx = data.wpnts[-1].id
        self.gb_max_s = data.wpnts[-1].s_m

    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        self.max_idx_updated = self.wpnts_updated[-1].id

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_max_s = data.wpnts[-1].s_m
        self.center_wpnts_max_idx = data.wpnts[-1].id
        
    def behavior_cb(self, data: BehaviorStrategy):
        self.local_wpnts_msg = data
        self.local_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.local_wpnts])

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.max_opp_idx = len(data.oppwpnts)-1
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data

    def dyn_param_cb(self, params: object):
        self.evasion_dist = self._get_param_or_default("dynamic_change_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = self._get_param_or_default("dynamic_change_tuner_node/obs_traj_tresh", 1.5)
        self.spline_bound_mindist = self._get_param_or_default("dynamic_change_tuner_node/spline_bound_mindist", 0.2)

        print(
            f"[Planner] Dynamic reconf triggered new spline params: \n"
            f" Evasion apex distance: {self.evasion_dist} [m],\n"
            f" Obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" Spline boundary mindist: {self.spline_bound_mindist} [m]\n"
        )
        
    def lookahead_callback(self, data):
        if data.id == 100:
            self.lookahead_point_x = data.pose.position.x
            self.lookahead_point_y = data.pose.position.y

    def smart_static_active_cb(self, data: Bool):
        self.smart_mode_active = data.data
        self.get_logger().info(f"[Change Avoidance] Smart mode active flag: {self.smart_mode_active}")

    def smart_static_otwpnts_cb(self, data: OTWpntArray):
        if len(data.wpnts) == 0:
            return

        self.smart_wpnts_msg = data
        self.smart_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.smart_max_idx = data.wpnts[-1].id
        self.smart_max_s = data.wpnts[-1].s_m

        if len(data.wpnts) > 1:
            self.smart_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

        # Smart converter 생성 (x, y 좌표로)
        smart_xy = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        if self.smart_converter is None or len(smart_xy) > 10:  # 충분한 포인트가 있을 때만 업데이트
            self.smart_converter = FrenetConverter(smart_xy[:, 0], smart_xy[:, 1])
            self.get_logger().info(f"[Change Avoidance] Created smart converter with {len(smart_xy)} points")

            # Load obstacles_only_nearby.png GridFilter for smart mode
            self.load_smart_gridfilter()


    ### Common Functions ###
    def load_smart_gridfilter(self):
        """
        Load GridFilter for obstacles_only_nearby.png map (for smart mode)
        This is called when smart converter is created
        """
        if not self.map_name:
            self.get_logger().warning("[Change Avoidance] Map name is empty, cannot load smart GridFilter")
            return

        obstacles_only_png_path = os.path.join(self.map_dir, f'{self.map_name}_obstacles_only_nearby.png')
        modified_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')

        if os.path.exists(obstacles_only_png_path):
            self.map_filter_fixed = GridFilter(node=self, map_topic=None, debug=False)
            if self.map_filter_fixed.load_from_file(obstacles_only_png_path, modified_yaml_path):
                self.map_filter_fixed.set_erosion_kernel_size(self.kernel_size)
                self.get_logger().info(f"[Change Avoidance] Loaded smart GridFilter from {obstacles_only_png_path}")
            else:
                self.get_logger().warning(f"[Change Avoidance] Failed to load smart GridFilter from {obstacles_only_png_path}")
                self.map_filter_fixed = None
        else:
            self.get_logger().warning(f"[Change Avoidance] Obstacles-only map not found at {obstacles_only_png_path}")
            self.map_filter_fixed = None

    def get_active_params(self):
        """
        Returns active parameters based on smart_mode_active flag.
        GB mode (False): 기존 global waypoints 기반
        Smart mode (True): smart static waypoints 기반
        """
        if self.smart_mode_active and self.smart_converter is not None:
            return {
                'converter': self.smart_converter,
                'scaled_wpnts': self.smart_wpnts,
                'scaled_wpnts_msg': self.smart_wpnts_msg,
                'scaled_max_idx': self.smart_max_idx,
                'scaled_max_s': self.smart_max_s,
                'scaled_delta_s': self.smart_delta_s,
                'current_s': self.current_s_fixed,
                'current_d': self.current_d_fixed,
                'current_vs': self.current_vs_fixed,
                'map_filter': self.map_filter_fixed if self.map_filter_fixed is not None else self.map_filter,
                'wpnts_updated': self.smart_wpnts_msg.wpnts if self.smart_wpnts_msg is not None else [],
                'max_idx_updated': self.smart_max_idx
            }
        else:
            # GB mode (기본 모드)
            return {
                'converter': self.converter,
                'scaled_wpnts': self.scaled_wpnts,
                'scaled_wpnts_msg': self.scaled_wpnts_msg,
                'scaled_max_idx': self.scaled_max_idx,
                'scaled_max_s': self.scaled_max_s,
                'scaled_delta_s': self.scaled_delta_s,
                'current_s': self.current_s,
                'current_d': self.current_d,
                'current_vs': self.current_vs,
                'map_filter': self.map_filter,
                'wpnts_updated': self.wpnts_updated,
                'max_idx_updated': self.max_idx_updated
            }

    def initialize_converter(self) -> bool:
            """
            Initialize the FrenetConverter object"""
            rospy.wait_for_message("/global_waypoints", WpntArray)

            # Initialize the FrenetConverter object
            converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
            self.get_logger().info("[Spliner] initialized FrenetConverter object")

            return converter

    def obstacle_preprocessing(self, obs: ObstacleArray):
        # Get active parameters based on mode
        params = self.get_active_params()

        # Sort obstacles by s_start (use fixed field if in smart mode)
        if self.smart_mode_active:
            obs.obstacles = sorted(obs.obstacles, key=lambda obs: obs.s_start_fixed)
        else:
            obs.obstacles = sorted(obs.obstacles, key=lambda obs: obs.s_start)

        considered_obs = []
        for obs in obs.obstacles:
            # Use appropriate s, d fields based on mode
            obs_s_start = obs.s_start_fixed if self.smart_mode_active else obs.s_start
            obs_d_center = obs.d_center_fixed if self.smart_mode_active else obs.d_center

            if (obs_s_start - params['current_s']) % params['scaled_max_s'] < self.lookahead and abs(obs_d_center - params['current_d']) < self.obs_traj_tresh:
                if False and obs.id == self.obs_prediction_pred.id and len(self.obs_prediction.obstacles) == 20: # 무시해
                    for opd in self.obs_prediction.obstacles:
                        considered_obs.append(opd)
                else:
                    considered_obs.append(obs)

        return considered_obs

    def more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        # Use appropriate obstacle fields based on mode
        obs_d_left = obstacle.d_left_fixed if self.smart_mode_active else obstacle.d_left
        obs_d_right = obstacle.d_right_fixed if self.smart_mode_active else obstacle.d_right

        left_gap = abs(gb_wpnts[gb_idxs[0]].d_left - obs_d_left)
        right_gap = abs(gb_wpnts[gb_idxs[0]].d_right + obs_d_right)
        min_space = self.spline_bound_mindist + self.width_car/2 + self.safety_margin

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obs_d_right - (self.width_car/2 + self.safety_margin + 0.2)
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0 and right_gap < abs(d_apex_right):
                d_apex_right = 0
            return "right", d_apex_right, left_gap, right_gap

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obs_d_left + (self.width_car/2 + self.safety_margin + 0.2)
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0 and left_gap < abs(d_apex_left):
                d_apex_left = 0
            return "left", d_apex_left, left_gap, right_gap
        elif left_gap < min_space and right_gap < min_space:
            # self.get_logger().warning("No enough gap!")
            return None, 0.0, left_gap, right_gap
        else:
            # self.get_logger().warning("This happen!")
            return None, 0.0, left_gap, right_gap


    ### Visualize SPL Rviz###
    def visualize_dynamic_spliner(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            for i in range(len(evasion_s)):
                mrk = Marker(header=rospy.Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.1
                mrk.scale.y = 0.1
                mrk.scale.z = evasion_v[i] / self.scaled_vmax
                mrk.color.a = 1.0
                mrk.color.g = 0.13
                mrk.color.r = 0.63
                mrk.color.b = 0.94

                mrk.id = i
                mrk.pose.position.x = evasion_x[i]
                mrk.pose.position.y = evasion_y[i]
                mrk.pose.position.z = evasion_v[i] / self.scaled_vmax / 2
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.mrks_pub.publish(mrks)
            
            
    def visualize_spline_samples(self, x_vals, y_vals):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "spline_samples"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1  # optional: small height
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.lifetime = rospy.Duration(0.2)  # stays for 0.2 sec
            marker_array.markers.append(marker)

        self.spline_sample_pub.publish(marker_array)


    ### Lane Change Avoidance ###
    def lane_change(self, considered_obs: list, cur_s: float):
        # Get active parameters based on mode
        params = self.get_active_params()

        # Use appropriate obstacle fields based on mode
        obs_s_start = considered_obs[0].s_start_fixed if self.smart_mode_active else considered_obs[0].s_start
        obs_s_end = considered_obs[-1].s_end_fixed if self.smart_mode_active else considered_obs[-1].s_end
        obs_s_center = considered_obs[0].s_center_fixed if self.smart_mode_active else considered_obs[0].s_center

        # Get the initial guess of the overtaking side
        initial_guess_object_start_idx = np.abs(params['scaled_wpnts'][:, 0] - obs_s_start).argmin()
        initial_guess_object_end_idx = np.abs(params['scaled_wpnts'][:, 0] - obs_s_end).argmin()

        # Get array of indexes of the global waypoints overlapping with the ROC
        gb_idxs = np.array(range(initial_guess_object_start_idx, initial_guess_object_start_idx + (initial_guess_object_end_idx - initial_guess_object_start_idx)%params['scaled_max_idx']))%params['scaled_max_idx']

        if len(gb_idxs) < 20:
            gb_idxs = [int(obs_s_center / params['scaled_delta_s'] + i) % params['scaled_max_idx'] for i in range(20)]

        side, initial_apex, left_gap, right_gap = self.more_space(considered_obs[0], params['scaled_wpnts_msg'].wpnts, gb_idxs)
        kappas = np.array([params['scaled_wpnts_msg'].wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
        max_kappa = np.max(np.abs(kappas))
        outside = "left" if np.sum(kappas) < 0 else "right"

        # Enlongate the ROC if our initial guess suggests that we are overtaking on the outside
        if side == outside:
            for i in range(len(considered_obs)):
                # Use appropriate fields based on mode
                obs_i_s_end = considered_obs[i].s_end_fixed if self.smart_mode_active else considered_obs[i].s_end
                obs_i_s_start = considered_obs[i].s_start_fixed if self.smart_mode_active else considered_obs[i].s_start

                elongation = (obs_i_s_end - obs_i_s_start) % params['scaled_max_s'] * max_kappa * (self.width_car + self.evasion_dist)

                # Update the appropriate field
                if self.smart_mode_active:
                    considered_obs[i].s_end_fixed = obs_i_s_end + elongation
                else:
                    considered_obs[i].s_end = obs_i_s_end + elongation

        min_s_obs_start = params['scaled_max_s']
        max_s_obs_end = 0

        for obs in considered_obs:
            # Use appropriate fields based on mode
            obs_s_start = obs.s_start_fixed if self.smart_mode_active else obs.s_start
            obs_s_end = obs.s_end_fixed if self.smart_mode_active else obs.s_end

            if obs_s_start < min_s_obs_start:
                min_s_obs_start = obs_s_start
            if obs_s_end > max_s_obs_end:
                max_s_obs_end = obs_s_end

        # Get local waypoints to check where we are and where we are heading
        # If we are closer than threshold to the opponent use the first two local waypoints as start points
        start_avoidance = min_s_obs_start
        end_avoidance = max_s_obs_end

        # Get a downsampled version for s avoidance points
        s_avoidance = np.arange(start_avoidance, end_avoidance + self.down_sampled_delta_s, self.down_sampled_delta_s)

        if s_avoidance.size == 0:
            self.get_logger().warning("s_avoidance is empty! Skipping avoidance.")
            return [], [], [], [], []

        # Get the closest scaled waypoint for every s avoidance point (down sampled)
        scaled_wpnts_indices = np.array([np.abs(params['scaled_wpnts'][:, 0] - s % params['scaled_max_s']).argmin() for s in s_avoidance])

        # Get the min radius
        # # Clip speed
        clipped_speed = np.clip(params['current_vs'], 1.0, a_max=None)
        # Get the minimum of clipped speed and the updated speed of the first waypoints
        radius_speed = min([clipped_speed, params['wpnts_updated'][(scaled_wpnts_indices[0])%params['max_idx_updated']].vx_mps])
        # Interpolate the min_radius with speeds between 0.2 and 7 m
        self.min_radius = np.interp(radius_speed, [1, 6, 7], [0.2, 2, 4])
        self.max_kappa = 1/self.min_radius

        initial_guess = np.zeros(len(s_avoidance))
        max_obs_idx_center = 0
        left_count = 0
        right_count = 0
        left_gap_sum = 0
        right_gap_sum = 0
        for obs in considered_obs:
            side, apex, left_gap, right_gap = self.more_space(obs, params['scaled_wpnts_msg'].wpnts, gb_idxs)
            left_gap_sum += left_gap
            right_gap_sum += right_gap
            if side is None:
                continue
            elif side == "left":
                left_count += 1
            elif side == "right":
                right_count += 1

            # Use appropriate fields based on mode
            obs_s_start = obs.s_start_fixed if self.smart_mode_active else obs.s_start
            obs_s_center = obs.s_center_fixed if self.smart_mode_active else obs.s_center
            obs_s_end = obs.s_end_fixed if self.smart_mode_active else obs.s_end

            obs_idx_start = np.abs(s_avoidance - obs_s_start).argmin()
            obs_idx_center = np.abs(s_avoidance - obs_s_center).argmin()
            obs_idx_end = np.abs(s_avoidance - obs_s_end).argmin()
            if obs_idx_start >= obs_idx_end or obs_idx_end >= len(s_avoidance):
                continue

            initial_guess[obs_idx_center] = apex

            if obs_idx_center > max_obs_idx_center:
                max_obs_idx_center = obs_idx_center
        
        left_gap_avg = left_gap_sum / left_count if left_count > 0 else 0
        right_gap_avg = right_gap_sum / right_count if right_count > 0 else 0
        
        if left_count > right_count and left_gap_avg > right_gap_avg and left_gap_avg > self.width_car:
            preferred_side = "left"
        elif right_count > left_count and right_gap_avg > left_gap_avg and right_gap_avg > self.width_car:
            preferred_side = "right"
        else:
            return [], [], [], [], []

        points=[]
        tangents=[]
                        
        extra_points_after_lookahead = 5
        if self.lookahead_point_x is not None and self.lookahead_point_y is not None:
            distances = [np.hypot(wpnt.x_m - self.lookahead_point_x, wpnt.y_m - self.lookahead_point_y) for wpnt in self.local_wpnts_msg.local_wpnts] # 로컬 웨이포인트는 알아서 s,d 맞게 나올거야.
            closest_idx = int(np.argmin(distances))

            cutoff_idx = min(closest_idx + extra_points_after_lookahead, len(self.local_wpnts_msg.local_wpnts))

            for i in range(cutoff_idx):
                wpnt = self.local_wpnts_msg.local_wpnts[i]
                points.append([wpnt.x_m, wpnt.y_m])
                tangents.append([np.cos(wpnt.psi_rad), np.sin(wpnt.psi_rad)])
        
        if preferred_side == "left":
            target_wpnts = self.inner_lane_wpnts_msg.wpnts
        elif preferred_side == "right":
            target_wpnts = self.outer_lane_wpnts_msg.wpnts

        for obs in considered_obs:
            # Use appropriate fields based on mode
            obs_s_center = obs.s_center_fixed if self.smart_mode_active else obs.s_center

            speed_scale = np.clip(params['current_vs'], 0.5, 5.0)
            temp_s_mod = (obs_s_center + speed_scale + 1.0) % params['scaled_max_s']
            temp_idx = np.argmin([abs(wpnt.s_m - temp_s_mod) for wpnt in target_wpnts])
            temp_s = target_wpnts[temp_idx].s_m
            temp_d = target_wpnts[temp_idx].d_m
            temp_psi = target_wpnts[temp_idx].psi_rad
            temp_resp = params['converter'].get_cartesian(temp_s, temp_d)
            temp_resp = temp_resp.T if temp_resp.ndim == 2 else temp_resp
            points.append(temp_resp.tolist())
            tangents.append([np.cos(temp_psi), np.sin(temp_psi)])

        extended_indices = [(temp_idx + i) % len(target_wpnts) for i in range(1, self.num_extra_points + 1)]
        for idx in extended_indices:
            wp = target_wpnts[idx]
            ext_s, ext_d = wp.s_m, wp.d_m
            ext_psi = wp.psi_rad
            ext_resp = params['converter'].get_cartesian(ext_s, ext_d)
            ext_resp = ext_resp.T if ext_resp.ndim == 2 else ext_resp
            points.append(ext_resp.tolist())
            tangents.append([np.cos(ext_psi), np.sin(ext_psi)])

        speed_scale = np.clip(params['current_vs'], 4.0, 6.0)
        extend_gb_idx = np.argmin([abs(wpnt.s_m - (ext_s + speed_scale)) for wpnt in params['scaled_wpnts_msg'].wpnts])
        extended_gb_indices = [(extend_gb_idx + i) % len(params['scaled_wpnts_msg'].wpnts) for i in range(1, self.num_extra_global_points + 1)]
        for idx in extended_gb_indices:
            wp = params['scaled_wpnts_msg'].wpnts[idx]
            ext_s, ext_d = wp.s_m, wp.d_m
            ext_psi = wp.psi_rad
            ext_resp = params['converter'].get_cartesian(ext_s, ext_d)
            ext_resp = ext_resp.T if ext_resp.ndim == 2 else ext_resp
            points.append(ext_resp.tolist())
            tangents.append([np.cos(ext_psi), np.sin(ext_psi)])
            
        tangents = np.dot(tangents, 1.0*np.eye(2))
        points = np.asarray(points)
        nPoints, dim = points.shape

        # Parametrization parameter s.
        dp = np.diff(points, axis=0)                 # difference between points
        dp = np.linalg.norm(dp, axis=1)              # distance between points
        d = np.cumsum(dp)                            # cumsum along the segments
        d = np.hstack([[0],d])                       # add distance from first point
        l = d[-1]                                    # length of point sequence
        nSamples =  int(l/params['scaled_delta_s'])  # number of samples
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
        try:
            for i in range(dim):
                poly = BPoly.from_derivatives(d, spline_result[:, i])
                samples[:, i] = poly(s)

        except Exception as e:
            return [], [], [], [], []

        for i in range(samples.shape[0]):
            inside = params['map_filter'].is_point_inside(samples[i, 0], samples[i, 1])

            if not inside:
                evasion_x = []
                evasion_y = []
                evasion_s = []
                evasion_d = []
                evasion_v = []
                return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v

        self.visualize_spline_samples(samples[:, 0], samples[:, 1])

        smoothed_xy_points = self.ccma.filter(samples)
        smoothed_sd_points = params['converter'].get_frenet(smoothed_xy_points[:, 0], smoothed_xy_points[:, 1])
        evasion_s, evasion_d = zip(*sorted(zip(smoothed_sd_points[0], smoothed_sd_points[1])))
        evasion_x = smoothed_xy_points[:, 0]
        evasion_y = smoothed_xy_points[:, 1]
        evasion_coords = np.column_stack((evasion_x, evasion_y))
        evasion_psi, evasion_kappa = tph.calc_head_curv_num.calc_head_curv_num(
            path=evasion_coords,
            el_lengths=0.1 * np.ones(len(evasion_coords) - 1),
            is_closed=False
        )
        evasion_psi += np.pi / 2
        evasion_v = np.zeros(len(evasion_s))

        temp_idx = np.argmin([abs(evs - params['current_s']) for evs in evasion_s])
        if abs(params['current_d'] - evasion_d[temp_idx]) > 0.5:
            # print(abs(params['current_d'] - evasion_d[temp_idx]))
            evasion_x = []
            evasion_y = []
            evasion_s = []
            evasion_d = []
            evasion_v = []
            return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v
        
        # Create a new evasion waypoint message
        evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=self.get_clock().now().to_msg(), frame_id="map"))
        evasion_wpnts = []
        evasion_wpnts = [Wpnt(id=len(evasion_wpnts), s_m=s, d_m=d, x_m=x, y_m=y, psi_rad=p, kappa_radpm=k, vx_mps= v) for x, y, s, d, p, k, v in zip(evasion_x, evasion_y, evasion_s, evasion_d, evasion_psi, evasion_kappa, evasion_v)]
        evasion_wpnts_msg.wpnts = evasion_wpnts
        # self.past_avoidance_d = initial_guess
        mean_d = np.mean(evasion_d)
        if mean_d > 0:
            self.last_ot_side = "left"
        else:
            self.last_ot_side = "right"

        self.evasion_pub.publish(evasion_wpnts_msg)
        self.visualize_dynamic_spliner(evasion_s, evasion_d, evasion_x, evasion_y, evasion_v) 

        return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v


    def resample_lane(self, xy: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        deltas = np.diff(xy, axis=0)
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        s = np.concatenate([[0], np.cumsum(dists)])
        s_new = np.arange(0, s[-1], resolution)
        x_new = np.interp(s_new, s, xy[:, 0])
        y_new = np.interp(s_new, s, xy[:, 1])
        return np.stack([x_new, y_new], axis=1)


    def generate_lanes(self, center_wpnts: WpntArray):
        original_center_wpnts = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in center_wpnts.wpnts])
        original_center_psi = np.array([wpnt.psi_rad for wpnt in center_wpnts.wpnts])
        
        min_center_left_gap = np.min([wpnt.d_left for wpnt in center_wpnts.wpnts])
        min_center_right_gap = np.min([wpnt.d_right for wpnt in center_wpnts.wpnts])

        # Map boundary min value show
        # print(f"min_center_left_gap: {min_center_left_gap}, min_center_right_gap: {min_center_right_gap}")
        
        normals = np.stack([np.sin(original_center_psi), -np.cos(original_center_psi)], axis=1)

        # 1. Map boundary min value
        # outer_lane = original_center_wpnts + normals * min_center_left_gap
        # inner_lane = original_center_wpnts - normals * min_center_right_gap

        # 2. Constant othogonal evasion
        outer_lane = original_center_wpnts + normals * 0.25
        inner_lane = original_center_wpnts - normals * 0.25
        
        outer_lane_resampled = self.resample_lane(outer_lane, resolution=0.1)
        inner_lane_resampled = self.resample_lane(inner_lane, resolution=0.1)

        # Use appropriate converter based on mode
        # Note: generate_lanes is called at init, so we always use GB converter here
        # The lane s,d will be recalculated dynamically when using smart mode
        active_converter = self.smart_converter if (self.smart_mode_active and self.smart_converter is not None) else self.converter

        outer_s, outer_d = active_converter.get_frenet(outer_lane_resampled[:, 0], outer_lane_resampled[:, 1])
        inner_s, inner_d = active_converter.get_frenet(inner_lane_resampled[:, 0], inner_lane_resampled[:, 1])
        
        outer_lane_resampled_psi, outer_lane_resampled_kappa = tph.calc_head_curv_num.calc_head_curv_num(
                path=outer_lane_resampled,
                el_lengths=0.1*np.ones(len(outer_lane_resampled)-1),
                is_closed=False,
                stepsize_curv_preview = 5.0,
                stepsize_curv_review = 5.0
            )
        outer_lane_resampled_psi += np.pi / 2

        inner_lane_resampled_psi, inner_lane_resampled_kappa = tph.calc_head_curv_num.calc_head_curv_num(
                path=inner_lane_resampled,
                el_lengths=0.1*np.ones(len(inner_lane_resampled)-1),
                is_closed=False,
                stepsize_curv_preview = 5.0,
                stepsize_curv_review = 5.0
            )
        inner_lane_resampled_psi += np.pi / 2
        
        self.outer_lane_wpnts_msg.header.stamp = self.get_clock().now().to_msg()
        self.inner_lane_wpnts_msg.header.stamp = self.get_clock().now().to_msg()
        
        self.outer_lane_wpnts_msg.header.frame_id = "map"
        self.inner_lane_wpnts_msg.header.frame_id = "map"
        
        for i in range(len(outer_lane_resampled)):
            wpnt = Wpnt()
            wpnt.id = i
            wpnt.x_m = outer_lane_resampled[i, 0]
            wpnt.y_m = outer_lane_resampled[i, 1]
            wpnt.s_m = outer_s[i]
            wpnt.d_m = outer_d[i]
            wpnt.psi_rad = outer_lane_resampled_psi[i]
            wpnt.kappa_radpm = outer_lane_resampled_kappa[i]
            wpnt.d_left = 0.0
            wpnt.d_right = 0.0
            wpnt.vx_mps = 0.0
            wpnt.ax_mps2 = 0.0
            self.outer_lane_wpnts_msg.wpnts.append(wpnt)
            
        for i in range(len(inner_lane_resampled)):
            wpnt = Wpnt()
            wpnt.id = i
            wpnt.x_m = inner_lane_resampled[i, 0]
            wpnt.y_m = inner_lane_resampled[i, 1]
            wpnt.s_m = inner_s[i]
            wpnt.d_m = inner_d[i]
            wpnt.psi_rad = inner_lane_resampled_psi[i]
            wpnt.kappa_radpm = inner_lane_resampled_kappa[i]
            wpnt.d_left = 0.0
            wpnt.d_right = 0.0
            wpnt.vx_mps = 0.0
            wpnt.ax_mps2 = 0.0
            self.inner_lane_wpnts_msg.wpnts.append(wpnt)
        
        # ------------- Plot -------------
        plt.figure(figsize=(8, 6))
        plt.plot(original_center_wpnts[:,0], original_center_wpnts[:,1], 'k-', label='Center Line')
        plt.plot(outer_lane_resampled[:,0], outer_lane_resampled[:,1], 'g--', label='Outer Lane')
        plt.plot(inner_lane_resampled[:,0], inner_lane_resampled[:,1], 'b--', label='Inner Lane')

        plt.scatter(original_center_wpnts[:,0], original_center_wpnts[:,1], c='k', s=10)
        plt.scatter(outer_lane_resampled[:,0], outer_lane_resampled[:,1], c='g', s=10)
        plt.scatter(inner_lane_resampled[:,0], inner_lane_resampled[:,1], c='b', s=10)

        plt.axis('equal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Lane Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()
        # ------------- Plot -------------

    ### Main Loop ###
    def loop(self):
        # Wait for critical Messages and services
        self.get_logger().info("[OBS Spliner] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dynamic_change_tuner_node/parameter_updates", Config)
        rospy.wait_for_message("/behavior_strategy", BehaviorStrategy)
        self.get_logger().info("[OBS Spliner] Ready!")
        
        while not rospy.is_shutdown():
            start_time = time.perf_counter()

            # Get active parameters based on mode
            params = self.get_active_params()

            # Obstacle pre-processing
            obs = deepcopy(self.obs_perception)
            considered_obs = self.obstacle_preprocessing(obs=obs)

            # If there is an obstacle and we are in OT section
            # if len(considered_obs) > 0 and self.ot_section_check == True:
            if len(considered_obs) > 0:
                evasion_x, evasion_y, evasion_s, evasion_d, evasion_v = self.lane_change(considered_obs, params['current_s'])
                # Publish merge reagion if evasion track has been found
                if len(evasion_s) > 0:
                    # Use appropriate obs field based on mode
                    obs_s_end = considered_obs[-1].s_end_fixed if self.smart_mode_active else considered_obs[-1].s_end
                    self.merger_pub.publish(Float32MultiArray(data=[obs_s_end%params['scaled_max_s'], evasion_s[-1]%params['scaled_max_s']]))
            # If there is no point in overtaking anymore delte all markers
            else:
                mrks = MarkerArray()
                del_mrk = Marker(header=rospy.Header(stamp=self.get_clock().now().to_msg()))
                del_mrk.action = Marker.DELETEALL
                mrks.markers = []
                mrks.markers.append(del_mrk)
                self.mrks_pub.publish(mrks)

            # publish latency
            if self.measure:
                self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))

            self.rate.sleep()


if __name__ == "__main__":
    ChangeAvoidance = ChangeAvoidanceNode()
    ChangeAvoidance.loop()