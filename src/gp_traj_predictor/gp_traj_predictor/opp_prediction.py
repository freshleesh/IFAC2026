#!/usr/bin/env python3

import numpy as np
from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import Odometry
from f110_msgs.msg import ObstacleArray, WpntArray, Wpnt, Obstacle, ObstacleArray, OpponentTrajectory, OppWpnt, Prediction, PredictionArray, OTWpntArray
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.srv import Glob2FrenetArr, Frenet2GlobArr
from frenet_converter.frenet_converter import FrenetConverter  # HJ ADDED: For smart static Frenet conversion
import time
import copy

from std_srvs.srv import SetBool, SetBoolResponse

OPP_TRAJ_USE_THRESHOLD = 0.35

class OppTrajPredictor:
    def __init__(self):
        # Initialize the node
        rospy.init_node('opponent_propagation_predictor', anonymous=True)

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        self.glob2frenet = rospy.ServiceProxy("convert_glob2frenetarr_service", Glob2FrenetArr)
        self.frenet2glob = rospy.ServiceProxy("convert_frenet2globarr_service", Frenet2GlobArr)
        self.loop_rate = 10 #Hz

        # HJ ADDED: Use vd-based prediction for reactive vehicles
        self.USE_V_D = self._get_param_or_default('~use_vd_prediction', True)
        self.get_logger().info(f"[Opp. Pred.] USE_V_D prediction: {self.USE_V_D}")

        # Publisher
        self.marker_pub_beginn = self.create_publisher(Marker, "/opponent_predict/beginn", 10)
        self.marker_pub_end = self.create_publisher(Marker, "/opponent_predict/end", 10)
        self.prediction_obs_pub = self.create_publisher(ObstacleArray, "/opponent_prediction/obstacles", 10)
        self.prediction_obs_pred_pub = self.create_publisher(PredictionArray, "/opponent_prediction/obstacles_pred", 10)
        self.force_trailing_pub = self.create_publisher(Bool, "/opponent_prediction/force_trailing", 10)
        
        self.opp_traj_gp_pub = self.create_publisher(OpponentTrajectory, '/opponent_trajectory', 10)
        self.opp_traj_marker_pub = self.create_publisher(MarkerArray, '/opponent_traj_markerarray', 10)
        self.opp_marker_pub = self.create_publisher(MarkerArray, '/opponent_prediction_markerarray', 10)

        # Callback data - initialize BEFORE subscribing to avoid race condition
        self.opponent_pos = ObstacleArray()
        self.car_odom = Odometry()
        self.wpnts_opponent = list()
        self.wpnts_updated = list()
        self.state = String()

        # HJ ADDED: Smart static mode support
        self.smart_static_active = False
        self.smart_static_wpnts = list()
        self.smart_converter = None  # FrenetConverter for smart static path
        self.smart_wpnts_gb_frenet = list()  # SMART waypoints converted to GB Frenet (s_m, d_m)

        self.speed_offset = 0 # m/s

        # Subscriber - register AFTER initializing callback data
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.opponent_state_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.odom_cb, 10)
        self.create_subscription(OpponentTrajectory, self.opponent_traj_topic, self.opponent_trajectory_cb, 10)
        # HJ MODIFIED: Use global_waypoints_scaled instead of global_waypoints_updated (not published)
        # self.create_subscription(WpntArray, '/global_waypoints_updated', self.wpnts_updated_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self.wpnts_updated_cb, 10)
        self.create_subscription(WpntArray, "/centerline_waypoints", self.center_wpnts_cb, 10)
        self.create_subscription(String, "/state_machine", self.state_cb, 10)
        self.create_subscription(Config, "/dynamic_prediction_tuner_node/parameter_updates", self.dyn_param_cb, 10)
        # HJ ADDED: Smart static mode support
        self.create_subscription(Bool, "/planner/avoidance/smart_static_active", self.smart_static_active_cb, 10)
        self.create_subscription(OTWpntArray, "/planner/avoidance/smart_static_otwpnts", self.smart_static_wpnts_cb, 10)

        # Service server
        rospy.Service('/init_opp_trajectory', SetBool, self.init_opp_bool)

        # Simulation parameters
        self.time_steps = 200
        self.dt = 0.02 # s
        self.save_distance_front = 0.6 # m
        self.save_distance_back = 0.4 # m
        self.max_v = 10 # m/s
        self.min_v = 0 # m/s
        self.max_a = 5.5 # m/s^2
        self.min_a = 5 # m/s^2
        self.max_expire_counter = 10

        # HJ ADDED: Obstacle size parameter
        self.obstacle_half_width = self._get_param_or_default('~obstacle_half_width', 0.3)  # m
        self.get_logger().info(f"[Opp. Pred.] Obstacle half width: {self.obstacle_half_width}m")

        # Number of time steps before prediction expires. Set when prediction is published.
        self.expire_counter = 0

        # Visualization
        self.marker_beginn = self.marker_init(a = 0.5, r = 0.63, g = 0.13, b = 0.94, id = 0)
        self.marker_end = self.marker_init(a = 0.5, r = 0.63, g = 0.13, b = 0.94, id = 1)

        # Opponent
        self.opponent_lap_count = None
        
        # Stanley params
        self.k = 0.5  # control gain
        self.Kp = 1.0  # speed proportional gain
        # self.dt = 0.1  # [s] time difference
        self.L = 0.33  # [m] Wheel base of vehicle
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        
        self.init_fixed_wpnts = False
        self.fixed_wpnts = []

    # Service function
    def init_opp_bool(self, req):
        if req.data:
            self.get_logger().info("Received request: ON")
            success = True
            message = "Feature turned ON"
            self.global_to_opptraj(self.wpnts_updated)
        else:
            self.get_logger().info("Received request: OFF")
            success = True
            message = "Feature turned OFF"
        return SetBoolResponse(success, message)
    
    def global_to_opptraj(self, wptlist: list):
        wpnts_opponent = [OppWpnt(s_m=wp.s_m, d_m=wp.d_m, x_m=wp.x_m, y_m=wp.y_m, proj_vs_mps=wp.vx_mps) for wp in wptlist]
        
        opp_traj_gp_msg = self.make_opponent_trajectory_msg(wpnts_opponent)
        opp_traj_marker_array = self.visualize_opponent_wpnts(wpnts_opponent)
        
        self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
        self.opp_traj_marker_pub.publish(opp_traj_marker_array)
        
    def make_opponent_trajectory_msg(self, oppwpnts_list: list):
        opponent_trajectory_msg = OpponentTrajectory()
        opponent_trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        opponent_trajectory_msg.oppwpnts = oppwpnts_list
        return opponent_trajectory_msg
        
    def visualize_opponent_wpnts(self, oppwpnts_list: list):
        opp_traj_marker_array = MarkerArray()
    
        i=0
        for i in range(len(oppwpnts_list)):
            marker_height = oppwpnts_list[i].proj_vs_mps/10.0

            marker = Marker(header=rospy.Header(frame_id="map"), id = i, type = Marker.CYLINDER)
            marker.pose.position.x = oppwpnts_list[i].x_m
            marker.pose.position.y = oppwpnts_list[i].y_m
            marker.pose.position.z = marker_height/2
            marker.pose.orientation.w = 1.0
            marker.scale.x = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.y = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.z = marker_height
            if oppwpnts_list[i].vs_var == 69:
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            opp_traj_marker_array.markers.append(marker)
        return opp_traj_marker_array

    
    ### CALLBACKS ###
    # TODO: Only sees the first dynamic obstacle as opponent...
    def opponent_state_cb(self, data: ObstacleArray):
        self.opponent_pos.header = data.header
        is_dynamic = False
        if len(data.obstacles) > 0:
            for obs in data.obstacles:
                if obs.is_static == False: # and obs.is_opponent == True: # Changed by Tino and Nicolas for compatibility with new obstacle message
                    self.opponent_pos.obstacles = [obs]
                    is_dynamic = True
                    break
        if is_dynamic == False:
            self.opponent_pos.obstacles = []

    def odom_cb(self, data: Odometry): self.car_odom = data

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.wpnts_opponent = data.oppwpnts  # exclude last point (because last point == first point) <- Hopefully this is still the case?
        self.max_s_opponent = self.wpnts_opponent[-1].s_m
        self.opponent_lap_count = data.lap_count

    def wpnts_updated_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m # Should be the same as self.max_s but just in case. Only used for wrap around

    def center_wpnts_cb(self, data: WpntArray):
        self.center_wpnts_msg = data
        self.center_wpnts_max_s = data.wpnts[-1].s_m
        self.center_wpnts_max_idx = data.wpnts[-1].id
        
        if not self.init_fixed_wpnts:
            self.fixed_wpnts = [[wp.x_m, wp.y_m] for wp in data.wpnts]
            self.cx, self.cy = zip(*self.fixed_wpnts)
            self.cyaw = [wp.psi_rad for wp in data.wpnts]
            self.init_fixed_wpnts = True

    def state_cb(self, data: String):
        self.state = data.data

        # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: Config):
        """
        Notices the change in the parameters and changes spline params
        """
        self.time_steps = self._get_param_or_default("dynamic_prediction_tuner_node/n_time_steps", 200)
        self.dt = self._get_param_or_default("dynamic_prediction_tuner_node/dt", 0.02)
        self.save_distance_front = self._get_param_or_default("dynamic_prediction_tuner_node/save_distance_front", 0.6)
        self.max_expire_counter = self._get_param_or_default("dynamic_prediction_tuner_node/max_expire_counter", 10)
        self.speed_offset = self._get_param_or_default("dynamic_prediction_tuner_node/speed_offset", 0)

        print(
            f"[Opp. Pred.] Dynamic reconf triggered new params:\n"
            f" N time stepts: {self.time_steps}, \n"
            f" dt: {self.dt} [s], \n"
            f" save_distance_front: {self.save_distance_front} [m], \n"
            f" max_expire_counter: {self.max_expire_counter}"
        )

    # HJ ADDED: Smart static mode callbacks
    def smart_static_active_cb(self, data: Bool):
        self.smart_static_active = data.data

    def smart_static_wpnts_cb(self, data: OTWpntArray):
        self.smart_static_wpnts = data.wpnts  # Extract Wpnt[] from OTWpntArray

        # Create FrenetConverter for smart static path (only once)
        if len(self.smart_static_wpnts) > 1 and self.smart_converter is None:
            try:
                x_arr = np.array([wpnt.x_m for wpnt in self.smart_static_wpnts])
                y_arr = np.array([wpnt.y_m for wpnt in self.smart_static_wpnts])
                z_arr = np.array([wpnt.z_m for wpnt in self.smart_static_wpnts])
                self.smart_converter = FrenetConverter(x_arr, y_arr, z_arr)
                self.get_logger().info(f"[Opp. Pred.] Created FrenetConverter for SMART static path ({len(self.smart_static_wpnts)} waypoints)")

                # Convert SMART waypoints to GB Frenet for convergence target using glob2frenet service
                self.smart_wpnts_gb_frenet = []
                x_list = [wpnt.x_m for wpnt in self.smart_static_wpnts]
                y_list = [wpnt.y_m for wpnt in self.smart_static_wpnts]

                # Call glob2frenet service to convert all SMART waypoints to GB Frenet at once
                result = self.glob2frenet(x_list, y_list)
                for i in range(len(result.s)):
                    self.smart_wpnts_gb_frenet.append({'s_m': result.s[i], 'd_m': result.d[i]})
                self.get_logger().info(f"[Opp. Pred.] Converted SMART waypoints to GB Frenet ({len(self.smart_wpnts_gb_frenet)} points)")

            except Exception as e:
                self.get_logger().error(f"[Opp. Pred.] Failed to create FrenetConverter for SMART path: {e}")
                self.smart_converter = None


    ### HELPER FUNCTIONS ###
    def marker_init(self, a = 1, r = 1, g = 0, b = 0, id = 0):
        marker = Marker(header=rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"),id = id, type = Marker.SPHERE)
        # Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
        marker.pose.orientation.w = 1.0
        # Set the marker's scale
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        # Set the marker's color
        marker.color.a = a  # Alpha (transparency)
        marker.color.r = r  # Red
        marker.color.g = g  # Green
        marker.color.b = b  # Blue
        return marker

    def delete_all(self) -> None:
        empty_marker = Marker(header=rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"),id = 0)
        empty_marker.action = Marker.DELETE
        self.marker_pub_beginn.publish(empty_marker)
        empty_marker.id = 1
        self.marker_pub_end.publish(empty_marker)

        empty_obs_arr = ObstacleArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"))
        self.prediction_obs_pub.publish(empty_obs_arr)

    ### MAIN LOOP ###
    def loop(self):
        rate = rospy.Rate(self.loop_rate)

        self.get_logger().info("[Opp. Pred.] Opponent Predictor wating...")
        rospy.wait_for_message("/global_waypoints", WpntArray)
        # HJ MODIFIED: Use global_waypoints_scaled instead of global_waypoints_updated (not published)
        # rospy.wait_for_message("/global_waypoints_updated", WpntArray)
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        self.get_logger().info("[Opp. Pred.] Scaled waypoints recived!")
        rospy.wait_for_message(self.opponent_traj_topic, OpponentTrajectory)
        self.get_logger().info("[Opp. Pred.] Opponent waypoints recived!")
        rospy.wait_for_message("/tracking/obstacles", ObstacleArray)
        self.get_logger().info("[Opp. Pred.] Obstacles reveived!")
        self.get_logger().info("[Opp. Pred.] Opponent Predictor ready!")

        while not rospy.is_shutdown():
            
            opponent_pos_copy = copy.deepcopy(self.opponent_pos)

            prediction_obs_pred_arr = PredictionArray()

            # HJ DEBUG
            rospy.logwarn_throttle(2.0, f"[Opp Pred DEBUG] opponent_pos obstacles count: {len(opponent_pos_copy.obstacles)}")

            if len(opponent_pos_copy.obstacles) != 0:
                # HJ ADDED: Check if SMART mode is active (for final Prediction conversion only)
                use_smart_frenet = self.smart_static_active and self.smart_converter is not None

                # All internal calculations use GB Frenet (original behavior)
                current_ego_s = self.car_odom.pose.pose.position.x
                current_opponent_s = opponent_pos_copy.obstacles[0].s_center

                # Handle wrap around
                if current_ego_s > self.max_s_updated * (2/3) and current_opponent_s < self.max_s_updated * (1/3):
                    current_opponent_s += self.max_s_updated

                current_opponent_d = opponent_pos_copy.obstacles[0].d_center

                # Find center d for convergence: centerline (GB) or smart path (SMART mode)
                if use_smart_frenet and len(self.smart_wpnts_gb_frenet) > 0:
                    # SMART mode: converge to SMART path (in GB Frenet coordinates)
                    smart_s_array_gb = np.array([wpnt['s_m'] for wpnt in self.smart_wpnts_gb_frenet])
                    approx_opponent_center_indx = np.abs(smart_s_array_gb - current_opponent_s).argmin()
                    opponent_approx_center_d = self.smart_wpnts_gb_frenet[approx_opponent_center_indx]['d_m']
                else:
                    # GB mode: converge to centerline
                    s_points_center_array = np.array([wpnt.s_m for wpnt in self.center_wpnts_msg.wpnts])
                    approx_opponent_center_indx = np.abs(s_points_center_array - current_opponent_s).argmin()
                    opponent_approx_center_d = self.center_wpnts_msg.wpnts[approx_opponent_center_indx].d_m

                current_opponent_v = opponent_pos_copy.obstacles[0].vs

                # HJ ADDED: Safety check for opponent trajectory availability
                has_opponent_raceline = len(self.wpnts_opponent) > 1 and self.opponent_lap_count is not None

                if has_opponent_raceline:
                    approx_s_points_global_array = np.array([wpnt.s_m for wpnt in self.wpnts_opponent])
                    opponent_approx_indx = np.abs(approx_s_points_global_array - current_opponent_s).argmin()
                    opponent_approx_raceline_d = self.wpnts_opponent[opponent_approx_indx].d_m
                else:
                    # No opponent trajectory yet - use current position as reference
                    approx_s_points_global_array = np.array([])
                    opponent_approx_raceline_d = current_opponent_d

                start = time.process_time()

                # ===== ORIGINAL FUNCTION (before HJ modification) =====
                # if abs(current_opponent_d - opponent_approx_raceline_d) > 0.25 or self.opponent_lap_count < 1:
                #     self.force_trailing_pub.publish(True)
                #
                #     obstacle_list = []
                #     prediction_list = []
                #
                #     opp_marker_array = MarkerArray()
                #
                #     for i in range(self.time_steps):
                #         w = i / (self.time_steps - 1)
                #
                #         interpolated_d = (1 - w) * current_opponent_d + w * opponent_approx_center_d
                #
                #         obs = Obstacle()
                #         obs.id = i
                #         obs.s_start = current_opponent_s
                #         obs.s_end = current_opponent_s
                #         obs.s_center = current_opponent_s + i * current_opponent_v * self.dt
                #         obs.d_center = interpolated_d
                #         obs.d_left = obs.d_center + 0.25
                #         obs.d_right = obs.d_center - 0.25
                #         obs.size = opponent_pos_copy.obstacles[0].size
                #         obs.vs = current_opponent_v
                #         obs.vd = 0
                #         obs.is_actually_a_gap = False
                #         obs.is_static = False
                #         obstacle_list.append(obs)
                #
                #         pds = Prediction()
                #         pds.id = i
                #         pds.pred_s = obs.s_center
                #         pds.pred_d = obs.d_center
                #         prediction_list.append(pds)
                # ===== ORIGINAL FUNCTION END =====

                ############################# HJ MODIFIED START #############################
                # Choose prediction method based on USE_V_D flag
                # Use original method for first lap (no learned trajectory yet)
                # Use vd method only after 1 lap AND when lateral deviation > 0.25m
                use_vd_method = self.USE_V_D and self.opponent_lap_count >= 1 and abs(current_opponent_d - opponent_approx_raceline_d) > OPP_TRAJ_USE_THRESHOLD
                use_original_method = (not self.USE_V_D or self.opponent_lap_count < 1) and (abs(current_opponent_d - opponent_approx_raceline_d) > OPP_TRAJ_USE_THRESHOLD or self.opponent_lap_count < 1)

                # ===== HJ MODIFIED: Blend learned raceline vd with current vd =====
                if use_vd_method:
                    self.force_trailing_pub.publish(True)

                    obstacle_list = []
                    prediction_list = []

                    opp_marker_array = MarkerArray()

                    # Get current lateral velocity from tracking
                    current_opponent_vd = opponent_pos_copy.obstacles[0].vd

                    # Decay factor for lateral velocity (reactive movements dampen over time)
                    vd_decay = 0.6  # vd reduces by 40% per timestep

                    # Blending weight: how much to trust current vd vs learned raceline
                    # 0.0 = pure raceline, 1.0 = pure current vd
                    vd_blend_weight = 0.3  # Favor current reactive behavior

                    # Check if opponent trajectory is available for blending
                    has_opponent_trajectory = len(self.wpnts_opponent) > 1 and hasattr(self, 'max_s_opponent')

                    for i in range(self.time_steps):
                        # Get opponent's future s position
                        future_s = current_opponent_s + i * current_opponent_v * self.dt

                        # Extract learned raceline's lateral velocity (only if trajectory available)
                        raceline_vd = 0.0
                        if has_opponent_trajectory:
                            future_opponent_idx = np.abs(approx_s_points_global_array - future_s % self.max_s_opponent).argmin()
                            next_opponent_idx = (future_opponent_idx + 1) % len(self.wpnts_opponent)

                            # Calculate raceline's d change rate (vd from learned trajectory)
                            raceline_d_current = self.wpnts_opponent[future_opponent_idx].d_m
                            raceline_d_next = self.wpnts_opponent[next_opponent_idx].d_m
                            raceline_vs = self.wpnts_opponent[future_opponent_idx].proj_vs_mps

                            # Avoid division by zero
                            if raceline_vs > 0.1:
                                s_step = self.wpnts_opponent[next_opponent_idx].s_m - self.wpnts_opponent[future_opponent_idx].s_m
                                if s_step < 0:  # Handle wrap-around
                                    s_step += self.max_s_opponent
                                raceline_vd = (raceline_d_next - raceline_d_current) / (s_step / raceline_vs) if s_step > 0 else 0.0

                        # Blend current vd with raceline vd, then apply decay
                        decayed_current_vd = current_opponent_vd * (vd_decay ** i)
                        blended_vd = vd_blend_weight * decayed_current_vd + (1 - vd_blend_weight) * raceline_vd

                        # Apply blended lateral velocity
                        lateral_displacement = 0.0
                        for j in range(i + 1):
                            step_vd = vd_blend_weight * current_opponent_vd * (vd_decay ** j) + (1 - vd_blend_weight) * raceline_vd
                            lateral_displacement += step_vd * self.dt

                        predicted_d = current_opponent_d + lateral_displacement

                        obs = Obstacle()
                        obs.id = i
                        obs.s_start = current_opponent_s
                        obs.s_end = current_opponent_s
                        obs.s_center = future_s
                        obs.d_center = predicted_d
                        obs.d_left = obs.d_center + self.obstacle_half_width
                        obs.d_right = obs.d_center - self.obstacle_half_width
                        obs.size = opponent_pos_copy.obstacles[0].size
                        obs.vs = current_opponent_v
                        obs.vd = blended_vd
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        # HJ MODIFIED: Use SMART Frenet if active, otherwise GB Frenet
                        if use_smart_frenet:
                            # Convert GB prediction to SMART Frenet using smart_converter
                            pos_cart = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                            result = self.smart_converter.get_frenet(np.array([pos_cart.x[0]]), np.array([pos_cart.y[0]]))
                            pds.pred_s = float(result[0])
                            pds.pred_d = float(result[1])
                        else:
                            # GB mode: use obs.s_center and obs.d_center directly
                            pds.pred_s = obs.s_center
                            pds.pred_d = obs.d_center
                        prediction_list.append(pds)

                        marker = Marker()
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        marker.pose.position.z = 0.1

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15
                        marker.color.a = 0.8
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)

                    prediction_obs_arr = ObstacleArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), obstacles = obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), id = opponent_pos_copy.obstacles[0].id, predictions = prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)

                    self.expire_counter = 0

                # ===== ORIGINAL METHOD: Linear interpolation to center =====
                elif use_original_method:
                    self.force_trailing_pub.publish(True)

                    obstacle_list = []
                    prediction_list = []

                    opp_marker_array = MarkerArray()

                    for i in range(self.time_steps):
                        w = i / (self.time_steps - 1)

                        interpolated_d = (1 - w) * current_opponent_d + w * opponent_approx_center_d

                        obs = Obstacle()
                        obs.id = i
                        obs.s_start = current_opponent_s
                        obs.s_end = current_opponent_s
                        obs.s_center = current_opponent_s + i * current_opponent_v * self.dt
                        obs.d_center = interpolated_d
                        obs.d_left = obs.d_center + self.obstacle_half_width
                        obs.d_right = obs.d_center - self.obstacle_half_width
                        obs.size = opponent_pos_copy.obstacles[0].size
                        obs.vs = current_opponent_v
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        # HJ MODIFIED: Use SMART Frenet if active, otherwise GB Frenet
                        if use_smart_frenet:
                            # Convert GB prediction to SMART Frenet using smart_converter
                            pos_cart = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                            result = self.smart_converter.get_frenet(np.array([pos_cart.x[0]]), np.array([pos_cart.y[0]]))
                            pds.pred_s = float(result[0])
                            pds.pred_d = float(result[1])
                        else:
                            # GB mode: use obs.s_center and obs.d_center directly
                            pds.pred_s = obs.s_center
                            pds.pred_d = obs.d_center
                        prediction_list.append(pds)

                        marker = Marker()
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        marker.pose.position.z = 0.1

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15
                        marker.color.a = 0.8
                        marker.color.r = 1.0  # Red for original method
                        marker.color.g = 0.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)

                    prediction_obs_arr = ObstacleArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), obstacles = obstacle_list)
                    self.prediction_obs_pub.publish(prediction_obs_arr)

                    prediction_obs_pred_arr = PredictionArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), id = opponent_pos_copy.obstacles[0].id, predictions = prediction_list)
                    self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                    self.opp_marker_pub.publish(opp_marker_array)

                    self.expire_counter = 0
                # ===== HJ MODIFIED END =====
                #############################
                else:
                    self.force_trailing_pub.publish(False)
                    
                    # Temp params
                    beginn = False
                    end = False
                    beginn_s = 0
                    end_s = 0
                    beginn_d = 0
                    end_d = 0
                    obstacle_list = []
                    prediction_list = []
                    
                    # Find begin of the prediction
                    if (beginn == False and ((current_opponent_s - current_ego_s)%self.max_s_updated < self.save_distance_front or abs(current_opponent_s - current_ego_s) < self.save_distance_front)):
                        beginn_s = current_opponent_s
                        beginn_d = current_opponent_d
                        beginn = True
                        
                    opp_marker_array = MarkerArray()
                    
                    for i in range(self.time_steps):
                        # Get the speed at position i + 1
                        opponent_approx_indx = np.abs(approx_s_points_global_array - current_opponent_s % self.max_s_opponent).argmin()
                        opponent_speed = self.wpnts_opponent[opponent_approx_indx].proj_vs_mps
                        current_opponent_s = (current_opponent_s + opponent_speed * self.dt)
                        opponent_d = self.wpnts_opponent[opponent_approx_indx].d_m

                        obs = Obstacle()
                        obs.id = i
                        obs.s_start = current_opponent_s
                        obs.s_end = current_opponent_s + opponent_speed * self.dt
                        obs.s_center = (obs.s_start + obs.s_end) / 2
                        obs.d_center = opponent_d
                        obs.d_left = opponent_d + self.obstacle_half_width
                        obs.d_right = opponent_d - self.obstacle_half_width
                        obs.size = opponent_pos_copy.obstacles[0].size
                        obs.vs = opponent_speed
                        obs.vd = 0
                        obs.is_actually_a_gap = False
                        obs.is_static = False
                        obstacle_list.append(obs)

                        pds = Prediction()
                        pds.id = i
                        # HJ MODIFIED: Use SMART Frenet if active, otherwise GB Frenet
                        if use_smart_frenet:
                            # Convert GB prediction to SMART Frenet using smart_converter
                            s_gb = (obs.s_start + obs.s_end) / 2
                            d_gb = opponent_d
                            pos_cart = self.frenet2glob([s_gb % self.max_s_updated], [d_gb])
                            result = self.smart_converter.get_frenet(np.array([pos_cart.x[0]]), np.array([pos_cart.y[0]]))
                            pds.pred_s = float(result[0])
                            pds.pred_d = float(result[1])
                        else:
                            # GB mode
                            pds.pred_s = (obs.s_start + obs.s_end) / 2
                            pds.pred_d = opponent_d
                        prediction_list.append(pds)

                        marker = Marker()
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.header.frame_id = "map"
                        marker.id = i
                        marker.type = Marker.CYLINDER
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0

                        pos = self.frenet2glob([obs.s_center % self.max_s_updated], [obs.d_center])
                        marker.pose.position.x = pos.x[0]
                        marker.pose.position.y = pos.y[0]
                        marker.pose.position.z = 0.1

                        marker.scale.x = 0.15
                        marker.scale.y = 0.15
                        marker.scale.z = 0.15  # height
                        marker.color.a = 0.8
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0

                        opp_marker_array.markers.append(marker)
                        
                    # Find the end of the prediction
                    if (beginn == True and end == False):
                        end_s = current_opponent_s
                        end_d = opponent_d
                        end = True
                        
                        prediction_obs_arr = ObstacleArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), obstacles = obstacle_list)
                        self.prediction_obs_pub.publish(prediction_obs_arr)

                        prediction_obs_pred_arr = PredictionArray(header = rospy.Header(stamp = self.get_clock().now().to_msg(), frame_id = "map"), id = opponent_pos_copy.obstacles[0].id, predictions = prediction_list)
                        self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

                        self.opp_marker_pub.publish(opp_marker_array)
                        
                        self.expire_counter = 0
                        
                        # Visualize the prediction (Watchout for wrap around)
                        position_beginn = self.frenet2glob([beginn_s%self.max_s_updated], [beginn_d])
                        self.marker_beginn.pose.position.x = position_beginn.x[0]
                        self.marker_beginn.pose.position.y = position_beginn.y[0]
                        self.marker_pub_beginn.publish(self.marker_beginn)

                        position_end = self.frenet2glob([end_s%self.max_s_updated], [end_d])
                        self.marker_end.pose.position.x = position_end.x[0]
                        self.marker_end.pose.position.y = position_end.y[0]
                        self.marker_pub_end.publish(self.marker_end)

            self.prediction_obs_pred_pub.publish(prediction_obs_pred_arr)

            self.expire_counter += 1
            if self.expire_counter >= self.max_expire_counter:
                self.expire_counter = self.max_expire_counter
                self.delete_all()

            # print("Time: {}".format(time.process_time() - start))

        rate.sleep()

if __name__ == '__main__':
    node = OppTrajPredictor()
    node.loop()
