#!/usr/bin/env python3
from __future__ import annotations

import math
import time
import numpy as np

from std_msgs.msg import Float32
from f110_msgs.msg import WpntArray, OTWpntArray
from sensor_msgs.msg import LaserScan
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter as EKF
from frenet_converter.frenet_converter import FrenetConverter
from nav_msgs.msg import Odometry
from scipy.linalg import block_diag
from visualization_msgs.msg import Marker,MarkerArray

from f110_msgs.msg import ObstacleArray,Obstacle

def normalize_s(s,track_length):
        s = s % (track_length)    
        if s > track_length/2:
            s -= track_length
        return s


class Opponent_state:
    """
    This class implements the opponent with a kalman filter
    """
    track_length = None
    waypoints = None
    rate = None #hz
    dt = None
    ttl = None
    P_vs = None
    P_d = None
    P_vd = None
    measurment_var_s = None
    measurment_var_d = None
    measurment_var_vs = None
    measurment_var_vd = None
    process_var_vs = None
    process_var_vd = None
    ratio_to_glob_path = None
    def __init__(self):
        """
        Initializes the kalman filter
        """
        self.id = None
        self.size = None
        self.isInitialised = False
        self.vs_list = []
        self.avg_vs = 0
        self.useTargetVel = False

        #----------------------------------------------------------------
        # --- initialization of the kalman filter ---
        # the state of the kalman filter is :
        # X = [s, v_s, d, v_d]
        #
        # the measurment of the kalman filter is :
        # Z = [s, d]
        #
        # the model of the kalman filter is constant velocity in the 
        # frenet frame added to that:
        # - A target velocity for the s direction to be proportional to 
        # the precaculated velocity of the path 
        # - A target velocity in the d direction that bring it back to 0
        # - A target position in the d direction that bring it back to 0
        #---------------------------------------------------------------

        self.dynamic_kf = EKF(dim_x=4, dim_z=2)
        self.dynamic_kf.F = np.array([[1., Opponent_state.dt, 0., 0.],
                                      [0., 1, 0, 0.],
                                      [0., 0., 1, Opponent_state.dt],
                                      [0., 0., 0., 1]])
        q1 =Q_discrete_white_noise(dim= 2, dt= 1./Opponent_state.rate ,var=Opponent_state.process_var_vs)
        q2 =Q_discrete_white_noise(dim= 2, dt=1./Opponent_state.rate, var=Opponent_state.process_var_vd)        
        self.dynamic_kf.Q = block_diag(q1, q2)
        self.dynamic_kf.H = np.identity(4)
        self.dynamic_kf.R = np.diag([Opponent_state.measurment_var_s, Opponent_state.measurment_var_vs, 
                                     Opponent_state.measurment_var_d, Opponent_state.measurment_var_vd])
        self.dynamic_kf.P = np.array([[Opponent_state.measurment_var_s, 0., 0., 0.],
                                      [0., Opponent_state.process_var_vs, 0., 0.],
                                      [0., 0., Opponent_state.measurment_var_d, 0.],
                                      [0., 0., 0., Opponent_state.process_var_vd]])
        self.dynamic_kf.B = np.identity(4)

        filter_length = 5
        self.vs_filt = np.zeros(filter_length)
        self.vd_filt = np.zeros(filter_length)

    # --- defining some utility functions ---
    def residual_h(a,b):
        y = a-b
        y[0] = normalize_s(y[0],Opponent_state.track_length)
        return y
    
    def Hjac(self):
        return np.identity(4)

    def hx(x):
        return np.array([normalize_s(x[0],
                         Opponent_state.track_length),x[1], x[2], x[3]])

    def target_velocity(self) :
        # Wrap s-coordinate first, then convert to waypoint index
        s_wrapped = self.dynamic_kf.x[0] % Opponent_state.track_length
        idx_closest_waypoint = int(s_wrapped * 10)  # Convert meters to waypoint index (0.1m spacing)
        idx_closest_waypoint = min(idx_closest_waypoint, len(Opponent_state.waypoints) - 1)  # Clamp to array bounds
        return Opponent_state.ratio_to_glob_path*Opponent_state.waypoints[idx_closest_waypoint].vx_mps

    # ---------------------------------------
    #     defining the predict and update 
    #     functions for the kalman filter 
    # ---------------------------------------

    def predict (self):
        if self.useTargetVel:
            self.dynamic_kf.predict(u=[0,Opponent_state.P_vs*(self.target_velocity()-self.dynamic_kf.x[1]),
                                    -Opponent_state.P_d*self.dynamic_kf.x[2],-Opponent_state.P_vd*self.dynamic_kf.x[3]])
        else:
            self.dynamic_kf.predict(u=[0, 0,
                                   -Opponent_state.P_d*self.dynamic_kf.x[2],-Opponent_state.P_vd*self.dynamic_kf.x[3]])
        self.dynamic_kf.x[0] = normalize_s(self.dynamic_kf.x[0],Opponent_state.track_length)

    def update(self, tracked_obstacle: ObstacleSD):

        # Calculate velocity with wrapping consideration
        ds_1 = normalize_s(tracked_obstacle.measurments_s[-1] - tracked_obstacle.measurments_s[-2], Opponent_state.track_length)
        ds_2 = normalize_s(tracked_obstacle.measurments_s[-2] - tracked_obstacle.measurments_s[-3], Opponent_state.track_length)
        vs = ((2/3 * ds_1 * self.rate) + (1/3 * ds_2 * self.rate))

        if not (vs > -1 and vs < 8):
            self.isInitialised = False
            return

        z = np.array([
            tracked_obstacle.measurments_s[-1],
            vs,
            tracked_obstacle.measurments_d[-1],
            (tracked_obstacle.measurments_d[-1] - tracked_obstacle.measurments_d[-2])*self.rate,
        ])

        self.dynamic_kf.update(
            np.array([
                normalize_s(z[0],Opponent_state.track_length), z[1], z[2], z[3]
            ]),
            Opponent_state.Hjac,
            Opponent_state.hx, 
            residual=Opponent_state.residual_h
        )
        self.dynamic_kf.x[0] = normalize_s(self.dynamic_kf.x[0],Opponent_state.track_length)

        self.vs_list.append(self.dynamic_kf.x[1])
        if(len(self.vs_list) > 20):
            self.vs_list = self.vs_list[-10:]

        self.avg_vs = 0
        for v in self.vs_list:
            self.avg_vs += v
        self.avg_vs /= len(self.vs_list)

        self.vs_filt[0] = self.dynamic_kf.x[1]
        self.vs_filt[1:] = self.vs_filt[:-1]  
        self.vd_filt[0] = self.dynamic_kf.x[3]
        self.vd_filt[1:] = self.vd_filt[:-1]

        if(len(self.vs_list) >=10):
            self.vs_list.pop(0)
        self.vs_list.append(self.dynamic_kf.x[1])

# ---------------------------------------------------------------- 
# create a class to define what are the properties of an obstacle
# ----------------------------------------------------------------
class ObstacleSD:
    """
    This Class implements the propoerties of the static/dynamic obstacles
    """
    min_nb_meas = None
    ttl = None
    min_std = None
    max_std = None

    def __init__(self, id, s_meas, d_meas, lap, size, isVisible, sector_id=-1, in_static_obs_sector=False):
        """
        Initialize the static/dynamic obstacle
        """
        # --- variable ---
        self.id = id
        self.measurments_s = [s_meas]
        self.measurments_d = [d_meas]
        self.mean = [s_meas,d_meas] # [mean_s. mean_d]
        self.static_count = 0
        self.total_count = 0
        self.nb_meas = 0
        self.ttl = ObstacleSD.ttl
        self.isInFront = True
        self.current_lap = lap
        self.staticFlag = None
        self.size = size
        self.nb_detection = 0
        self.isVisible = isVisible
        # ===== HJ ADDED: Store sector information =====
        self.sector_id = sector_id
        self.in_static_obs_sector = in_static_obs_sector
        # ===== HJ ADDED END =====

        self.dynamic_state = Opponent_state()

    def update_mean(self,track_length):
        if (self.nb_meas == 0):
            self.mean = [self.measurments_s[-1],self.measurments_d[-1]]
        else:
            # ------------------------------------------------------------------------------------
            # since we know the number of measurments and the previous mean, to not loop 
            # through all the data the current mean is just a weighted sum between the 
            # previous mean weighted by the number of measurments and the new measurment 
            # ------------------------------------------------------------------------------------
            
            self.mean[1] = (self.mean[1]*self.nb_meas+self.measurments_d[-1])/(self.nb_meas+1)

            # ------------------------------------------------------------------------------------
            # to account for the wrapping in the process we can transform the s measurments into
            # angles ranging from 0 to 2 pi and do the weighted sum over unitary vectors with those 
            # angles and then convert the resulting angle back to an s measurment 
            # ------------------------------------------------------------------------------------

            previous_mean_rad = self.mean[0]*2*math.pi/track_length
            current_meas_rad = self.measurments_s[-1]*2*math.pi/track_length
            cos_mean_angle = (math.cos(previous_mean_rad)*self.nb_meas+math.cos(current_meas_rad))/(self.nb_meas+1)
            sin_mean_angle = (math.sin(previous_mean_rad)*self.nb_meas+math.sin(current_meas_rad))/(self.nb_meas+1)
            mean_angle = math.atan2(sin_mean_angle,cos_mean_angle)
            mean_s = mean_angle*track_length/2/math.pi
            self.mean[0] = mean_s if mean_s>=0 else mean_s+track_length

    def std_s (self,track_length):
        sum = 0
        mean_s = self.mean[0]
        for s in self.measurments_s:
            sum += normalize_s(s-mean_s,track_length)**2
        return (sum/len(self.measurments_s))**0.5

    def std_d (self):
        return np.std(self.measurments_d)

    def isStatic(self,track_length):
        # --- get a representative data set for the obstacle ---
        if self.nb_meas > ObstacleSD.min_nb_meas:
            std_s = self.std_s(track_length)
            std_d = self.std_d()
            # --- create a voting system so that the outliers don't affect much the result ---
            if (std_s < ObstacleSD.min_std and std_d < ObstacleSD.min_std):
                self.static_count = self.static_count +1
            # --- assert for sure that an obstacle is dynamic and not static ---
            elif (std_s > ObstacleSD.max_std or std_d > ObstacleSD.max_std):
                self.static_count = 0
            self.total_count = self.total_count +1
            self.staticFlag = self.static_count/self.total_count >= 0.5

        else:
            self.staticFlag = None

class StaticDynamic:
    """
    This class implements a ROS node that classifies an publishes obstacles.

    It subscribes to the following topics:
        - `/raw_obstacles`: Publishes the raw obstacle data.
        - `/global_waypoints_scaled`: Publishes the global waypoints.
        - `/odom_frenet`: Publishes the car state in frenet frame.
        - `/odom`: Publishes the car state in glob frame.
        - `/scan`: Publishes the lidar scans.

    It publishes the following topics:
        - `/static_dynamic_marker_pub`: Publishes the obstacle markers
        - `/obstacles`: Publishes the obstacles
        - `/raw_obstacles`: Publishes the obstacles without Kalman Filtering
    """
    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # --- Node properties ---
        rospy.init_node('StaticDynamic', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        # --- Variables ---
        self.meas_obstacles = []
        self.tracked_obstacles = [] 
        self.waypoints = None
        self.car_s = None 
        self.car_position = None
        self.car_orientation = None
        self.last_car_s = 0
        self.current_lap = 0
        self.globalpath = None
        self.track_length = None
        self.opponent_obstacles = []
        self.current_stamp = None
        self.scans = None
        self.current_id = 1
        self.from_bag = self._get_param_or_default("/from_bag", False)
        self.measuring = self._get_param_or_default("/measure", False)

        # ===== HJ ADDED: Fixed path Frenet converter =====
        self.fixed_converter = None
        self.fixed_converter_initialized = False
        self.fixed_track_length = None
        # ===== HJ ADDED END =====

        # --- Subscribers ---
        self.create_subscription(ObstacleArray, '/detect/raw_obstacles', self.obstacleCallback, 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self.pathCallback, 10)
        self.create_subscription(OTWpntArray, '/planner/avoidance/smart_static_otwpnts', self.fixedPathCallback, 10)  # ===== HJ ADDED =====
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.carStateCallback, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.carStateGlobCallback, 10)
        self.create_subscription(LaserScan, '/scan', self.scansCallback, 10)
        if not self.from_bag:
            self.create_subscription(Config, "/dyn_perception/parameter_updates", self.dyn_param_cb, 10)

        # --- Publisher ---
        self.static_dynamic_marker_pub = self.create_publisher(MarkerArray, '/tracking/static_dynamic_marker_pub', 5)
        self.estimated_obstacles_pub = self.create_publisher(ObstacleArray, '/tracking/obstacles', 5)
        self.raw_opponent_pub = self.create_publisher(ObstacleArray, '/tracking/raw_obstacles', 5)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, '/tracking/latency', 10)

        # --- Tunable Parameters ---
        self.rate = self._get_param_or_default("/perception/rate_tracking")

        Opponent_state.rate = self.rate
        Opponent_state.dt = 1/self.rate
        Opponent_state.P_vs = self._get_param_or_default("/perception/P_vs")
        Opponent_state.P_d  = self._get_param_or_default("/perception/P_d")
        Opponent_state.P_vd = self._get_param_or_default("/perception/P_vd")
        Opponent_state.measurment_var_s = self._get_param_or_default("/perception/measurment_var_s")
        Opponent_state.measurment_var_d = self._get_param_or_default("/perception/measurment_var_d")
        Opponent_state.measurment_var_vs = self._get_param_or_default("/perception/measurment_var_vs")
        Opponent_state.measurment_var_vd = self._get_param_or_default("/perception/measurment_var_vd")
        Opponent_state.process_var_vs = self._get_param_or_default("/perception/process_var_vs")
        Opponent_state.process_var_vd = self._get_param_or_default("/perception/process_var_vd")

        self.max_dist       = self._get_param_or_default("/perception/max_dist")
        self.var_pub        = self._get_param_or_default("/perception/var_pub")
        self.aggro_multiplier = self._get_param_or_default("/perception/aggro_multi")
        self.dist_deletion = self._get_param_or_default("/perception/dist_deletion")
        self.dist_infront = self._get_param_or_default("/perception/dist_infront")
        self.max_std = self._get_param_or_default("/perception/max_std")
        self.min_std = self._get_param_or_default("/perception/min_std")
        self.min_nb_meas = self._get_param_or_default("/perception/min_nb_meas")
        self.noMemoryMode = self._get_param_or_default("/perception/noMemoryMode")
        self.debug_mode = self._get_param_or_default("/perception/debug_mode")
        self.publish_static = self._get_param_or_default("/perception/publish_static")
        self.ratio_to_glob_path = self._get_param_or_default("/perception/ratio_to_glob_path")
        self.publish_static = self._get_param_or_default("/perception/publish_static")
        self.ttl_dynamic = self._get_param_or_default("/perception/ttl_dynamic")
        self.ttl_static = self._get_param_or_default("/perception/ttl_static")
        self.ttl_static_sector = self._get_param_or_default("/perception/ttl_static_sector", 3)
        self.vs_reset = self._get_param_or_default("/perception/vs_reset")

        
        # dyn params sub
        Opponent_state.ttl = self.ttl_dynamic
        Opponent_state.ratio_to_glob_path = self.ratio_to_glob_path
        ObstacleSD.ttl = self.ttl_static
        ObstacleSD.min_nb_meas = self.min_nb_meas
        ObstacleSD.min_std = self.min_std
        ObstacleSD.max_std = self.max_std
        self.vs_reset = self.vs_reset


        while(self.waypoints is None):
            rospy.sleep(0.1)
        self.converter = self.initialize_converter()

    def dyn_param_cb(self, params: Config):
        self.ttl_dynamic = self._get_param_or_default("dyn_perception/ttl_dynamic")
        self.ratio_to_glob_path = self._get_param_or_default("dyn_perception/ratio_to_glob_path")
        self.ttl_static = self._get_param_or_default("dyn_perception/ttl_static")
        self.ttl_static_sector = self._get_param_or_default("dyn_perception/ttl_static_sector", 3)
        self.min_nb_meas = self._get_param_or_default("dyn_perception/min_nb_meas")
        self.dist_deletion = self._get_param_or_default("dyn_perception/dist_deletion")
        self.dist_infront = self._get_param_or_default("dyn_perception/dist_infront")
        self.min_std = self._get_param_or_default("dyn_perception/min_std")
        self.max_std = self._get_param_or_default("dyn_perception/max_std")
        self.vs_reset = self._get_param_or_default("dyn_perception/vs_reset")
        self.aggro_multiplier = self._get_param_or_default("dyn_perception/aggro_multi")
        self.debug_mode = self._get_param_or_default("dyn_perception/debug_mode")
        self.publish_static = self._get_param_or_default("dyn_perception/publish_static")
        self.noMemoryMode = self._get_param_or_default("dyn_perception/noMemoryMode")
        
        Opponent_state.ttl = self.ttl_dynamic
        Opponent_state.ratio_to_glob_path = self.ratio_to_glob_path

        ObstacleSD.ttl = self.ttl_static
        ObstacleSD.min_nb_meas = self.min_nb_meas
        ObstacleSD.min_std = self.min_std
        ObstacleSD.max_std = self.max_std
        
        obstacle_params = [ObstacleSD.ttl, ObstacleSD.min_nb_meas, ObstacleSD.min_std, ObstacleSD.max_std]
        print(f'[Tracking] Dynamic reconf triggered new tracking params: Tracking TTL: {Opponent_state.ttl}, Ratio to glob path: {Opponent_state.ratio_to_glob_path}\n'
              f'ObstacleSD ttl, min_nb_meas, min_std, max_std: {obstacle_params},\n',
              f'dist_deletion: {self.dist_deletion} [m], dist_infront: {self.dist_infront} [m], vs_reset: {self.vs_reset},\n',
              f'Publish static obstacles: {self.publish_static}, no memory mode: {self.noMemoryMode}'
              )
        

    def shutdown(self):
        self.get_logger().warning('Tracking is shutdown')

    # --- Callbacks ---

    def obstacleCallback(self,data):
        self.meas_obstacles = data.obstacles
        self.current_stamp = data.header.stamp

    def pathCallback(self, data):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
        if self.track_length is None:
            self.get_logger().info('[Tracking] received global path')
            self.globalpath = data.wpnts
            self.track_length = data.wpnts[-1].s_m
            Opponent_state.track_length = self.track_length
            Opponent_state.waypoints = self.globalpath

    # ===== HJ ADDED: Fixed path callback (initialize converter only once) =====
    def fixedPathCallback(self, data):
        if not data.wpnts:
            # Fixed path not available - normal before Smart path is generated
            return

        # Initialize converter only once when Fixed path first becomes available
        if not self.fixed_converter_initialized:
            fixed_waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
            self.fixed_converter = FrenetConverter(fixed_waypoints[:, 0], fixed_waypoints[:, 1], fixed_waypoints[:, 2])
            self.fixed_track_length = data.wpnts[-1].s_m
            self.fixed_converter_initialized = True
            self.get_logger().info('[Tracking] Fixed path Frenet converter initialized: %d waypoints, track_length: %.2f', len(data.wpnts), self.fixed_track_length)

    def convert_frenet_velocity(self, s_gb, d_gb, vs_gb, vd_gb, x_m, y_m):
        """
        Convert velocity from GB Frenet to Fixed Frenet coordinates.

        Args:
            s_gb, d_gb: GB Frenet position
            vs_gb, vd_gb: GB Frenet velocity
            x_m, y_m: XY position

        Returns:
            vs_fixed, vd_fixed: Fixed Frenet velocity
        """
        if not self.fixed_converter_initialized or self.fixed_converter is None:
            return 0.0, 0.0

        # Get tangent angles for both reference paths
        psi_gb = np.arctan2(*self.converter.get_derivative(s_gb)[::-1])  # dy/ds, dx/ds
        s_fixed_approx = self.fixed_converter.get_approx_s(np.array([x_m]), np.array([y_m]))[0]
        psi_fixed = np.arctan2(*self.fixed_converter.get_derivative(s_fixed_approx)[::-1])

        # Convert GB Frenet velocity to XY velocity
        vx = vs_gb * np.cos(psi_gb) - vd_gb * np.sin(psi_gb)
        vy = vs_gb * np.sin(psi_gb) + vd_gb * np.cos(psi_gb)

        # Convert XY velocity to Fixed Frenet velocity
        vs_fixed = vx * np.cos(psi_fixed) + vy * np.sin(psi_fixed)
        vd_fixed = -vx * np.sin(psi_fixed) + vy * np.cos(psi_fixed)

        return float(vs_fixed), float(vd_fixed)
    # ===== HJ ADDED END =====

    def initialize_converter(self) -> bool:
        """
        Initialize the FrenetConverter object"""
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.get_logger().info("[Tracking] initialized FrenetConverter object")

        return converter

    def carStateCallback (self, data):
        self.car_s = data.pose.pose.position.x
        if self.last_car_s is None:
            self.last_car_s = data.pose.pose.position.x

    def carStateGlobCallback(self,data):
        self.car_position = np.array([data.pose.pose.position.x,data.pose.pose.position.y])
        angles = euler_from_quaternion([data.pose.pose.orientation.x,
                                        data.pose.pose.orientation.y,
                                        data.pose.pose.orientation.z,
                                        data.pose.pose.orientation.w])
        theta = angles[2]
        self.car_orientation = np.array([np.cos(theta),np.sin(theta)])

    def scansCallback(self, data):
        self.scans = data.ranges


    # --- Utility Functions ---
    def lap_update (self, car_s):
        if car_s - self.last_car_s <-self.track_length/2:
            self.current_lap +=1
        self.last_car_s = car_s

    def get_closest_pos(self, max_dist, obstacle_position, meas_obstacles_copy):
        potential_obs = []
        dists = []
        for meas_obstacle in meas_obstacles_copy:
            meas_obstacle_position=[meas_obstacle.s_center, 
                                    meas_obstacle.d_center]
            dist = math.dist(obstacle_position,meas_obstacle_position)
            if (dist<max_dist):
                potential_obs.append(meas_obstacle)
                dists.append(dist)
        return potential_obs, dists
    
    def verify_position(self,obstacle,meas_obstacles_copy):
        """
        Verifies if an obstacle with a certain position is tracked or not. Chooses among all possible obstacles the nearest one
        """
        # for dynamic obstacles we use the predicted position to get better accuracy
        max_dist = self.max_dist
        if(obstacle.staticFlag == False):
            obstacle_position = [obstacle.dynamic_state.dynamic_kf.x[0]%self.track_length, obstacle.dynamic_state.dynamic_kf.x[2]]
            max_dist *= self.aggro_multiplier
        else:
            obstacle_position = [obstacle.mean[0], obstacle.mean[1]]
        potential_obs, dists =  self.get_closest_pos(max_dist, obstacle_position, meas_obstacles_copy)
        if (len(dists)>0):
            min_idx = np.argmin(dists)
            return True, potential_obs[min_idx]
        
        # maybe kalman was wrong, the obstacles can't just be gone
        elif(obstacle.staticFlag == False):
            obstacle_position = [obstacle.mean[0], obstacle.mean[1]] 
            potential_obs, dists =  self.get_closest_pos(max_dist, obstacle_position, meas_obstacles_copy)
            if (len(dists)>0):
                min_idx = np.argmin(dists)
                return True, potential_obs[min_idx]
            
        return False, None
    
    def angle_to_obs(self, vec_to_obstacle: np.array, car_orientation: np.array) -> float:
        norm_vec_to_obs = vec_to_obstacle/np.linalg.norm(vec_to_obstacle)
        norm_car_orientation = car_orientation/np.linalg.norm(car_orientation)

        theta = np.arctan2(norm_car_orientation[1], norm_car_orientation[0])
        if theta < 0:
            theta += 2*np.pi
        
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        vec_to_obs_rot = np.dot(rot, norm_vec_to_obs)

        phi = np.degrees(np.arctan2(vec_to_obs_rot[1], vec_to_obs_rot[0]))
        
        angle = 135 + phi # because lidar has a range of 270 deg

        return angle

    def update_tracked_obstacle(self, tracked_obstacle: ObstacleSD, meas_obstacle):
        tracked_obstacle.measurments_s.append(meas_obstacle.s_center)
        tracked_obstacle.measurments_d.append(meas_obstacle.d_center)
        # handle list lenght
        if(len(tracked_obstacle.measurments_s) > 30):
            tracked_obstacle.measurments_s = tracked_obstacle.measurments_s[-20:]
            tracked_obstacle.measurments_d = tracked_obstacle.measurments_d[-20:]
        tracked_obstacle.update_mean(self.track_length)
        tracked_obstacle.nb_meas += 1
        tracked_obstacle.isInFront = True
        tracked_obstacle.isVisible = True
        tracked_obstacle.current_lap = self.current_lap
        tracked_obstacle.size = meas_obstacle.size
        tracked_obstacle.isStatic(self.track_length)

        # ===== HJ MODIFIED: Use longer TTL for obstacles in static sector =====
        # Update sector information from latest measurement
        tracked_obstacle.sector_id = meas_obstacle.sector_id
        tracked_obstacle.in_static_obs_sector = meas_obstacle.in_static_obs_sector

        # Use different TTL based on whether obstacle is in smart static sector
        if tracked_obstacle.in_static_obs_sector:
            tracked_obstacle.ttl = self.ttl_static_sector  # Long TTL (e.g., 600 = 15 sec @ 40Hz)
        else:
            tracked_obstacle.ttl = ObstacleSD.ttl  # Normal TTL (e.g., 3)
        # ===== HJ MODIFIED END =====

        return tracked_obstacle
    
    def initialize_dynamic_obstacle(self, tracked_obstacle):
        tracked_obstacle.dynamic_state.dynamic_kf.x = np.array([
            tracked_obstacle.measurments_s[-1],
            (tracked_obstacle.measurments_s[-1]-tracked_obstacle.measurments_s[-2])*Opponent_state.rate,
            tracked_obstacle.measurments_d[-1],
            (tracked_obstacle.measurments_d[-1]-tracked_obstacle.measurments_d[-2])*Opponent_state.rate
        ])
        tracked_obstacle.dynamic_state.isInitialised = True
        tracked_obstacle.dynamic_state.id = tracked_obstacle.id
        tracked_obstacle.dynamic_state.ttl = Opponent_state.ttl
        tracked_obstacle.dynamic_state.size = tracked_obstacle.size
        tracked_obstacle.dynamic_state.avg_vs = 0
        tracked_obstacle.dynamic_state.vs_list = []

    def check_in_front(self, tracked_obstacle, car_s_copy) -> bool:
        obj_dist_in_front = normalize_s(
            tracked_obstacle.measurments_s[-1] - car_s_copy, 
            self.track_length)
        
        return 0 < obj_dist_in_front < self.dist_infront
    
    def calc_distance_obs_car(self, tracked_obstacle, car_s):
        distance_obs_car = (tracked_obstacle.measurments_s[-1] - car_s) % self.track_length

        if self.debug_mode:
            print("DIST CAR TO OBS")
            print("car s: ", car_s)
            print("obsatcle s: ", tracked_obstacle.measurments_s[-1])
            print("track length: ", self.track_length)
            print("current lap: ", self.current_lap)
            print("obstacle lap: ", tracked_obstacle.current_lap)
            print("distance: ", distance_obs_car)
            print("----------------------------------")

        return distance_obs_car

    def check_in_field_of_view(self, vec_car_to_obs, car_orientation_copy, dist_to_obs) -> bool:
        """
        Checks if an obstacle is in the field of view by checking the corresponding lidar beams
        """
        dist_to_obs = np.linalg.norm(vec_car_to_obs)
        
        angle = self.angle_to_obs(vec_car_to_obs, car_orientation_copy)

        max_angle = len(self.scans) - 1
        fov = int(max_angle/4)
        if(angle > 135 + fov/2 or angle < 135 - fov/2): 
            return False
        
        removed_deg = 135 - int(fov/2)
        
        idx = int(round(4* (angle - removed_deg))) # because scans has 4*270 entries
        if idx < 0 or idx > max_angle: return False

        if self.debug_mode:
            print("FIELD OF VIEW:")
            print("index: ", idx)
            print("angle: ", angle)
            print("Other lidar scans: ", str([f"{sc:.2f}" for sc in self.scans[idx-10:idx+10]]))
            print("dist: ", dist_to_obs)
            print("----------------------------------")

        if(dist_to_obs < min(self.scans[max(0, idx -4) : min(idx + 4, max_angle)])): 
            return True

        return False

    # --- update tracked obstacles, add new obstacles and remove unecessary ---
    def update(self):
        meas_obstacles_copy = self.meas_obstacles.copy()
        car_s_copy = self.car_s
        car_position_copy = np.copy(self.car_position)
        car_orientation_copy = np.copy(self.car_orientation)
        self.lap_update(car_s_copy)
        removal_list = []
        num_dyn_obs = 0
        for tracked_obstacle in self.tracked_obstacles:
            # --- verify if the obstacle is tracked by position and update the associated obstacle ---
            isTracked, meas_obstacle = self.verify_position(tracked_obstacle, meas_obstacles_copy)

            if isTracked:
                tracked_obstacle = self.update_tracked_obstacle(tracked_obstacle, meas_obstacle)
                # obstacle is classified as moving
                if(tracked_obstacle.staticFlag == False):
                    if tracked_obstacle.dynamic_state.isInitialised:
                        tracked_obstacle.dynamic_state.useTargetVel = False
                        if(tracked_obstacle.dynamic_state.avg_vs < self.vs_reset and len(tracked_obstacle.dynamic_state.vs_list) > 10 and self.publish_static):
                            tracked_obstacle.dynamic_state.isInitialised = False
                            tracked_obstacle.staticFlag = True
                            tracked_obstacle.static_count = 0
                            tracked_obstacle.total_count = 0
                            tracked_obstacle.nb_meas = 0
                        else:
                            tracked_obstacle.dynamic_state.update(tracked_obstacle)
                            # tracked_obstacle.dynamic_state.id = tracked_obstacle.id
                            tracked_obstacle.dynamic_state.ttl = Opponent_state.ttl
                            # tracked_obstacle.dynamic_state.size = tracked_obstacle.size
                    else:
                        self.initialize_dynamic_obstacle(tracked_obstacle)
                        
                    self.tracked_obstacles.remove(tracked_obstacle)
                    self.tracked_obstacles.insert(num_dyn_obs, tracked_obstacle)
                    num_dyn_obs += 1 
                        
                meas_obstacles_copy.remove(meas_obstacle)

            else:
                # --- remove obstacle with dead ttl ---
                if tracked_obstacle.ttl <= 0:
                    if(tracked_obstacle.staticFlag == False):
                        tracked_obstacle.dynamic_state.useTargetVel = True
                    removal_list.append(tracked_obstacle)
                elif tracked_obstacle.staticFlag is None:
                    tracked_obstacle.ttl-=1
                else:
                    tracked_obstacle.isInFront = self.check_in_front(tracked_obstacle, car_s_copy)
                    distance_obstacle_car = self.calc_distance_obs_car(tracked_obstacle, car_s_copy)

                    # ===== HJ MODIFIED: Smart static sector obstacles decrease TTL rapidly when not detected =====
                    if(tracked_obstacle.staticFlag and tracked_obstacle.in_static_obs_sector):
                        # Smart static sector: decrease TTL very rapidly when not detected
                        tracked_obstacle.ttl -= 1  # 600 ttl
                        tracked_obstacle.isVisible = False
                    
                    elif(tracked_obstacle.staticFlag and self.noMemoryMode):
                        tracked_obstacle.ttl -= 1
                    # ===== HJ MODIFIED END =====
                    # --- if obstacle is near enough check if we can see it ---
                    elif(distance_obstacle_car < self.dist_deletion and tracked_obstacle.staticFlag):
                        try:
                            resp = self.converter.get_cartesian(tracked_obstacle.mean[0],tracked_obstacle.mean[1])
                        except:
                            continue
                        vec_car_to_obs = resp - car_position_copy
                        if (self.check_in_field_of_view(vec_car_to_obs, car_orientation_copy, distance_obstacle_car)):
                            tracked_obstacle.ttl-=1
                            tracked_obstacle.isVisible = True
                        else:
                            tracked_obstacle.isVisible = False
                    # update ttl of moving obstacles
                    elif(not tracked_obstacle.staticFlag):
                        tracked_obstacle.ttl -= 1
                    else:
                        tracked_obstacle.isVisible = False

            # update ttl of dynamic obstacle
            if(tracked_obstacle.dynamic_state.isInitialised):
                if(tracked_obstacle.dynamic_state.ttl <= 0):
                    tracked_obstacle.dynamic_state.isInitialised = False
                    tracked_obstacle.dynamic_state.useTargetVel = False
                else:
                    tracked_obstacle.dynamic_state.ttl -= 1

        for el in removal_list:
            self.tracked_obstacles.remove(el)

        for meas_obstacle in meas_obstacles_copy:
            # update the init function and append a new obstacle to the new_obstacles
            # ===== HJ EDITED: Pass sector information from detect =====
            self.tracked_obstacles.append(ObstacleSD(
                id = self.current_id,
                s_meas = meas_obstacle.s_center,
                d_meas = meas_obstacle.d_center,
                lap = self.current_lap,
                size = meas_obstacle.size,
                isVisible = True,
                sector_id = meas_obstacle.sector_id,
                in_static_obs_sector = meas_obstacle.in_static_obs_sector
            ))
            self.current_id += 1

    def clearmarkers(self):
        marker = Marker()
        marker.action = 3
        return [marker]

    def publish_Marker(self):
        markers_array = []
        for tracked_obstacle in self.tracked_obstacles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.current_stamp
            marker.id = tracked_obstacle.id
            marker.type = marker.SPHERE

            if tracked_obstacle.isInFront :
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
            else :
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.25

            marker.color.a = 0.5
            if tracked_obstacle.staticFlag is None:
                marker.type = marker.SPHERE
                marker.color.g = 0.
                marker.color.r = 1.
                marker.color.b = 0.
                x, y = self.converter.get_cartesian(tracked_obstacle.measurments_s[-1],tracked_obstacle.measurments_d[-1])
                
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.orientation.w = 1

            elif tracked_obstacle.staticFlag :
                marker.color.g = 1.
                marker.color.r = 0.
                marker.color.b = 0.
                x, y = self.converter.get_cartesian(tracked_obstacle.mean[0],tracked_obstacle.mean[1])
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.orientation.w = 1
            else:
                if tracked_obstacle.dynamic_state.isInitialised:
                    marker.color.g = 0.
                    marker.color.r = 0.
                    marker.color.b = 1.
                    if tracked_obstacle.dynamic_state.dynamic_kf.P[0][0]<self.var_pub:
                        marker.scale.x = 0.5
                        marker.scale.y = 0.5
                        marker.scale.z = 0.5
                    else:
                        marker.scale.x = 0.25
                        marker.scale.y = 0.25
                        marker.scale.z = 0.25
                    x, y = self.converter.get_cartesian(tracked_obstacle.dynamic_state.dynamic_kf.x[0]%self.track_length, tracked_obstacle.dynamic_state.dynamic_kf.x[2])
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.orientation.w = 1
                    markers_array.append(marker)
                
            if tracked_obstacle.staticFlag is None and self.publish_static:
                markers_array.append(marker)
            elif tracked_obstacle.staticFlag and self.publish_static:
                markers_array.append(marker)            

        self.static_dynamic_marker_pub.publish(self.clearmarkers())
        self.static_dynamic_marker_pub.publish(markers_array)

    def publishObstacles (self):
        obstaclearray_temp = ObstacleArray()
        obstaclearray_temp.header.frame_id = 'map'
        obstaclearray_temp.header.stamp = self.current_stamp
        obstacle_array = []
        raw_opponent_array = []
        for obs in self.tracked_obstacles:
            obs_msg = Obstacle()

            obs_msg.id    = obs.id
            obs_msg.size  = obs.size
            obs_msg.vs    = 0
            obs_msg.vd    = 0
            obs_msg.is_static         = True
            obs_msg.is_actually_a_gap = False
            obs_msg.is_visible = obs.isVisible
            # ===== HJ ADDED: Copy sector information =====
            obs_msg.sector_id = obs.sector_id
            obs_msg.in_static_obs_sector = obs.in_static_obs_sector
            # ===== HJ ADDED END =====

            if obs.staticFlag is None:
                obs_msg.s_center = obs.measurments_s[-1]%self.track_length
                obs_msg.d_center = obs.measurments_d[-1]
            elif obs.staticFlag :
                obs_msg.s_center = obs.mean[0]
                obs_msg.d_center = obs.mean[1]
            else:
                if obs.dynamic_state.isInitialised:
                    if obs.dynamic_state.dynamic_kf.P[0][0]<self.var_pub:

                        obs_msg.id    = obs.dynamic_state.id
                        obs_msg.size  = obs.dynamic_state.size
                        obs_msg.vs    = np.mean(obs.dynamic_state.vs_filt)
                        obs_msg.vd    = np.mean(obs.dynamic_state.vd_filt)
                        obs_msg.is_static         = False
                        obs_msg.is_actually_a_gap = False
                        obs_msg.is_visible        = True
                        obs_msg.s_center = obs.dynamic_state.dynamic_kf.x[0]%self.track_length
                        obs_msg.d_center = obs.dynamic_state.dynamic_kf.x[2]
                        obs_msg.s_var = obs.dynamic_state.dynamic_kf.P[0][0]
                        obs_msg.vs_var = obs.dynamic_state.dynamic_kf.P[1][1]
                        obs_msg.d_var = obs.dynamic_state.dynamic_kf.P[2][2]
                        obs_msg.vd_var = obs.dynamic_state.dynamic_kf.P[3][3]                
                        obstacle_array.append(obs_msg)
                else:
                    obs_msg.s_center = obs.measurments_s[-1]%self.track_length
                    obs_msg.d_center = obs.measurments_d[-1]

            obs_msg.s_start = (obs_msg.s_center-obs_msg.size/2)%self.track_length
            obs_msg.s_end   = (obs_msg.s_center+obs_msg.size/2)%self.track_length
            obs_msg.d_right = obs_msg.d_center-obs_msg.size/2
            obs_msg.d_left  = obs_msg.d_center+obs_msg.size/2

            obs_msg.x_m, obs_msg.y_m = self.converter.get_cartesian(obs_msg.s_center,obs_msg.d_center)

            # ===== HJ ADDED: Calculate Fixed path Frenet coordinates =====
            if self.fixed_converter_initialized and self.fixed_converter is not None:
                # Get Fixed Frenet position
                s_fixed, d_fixed = self.fixed_converter.get_frenet(np.array([obs_msg.x_m]), np.array([obs_msg.y_m]))
                obs_msg.s_center_fixed = float(s_fixed[0])
                obs_msg.d_center_fixed = float(d_fixed[0])
                obs_msg.s_start_fixed = (obs_msg.s_center_fixed - obs_msg.size / 2) % self.fixed_track_length
                obs_msg.s_end_fixed = (obs_msg.s_center_fixed + obs_msg.size / 2) % self.fixed_track_length
                obs_msg.d_right_fixed = obs_msg.d_center_fixed - obs_msg.size / 2
                obs_msg.d_left_fixed = obs_msg.d_center_fixed + obs_msg.size / 2

                # Convert velocity from GB Frenet to Fixed Frenet
                vs_fixed, vd_fixed = self.convert_frenet_velocity(
                    obs_msg.s_center, obs_msg.d_center,
                    obs_msg.vs, obs_msg.vd,
                    obs_msg.x_m, obs_msg.y_m
                )
                obs_msg.vs_fixed = vs_fixed
                obs_msg.vd_fixed = vd_fixed

                # Copy variance (coordinate-system independent approximation)
                obs_msg.s_var_fixed = obs_msg.s_var if hasattr(obs_msg, 's_var') else 0.0
                obs_msg.d_var_fixed = obs_msg.d_var if hasattr(obs_msg, 'd_var') else 0.0
                obs_msg.vs_var_fixed = obs_msg.vs_var if hasattr(obs_msg, 'vs_var') else 0.0
                obs_msg.vd_var_fixed = obs_msg.vd_var if hasattr(obs_msg, 'vd_var') else 0.0
            else:
                # Fixed path not available - fill with zeros
                obs_msg.s_center_fixed = 0.0
                obs_msg.d_center_fixed = 0.0
                obs_msg.s_start_fixed = 0.0
                obs_msg.s_end_fixed = 0.0
                obs_msg.d_right_fixed = 0.0
                obs_msg.d_left_fixed = 0.0
                obs_msg.vs_fixed = 0.0
                obs_msg.vd_fixed = 0.0
                obs_msg.s_var_fixed = 0.0
                obs_msg.d_var_fixed = 0.0
                obs_msg.vs_var_fixed = 0.0
                obs_msg.vd_var_fixed = 0.0
            # ===== HJ ADDED END =====

            if obs.staticFlag is None and self.publish_static:
                raw_opponent_array.append(obs_msg)

            elif obs.staticFlag and self.publish_static:
                obstacle_array.append(obs_msg)
            else:
                raw_opponent_array.append(obs_msg)

        obstaclearray_temp.obstacles=obstacle_array
        self.estimated_obstacles_pub.publish(obstaclearray_temp)
        obstaclearray_temp.obstacles=raw_opponent_array
        self.raw_opponent_pub.publish(obstaclearray_temp)
        
    def opponents_predict(self):
        for obs in self.tracked_obstacles:
            if obs.dynamic_state.isInitialised:
                obs.dynamic_state.predict()
        
        
    def timer_callback(self, event):
        if self.measuring:
            start = time.perf_counter()
        self.opponents_predict()
        
        self.update()
        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(1/(end-start))
        self.publishObstacles()
        self.publish_Marker()
        
    def main (self):
        rate = rospy.Rate(self.rate) # hz
        self.get_logger().info('[Opponent Tracking]: Waiting for global wpnts...')
        rospy.wait_for_message('/global_waypoints_scaled', WpntArray)
        rospy.wait_for_message('/car_state/odom_frenet', Odometry)
        rospy.wait_for_message('/detect/raw_obstacles', ObstacleArray)
        self.get_logger().info('[Opponent Tracking]: Ready!')

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.timer_callback)
        rospy.spin()


if __name__ == '__main__':
    sd = StaticDynamic()
    sd.main()