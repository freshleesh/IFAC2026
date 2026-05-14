#!/usr/bin/env python3

from typing import Any
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
import os
import copy
import math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import ObstacleArray, PidData, WpntArray, BehaviorStrategy, Wpnt
from sensor_msgs.msg import LaserScan
from frenet_conversion.frenet_converter import FrenetConverter
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from controller.Controller import Controller
from controller.ftg import FTG

class Controller_manager(Node):
    """This class is the main controller manager for the car. It is responsible for selecting the correct controller $
    and publishing the corresponding commands to the actuators.
    
    It subscribes to the following topics:
    - /car_state/odom:  get ego car speed
    - /car_state/pose:  get ego car position (x, y, theta)
    - /local_waypoints: get waypoints starting at car's position in map frame
    - /vesc/sensors/imu/raw: get acceleration for steer scaling
    - /car_state/odom_frenet: get ego car frenet coordinates
    - /tracking/obstacles: get opponent information (position, speed, static/dynamic)
    - /state_machine: get state of the car
    - /scan: get lidar scan data

    It publishes the following topics:
    - /lookahead_point: publish the lookahead point for visualization
    - /trailing_opponent_marker: publish the trailing opponent marker for visualization
    - /my_waypoints: publish the waypoints for visualization
    - /l1_distance: publish the l1 distance from the Controller for visualization
    - /vesc/high_level/ackermann_cmd_mux/input/nav_1: publish the steering and speed command
    - /controller/latency: publish the latency of the controller for measuring if launched with measure:=true

    """
    def _get_param_or_default(self, name, default=None):
        """ROS1 의 rospy.get_param 호환 helper (state_machine_init 와 동일 패턴).

        파라미터 미선언 시에만 fallback (ParameterNotDeclaredException).
        다른 예외 (잘못된 타입 등) 는 의도적으로 전파 — 디버깅 가시성 우선.
        """
        from rclpy.exceptions import ParameterNotDeclaredException, ParameterAlreadyDeclaredException
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
            except ParameterNotDeclaredException:
                continue
        if default is None:
            return None
        try:
            self.declare_parameter(name, default)
        except ParameterAlreadyDeclaredException:
            pass
        return self.get_parameter(name).value

    def __init__(self):
        self.name = "control_node"
        Node.__init__(
            self, self.name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.lock = threading.Lock()
        self.loop_rate = 50 # rate in hertz
        # self.ros_time 사용 위치 확인 필요 (ROS2 에서는 메시지가 아닌 객체 필요할 수도)
        self.ros_time = self.get_clock().now()
        self.scan = None
        
        self.mapping = self._get_param_or_default('controller_manager/mapping', False)
        if self.mapping:
            self.init_mapping()
        else:
            self.init_controller()


    def init_controller(self):
        self.racecar_version = self._get_param_or_default('racecar_version') # NUCX
        self.ctrl_algo = self._get_param_or_default('controller_manager/ctrl_algo', 'PP') # default controller

        # only for MAP Controller
        self.LUT_name = self._get_param_or_default('controller_manager/LU_table') # name of lookup table
        self.get_logger().info(f"[{self.name}] Using {self.LUT_name}")
        
        self.use_sim = self._get_param_or_default('/sim')
        self.wheelbase = self._get_param_or_default('/vesc/wheelbase', 0.33) # NUCX
        self.measuring = self._get_param_or_default('/measure', False)
        
        self.state_machine_rate = self._get_param_or_default('state_machine/rate') #rate in hertz
        self.position_in_map = [] # current position in map frame
        self.position_z = 0.0  # ### HJ : z coordinate for 3D nearest waypoint search
        # ===== HJ MODIFIED: Dual Frenet position system =====
        self.position_in_map_frenet = [] # current position in frenet coordinates (GB or Fixed depending on mode)
        self.position_in_map_frenet_gb = [] # GB Frenet position
        self.position_in_map_frenet_fixed = [] # Fixed Frenet position
        # ===== HJ MODIFIED END =====
        self.waypoint_list_in_map = [] # waypoints starting at car's position in map frame
        self.speed_now = 0 # current speed
        self.acc_now = np.zeros(10) # last 5 accleration values
        self.speed_now_y =0 
        self.yaw_rate = 0 
        self.waypoint_safety_counter = 0

        # Trailing related variables
        self.opponent = [0,0,0,False, True] #s, d, vs, is_static
        self.state = ""
        self.trailing_command = 2
        self.i_gap = 0

        # ===== HJ ADDED: Dual Frenet converter system =====
        self.converter = None  # GB Frenet converter (will be converter_gb)
        self.converter_gb = None  # GB raceline converter
        self.converter_fixed = None  # Smart Static Fixed path converter
        self.smart_static_active = False  # Current Smart Static mode state
        self._prev_smart_static_active = False  # Track mode changes
        # ===== HJ ADDED END =====

        # initializing l1 parameter
        # This step could be removed with rospy.wait_for_message() in control loop
        self.t_clip_min = self._get_param_or_default('L1_controller/t_clip_min')
        self.t_clip_max = self._get_param_or_default('L1_controller/t_clip_max')
        self.m_l1 = self._get_param_or_default('L1_controller/m_l1')
        self.q_l1 = self._get_param_or_default('L1_controller/q_l1')
        self.speed_lookahead = self._get_param_or_default('L1_controller/speed_lookahead')
        self.lat_err_coeff = self._get_param_or_default('L1_controller/lat_err_coeff')
        self.acc_scaler_for_steer = self._get_param_or_default('L1_controller/acc_scaler_for_steer')
        self.dec_scaler_for_steer = self._get_param_or_default('L1_controller/dec_scaler_for_steer')
        self.start_scale_speed = self._get_param_or_default('L1_controller/start_scale_speed')
        self.end_scale_speed = self._get_param_or_default('L1_controller/end_scale_speed')
        self.downscale_factor = self._get_param_or_default('L1_controller/downscale_factor')
        self.speed_lookahead_for_steer = self._get_param_or_default('L1_controller/speed_lookahead_for_steer')
        self.trailing_gap = self._get_param_or_default('L1_controller/trailing_gap')
        self.trailing_vel_gain = self._get_param_or_default('L1_controller/trailing_vel_gain')
        self.trailing_p_gain = self._get_param_or_default('L1_controller/trailing_p_gain')
        self.trailing_i_gain = self._get_param_or_default('L1_controller/trailing_i_gain')
        self.trailing_d_gain = self._get_param_or_default('L1_controller/trailing_d_gain')
        self.blind_trailing_speed = self._get_param_or_default('L1_controller/blind_trailing_speed')
        
        # L1 dist calc param
        self.curvature_factor = self._get_param_or_default('L1_controller/curvature_factor')

        self.speed_factor_for_lat_err = self._get_param_or_default('L1_controller/speed_factor_for_lat_err')
        self.speed_factor_for_curvature = self._get_param_or_default('L1_controller/speed_factor_for_curvature')

        # steering_compensation
        self.KP = self._get_param_or_default('L1_controller/KP')
        self.KI = self._get_param_or_default('L1_controller/KI')
        self.KD = self._get_param_or_default('L1_controller/KD')

        self.heading_error_thres = self._get_param_or_default('L1_controller/heading_error_thres')
        self.steer_gain_for_speed = self._get_param_or_default('L1_controller/steer_gain_for_speed')

        self.future_constant = self._get_param_or_default('L1_controller/future_constant')
        
        self.AEB_thres = self._get_param_or_default('L1_controller/AEB_thres')


        self.speed_diff_thres = self._get_param_or_default('L1_controller/speed_diff_thres')
        self.start_speed = self._get_param_or_default('L1_controller/start_speed')
        self.start_curvature_factor = self._get_param_or_default('L1_controller/start_curvature_factor')

        # Parameters
        for i in range(5):
            # waiting for this message twice, as the republisher needs it first to compute the wanted param
            _ok, waypoints = wait_for_message(WpntArray, self, '/global_waypoints', time_to_wait=10.0)
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in waypoints.wpnts])

        # ===== HJ MODIFIED: Dual track length for GB and Fixed =====
        self.track_length_gb = self._get_param_or_default("/global_republisher/track_length")
        self.track_length_fixed = 0.0  # Will be set when Fixed path arrives
        self.track_length = self.track_length_gb  # Default to GB, updated dynamically in controller_cycle
        self.get_logger().info(f"[{self.name}] GB track length: {self.track_length_gb:.2f}m")
        # ===== HJ MODIFIED END =====

        # ===== HJ MODIFIED: Initialize GB converter and set as default =====
        self.converter_gb = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.converter = self.converter_gb  # Default to GB
        self.get_logger().info(f"[{self.name}] Initialized GB Frenet converter")
        # ===== HJ MODIFIED END =====


        # FTG
        self.ftg_controller = FTG()
        #  initialize controller

        self.controller = Controller(
            self.t_clip_min, 
            self.t_clip_max, 
            self.m_l1, 
            self.q_l1, 
            
            self.curvature_factor,
            
            self.KP,
            self.KI,
            self.KD,
            self.heading_error_thres,
            self.steer_gain_for_speed,

            self.future_constant,

            self.speed_lookahead, 
            self.lat_err_coeff, 
            self.acc_scaler_for_steer, 
            self.dec_scaler_for_steer, 
            self.start_scale_speed, 
            self.end_scale_speed, 
            self.downscale_factor, 
            self.speed_lookahead_for_steer,

            self.trailing_gap,
            self.trailing_vel_gain,
            self.trailing_p_gain,
            self.trailing_i_gain,
            self.trailing_d_gain,
            self.blind_trailing_speed,

            self.loop_rate,
            self.LUT_name,
            self.wheelbase,

            self.speed_factor_for_lat_err,
            self.speed_factor_for_curvature,
            self.ctrl_algo,

            self.speed_diff_thres,
            self.start_speed,
            self.start_curvature_factor,

            self.AEB_thres,

            self.converter,

            node=self,  # ROS2 포팅: Controller 가 self.create_publisher 등 호출 위해 노드 ref
            logger_info=self.get_logger().info,
            logger_warn=self.get_logger().warning
        )


        # Publishers to view data
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)
        self.future_position_pub = self.create_publisher(Marker, 'future_position', 10)
        self.trailing_pub = self.create_publisher(Marker, 'trailing_opponent_marker', 10)

        self.l1_pub = self.create_publisher(Point, 'l1_distance', 10)    
        # Publisher for steering and speed command
        self.publish_topic = self._get_param_or_default("~drive_topic", '/vesc/high_level/ackermann_cmd_mux/input/nav_1')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        if self.measuring:
            self.measure_pub = self.create_publisher(Float32, '/controller/latency', 10)

        ### HJ : current_brake control
        self.enable_brake_ctrl = False
        self.brake_mode = 0  # 0=jerk512, 1=direct brake topic
        self.brake_speed_diff_thres = 0.5  # [m/s]
        self.brake_current = 15.0  # [A] target brake decel strength
        self.brake_current_min = 3.0  # [A] min brake decel strength
        # Direct brake mode publishers
        from std_msgs.msg import Float64 as Float64Msg
        self.Float64Msg = Float64Msg
        self.brake_pub = self.create_publisher(Float64Msg, '/vesc/commands/motor/brake', 10)
        self.servo_pub = self.create_publisher(Float64Msg, '/vesc/commands/servo/position', 10)
        self.steering_to_servo_gain = self._get_param_or_default('/vesc/steering_angle_to_servo_gain', -1.2135)
        self.steering_to_servo_offset = self._get_param_or_default('/vesc/steering_angle_to_servo_offset', 0.5304)
        ### HJ : end

        ### HJ : friction sector → accel limiter ay_max sync
        # TODO C-5 패턴: dyn_sector_tuner 미포팅이라 sub 비활성. ROS2 native parameter
        # callback 으로 향후 통합 시 활성화.
        # self.create_subscription(DynConfig, '/dyn_sector_tuner/friction/parameter_updates', self.friction_sector_cb, 10)
        ### HJ : end

        # Subscribers
        self.create_subscription(BehaviorStrategy, '/behavior_strategy', self.behavior_cb, 10) # waypoints (x, y, v, norm trackbound, s, kappa)
        self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 10) # car speed
        self.create_subscription(PoseStamped, '/car_state/pose', self.car_state_cb, 10) # car position (x, y, theta)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10) # acceleration subscriber for steer change
        # ===== HJ MODIFIED: Dual Frenet odom subscribers =====
        self.create_subscription(Odometry, '/car_state/odom_frenet', self.car_state_frenet_gb_cb, 10) # GB frenet coordinates
        self.create_subscription(Odometry, '/car_state/odom_frenet_fixed', self.car_state_frenet_fixed_cb, 10) # Fixed frenet coordinates
        self.create_subscription(Bool, '/smart_static_active', self.smart_static_active_cb, 10) # Smart Static mode flag
        self.create_subscription(WpntArray, '/smart_static_avoidance_wpnts', self.smart_static_wpnts_cb, 10) # Fixed path waypoints
        # ===== HJ MODIFIED END =====
        # TODO C-5 패턴: /dyn_controller 미포팅 — l1 param 변경은 ros2 param set 으로 (native callback)
        # self.create_subscription(Config, "/dyn_controller/parameter_updates", self.l1_params_cb, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/vesc/odom", self.vesc_odom_cb, 10)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, 10)

        self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")
        
    def init_mapping(self):
        self.get_logger().warning(f"[{self.name}] Initializing for mapping")
        # Use FTG for mapping
        self.ftg_controller = FTG(mapping=False)
        
        # Publisher
        self.publish_topic = '/vesc/high_level/ackermann_cmd_mux/input/nav_1'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        
        # Subscribers
        self.create_subscription(Odometry, '/car_state/odom', self.odom_mapping_cb, 10) # car speed
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        
        
        self.get_logger().info(f"[{self.name}] initialized for mapping")

    ############################################CALLBACKS############################################
    def save_start_traj_cb(self, msg):
        self.controller.boost_mode = True
        self.controller.cur_state_speed = self.controller.start_speed
        
        
    def scan_cb(self, data: LaserScan):
        self.scan = data
          
    def l1_params_cb(self, params:Any):
        """
        Here the l1 parameters are updated if changed with rqt (dyn reconfigure)
        Values from .yaml file are set in l1_params_server.py      
        """
        self.t_clip_min = self._get_param_or_default('dyn_controller/t_clip_min')
        self.t_clip_max = self._get_param_or_default('dyn_controller/t_clip_max')
        self.m_l1 = self._get_param_or_default('dyn_controller/m_l1')
        self.q_l1 = self._get_param_or_default('dyn_controller/q_l1')
        self.speed_lookahead = self._get_param_or_default('dyn_controller/speed_lookahead')
        self.lat_err_coeff = self._get_param_or_default('dyn_controller/lat_err_coeff')
        self.acc_scaler_for_steer = self._get_param_or_default('dyn_controller/acc_scaler_for_steer')
        self.dec_scaler_for_steer = self._get_param_or_default('dyn_controller/dec_scaler_for_steer')
        self.start_scale_speed = self._get_param_or_default('dyn_controller/start_scale_speed')
        self.end_scale_speed = self._get_param_or_default('dyn_controller/end_scale_speed')
        self.downscale_factor = self._get_param_or_default('dyn_controller/downscale_factor')
        self.speed_lookahead_for_steer = self._get_param_or_default('dyn_controller/speed_lookahead_for_steer')
        self.trailing_gap = self._get_param_or_default('dyn_controller/trailing_gap')
        self.trailing_vel_gain = self._get_param_or_default('dyn_controller/trailing_vel_gain')
        self.trailing_p_gain = self._get_param_or_default('dyn_controller/trailing_p_gain')
        self.trailing_i_gain = self._get_param_or_default('dyn_controller/trailing_i_gain')
        self.trailing_d_gain = self._get_param_or_default('dyn_controller/trailing_d_gain')
        self.blind_trailing_speed = self._get_param_or_default('dyn_controller/blind_trailing_speed')
        self.future_constant = self._get_param_or_default('dyn_controller/future_constant')
        
        self.speed_diff_thres = self._get_param_or_default('dyn_controller/speed_diff_thres')
        self.start_speed = self._get_param_or_default('dyn_controller/start_speed')

        # steering_compensation
        self.KP = self._get_param_or_default('dyn_controller/KP')
        self.KI = self._get_param_or_default('dyn_controller/KI')
        self.KD = self._get_param_or_default('dyn_controller/KD')

        self.heading_error_thres = self._get_param_or_default('dyn_controller/heading_error_thres')
        self.steer_gain_for_speed = self._get_param_or_default('dyn_controller/steer_gain_for_speed')

        # L1 dist calc param
        self.curvature_factor = self._get_param_or_default('dyn_controller/curvature_factor')

        self.AEB_thres = self._get_param_or_default('dyn_controller/AEB_thres')

        self.speed_factor_for_lat_err = self._get_param_or_default('dyn_controller/speed_factor_for_lat_err')
        self.speed_factor_for_curvature = self._get_param_or_default('dyn_controller/speed_factor_for_curvature')

        ## Updating params for map and pp controller
        ## Lateral Control Parameters
        self.controller.t_clip_min = self.t_clip_min  
        self.controller.t_clip_max = self.t_clip_max   
        self.controller.m_l1 = self.m_l1
        self.controller.q_l1 = self.q_l1
        
        self.controller.curvature_factor = self.curvature_factor     

        self.controller.speed_factor_for_lat_err = self.speed_factor_for_lat_err
        self.controller.speed_factor_for_curvature = self.speed_factor_for_curvature
        
        self.controller.KP = self.KP 
        self.controller.KI = self.KI 
        self.controller.KD = self.KD 

        self.controller.heading_error_thres = self.heading_error_thres 
        self.controller.steer_gain_for_speed = self.steer_gain_for_speed 
        
        self.controller.speed_lookahead = self.speed_lookahead
        self.controller.lat_err_coeff = self.lat_err_coeff
        self.controller.acc_scaler_for_steer = self.acc_scaler_for_steer
        self.controller.dec_scaler_for_steer = self.dec_scaler_for_steer
        self.controller.start_scale_speed = self.start_scale_speed
        self.controller.end_scale_speed = self.end_scale_speed
        self.controller.downscale_factor = self.downscale_factor
        self.controller.speed_lookahead_for_steer = self.speed_lookahead_for_steer
        self.controller.future_constant = self.future_constant

        self.controller.speed_diff_thres = self.speed_diff_thres
        self.controller.start_speed = self.start_speed
        self.controller.start_curvature_factor = self.start_curvature_factor

        self.controller.AEB_thres = self.AEB_thres

        ### HJ : lateral correction params from dyn_reconfigure
        lat_mode_int = self._get_param_or_default('dyn_controller/lat_correction_mode', 0)
        self.controller.lat_correction_mode = ['none', 'stanley', 'predictive'][lat_mode_int]
        self.controller.lat_K_stanley = self._get_param_or_default('dyn_controller/lat_K_stanley', 1.5)
        self.controller.lat_pred_horizon = self._get_param_or_default('dyn_controller/lat_pred_horizon', 0.3)
        self.controller.lat_pred_alpha = self._get_param_or_default('dyn_controller/lat_pred_alpha', 0.3)
        self.controller.speed_ff_gain_accel = self._get_param_or_default('dyn_controller/speed_ff_gain_accel', 0.0)
        self.controller.speed_ff_gain_brake = self._get_param_or_default('dyn_controller/speed_ff_gain_brake', 0.0)
        self.controller.ff_accel_lookahead = self._get_param_or_default('dyn_controller/ff_accel_lookahead', 0.0)
        self.controller.ff_brake_lookahead = self._get_param_or_default('dyn_controller/ff_brake_lookahead', 0.0)
        ### HJ : friction-ellipse accel limiter (scale both axes by sector friction)
        self.controller.accel_limiter_enabled = self._get_param_or_default('dyn_controller/accel_limiter_enabled', True)
        friction = self._get_current_friction()
        self.controller.accel_lim_ax_max = self._get_param_or_default('dyn_controller/accel_lim_ax_max', 5.0) * friction
        self.controller.accel_lim_ay_max = self._get_param_or_default('dyn_controller/accel_lim_ay_max', 4.5) * friction
        self.controller.accel_lim_horizon = self._get_param_or_default('dyn_controller/accel_lim_horizon', 0.3)
        self.controller.accel_lim_lookahead = self._get_param_or_default('dyn_controller/accel_lim_lookahead', 0.3)
        ### HJ : end

        ### HJ : GP residual + yaw rate feedback from dyn_reconfigure
        self.controller.gp_steer_enabled = self._get_param_or_default('dyn_controller/gp_steer_enabled', False)
        self.controller.gp_max_correction = self._get_param_or_default('dyn_controller/gp_max_correction', 0.05)
        self.controller.gp_uncertainty_thres = self._get_param_or_default('dyn_controller/gp_uncertainty_thres', 0.1)
        self.controller.K_yr = self._get_param_or_default('dyn_controller/K_yr', 0.0)
        self.controller.K_yr_sat = self._get_param_or_default('dyn_controller/K_yr_sat', 0.05)
        self.controller.K_us = self._get_param_or_default('dyn_controller/K_us', 0.0)
        ### HJ : end

        ### HJ : brake control params from dyn_reconfigure
        self.enable_brake_ctrl = self._get_param_or_default('dyn_controller/enable_brake_ctrl', False)
        self.brake_mode = self._get_param_or_default('dyn_controller/brake_mode', 0)
        self.brake_speed_diff_thres = self._get_param_or_default('dyn_controller/brake_speed_diff_thres', 0.5)
        self.brake_current = self._get_param_or_default('dyn_controller/brake_current', 15.0)
        self.brake_current_min = self._get_param_or_default('dyn_controller/brake_current_min', 3.0)
        ### HJ : end

        ## Trailing Control Parameters
        self.controller.trailing_gap = self.trailing_gap # Distance in meters
        self.controller.trailing_vel_gain = self.trailing_vel_gain # Distance in meters
        self.controller.trailing_p_gain = self.trailing_p_gain
        self.controller.trailing_i_gain = self.trailing_i_gain
        self.controller.trailing_d_gain = self.trailing_d_gain
        self.controller.blind_trailing_speed = self.blind_trailing_speed

    def odom_mapping_cb(self, data: Odometry):
        # velocity for follow the gap (needed to set gap radius)
        self.ftg_controller.set_vel(data.twist.twist.linear.x)

    def odom_cb(self, data: Odometry):
        self.speed_now = data.twist.twist.linear.x
        self.speed_now_y = data.twist.twist.linear.y
        self.controller.speed_now = self.speed_now

        ### HJ : yaw rate from odom (IMU /imu/data dead) — ENU, left+
        self.yaw_rate = data.twist.twist.angular.z
        self.controller.yaw_rate = self.yaw_rate
        ### HJ : end

        # velocity for follow the gap (needed to set gap radius)
        self.ftg_controller.set_vel(data.twist.twist.linear.x)
        
    def vesc_odom_cb(self, data: Odometry):
        self.wheelspeed_now = data.twist.twist.linear.x
        
        # velocity for follow the gap (needed to set gap radius)
        self.ftg_controller.set_vel(data.twist.twist.linear.x)

    def car_state_cb(self, data: PoseStamped):
        x = data.pose.position.x
        y = data.pose.position.y
        # ROS2 포팅: tf_transformations.euler_from_quaternion → 직접 atan2 (yaw)
        q = data.pose.orientation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.position_in_map = np.array([x, y, theta])[np.newaxis]
        ### HJ : store z separately for 3D nearest waypoint search
        self.position_z = data.pose.position.z
        ### HJ : end

    # ===== HJ MODIFIED: Split Frenet callbacks for GB and Fixed =====
    def car_state_frenet_gb_cb(self, data: Odometry):
        """GB Frenet odom callback"""
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet_gb = np.array([s, d, vs, vd])

        # Update active frenet position if in GB mode
        if not self.smart_static_active:
            self.position_in_map_frenet = self.position_in_map_frenet_gb

    def car_state_frenet_fixed_cb(self, data: Odometry):
        """Fixed Frenet odom callback"""
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet_fixed = np.array([s, d, vs, vd])

        # Update active frenet position if in Smart Static mode
        if self.smart_static_active:
            self.position_in_map_frenet = self.position_in_map_frenet_fixed

    def smart_static_active_cb(self, data: Bool):
        """Smart Static mode flag callback - switches between GB and Fixed Frenet"""
        prev_state = self.smart_static_active
        self.smart_static_active = data.data

        # Detect mode changes and switch converter + frenet position + track_length
        if self.smart_static_active != prev_state:
            if self.smart_static_active:
                # Switching to Smart Static mode
                if self.converter_fixed is not None:
                    self.controller.converter = self.converter_fixed
                    self.track_length = self.track_length_fixed
                    self.get_logger().info(f"[{self.name}] Switched to Fixed Frenet (length={self.track_length_fixed:.2f}m)")
                if len(self.position_in_map_frenet_fixed) > 0:
                    self.position_in_map_frenet = self.position_in_map_frenet_fixed
            else:
                # Switching to GB mode
                self.controller.converter = self.converter_gb
                self.track_length = self.track_length_gb
                self.get_logger().info(f"[{self.name}] Switched to GB Frenet (length={self.track_length_gb:.2f}m)")
                if len(self.position_in_map_frenet_gb) > 0:
                    self.position_in_map_frenet = self.position_in_map_frenet_gb

        self._prev_smart_static_active = self.smart_static_active

    def smart_static_wpnts_cb(self, data: WpntArray):
        """Smart Static waypoints callback - creates Fixed Frenet converter only once"""
        if len(data.wpnts) == 0:
            return

        # Only create if not already created
        if self.converter_fixed is None:
            fixed_wpnts = np.array([[wpnt.x_m, wpnt.y_m, wpnt.z_m] for wpnt in data.wpnts])
            self.converter_fixed = FrenetConverter(fixed_wpnts[:, 0], fixed_wpnts[:, 1], fixed_wpnts[:, 2])

            # Calculate Fixed track length (last waypoint's s value)
            self.track_length_fixed = data.wpnts[-1].s_m
            self.get_logger().info(f"[{self.name}] Created Fixed Frenet converter ({len(fixed_wpnts)} waypoints, length={self.track_length_fixed:.2f}m)")

            # If currently in Smart Static mode, apply it immediately
            if self.smart_static_active:
                self.controller.converter = self.converter_fixed
                self.track_length = self.track_length_fixed
                self.get_logger().info(f"[{self.name}] Applied Fixed Frenet converter (Smart mode active)")
    # ===== HJ MODIFIED END ===== 


    def behavior_cb(self, data: BehaviorStrategy):
        if len(data.trailing_targets) != 0:
            opponent= data.trailing_targets[0]
            opponent_s = opponent.s_center
            opponent_d = opponent.d_center
            opponent_vs = opponent.vs
            opponent_visible = opponent.is_visible
            opponent_static = opponent.is_static
            # ===== HJ ADDED: Add static sector info for differential trailing control =====
            opponent_in_static_sector = opponent.in_static_obs_sector
            self.opponent = [opponent_s, opponent_d, opponent_vs, opponent_static, opponent_visible, opponent_in_static_sector]
            # Index:          [0]        [1]        [2]       [3]              [4]               [5]
            # ===== HJ ADDED END =====
        else:
            self.opponent = None

        self.waypoint_list_in_map = []
        
        ### HJ : waypoint layout [x, y, z, speed, safety_ratio, s, kappa, psi, ax, d]
        ###       indices:        0  1  2  3      4              5  6      7    8   9
        for waypoint in data.local_wpnts:
            speed = waypoint.vx_mps
            if waypoint.d_right + waypoint.d_left != 0:
                safety_ratio = min(waypoint.d_left, waypoint.d_right) / (waypoint.d_right + waypoint.d_left)
            else:
                safety_ratio = 0
            self.waypoint_list_in_map.append([
                waypoint.x_m,         # 0
                waypoint.y_m,         # 1
                waypoint.z_m,         # 2
                speed,                # 3
                safety_ratio,         # 4
                waypoint.s_m,         # 5
                waypoint.kappa_radpm, # 6
                waypoint.psi_rad,     # 7
                waypoint.ax_mps2,     # 8
                waypoint.d_m,         # 9
            ])
        ### HJ : end
        self.waypoint_array_in_map = np.array(self.waypoint_list_in_map)
        self.waypoint_safety_counter = 0
        self.state = data.state
        
    ### HJ : friction sector → update accel limiter ay_max per sector
    def friction_sector_cb(self, msg):
        """Friction sector params changed — reload from rosparam"""
        try:
            n_sec = self._get_param_or_default('/friction_map_params/n_sectors', 0)
            if n_sec > 0:
                self._friction_sectors = []
                for si in range(n_sec):
                    self._friction_sectors.append({
                        's_start': self._get_param_or_default(f'/friction_map_params/Sector{si}/s_start', -1.0),
                        's_end': self._get_param_or_default(f'/friction_map_params/Sector{si}/s_end', -1.0),
                        'start': self._get_param_or_default(f'/friction_map_params/Sector{si}/start', 0),
                        'end': self._get_param_or_default(f'/friction_map_params/Sector{si}/end', 0),
                        'friction': self._get_param_or_default(f'/friction_map_params/Sector{si}/friction', 1.0),
                    })
                self._friction_global_limit = self._get_param_or_default('/friction_map_params/global_friction_limit', 1.0)
        except Exception:
            pass

    def _get_current_friction(self):
        """Get friction scale for current s position"""
        if not hasattr(self, '_friction_sectors') or not self._friction_sectors:
            return 1.0
        if len(self.position_in_map_frenet) == 0:
            return 1.0
        s_now = self.position_in_map_frenet[0]
        for sec in self._friction_sectors:
            if sec.get('s_start', -1) >= 0:
                if sec['s_start'] <= s_now <= sec['s_end']:
                    return min(sec['friction'], self._friction_global_limit)
            else:
                # fallback: index-based (cannot use here, return global)
                return 1.0
        return 1.0
    ### HJ : end

    def imu_cb(self, data):
        self.acc_now[1:] = self.acc_now[:-1]
        # self.acc_now[0] = -data.linear_acceleration.x # Micro Strain

        self.acc_now[0] = -data.linear_acceleration.y # vesc is rotated 90 deg -y is +x dir

        self.yaw_rate = -data.angular_velocity.z # vesc is rotated 90 deg, so (-acc_y) == (long_acc)
        self.controller.yaw_rate = self.yaw_rate

    ############################################MAIN LOOP############################################

    def control_loop(self):
        """Timer callback (50Hz). 첫 호출에서만 wait + setup, 그 후엔 tick body 만."""
        if not getattr(self, "_loop_inited", False):
            if self.mapping:
                self._init_mapping_loop()
            else:
                self._init_controller_loop()
            self._loop_inited = True
        if self.mapping:
            self.mapping_loop()
        else:
            self.controller_loop()

    def _init_mapping_loop(self):
        wait_for_message(LaserScan, self, '/scan', time_to_wait=10.0)
        wait_for_message(Odometry, self, '/car_state/odom', time_to_wait=10.0)
        self.get_logger().info(f"[{self.name}] Ready for mapping!")

    def _init_controller_loop(self):
        self.get_logger().info(f"[{self.name}] Waiting for behavior_strategy")
        wait_for_message(BehaviorStrategy, self, '/behavior_strategy', time_to_wait=10.0)
        wait_for_message(WpntArray, self, '/global_waypoints', time_to_wait=10.0)
        wait_for_message(Odometry, self, '/car_state/odom', time_to_wait=10.0)
        self.get_logger().info(f"[{self.name}] BehaviorStrategy received")
        self.get_logger().info(f"[{self.name}] Waiting for car_state/pose")
        wait_for_message(PoseStamped, self, '/car_state/pose', time_to_wait=10.0)
        self.track_length = self._get_param_or_default("/global_republisher/track_length")
        self.get_logger().info(f"[{self.name}] Ready!")

    def mapping_loop(self):
        # ready check
        if self.scan is None:
            return
        if True:  # 한 번 — timer 가 50Hz 로 호출
            speed, acceleration, jerk, steering_angle = 0, 0, 0, 0
            speed, steering_angle = self.ftg_controller.process_lidar(self.scan.ranges)
            ack_msg = self.create_ack_msg(speed, acceleration, jerk, steering_angle)
            self.drive_pub.publish(ack_msg)

    def controller_loop(self):
        # ready check — callback 으로 set 되는 attribute 가 모두 채워졌나
        # (numpy array 는 truth 가 ambiguous → 명시적 length check)
        def _empty(v):
            if v is None:
                return True
            try:
                return len(v) == 0
            except TypeError:
                return False
        if not hasattr(self, "waypoint_array_in_map") or _empty(self.waypoint_array_in_map):
            return
        if _empty(getattr(self, "position_in_map_frenet", None)):
            return
        if _empty(getattr(self, "position_in_map", None)):
            return

        if True:  # 한 번 — timer 가 50Hz 로 호출
            if self.measuring:
                start = time.perf_counter()
            speed, acceleration, jerk, steering_angle = 0, 0, 0, 0

            # Logic to select controller — no silent fallback.
            # 예외 발생 시 노드가 죽도록 의도적으로 try/except 제거 (디버깅 가시성 우선).
            if self.state != "FTGONLY":
                speed, acceleration, jerk, steering_angle = self.controller_cycle()
            else:
                speed, steering_angle = self.ftg_cycle()
                
            if self.measuring:
                end = time.perf_counter()
                self.measure_pub.publish(end-start)
                
            ### HJ : current_brake switching logic
            # brake_mode 0 = jerk512 (acceleration → current via ackermann pipeline)
            # brake_mode 1 = direct /vesc/commands/motor/brake (exact current you set)
            brake_active = False
            if self.enable_brake_ctrl and self.speed_now > 0.3:
                speed_diff = self.speed_now - speed  # positive when need to decelerate
                if speed_diff > self.brake_speed_diff_thres:
                    alpha = min(speed_diff / max(self.speed_now, 1.0), 1.0)
                    brake_val = self.brake_current_min + alpha * (self.brake_current - self.brake_current_min)
                    brake_active = True

                    if self.brake_mode == 0:
                        # jerk512: send negative accel through ackermann pipeline
                        ack_msg = self.create_ack_msg(speed, -brake_val, 512, steering_angle)
                        self.drive_pub.publish(ack_msg)
                    else:
                        # direct: exact brake current to VESC, steering via servo
                        self.brake_pub.publish(self.Float64Msg(data=brake_val))
                        servo_msg = self.Float64Msg(
                            data=self.steering_to_servo_gain * steering_angle + self.steering_to_servo_offset)
                        self.servo_pub.publish(servo_msg)
            ### HJ : end

            if not brake_active:
                ack_msg = self.create_ack_msg(speed, acceleration, jerk, steering_angle)

                # #-------------------------------Force Speed--------------------------------
                # ack_msg = self.create_ack_msg(2.5, acceleration, jerk, steering_angle)
                # #-------------------------------Force Speed--------------------------------

                self.drive_pub.publish(ack_msg)
            if self.measuring:
                end = time.perf_counter()
                self.measure_pub.publish(1/(end-start))
    def controller_cycle(self):
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint, curvature_waypoints, future_position = self.controller.main_loop(self.state, 
                                                                                                                    self.position_in_map, 
                                                                                                                    self.waypoint_array_in_map, 
                                                                                                                    self.speed_now, 
                                                                                                                    self.opponent, 
                                                                                                                    self.position_in_map_frenet, 
                                                                                                                    self.acc_now,
                                                                                                                    self.track_length)
                
        # D-1d: viz 재활성 (모든 numeric 필드 float() cast 명시 후 — D-1c 에서 작업)
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.visualize_trailing_opponent()
        self.viz_future_position(future_position, 200)

        self.curvature_waypoints = curvature_waypoints
        # ROS2 strict 타입: Point.x/y/z 는 float64. int / numpy 모두 float() cast.
        self.l1_pub.publish(Point(
            x=float(idx_nearest_waypoint),
            y=float(L1_distance),
            z=float(self.curvature_waypoints),
        ))

        
        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.loop_rate/self.state_machine_rate * 10:
            self.get_logger().error(f"[{self.name}] Received no local wpnts. STOPPING!!")
            speed = 0
            steering_angle = 0

        return speed, acceleration, jerk, steering_angle
    

    def ftg_cycle(self):
        speed, steer = self.ftg_controller.process_lidar(self.scan.ranges)
        self.get_logger().warning(f"[{self.name}] FTGONLY!!!")
        return speed, steer 
        
    def create_ack_msg(self, speed, acceleration, jerk, steering_angle):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = 'base_link'
        # ROS2 strict: numpy → float() cast 명시 (AckermannDrive 의 필드는 float32)
        ack_msg.drive.steering_angle = float(steering_angle)
        ack_msg.drive.speed = float(speed)
        ack_msg.drive.jerk = float(jerk)
        ack_msg.drive.acceleration = float(acceleration)
        return ack_msg

############################################MSG CREATION############################################
# visualization utilities
    def visualize_steering(self, theta):
        _half = float(theta) * 0.5
        quaternions = (0.0, 0.0, math.sin(_half), math.cos(_half))

        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "base_link"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = Marker.ARROW
        lookahead_marker.id = 50
        lookahead_marker.scale.x = 0.6
        lookahead_marker.scale.y = 0.05
        lookahead_marker.scale.z = 0.0
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        from rclpy.duration import Duration as _Duration
        lookahead_marker.lifetime = _Duration().to_msg()
        lookahead_marker.pose.position.x = 0.0
        lookahead_marker.pose.position.y = 0.0
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = float(quaternions[0])
        lookahead_marker.pose.orientation.y = float(quaternions[1])
        lookahead_marker.pose.orientation.z = float(quaternions[2])
        lookahead_marker.pose.orientation.w = float(quaternions[3])
        self.lookahead_pub.publish(lookahead_marker)

    def set_lookahead_marker(self, lookahead_point, id):

        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = id
        lookahead_marker.scale.x = 0.35
        lookahead_marker.scale.y = 0.35
        lookahead_marker.scale.z = 0.35
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = float(lookahead_point[0])
        lookahead_marker.pose.position.y = float(lookahead_point[1])
        lookahead_marker.pose.position.z = float(lookahead_point[2]) if len(lookahead_point) > 2 else 0.0

        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0

        self.lookahead_pub.publish(lookahead_marker)

    def viz_future_position(self, future_position,id):

        _half = float(future_position[0,2]) * 0.5
        quaternions = (0.0, 0.0, math.sin(_half), math.cos(_half))

        future_position_marker = Marker()
        future_position_marker.header.frame_id = "map"
        future_position_marker.header.stamp = self.get_clock().now().to_msg()
        future_position_marker.type = Marker.ARROW
        future_position_marker.id = id
        future_position_marker.scale.x = 1.2
        future_position_marker.scale.y = 0.06
        future_position_marker.scale.z = 0.0
        future_position_marker.color.r = 0.5
        future_position_marker.color.g = 0.0
        future_position_marker.color.b = 0.5
        future_position_marker.color.a = 1.0
        future_position_marker.pose.position.x = float(future_position[0,0])
        future_position_marker.pose.position.y = float(future_position[0,1])
        future_position_marker.pose.position.z = float(self.controller.future_position_z)

        future_position_marker.pose.orientation.x = float(quaternions[0])
        future_position_marker.pose.orientation.y = float(quaternions[1])
        future_position_marker.pose.orientation.z = float(quaternions[2])
        future_position_marker.pose.orientation.w = float(quaternions[3])

        self.future_position_pub.publish(future_position_marker)

    def set_test_lookahead_marker(self, lookahead_point, id):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = id
        lookahead_marker.scale.x = 0.35
        lookahead_marker.scale.y = 0.35
        lookahead_marker.scale.z = 0.35
        lookahead_marker.color.r = 0.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 1.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = float(lookahead_point[0])
        lookahead_marker.pose.position.y = float(lookahead_point[1])
        lookahead_marker.pose.position.z = float(lookahead_point[2]) if len(lookahead_point) > 2 else 0.0
        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)

    def visualize_trailing_opponent(self):
        if(self.state == "TRAILING" and (self.opponent is not None)):
            on = True
        else:
            on = False
        opponent_marker = Marker()
        opponent_marker.header.frame_id = "map"
        opponent_marker.header.stamp = self.get_clock().now().to_msg()
        opponent_marker.type = 2
        opponent_marker.scale.x = 0.3
        opponent_marker.scale.y = 0.3
        opponent_marker.scale.z = 0.3
        opponent_marker.color.r = 1.0
        opponent_marker.color.g = 0.0
        opponent_marker.color.b = 0.0
        opponent_marker.color.a = 1.0
        if self.opponent is not None:
            # numpy 2.x: float(ndarray) requires 0-dim. Pass scalars (not lists) so
            # get_cartesian_3d returns 1-D (3,) array → pos[i] is scalar.
            pos = self.converter.get_cartesian_3d(self.opponent[0], self.opponent[1])
            opponent_marker.pose.position.x = float(pos[0])
            opponent_marker.pose.position.y = float(pos[1])
            opponent_marker.pose.position.z = float(pos[2])

        opponent_marker.pose.orientation.x = 0.0
        opponent_marker.pose.orientation.y = 0.0
        opponent_marker.pose.orientation.z = 0.0
        opponent_marker.pose.orientation.w = 1.0
        if on == False:
            opponent_marker.action = Marker.DELETE
        self.trailing_pub.publish(opponent_marker)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = Controller_manager()
    except (RuntimeError, FileNotFoundError, ImportError) as e:
        print(f"[controller_manager] startup error: {e}")
        rclpy.shutdown()
        return
    # control_loop 의 init 단계 (wait_for_message + 첫 setup) 는 첫 timer callback
    # 안에서 처리 (_loop_inited flag). 그 후 매 50Hz tick.
    try:
        node.create_timer(1.0 / node.loop_rate, node.control_loop)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
 
