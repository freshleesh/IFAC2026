#!/usr/bin/env python3
import time
from typing import List, Any, Tuple, Optional, Dict
import copy
import os
import sys
import csv
import json
import threading
import math
import numpy as np
import cv2
import yaml
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import BPoly
from scipy.signal import argrelextrema, savgol_filter
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from f110_msgs.msg import Obstacle, ObstacleArray, OTWpntArray, Wpnt, WpntArray, BehaviorStrategy
from frenet_converter.frenet_converter import FrenetConverter
import tf.transformations as tf_trans
from grid_filter.grid_filter import GridFilter
import trajectory_planning_helpers as tph

# ===== HJ ADDED: GB-aware algorithm parameters =====
OPPOSITE_SPACE_THRESHOLD = 0.6 # meters, if opposite wall has > this space, shift toward it

# Obstacle size
OBSTACLE_RADIUS = 0.25  # meters, obstacle radius for all calculations

# Safety width override
OVERRIDE_SAFETY_WIDTH = True  # True: use SAFETY_WIDTH_VALUE, False: use ROS parameter
SAFETY_WIDTH_VALUE = 1.0  # Safety Boundary to Wall for OPT, meters, used only if OVERRIDE_SAFETY_WIDTH is True

# Fixed path post-smoothing (applied after optimization)
SMOOTH_OPT_OUTPUT = True  # Enable Savitzky-Golay smoothing on optimizer output
SMOOTH_WINDOW = 51 # Savitzky-Golay window size (must be odd, larger = smoother)
SMOOTH_POLYORDER = 2  # Polynomial order (3 = cubic, lower = smoother but less accurate)
# ===== HJ ADDED END =====

# Set to True to use conservative bounds from _for_spliner.png and obstacle-added map
# Set to False to use original GB waypoint bounds (d_left, d_right from optimization)
USE_SAFE_GRID = True
# Set to True to use conservative bounds (_for_spliner.png + obstacles) for optimization input
# Set to False to use original bounds (map.png + obstacles) for optimization input
# Only effective when USE_SAFE_GRID=True
USE_CONSERVATIVE_BOUNDS_FOR_OPT = True
# Set to True to visualize d_left/d_right endpoints as spheres in RViz
# GB mode: Blue spheres, Smart mode: Cyan spheres
DEBUG_MORE_SPACE = True

# Set to True to disable Fixed path optimization entirely (use only GB path with dynamic avoidance)
USE_ONLY_GB = False

# Joy button to force GB mode (ignore obstacles)
JOY_FORCE_GB_BUTTON = 7  # Button index to force GB mode (set to -1 to disable)

# Local optimization mode (instead of global optimization)
# True: optimize only around obstacles (±5m), merge with GB raceline, then smooth
# False: global optimization (original behavior)
ENABLE_LOCAL_OPT = True
LOCAL_OPT_RANGE = 4  # meters before/after obstacle for local optimization / # make it small when the map is small
LOCAL_OPT_WALL_MARGIN = 0.6  # Minimum distance from wall for fixed path spline (meters)
LOCAL_OPT_EVASION_DIST = 0.5  # meters, distance from obstacle edge for local optimization
LOCAL_OPT_MID_D_RATIO = 0.5  # d value ratio for intermediate points (1/4, 3/4): 0.7 = 70% of d_apex

RECOMPUTE_LOCAL_OPT_SPEED = True  # True: recompute speed from kappa (like centerline opt), False: use GB speed
KAPPA_SMOOTH_WINDOW = 11  # Savgol filter window for kappa smoothing before velocity calc (odd number, 0=disabled)

LOCAL_OPT_FALLBACK = True  # True: fallback to LOCAL_OPT when GLOBAL optimization fails, False: fallback to GB mode

# Transition smoothing parameters (for LOCAL_OPT mode)
SMOOTH_TRANS_MARGIN = 30  # waypoints before/after transition point
SMOOTH_TRANS_NUM_CTRL_PTS = 9  # number of control points (odd recommended: 3, 5, 7, 9, ...)
SMOOTH_TRANS_SCALE = 1.0  # tangent scale: larger = rounder curve

##########################################################################
############################### HJ Edited ################################
##########################################################################

# Velocity blending parameters (for LOCAL_OPT mode)
# Blends kappa-based velocity with GB velocity near obstacles for smoother profile
USE_VEL_BLENDING = True  # True: blend kappa+GB velocity, False: use kappa velocity only
VEL_BLEND_RANGE = 10.0  # meters around obstacle where blending is applied
VEL_BLEND_TRANSITION = 5.0  # meters for smooth transition from blend zone to GB

VEL_BLEND_GB_WEIGHT = 0.4  # GB weight: 0.0 = 100% kappa (slow), 0.5 = average, 1.0 = 100% GB (fast)

# Real-time avoidance path smoothing (for do_spline / otwpnts output)
SMOOTH_OTWPNTS = True  # Enable Savitzky-Golay smoothing on real-time avoidance path
SMOOTH_OTWPNTS_WINDOW = 51  # Window size (must be odd, larger = smoother, 11 is good balance)
SMOOTH_OTWPNTS_POLYORDER = 2  # Polynomial order

##########################################################################
############################### HJ Edited ################################
##########################################################################


# GB optimizer speed tuning parameters
OPT_WIDTH_OPT = 0.8  # Optimization width factor (lower = faster, less aggressive)
OPT_IQP_ITERS_MIN = 5  # Minimum IQP iterations (lower = faster, less precise)
OPT_IQP_CURVERROR_ALLOWED = 0.05 # Allowed curvature error (higher = faster, less smooth) #--Very Important--#

OPT_STEPSIZE_PREP = 0.3  # Spline fitting step size in meters (higher = faster)  #--Very Important--#
OPT_STEPSIZE_REG = 0.3  # Optimization step size in meters (higher = faster)
OPT_STEPSIZE_INTERP = 0.1  # Final interpolation step size in meters (waypoint spacing)
OPT_IQP_ITERS_MAX = 30  # Maximum IQP iterations - prevents infinite loop on non-convergence

# Add GB optimizer path to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gb_optimizer/src'))
from global_racetrajectory_optimization.trajectory_optimizer import trajectory_optimizer

class ObstacleSpliner:
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/tracking/obstacles`: Subscribes to the obstacle array.
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
    
    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes real-time spline markers (pre-fix mode).
        - `/planner/avoidance/otwpnts`: Publishes splined waypoints.
        - `/planner/avoidance/considered_OBS`: Publishes markers for the closest obstacle.
        - `/planner/avoidance/propagated_obs`: Publishes markers for the propagated obstacle.
        - `/planner/avoidance/latency`: Publishes the latency of the spliner node. (only if measuring is enabled)
        - `/planner/avoidance/smart_markers`: Publishes GB optimizer fixed path markers (post-fix mode).
    """

    # ===== HJ ADDED: Global toggle for Frenet coordinate system =====
    # Set to True for Fixed path arc length (proper), False for GB Frenet (debugging)
    # This affects both do_spline and _generate_fixed_path waypoint s,d coordinates

    USE_FIXED_PATH_FRENET = True

    # ===== HJ ADDED END =====

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        self.name = "obs_spliner_node"
        rospy.init_node(self.name)

        # initialize the instance variable
        # self.obs = ObstacleArray()
        self.obs_in_interest = None
        self.gb_wpnts = WpntArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.cur_s = 0
        self.cur_d = 0
        self.cur_vs = 0
        self.gb_scaled_wpnts = WpntArray()
        self.lookahead = 10  # in meters [m]

        # ===== HJ ADDED: Race start detection (ignore obstacles before race starts) =====
        self.race_started = False  # One-time flag: becomes True when vehicle speed exceeds threshold
        self.race_start_speed_threshold = self._get_param_or_default("~race_start_speed_threshold", 0.5)  # m/s
        self.get_logger().info(f"[{self.name}] Race start speed threshold: {self.race_start_speed_threshold} m/s (obstacles ignored until race starts)")
        # ===== HJ ADDED END =====
        self.last_switch_time = self.get_clock().now().to_msg()
        self.last_ot_side = ""
        self.from_bag = self._get_param_or_default("/from_bag", False)
        self.measuring = self._get_param_or_default("/measure", False)
        
        self.sampling_dist = self._get_param_or_default("/sampling_dist", 20.0)
        self.spline_scale = self._get_param_or_default("/spline_scale", 0.8)
        self.post_min_dist = self._get_param_or_default("/post_min_dist", 1.5)
        self.post_max_dist = self._get_param_or_default("/post_max_dist", 5.0)
        self.kernel_size = self._get_param_or_default("/kernel_size", 4)
        
        # ===== HJ ADDED: Dual grid filters =====
        self.map_filter = GridFilter(map_topic="/map", debug=False)  # Original map
        self.map_filter.set_erosion_kernel_size(self.kernel_size)
        self.map_filter_fixed = None  # Modified map (with obstacles), created after fixed path generation
        self.map_filter_conservative_base = None  # Conservative base map (_for_spliner.png), always maintained
        self.map_filter_conservative_with_obs = None  # Conservative map with obstacles (_for_spliner_with_obs.png), created when obstacle added
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Reverse mapping parameter (from global_planner) =====
        self.reverse_mapping = self._get_param_or_default('/global_planner/reverse_mapping', False)
        self.get_logger().info(f"[{self.name}] Reverse mapping: {self.reverse_mapping}")
        # ===== HJ ADDED END =====

        # ===== HJ EDITED START: Smart static avoidance memory system =====
        # IMPORTANT: Initialize BEFORE subscribing to /tracking/obstacles to avoid AttributeError
        # Phase B: Memory structure - (sector_id, obs_id) tuple keys for individual obstacle tracking
        self.static_obs_memory = {}  # {(sector_id, obs_id): {'s_history': [], 'd_history': [], 's_history_fixed': [], 'd_history_fixed': [], 'obs_count': 0, 'interferes': bool, 'interferes_fixed': bool, 'last_seen': rospy.Time}}
        self.fixed_path_generated = False  # Static flag: Has fixed path been generated? (Once True, stays True)
        # If USE_ONLY_GB=True, skip optimization entirely by marking as already attempted
        self.fixed_path_generation_attempted = USE_ONLY_GB  # Static flag: Has generation been attempted? (prevent infinite retry)
        if USE_ONLY_GB:
            self.get_logger().warning(f"[{self.name}] ===== USE_ONLY_GB = True =====")
            self.get_logger().warning(f"[{self.name}] Fixed path optimization DISABLED")
            self.get_logger().warning(f"[{self.name}] Will use GB path only with dynamic obstacle avoidance (do_spline)")
        else:
            self.get_logger().info(f"[{self.name}] USE_ONLY_GB = False (Fixed path optimization enabled)")
        self.use_fixed_path = False  # Dynamic flag: Currently using fixed path? (Can toggle based on obstacles)
        self.fixed_path_generating = False  # Flag to track if generation is in progress
        self.fixed_path_wpnts = OTWpntArray()
        self.fixed_path_markers = MarkerArray()  # Markers for fixed path visualization
        self.fixed_converter = None  # FrenetConverter for fixed path (created when path is generated)
        self.gb_revert_threshold_m = 1.0  # All obstacles must be >1.0m away (lateral) to revert to GB (noise tolerance)
        # ===== HJ ADDED: Low-rate publishing for fixed path =====
        self.fixed_path_last_pub_time = rospy.Time(0)
        self.fixed_path_pub_rate = 2.0  # seconds (0.5Hz) - static data doesn't need high rate
        # ===== HJ ADDED END =====
        # ===== HJ ADDED: State-aware flag management =====
        self.pending_flag_true = False   # Want to publish True but waiting for safe conditions
        self.pending_flag_false = False  # Want to publish False but waiting for safe state
        self.current_state_name = "GB_TRACK"  # State Machine's current state from behavior_strategy
        self.published_flag = False  # Currently published flag value (actual topic value)
        # ===== HJ ADDED END =====
        self.min_stable_observations = 5  # 0.25 seconds at 20Hz (enough for static obs with std check)
        self.position_std_threshold = 0.05  # 5cm position stability threshold
        self.min_sectors_with_stable_obs = 2  # Need stable obstacles in 2 DIFFERENT sectors
        self.max_history_length = 100  # Keep recent 100 observations (5 seconds)

        # ===== HJ ADDED: Adaptive memory timeout =====
        self.memory_timeout_sec = 15.0  # Initial: 15 seconds (before fixed path)
        self.memory_timeout_min = 1.0   # Minimum: 1 second (after fixed path stabilizes)
        self.memory_timeout_decay_interval = 3.0  # Decay every 5 seconds after fixed path
        self.memory_timeout_decay_amount = 0.0    # Reduce by 1 second each interval
        self.last_timeout_decay_time = None  # Track last decay (set when fixed path generated)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Load static obstacle sectors for position-based active check =====
        self.static_obs_sectors = {}  # {sector_id: {'s_start': float, 's_end': float, 'static_obs_section': bool}}
        self.track_length = 0.0  # Will be set from GB waypoints
        self._load_static_obs_sectors()

        # Detection zone monitoring: track if vehicle has passed through zone without obstacles
        self.sector_monitoring = {}  # {sector_id: {'in_zone': bool, 'had_obstacles': bool, 'entered_at_s': float}}
        self.consecutive_cleared_sectors = 0  # Track how many consecutive sectors were cleared without obstacles

        # Sector-specific interference tracking: remember which sectors had interfering obstacles during path generation
        self.sectors_with_interfering_obs = set()  # Set of sector_ids that had interfering obstacles when fixed path was generated
        # ===== HJ ADDED END =====

        # Phase B: Position-based obstacle merging (in case TTL causes ID changes)
        self.use_position_based_merge = True  # Set False if tracking TTL is long enough
        self.position_merge_threshold = 0.05  # 5cm - obstacles closer than this are considered same

        # Phase A: Callback lightweight variables
        self.latest_obstacles = []
        self.obstacles_timestamp = self.get_clock().now().to_msg()
        self.obstacles_updated = False

        # Phase 3: GB optimizer configuration
        self.obstacle_shift_s_range = 2.0  # [m] shift waypoints within ±2m of obstacle s position
        self.obstacle_wall_margin = 0.1  # [m] margin when adding obstacle perimeter to wall boundary
        self.obstacle_wall_points = 100  # number of points on obstacle perimeter
        self.gb_optimizer_plot = self._get_param_or_default(f'/{self.name}/gb_optimizer_plot', False)  # Show plots for debugging

        # Get racecar_version and map_name for locating config files (same as mapping.launch)
        self.racecar_version = self._get_param_or_default('racecar_version', 'SRX1')
        self.map_name = self._get_param_or_default('/map')  # Map name from /map parameter
        self.map_dir = os.path.join(
            os.path.dirname(__file__),
            '../../../stack_master/maps',
            self.map_name
        )

        # Load GB optimizer parameters (loaded from global_planner_params.yaml in launch file)
        # ===== HJ MODIFIED: Allow global override of safety_width =====
        if OVERRIDE_SAFETY_WIDTH:
            self.safety_width = SAFETY_WIDTH_VALUE
            self.get_logger().warning(f"[{self.name}] OVERRIDE_SAFETY_WIDTH=True → Using safety_width={SAFETY_WIDTH_VALUE}m (ignoring ROS param)")
        else:
            self.safety_width = self._get_param_or_default('~safety_width', 1.2)  # [m] safety width for GB optimizer
            self.get_logger().info(f"[{self.name}] Using ROS parameter safety_width={self.safety_width}m")
        # ===== HJ MODIFIED END =====

        # ===== HJ ADDED: Load vehicle dynamics parameters for velocity calculation =====
        import configparser
        parser = configparser.ConfigParser()
        ini_path = os.path.join(get_package_share_directory('stack_master'), 'config', self.racecar_version, 'racecar_f110.ini')
        if not parser.read(ini_path):
            self.get_logger().error(f"[{self.name}] Failed to read {ini_path}")
        else:
            self.pars = {}
            self.pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
            self.pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

            # Load vehicle dynamics info (ggv, ax_max_machines)
            ggv_path = os.path.join(get_package_share_directory('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ggv.csv")
            ax_max_path = os.path.join(get_package_share_directory('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "ax_max_machines.csv")
            b_ax_max_path = os.path.join(get_package_share_directory('stack_master'), 'config', self.racecar_version, "veh_dyn_info", "b_ax_max_machines.csv")

            self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
                ggv_import_path=ggv_path,
                ax_max_machines_import_path=ax_max_path
            )
            _, self.b_ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
                ggv_import_path=ggv_path,
                ax_max_machines_import_path=b_ax_max_path
            )

            self.get_logger().info(f"[{self.name}] Loaded vehicle dynamics parameters from {ini_path}")
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Safe grid filtering with conservative bounds (MUST be before Subscribers) =====
        self.conservative_map_base = None  # _for_spliner.png (loaded once)
        self.conservative_map_with_obs = None  # _for_spliner_with_obs.png (updated when obstacle added)
        self.conservative_bounds = {}  # {s_idx: (d_left, d_right)} for GB waypoints
        self.conservative_map_loaded = False
        self.obstacle_added_to_map = False  # Flag: has obstacle been added to map?
        self.conservative_map_base_path = ""  # Path to _for_spliner.png
        self.conservative_map_with_obs_path = ""  # Path to _for_spliner_with_obs.png
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Joy button to force GB mode =====
        self.joy_force_gb = False  # True when button pressed → force GB mode
        self.joy_force_gb_prev = False  # Previous button state for edge detection
        # ===== HJ ADDED END =====

        # Subscribe to the topics
        # ===== HJ EDITED START: Subscribe to tracking obstacles for smart static avoidance =====
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.tracking_obs_cb, 10)
        # ===== HJ EDITED END =====
        self.create_subscription(BehaviorStrategy, "/behavior_strategy", self.behavior_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.state_frenet_cb, 10)
        self.create_subscription(Odometry, "/car_state/odom", self.state_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints", self.gb_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.gb_scaled_cb, 10)
        # ===== HJ ADDED: Joy subscriber for force GB mode =====
        if JOY_FORCE_GB_BUTTON >= 0:
            self.create_subscription(Joy, "/joy", self.joy_cb, 10)
            self.get_logger().info(f"[{self.name}] Joy force GB mode enabled: button {JOY_FORCE_GB_BUTTON}")
        # ===== HJ ADDED END =====
        # dyn params sub
        self.evasion_dist = 0.65
        self.obs_traj_tresh = 0.3
        self.spline_bound_mindist = 0.2
        self.n_loc_wpnts = 80
        self.width_car = 0.30

        # ===== HJ EDITED START: Params for obstacle interference check (same as state_machine) =====
        self.gb_ego_width_m = self._get_param_or_default("state_machine/gb_ego_width_m", 0.3)
        self.lateral_width_gb_m = self._get_param_or_default("state_machine/lateral_width_gb_m", 0.1)
        # ===== HJ EDITED END =====

        if not self.from_bag:
            self.create_subscription(Config, "/dynamic_spline_tuner_node/parameter_updates", self.dyn_param_cb, 10)
            # ===== HJ EDITED START: Subscribe to state machine dynamic params =====
            self.create_subscription(Config, "/dyn_statemachine/parameter_updates", self.state_machine_dyn_param_cb, 10)
            # ===== HJ EDITED END =====

        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/otwpnts", 10)
        self.closest_obs_pub = self.create_publisher(Marker, "/planner/avoidance/considered_OBS", 10)
        self.pub_propagated = self.create_publisher(Marker, "/planner/avoidance/propagated_obs", 10)
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/planner/avoidance/latency", 10)

        # Smart static avoidance publishers (fixed path)
        self.fixed_path_pub = self.create_publisher(OTWpntArray, "/planner/avoidance/smart_static_otwpnts", 10)
        self.fixed_path_mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/smart_markers", 10)
        # ===== HJ ADDED: Publisher for use_fixed_path flag =====
        self.use_fixed_path_pub = self.create_publisher(Bool, "/planner/avoidance/smart_static_active", 10)
        self.smart_active_marker_pub = self.create_publisher(MarkerArray, "/planner/avoidance/smart_active_marker", 10)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Publisher for obstacles-only map visualization =====
        from nav_msgs.msg import OccupancyGrid
        self.map_with_obs_pub = rospy.Publisher("/map_with_obs", OccupancyGrid, queue_size=1, latch=True)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Publisher for do_spline samples visualization =====
        self.temp_spline_markers_pub = self.create_publisher(MarkerArray, "/planner/avoidance/temp_do_spline_markers", 10)
        # ===== HJ ADDED END =====

        # ===== HJ ADDED: Publisher for _more_space() d_left/d_right bounds debug visualization =====
        self.more_space_debug_pub = self.create_publisher(MarkerArray, "/planner/avoidance/more_space_debug", 10)
        # ===== HJ ADDED END =====

        self.converter = self.initialize_converter()

        # Set the rate at which the loop runs
        self.rate = rospy.Rate(20)  # Hz

        # ===== HJ DEBUG: Counter for saving do_spline inputs =====
        self._debug_save_counter = 0
        # ===== HJ DEBUG END =====

        # ===== HJ ADDED: Cache for original wall boundaries =====
        self._original_bound_r = None
        self._original_bound_l = None
        # ===== HJ ADDED END =====

    #############
    # CALLBACKS #
    #############
    # ===== HJ EDITED START: Tracking obstacles callback for smart static avoidance =====
    def tracking_obs_cb(self, data: ObstacleArray):
        """
        Callback for /tracking/obstacles topic.
        Phase A: Lightweight callback - only store data, no computation.
        All processing moved to _process_obstacles() called from main loop.
        """
        # Simply store latest data and set flag
        self.latest_obstacles = data.obstacles
        self.obstacles_timestamp = data.header.stamp
        self.obstacles_updated = True
    # ===== HJ PHASE A: Callback lightened =====

    # ===== HJ ADDED: Joy callback for force GB mode =====
    def joy_cb(self, data: Joy):
        """
        Callback for /joy topic.

        Force GB mode permanently when JOY_FORCE_GB_BUTTON is pressed (one-shot).
        - Before optimization: prevent optimization from ever running
        - After optimization: revert to GB mode permanently
        Once activated, cannot be undone (restart node to reset).
        """
        if JOY_FORCE_GB_BUTTON < 0 or JOY_FORCE_GB_BUTTON >= len(data.buttons):
            return

        # Already forced → ignore further presses
        if self.joy_force_gb:
            return

        button_pressed = data.buttons[JOY_FORCE_GB_BUTTON] == 1

        # Edge detection: only activate on rising edge (button press)
        if button_pressed and not self.joy_force_gb_prev:
            self.joy_force_gb = True  # Permanent, no toggle back

            self.get_logger().warning(f"[{self.name}] ===== JOY FORCE GB MODE: ACTIVATED (PERMANENT) =====")
            if self.fixed_path_generated:
                # Already optimized → revert to GB permanently
                self.get_logger().warning(f"[{self.name}] Fixed path exists → Reverting to GB mode permanently")
                self.use_fixed_path = False
                self.published_flag = False
                self.use_fixed_path_pub.publish(Bool(data=False))
            else:
                # Not yet optimized → prevent optimization forever
                self.get_logger().warning(f"[{self.name}] No fixed path yet → Optimization will be permanently skipped")

        self.joy_force_gb_prev = button_pressed
    # ===== HJ ADDED END =====

    def behavior_cb(self, data: BehaviorStrategy):
        """
        Callback for /behavior_strategy topic.

        Track current state for safe flag transition timing.
        """
        # ===== HJ ADDED: Track state for safe flag management =====
        self.current_state_name = data.state
        # ===== HJ ADDED END =====


    def state_frenet_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x

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
            # ===== HJ ADDED: Set track length for sector checking =====
            self.track_length = self.gb_max_s
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Load original bounds for _more_space() fallback =====
            if not hasattr(self, 'bound_l_original'):
                self.get_logger().info(f"[{self.name}] Loading original bounds for _more_space() function...")
                bound_r_original, bound_l_original = self._get_original_wall_boundaries()
                if bound_r_original is not None and bound_l_original is not None:
                    self.bound_r_original = bound_r_original
                    self.bound_l_original = bound_l_original
                    self.get_logger().info(f"[{self.name}] Loaded original bounds: {len(bound_r_original)} right, {len(bound_l_original)} left")
                else:
                    self.get_logger().warning(f"[{self.name}] Failed to load original bounds")
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Calculate GB conservative bounds from _for_spliner.png =====
            if USE_SAFE_GRID and not self.conservative_map_loaded:
                conservative_base_path = os.path.join(self.map_dir, f'{self.map_name}_for_spliner.png')
                yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')

                # Calculate GB conservative bounds
                self.gb_conservative_d_left, self.gb_conservative_d_right = \
                    self._calculate_conservative_bounds_from_map(
                        map_path=conservative_base_path,
                        waypoints=data.wpnts,
                        mode="GB"
                    )
                self.get_logger().info(f"[{self.name}] GB conservative bounds calculated: {len(self.gb_conservative_d_left)} waypoints")

                # Create GB conservative GridFilter from _for_spliner.png
                self.map_filter_conservative_base = GridFilter(map_topic=None, debug=False)
                if self.map_filter_conservative_base.load_from_file(conservative_base_path, yaml_path):
                    self.map_filter_conservative_base.set_erosion_kernel_size(self.kernel_size)
                    self.get_logger().info(f"[{self.name}] Created GB conservative GridFilter from {conservative_base_path}")
                else:
                    self.get_logger().warning(f"[{self.name}] Failed to load GB conservative GridFilter")
                    self.map_filter_conservative_base = None

                # Load conservative bounds (no obstacles) for GB mode USE_SAFE_GRID=True
                bound_r_conservative, bound_l_conservative = self._get_conservative_boundaries()
                if bound_r_conservative is not None and bound_l_conservative is not None:
                    self.bound_r_conservative = bound_r_conservative
                    self.bound_l_conservative = bound_l_conservative
                    self.get_logger().info(f"[{self.name}] Loaded conservative bounds: {len(bound_r_conservative)} right, {len(bound_l_conservative)} left")
                else:
                    self.get_logger().warning(f"[{self.name}] Failed to load conservative bounds")

                self.conservative_map_loaded = True
            # ===== HJ ADDED END =====

    # Callback for scaled global waypoint topic
    def gb_scaled_cb(self, data: WpntArray):
        self.gb_scaled_wpnts = data

    # ===== HJ ADDED: Load static obstacle sectors from ROS params =====
    def _load_static_obs_sectors(self):
        """Load static obstacle sectors from ROS parameters (same as detect.cpp)"""
        self.static_obs_sectors.clear()

        n_sectors = self._get_param_or_default("/static_obs_map_params/n_sectors", 0)
        if n_sectors == 0:
            self.get_logger().warning(f"[{self.name}] Static obs sectors param '/static_obs_map_params/n_sectors' not found, defaulting to 0")
            return

        for i in range(n_sectors):
            sector_key = f"/static_obs_map_params/Static_Obs_sector{i}"

            try:
                s_start = self._get_param_or_default(f"{sector_key}/s_start")
                s_end = self._get_param_or_default(f"{sector_key}/s_end")
                static_obs_section = self._get_param_or_default(f"{sector_key}/static_obs_section")

                self.static_obs_sectors[i] = {
                    's_start': s_start,
                    's_end': s_end,
                    'static_obs_section': static_obs_section
                }

                self.get_logger().info(f"[{self.name}] Loaded sector {i}: s_start={s_start:.2f}, s_end={s_end:.2f}, static_obs_section={static_obs_section}")
            except KeyError as e:
                self.get_logger().warning(f"[{self.name}] Failed to load sector {i} parameters: {e}")

        self.get_logger().info(f"[{self.name}] Loaded {len(self.static_obs_sectors)} static obstacle sectors")

    def _is_position_in_sector(self, s_position, sector_data):
        """Check if s_position is within sector bounds (handling wrap-around)

        Args:
            s_position: Current s coordinate to check
            sector_data: Dict with 's_start', 's_end', 'static_obs_section'

        Returns:
            bool: True if position is in sector
        """
        if self.track_length == 0.0:
            return False

        # Normalize s to [0, track_length)
        s_normalized = s_position
        while s_normalized < 0:
            s_normalized += self.track_length
        while s_normalized >= self.track_length:
            s_normalized -= self.track_length

        s_start = sector_data['s_start']
        s_end = sector_data['s_end']

        # Check if position is within sector bounds
        if s_start <= s_end:
            # Normal case: sector doesn't wrap around
            in_sector = (s_normalized >= s_start and s_normalized <= s_end)
        else:
            # Wrap-around case: sector crosses track start/end
            in_sector = (s_normalized >= s_start or s_normalized <= s_end)

        return in_sector
    # ===== HJ ADDED END =====

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: Config):
        """
        Notices the change in the parameters and changes spline params
        """
        self.evasion_dist = self._get_param_or_default("dynamic_spline_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = self._get_param_or_default("dynamic_spline_tuner_node/obs_traj_tresh", 0.3)
        self.spline_bound_mindist = self._get_param_or_default("dynamic_spline_tuner_node/spline_bound_mindist", 0.2)

        self.kd_obs_pred = self._get_param_or_default("dynamic_spline_tuner_node/kd_obs_pred")
        self.fixed_pred_time = self._get_param_or_default("dynamic_spline_tuner_node/fixed_pred_time")
        self.sampling_dist = self._get_param_or_default("dynamic_spline_tuner_node/post_sampling_dist")
        self.spline_scale = self._get_param_or_default("dynamic_spline_tuner_node/spline_scale")
        self.post_min_dist = self._get_param_or_default("dynamic_spline_tuner_node/post_min_dist")
        self.post_max_dist = self._get_param_or_default("dynamic_spline_tuner_node/post_max_dist")
        self.kernel_size = self._get_param_or_default("dynamic_spline_tuner_node/kernel_size")
        
        self.map_filter.set_erosion_kernel_size(self.kernel_size)

        self.get_logger().info(
            f"[{self.name}] evasion apex distance: {self.evasion_dist} [m],\n"
            f" obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" obstacle prediciton k_d: {self.kd_obs_pred},    obstacle prediciton constant time: {self.fixed_pred_time} [s] "
        )

    # ===== HJ EDITED START: State machine dynamic param callback =====
    def state_machine_dyn_param_cb(self, params: Config):
        """
        Update params from state_machine dynamic reconfigure (same params for consistency)
        """
        self.lateral_width_gb_m = self._get_param_or_default("dyn_statemachine/lateral_width_gb_m", 0.1)
        self.get_logger().info(f"[{self.name}] Updated lateral_width_gb_m: {self.lateral_width_gb_m}")
    # ===== HJ EDITED END =====

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        # Wait for critical Messages and services
        self.get_logger().info(f"[{self.name}] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dynamic_spline_tuner_node/parameter_updates", Config)
        self.get_logger().info(f"[{self.name}] Ready!")

        while not rospy.is_shutdown():
            if self.measuring:
                start = time.perf_counter()

            # ===== HJ PHASE A: Process obstacles in main loop =====
            self._process_obstacles()
            # ===== HJ PHASE A END =====

            # ===== HJ ADDED: State-aware flag management - publish when safe =====
            # Check if we have pending flag changes and can safely publish them
            # ===== HJ MODIFIED: Joy force GB overrides use_fixed_path =====
            desired_flag = self.use_fixed_path and self.fixed_path_generated and not self.joy_force_gb

            # Detect if internal state changed (want to publish new value)
            if desired_flag != self.published_flag:
                # Check if current state is safe for flag transition (not OVERTAKE)
                if self.current_state_name != "OVERTAKE":
                    # Safe to publish - do it now
                    self.get_logger().info(f"[{self.name}] Publishing smart_static_active={desired_flag} (state={self.current_state_name})")
                    self.use_fixed_path_pub.publish(Bool(data=desired_flag))
                    self.published_flag = desired_flag
                    # Clear pending flags
                    self.pending_flag_true = False
                    self.pending_flag_false = False
                else:
                    # Not safe (in OVERTAKE) - set pending flag and wait
                    if desired_flag:
                        if not self.pending_flag_true:
                            self.get_logger().warning(f"[{self.name}] Want to activate flag but in OVERTAKE state - waiting...")
                        self.pending_flag_true = True
                        self.pending_flag_false = False
                    else:
                        if not self.pending_flag_false:
                            self.get_logger().warning(f"[{self.name}] Want to deactivate flag but in OVERTAKE state - waiting...")
                        self.pending_flag_false = True
                        self.pending_flag_true = False
            # ===== HJ ADDED END =====

            # Sample data
            gb_scaled_wpnts = self.gb_scaled_wpnts.wpnts
            wpnts = OTWpntArray()
            mrks = MarkerArray()

            # ===== HJ EDITED START: Smart static avoidance mode switching =====
            # Check if we should generate fixed path (only if not attempted AND not generating AND not joy force GB)
            if not self.fixed_path_generation_attempted and not self.fixed_path_generating and not self.joy_force_gb and self._check_obs_is_ready_for_path_gen():
                self.get_logger().info(f"[{self.name}] ========================================")
                self.get_logger().info(f"[{self.name}] OBSTACLES VERIFIED! Starting fixed path generation in background...")
                self.get_logger().info(f"[{self.name}] ========================================")

                # Get verified obstacles for path generation
                verified_obs, nearby_obs = self._get_verified_obstacles()

                self.get_logger().info(f"[{self.name}] Verified interfering obstacles: {len(verified_obs)}")
                for sector_id, obs_id, s, d in verified_obs:
                    self.get_logger().info(f"[{self.name}]   Sector {sector_id}, Obs {obs_id}: s={s:.2f}m, d={d:.2f}m")

                # Set generating flag BEFORE starting thread
                self.fixed_path_generating = True

                # Phase 3: Generate fixed optimized path using GB optimizer in SEPARATE THREAD
                # This allows main loop to continue running at 20Hz for real-time spline avoidance
                generation_thread = threading.Thread(
                    target=self._generate_fixed_path_async,
                    args=(verified_obs, nearby_obs),
                    daemon=True
                )
                generation_thread.start()
                self.get_logger().info(f"[{self.name}] Path generation thread started. Continuing real-time avoidance...")

            # Real-time spline avoidance (always active)
            if self.obs_in_interest is not None:
                # Select reference path based on use_fixed_path flag
                # ===== HJ MODIFIED: Extract .wpnts list from OTWpntArray =====
                # ===== HJ MODIFIED: Joy force GB overrides use_fixed_path =====
                use_fixed = self.use_fixed_path and not self.joy_force_gb
                if use_fixed:
                    reference_wpnts = self.fixed_path_wpnts.wpnts  # Extract list from OTWpntArray
                    rospy.loginfo_throttle(5.0, f"[{self.name}] POST-FIX (Fixed mode): Spline avoidance around FIXED path")
                else:
                    reference_wpnts = gb_scaled_wpnts  # Already a list
                    if self.joy_force_gb:
                        rospy.logwarn_throttle(5.0, f"[{self.name}] JOY FORCE GB: Using GB raceline (ignoring obstacles)")
                    elif self.fixed_path_generating:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] PRE-FIX: Generating... Using GB raceline")
                    elif self.fixed_path_generated:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] POST-FIX (GB mode): Using GB raceline")
                    else:
                        rospy.loginfo_throttle(5.0, f"[{self.name}] PRE-FIX: Using GB raceline")
                # ===== HJ MODIFIED END =====

                # ===== HJ DEBUG: Save do_spline inputs for offline analysis =====
                if self.use_fixed_path:
                    self._debug_save_counter += 1
                    if self._debug_save_counter == 50:  # Save only first successful call
                        import pickle
                        debug_data = {
                            'obs': copy.deepcopy(self.obs_in_interest),
                            'gb_wpnts': reference_wpnts[:],  # Fixed path waypoints (copy list)
                            'use_fixed_path': self.use_fixed_path,
                            'cur_x': self.cur_x,
                            'cur_y': self.cur_y,
                            'cur_yaw': self.cur_yaw,  # Add heading!
                            'cur_s': self.cur_s,
                            'cur_d': self.cur_d,
                            'boundary_left_xy': copy.deepcopy(self.fixed_boundary_left_xy) if hasattr(self, 'fixed_boundary_left_xy') else None,
                            'boundary_right_xy': copy.deepcopy(self.fixed_boundary_right_xy) if hasattr(self, 'fixed_boundary_right_xy') else None
                        }
                        debug_path = '/tmp/do_spline_input_fixed.pkl'
                        with open(debug_path, 'wb') as f:
                            pickle.dump(debug_data, f)
                        # self.get_logger().warning(f"[{self.name}] Saved do_spline input to {debug_path}")
                # ===== HJ DEBUG END =====

                wpnts, mrks = self.do_spline(obs=copy.deepcopy(self.obs_in_interest), gb_wpnts=reference_wpnts)

                # ===== HJ ADDED: Debug log for do_spline result =====
                # if len(wpnts.wpnts) == 0:
                #     rospy.logwarn_throttle(2.0,
                #         f"[{self.name}] DEBUG: do_spline returned EMPTY waypoints! "
                #         f"Mode: {'FIXED' if self.use_fixed_path else 'GB'}, "
                #         f"Obs s={self.obs_in_interest.s_center:.2f}, d={self.obs_in_interest.d_center:.2f}")
                # else:
                #     rospy.loginfo_throttle(2.0,
                #         f"[{self.name}] DEBUG: do_spline SUCCESS! "
                #         f"Generated {len(wpnts.wpnts)} waypoints, "
                #         f"Mode: {'FIXED' if self.use_fixed_path else 'GB'}")
                # ===== HJ ADDED END =====
            else:
                # No obstacle in interest, delete markers
                del_mrk = Marker()
                del_mrk.header.stamp = self.get_clock().now().to_msg()
                del_mrk.action = Marker.DELETEALL
                mrks.markers.append(del_mrk)

            # Publish real-time spline wpnts and markers
            if self.measuring:
                end = time.perf_counter()
                self.latency_pub.publish(end - start)
            self.evasion_pub.publish(wpnts)
            self.mrks_pub.publish(mrks)

            # Publish fixed path separately (if generated) at low rate (0.5Hz)
            # ===== HJ MODIFIED: Low-rate publishing for static data =====
            if self.fixed_path_generated:
                time_since_last_pub = (self.get_clock().now().to_msg() - self.fixed_path_last_pub_time).to_sec()
                if time_since_last_pub > self.fixed_path_pub_rate:
                    # ===== HJ REMOVED: Don't update timestamp on republish (Smart path never changes) =====
                    # Timestamp is set once at creation (line 3398), then kept constant
                    # This prevents conflicts with global_velocity_planner republishing with same timestamp
                    # self.fixed_path_wpnts.header.stamp = self.get_clock().now().to_msg()
                    # self.fixed_path_markers.markers[0].header.stamp = self.get_clock().now().to_msg() if len(self.fixed_path_markers.markers) > 0 else self.get_clock().now().to_msg()
                    # ===== HJ REMOVED END =====
                    # self.get_logger().info(f"[{self.name}] Publishing FIXED optimized path ({len(self.fixed_path_wpnts.wpnts)} waypoints) at {1.0/self.fixed_path_pub_rate:.1f}Hz")
                    self.fixed_path_pub.publish(self.fixed_path_wpnts)
                    self.fixed_path_mrks_pub.publish(self.fixed_path_markers)
                    self.fixed_path_last_pub_time = self.get_clock().now().to_msg()
            # ===== HJ MODIFIED END =====

            # ===== HJ ADDED: Publish SMART_ACTIVE marker next to STATE MARKER =====
            # NOTE: use_fixed_path flag is now published in state-aware section above (lines 419-446)
            self._publish_smart_active_marker()
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Publish _more_space() bounds debug visualization =====
            if DEBUG_MORE_SPACE:
                self._publish_more_space_debug()
            # ===== HJ ADDED END =====

            # ===== HJ EDITED END =====
            self.rate.sleep()
    
    #########
    # UTILS #
    #########
    # ===== HJ EDITED START: Check if obstacle interferes with global raceline =====
    def _check_obstacle_interference(self, obs: Obstacle) -> bool:
        """
        Check if obstacle interferes with global raceline.
        Uses same logic as state_machine's _check_free() for consistency,
        but with FIXED scaling_factor=1.0 (no distance-based scaling).

        This ensures we detect interference early and generate path once.

        NOTE: Potential issue - obstacles that don't interfere with global raceline
        might interfere with the optimized path we generate. This needs to be monitored.

        Returns True if obstacle interferes (needs avoidance).
        """
        if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            return False

        # Get global waypoint at obstacle's s position
        wpnt_dist = self.gb_wpnts.wpnts[1].s_m - self.gb_wpnts.wpnts[0].s_m
        obs_s_idx = int(obs.s_center / wpnt_dist) % self.gb_max_idx
        gb_wpnt = self.gb_wpnts.wpnts[obs_s_idx]

        # Distance from raceline (d=0) to obstacle center
        # For global waypoints, d_m is the lateral offset from centerline
        min_dist = abs(gb_wpnt.d_m - obs.d_center) if hasattr(gb_wpnt, 'd_m') else abs(0 - obs.d_center)

        # Free distance calculation (same as state_machine)
        free_dist = min_dist - obs.size/2 - self.gb_ego_width_m/2

        # Fixed scaling_factor = 1.0 (no distance-based scaling for static obstacles)
        # This ensures consistent interference detection regardless of distance
        scaling_factor = 1.0

        # Check interference (same logic as state_machine)
        if free_dist < self.lateral_width_gb_m * scaling_factor:
            return True  # Interference detected!
        else:
            return False  # No interference
    # ===== HJ EDITED END =====

    # ===== HJ EDITED START: Check if obstacle interferes with fixed path =====
    def _check_interference_fixed_path(self, obs: Obstacle) -> bool:
        """
        Check if obstacle interferes with fixed path (Single Frenet approach).
        Uses FrenetConverter to calculate lateral distance from fixed path.

        This is used to detect NEW obstacles that interfere with the GENERATED fixed path,
        not the original GB raceline. When interference is detected, regeneration is triggered.

        Returns True if obstacle interferes with fixed path.
        """
        # Fixed path not generated yet
        if self.fixed_converter is None:
            return False

        # Convert obstacle position to fixed path Frenet coordinates
        result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
        s_fixed, d_fixed = result[0], result[1]

        # Calculate free distance from fixed path
        free_dist = abs(d_fixed) - obs.size/2 - self.gb_ego_width_m/2

        # Check interference (same threshold as GB raceline check)
        scaling_factor = 1.0  # Fixed scaling
        if free_dist < self.lateral_width_gb_m * scaling_factor:
            return True  # Interference with fixed path!
        else:
            return False  # No interference
    # ===== HJ EDITED END =====

    # ===== HJ PHASE D: Get verified obstacles from memory (interfering only) =====
    def _get_verified_obstacles(self) -> Tuple[List[Tuple[int, int, float, float]], List[Tuple[int, int, float, float]]]:
        """
        Phase D: Extract verified obstacles from memory for path generation.

        Returns TWO lists:
        1. verified_obs: INTERFERING obstacles for GB optimizer (blocking path)
        2. nearby_obs: ALL stable obstacles within 2.0m of GB raceline (for spline occupancy grid)

        Both lists contain tuples: [(sector_id, obs_id, s_mean, d_mean), ...]
        Only includes obstacles that are stable (verified).

        SIDE EFFECT: Stores sector IDs with interfering obstacles in self.sectors_with_interfering_obs
        """
        verified_obs = []  # Interfering obstacles for GB optimizer
        nearby_obs = []    # All nearby obstacles for spline
        interfering_sectors = set()  # Track which sectors have interfering obstacles

        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            # Check if this obstacle has enough observations
            if mem['obs_count'] < self.min_stable_observations:
                continue

            # Check position stability
            if len(mem['s_history']) < self.min_stable_observations:
                continue

            s_std = np.std(mem['s_history'])
            d_std = np.std(mem['d_history'])

            # Only process verified (stable) obstacles
            if s_std < self.position_std_threshold and d_std < self.position_std_threshold:
                s_mean = np.mean(mem['s_history'])
                d_mean = np.mean(mem['d_history'])

                # Add to verified_obs if interfering (original logic)
                if mem['interferes']:
                    verified_obs.append((sector_id, obs_id, s_mean, d_mean))
                    interfering_sectors.add(sector_id)
                    self.get_logger().info(f"[{self.name}] Verified INTERFERING obstacle: "
                                 f"sector={sector_id}, id={obs_id}, s={s_mean:.2f}, d={d_mean:.2f}")

                # Add to nearby_obs if within 2.5m of GB raceline (for spline)
                if abs(d_mean) <= 2.5:
                    nearby_obs.append((sector_id, obs_id, s_mean, d_mean))
                    rospy.logdebug(f"[{self.name}] Nearby obstacle for spline: "
                                  f"sector={sector_id}, id={obs_id}, s={s_mean:.2f}, d={d_mean:.2f}")

        # Store which sectors had interfering obstacles (will be used in zone passage monitoring)
        self.sectors_with_interfering_obs = interfering_sectors
        self.get_logger().info(f"[{self.name}] Sectors with interfering obstacles: {interfering_sectors}")
        self.get_logger().info(f"[{self.name}] Obstacle counts: verified={len(verified_obs)}, nearby={len(nearby_obs)}")

        return verified_obs, nearby_obs
    # ===== HJ PHASE D END =====

    # ===== HJ PHASE C: Check if obstacles are ready for path generation (2-stage condition) =====
    def _check_obs_is_ready_for_path_gen(self) -> bool:
        """
        Phase C: 2-stage path generation condition.

        Stage 1 (is_ready): 2+ sectors with stable obstacles (interference irrelevant)
        Stage 2 (should_generate): Has interfering obstacles to optimize for

        Returns True only if BOTH conditions met.
        """
        # Count sectors with verified obstacles
        sectors_with_verified_obs = set()
        interfering_count = 0

        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            # Check if this obstacle is verified (stable)
            if mem['obs_count'] < self.min_stable_observations:
                continue  # Not enough observations yet

            if len(mem['s_history']) < self.min_stable_observations:
                continue  # Not enough history

            s_std = np.std(mem['s_history'])
            d_std = np.std(mem['d_history'])

            # Check if position is verified (stable)
            if s_std < self.position_std_threshold and d_std < self.position_std_threshold:
                sectors_with_verified_obs.add(sector_id)

                if mem['interferes']:
                    interfering_count += 1

                # rospy.loginfo_throttle(2.0,
                #     f"[{self.name}] Sector {sector_id}, Obs {obs_id} VERIFIED: "
                #     f"s_std={s_std:.4f}, d_std={d_std:.4f}, interferes={mem['interferes']}")

        # Stage 1: is_ready
        is_ready = len(sectors_with_verified_obs) >= self.min_sectors_with_stable_obs

        # Stage 2: should_generate
        should_generate = interfering_count > 0

        if is_ready and should_generate:
            self.get_logger().info(f"[{self.name}] READY & SHOULD GENERATE! "
                         f"{len(sectors_with_verified_obs)} sectors, {interfering_count} interfering obstacles")
            return True
        elif is_ready and not should_generate:
            # rospy.loginfo_throttle(5.0,
            #     f"[{self.name}] Ready but no interference "
            #     f"({len(sectors_with_verified_obs)} sectors, 0 interfering)")
            return False
        else:
            # rospy.loginfo_throttle(5.0,
            #     f"[{self.name}] Waiting: {len(sectors_with_verified_obs)}/{self.min_sectors_with_stable_obs} sectors")
            return False
    # ===== HJ PHASE C END =====

    # ===== HJ PHASE B: Find existing obstacle key by position =====
    def _find_obstacle_key_by_position(self, obs: Obstacle):
        """
        Find existing obstacle in memory by position (in case ID changed due to TTL).
        Returns existing key if nearby obstacle found, otherwise returns new key.

        This handles the case where tracking loses an obstacle temporarily and
        reassigns a different ID when it reappears at the same location.
        """
        if not self.use_position_based_merge:
            return (obs.sector_id, obs.id)  # Just use current ID

        # Search for nearby obstacles in same sector
        for (sector_id, old_id), mem in self.static_obs_memory.items():
            if sector_id != obs.sector_id:
                continue  # Different sector

            # Check if position is very close to last observation
            if len(mem['s_history']) > 0:
                last_s = mem['s_history'][-1]
                last_d = mem['d_history'][-1]

                # Euclidean distance in Frenet space
                dist = np.sqrt((obs.s_center - last_s)**2 + (obs.d_center - last_d)**2)

                if dist < self.position_merge_threshold:
                    # Same obstacle, different ID!
                    # rospy.loginfo_throttle(5.0,
                    #     f"[{self.name}] Merged obstacle: old_id={old_id} → new_id={obs.id} "
                    #     f"(sector={sector_id}, dist={dist:.3f}m)")
                    return (sector_id, old_id)  # Reuse old key

        # No nearby obstacle found, use new key
        return (obs.sector_id, obs.id)
    # ===== HJ PHASE B END =====

    # ===== HJ PHASE A: Process obstacles function =====
    def _process_obstacles(self):
        """
        Process latest obstacles from callback.
        Called from main loop to avoid callback overload.

        Tasks:
        1. Filter obstacles (in_static_obs_sector AND is_static)
        2. Update memory for ALL filtered obstacles (interference check stored)
        3. Set obs_in_interest for PRE-FIX mode
        """
        if not self.obstacles_updated:
            return

        # ===== HJ ADDED: Race start detection - ignore obstacles before race starts =====
        if not self.race_started:
            if self.cur_vs >= self.race_start_speed_threshold:
                self.race_started = True
                self.get_logger().warning(f"[{self.name}] ====== RACE STARTED! ======")
                self.get_logger().warning(f"[{self.name}] Vehicle speed: {self.cur_vs:.2f} m/s (threshold: {self.race_start_speed_threshold} m/s)")
                self.get_logger().warning(f"[{self.name}] Static obstacle detection now ACTIVE")
            else:
                # Race not started yet, ignore all obstacles
                rospy.loginfo_throttle(5.0,
                    f"[{self.name}] Waiting for race start... "
                    f"Speed: {self.cur_vs:.2f} m/s < {self.race_start_speed_threshold} m/s. "
                    f"Ignoring all obstacles.")
                return
        # ===== HJ ADDED END =====

        # ===== DEBUG: Log all obstacles (throttled) =====
        # if len(self.latest_obstacles) > 0:
        #     rospy.loginfo_throttle(2.0, f"[{self.name}] ===== OBSTACLE DEBUG =====")
        #     for obs in self.latest_obstacles:
        #         rospy.loginfo_throttle(2.0,
        #             f"[{self.name}] OBS: id={obs.id}, "
        #             f"sector_id={obs.sector_id}, "
        #             f"in_static_obs_sector={obs.in_static_obs_sector}, "
        #             f"is_static={obs.is_static}, "
        #             f"s={obs.s_center:.2f}m, d={obs.d_center:.2f}m")

        # Filter obstacles into two lists:
        # 1. static_obs_in_sector: For memory tracking (use latest_obstacles)
        # 2. static_obs_for_decision: For GB revert decision (use memory, more stable)
        # 3. all_static_obs: For do_spline (includes obstacles outside sector)

        # ===== HJ MODIFIED: Separate memory update and decision logic =====
        # For memory update: use latest_obstacles (real-time)
        static_obs_in_sector = [obs for obs in self.latest_obstacles
                                if obs.in_static_obs_sector and obs.is_static]

        # For decision: use memory (stable across callbacks)
        obstacles_from_memory = []
        for (sector_id, obs_id), mem in self.static_obs_memory.items():
            if len(mem['s_history']) > 0:
                # Create obstacle object from latest memory data
                obs = Obstacle()
                obs.sector_id = sector_id
                obs.id = obs_id
                obs.s_center = mem['s_history'][-1]  # Latest s
                obs.d_center = mem['d_history'][-1]  # Latest d
                obs.is_static = True
                obs.in_static_obs_sector = True
                obstacles_from_memory.append(obs)

        # ===== HJ ADDED: Debug logging =====
        # if self.fixed_path_generated:
        #     rospy.loginfo_throttle(1.0,
        #         f"[{self.name}] DEBUG Filter: latest_obstacles={len(self.latest_obstacles)}, "
        #         f"static_in_sector={len(static_obs_in_sector)}, "
        #         f"memory_obstacles={len(obstacles_from_memory)}, "
        #         f"use_fixed_path={self.use_fixed_path}")
        # ===== HJ ADDED END =====

        # For do_spline: use latest_obstacles (real-time avoidance)
        all_static_obs = [obs for obs in self.latest_obstacles
                          if obs.is_static]
        # ===== HJ MODIFIED END =====

        # ===== DEBUG: Log filtering results =====
        # rospy.logwarn_throttle(2.0,
        #     f"[{self.name}] Total: {len(self.latest_obstacles)}, "
        #     f"In static sector: {len(static_obs_in_sector)}, "
        #     f"All static: {len(all_static_obs)}")

        # Count obstacles per sector (only for static_obs_section: true sectors)
        active_sector_ids = [sid for sid, data in self.static_obs_sectors.items() if data.get('static_obs_section', False)]
        sector_obs_count = {sid: 0 for sid in active_sector_ids}

        for obs in static_obs_in_sector:
            sector_id = obs.sector_id
            if sector_id in sector_obs_count:
                sector_obs_count[sector_id] += 1

        # Log active sectors info
        sector_info = ", ".join([f"S{sid}:{count}" for sid, count in sorted(sector_obs_count.items())])
        rospy.logwarn_throttle(2.0,
            f"[{self.name}] Active sectors: {len(active_sector_ids)} sectors (static_obs_section=true) → {sector_info}")

        # Early return if not initialized
        if self.gb_max_s is None or self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            self.obstacles_updated = False
            return

        # Phase B: Update memory for obstacles IN SECTOR (not just interfering ones)
        # Use (sector_id, obs_id) tuple keys for individual obstacle tracking
        for obs in static_obs_in_sector:
            # Check interference with GB raceline (for path generation trigger)
            interferes_gb = self._check_obstacle_interference(obs)

            # Check interference with fixed path (for regeneration trigger)
            interferes_fixed = self._check_interference_fixed_path(obs)

            # Find appropriate key (may merge with existing obstacle if position matches)
            key = self._find_obstacle_key_by_position(obs)

            # Initialize obstacle memory if first time
            if key not in self.static_obs_memory:
                self.static_obs_memory[key] = {
                    's_history': [],
                    'd_history': [],
                    's_history_fixed': [],  # Fixed path Frenet s coordinates
                    'd_history_fixed': [],  # Fixed path Frenet d coordinates
                    'size_history': [],  # Store obstacle size for GB optimizer wall modification
                    'obs_count': 0,
                    'interferes': False,  # GB raceline interference
                    'interferes_fixed': False,  # Fixed path interference
                    'last_seen': self.get_clock().now().to_msg()
                }

            mem = self.static_obs_memory[key]
            mem['s_history'].append(obs.s_center)
            mem['d_history'].append(obs.d_center)
            mem['size_history'].append(obs.size)  # Store size
            mem['obs_count'] += 1
            mem['interferes'] = interferes_gb  # Update GB interference status
            mem['interferes_fixed'] = interferes_fixed  # Update fixed path interference status
            mem['last_seen'] = self.get_clock().now().to_msg()

            # Store fixed path Frenet coordinates (if fixed converter available)
            if self.fixed_converter is not None:
                result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
                s_fixed, d_fixed = result[0], result[1]
                mem['s_history_fixed'].append(s_fixed)
                mem['d_history_fixed'].append(d_fixed)

            # Keep only recent history
            if len(mem['s_history']) > self.max_history_length:
                mem['s_history'].pop(0)
                mem['d_history'].pop(0)
                mem['size_history'].pop(0)
                if len(mem['s_history_fixed']) > 0:
                    mem['s_history_fixed'].pop(0)
                    mem['d_history_fixed'].pop(0)

        # Phase B: Remove old obstacles (not seen for timeout duration)
        current_time = self.get_clock().now().to_msg()

        # ===== HJ ADDED: Adaptive timeout decay (after fixed path generated) =====
        if self.last_timeout_decay_time is not None:
            time_since_last_decay = (current_time - self.last_timeout_decay_time).to_sec()
            if time_since_last_decay >= self.memory_timeout_decay_interval:
                # Reduce timeout by 1 second
                old_timeout = self.memory_timeout_sec
                self.memory_timeout_sec = max(self.memory_timeout_min,
                                             self.memory_timeout_sec - self.memory_timeout_decay_amount)
                self.last_timeout_decay_time = current_time
                # self.get_logger().warning(f"[{self.name}] Memory timeout decay: {old_timeout:.1f}s → {self.memory_timeout_sec:.1f}s")
        # ===== HJ ADDED END =====

        timeout = rospy.Duration(self.memory_timeout_sec)

        # ===== HJ ADDED: Debug memory state =====
        # rospy.loginfo_throttle(1.0,
        #     f"[{self.name}] DEBUG MEMORY: {len(self.static_obs_memory)} obstacles in memory, "
        #     f"timeout={self.memory_timeout_sec}s")

        keys_to_remove = []
        for key, mem in self.static_obs_memory.items():
            time_diff = (current_time - mem['last_seen']).to_sec()
            # rospy.loginfo_throttle(1.0,
            #     f"[{self.name}] DEBUG MEM OBS: sector={key[0]}, id={key[1]}, "
            #     f"time_since_last_seen={time_diff:.2f}s, will_remove={time_diff > self.memory_timeout_sec}")

            if (current_time - mem['last_seen']) > timeout:
                keys_to_remove.append(key)
                self.get_logger().warning(f"[{self.name}] DEBUG TIMEOUT: Removing obstacle sector={key[0]}, id={key[1]}, "
                             f"time_diff={time_diff:.2f}s, timeout={self.memory_timeout_sec}s")
        # ===== HJ ADDED END =====

        for key in keys_to_remove:
            del self.static_obs_memory[key]
            # self.get_logger().info(f"[{self.name}] Removed stale obstacle: sector={key[0]}, id={key[1]}")

        # ===== Phase: Decide use_fixed_path and set obs_in_interest =====
        if self.fixed_path_generated:
            # ===== HJ ADDED: Vehicle position-based active check =====
            # POST-FIX mode: Revert to GB only if vehicle is in sector but no interfering obstacles detected
            # If vehicle is in sector but no interfering obstacles detected, revert to GB
            # (Handles "memory forgot" scenario after vehicle stop/restart)
            if self.use_fixed_path and len(self.static_obs_sectors) > 0:
                vehicle_in_sector = False
                vehicle_sector_id = None

                # Check if vehicle is in any ACTIVE sector's detection zone
                # Detection zone: [3m before sector start, 15% into sector]
                # NEW Logic: Monitor zone passage - only revert if vehicle passes ENTIRE zone without obstacles
                sector_start_margin = 0 # meters - start checking ~m BEFORE sector start
                sector_check_threshold = 0.5  # Check up to ~% into sector

                for sector_id, sector_data in self.static_obs_sectors.items():
                    if not sector_data['static_obs_section']:
                        continue

                    # ===== HJ ADDED: Only monitor sectors that had interfering obstacles during path generation =====
                    # Skip sectors that never had interfering obstacles - they shouldn't trigger GB reversion
                    if sector_id not in self.sectors_with_interfering_obs:
                        # rospy.loginfo_throttle(5.0,
                        #     f"[{self.name}] Skipping sector {sector_id} zone monitoring (no interfering obstacles during path generation)")
                        continue
                    # ===== HJ ADDED END =====

                    s_start = sector_data['s_start']
                    s_end = sector_data['s_end']

                    # Calculate sector length
                    if s_start <= s_end:
                        # Normal case: no wrap-around
                        sector_length = s_end - s_start
                    else:
                        # Wrap-around case: sector crosses track boundary
                        sector_length = (self.track_length - s_start) + s_end

                    # Define detection zone boundaries
                    check_start = (s_start - sector_start_margin) % self.track_length
                    check_end = (s_start + sector_length * sector_check_threshold) % self.track_length

                    # Normalize current position
                    cur_s_norm = self.cur_s % self.track_length

                    # Check if vehicle is in detection zone (handle wrap-around)
                    if check_start <= check_end:
                        # Normal case: zone doesn't wrap
                        in_zone = (cur_s_norm >= check_start and cur_s_norm <= check_end)
                    else:
                        # Wrap-around case: zone crosses track boundary
                        in_zone = (cur_s_norm >= check_start or cur_s_norm <= check_end)

                    # Initialize monitoring state for this sector if needed
                    if sector_id not in self.sector_monitoring:
                        self.sector_monitoring[sector_id] = {
                            'in_zone': False,
                            'had_obstacles': False,
                            'entered_at_s': None
                        }

                    monitor = self.sector_monitoring[sector_id]

                    # Check for interfering obstacles IN THIS SPECIFIC SECTOR
                    # Filter to only obstacles in this sector (using sector_id attribute)
                    # DEBUG: First check all obstacles in this sector
                    sector_static_obs = [obs for obs in self.latest_obstacles
                                        if obs.in_static_obs_sector
                                        and obs.is_static
                                        and obs.sector_id == sector_id]

                    interfering_obs = [obs for obs in sector_static_obs
                                      if self._check_obstacle_interference(obs)]

                    if in_zone:
                        # Vehicle entered or is in zone
                        if not monitor['in_zone']:
                            # Just entered zone
                            monitor['in_zone'] = True
                            monitor['had_obstacles'] = False  # Reset
                            monitor['entered_at_s'] = cur_s_norm
                            self.get_logger().warning(
                                f"[{self.name}] ===== ZONE ENTERED ===== "
                                f"sector={sector_id}, cur_s={self.cur_s:.2f}m, "
                                f"zone=[{check_start:.2f}, {check_end:.2f}]m")

                        # Track previous had_obstacles flag for change detection
                        prev_had_obstacles = monitor['had_obstacles']

                        # Mark if obstacles detected
                        if interfering_obs:
                            monitor['had_obstacles'] = True
                            # ===== HJ ADDED: Reset counter when obstacles detected =====
                            self.consecutive_cleared_sectors = 0
                            # ===== HJ ADDED END =====

                        # Warn only when flag changes (or first time)
                        if monitor['had_obstacles'] != prev_had_obstacles:
                            self.get_logger().warning(
                                f"[{self.name}] >>> Sector {sector_id}: had_obstacles changed to {monitor['had_obstacles']} "
                                f"(interfering_obs={len(interfering_obs)}, sector_static_obs={len(sector_static_obs)})")

                        vehicle_in_sector = True
                        vehicle_sector_id = sector_id
                        break

                    else:
                        # Vehicle exited zone
                        if monitor['in_zone']:
                            # Just exited - check if we can revert
                            monitor['in_zone'] = False

                            if not monitor['had_obstacles']:
                                # ===== HJ MODIFIED: Require 2+ consecutive cleared sectors =====
                                # Passed through zone without obstacles → increment counter
                                self.consecutive_cleared_sectors += 1
                                self.get_logger().warning(
                                    f"[{self.name}] POST-FIX: Vehicle EXITED sector {sector_id} zone "
                                    f"(s={self.cur_s:.2f}m) WITHOUT obstacles detected during passage "
                                    f"→ Consecutive cleared sectors: {self.consecutive_cleared_sectors}")

                                # Only revert to GB after 2+ consecutive cleared sectors
                                if self.consecutive_cleared_sectors >= 2:
                                    self.use_fixed_path = False
                                    self.get_logger().warning(
                                        f"[{self.name}] POST-FIX: {self.consecutive_cleared_sectors} consecutive sectors cleared "
                                        f"→ Reverting to GB")
                                # ===== HJ MODIFIED END =====
                            # else:
                            #     self.get_logger().info(
                            #         f"[{self.name}] Vehicle exited sector {sector_id} zone, "
                            #         f"but had obstacles during passage → Keep using fixed path")
            # ===== HJ ADDED END =====
            # ===== HJ MODIFIED END =====

        # Set obs_in_interest from ALL static obstacles (not just in sector)
        # Filter by interference with current reference path (GB or Fixed)
        if self.use_fixed_path and self.fixed_converter is not None:
            # Using fixed path: Check interference and find closest
            interfering_obs_fixed = [obs for obs in all_static_obs
                                    if self._check_interference_fixed_path(obs)]

            if interfering_obs_fixed:
                # Get vehicle position in fixed path Frenet
                result_ego = self.fixed_converter.get_frenet(np.array([self.cur_x]), np.array([self.cur_y]))
                cur_s_fixed = result_ego[0]

                # Find closest using FIXED path s coordinates
                closest_obs = None
                min_dist = float('inf')
                ref_max_s = self.fixed_converter.raceline_length

                for obs in interfering_obs_fixed:
                    result = self.fixed_converter.get_frenet(np.array([obs.x_m]), np.array([obs.y_m]))
                    s_fixed, d_fixed = result[0], result[1]

                    # Calculate forward distance with wrap-around handling
                    forward_dist = (s_fixed - cur_s_fixed) % ref_max_s

                    # Skip obstacles behind (distance > half track length)
                    # Explicit check to handle wrap-around boundary correctly
                    if forward_dist > ref_max_s / 2:
                        continue  # Obstacle is behind, skip

                    if forward_dist < min_dist:
                        min_dist = forward_dist
                        closest_obs = obs
                        closest_s_fixed = s_fixed
                        closest_d_fixed = d_fixed

                # Check if we found any forward obstacle
                if closest_obs is not None:
                    # Create obstacle with fixed Frenet coordinates for do_spline
                    # ===== HJ MODIFIED: Convert all Frenet coordinates to Fixed frame =====
                    obs_fixed = copy.deepcopy(closest_obs)
                    obs_fixed.s_center = float(closest_s_fixed)  # Convert to Python float for ROS message
                    obs_fixed.d_center = float(closest_d_fixed)  # Convert to Python float for ROS message

                    # Convert d_left, d_right to Fixed Frenet (method 2: use size)
                    # Assuming circular obstacle, size doesn't change with coordinate system
                    obs_fixed.d_left = float(closest_d_fixed) + closest_obs.size / 2.0
                    obs_fixed.d_right = float(closest_d_fixed) - closest_obs.size / 2.0

                    self.obs_in_interest = obs_fixed
                    # rospy.loginfo_throttle(2.0,
                    #     f"[{self.name}] POST-FIX (Fixed mode): {len(interfering_obs_fixed)} obstacles, "
                    #     f"closest at fixed s={float(closest_s_fixed):.2f}, d={float(closest_d_fixed):.2f}, "
                    #     f"d_left={obs_fixed.d_left:.2f}, d_right={obs_fixed.d_right:.2f}")
                    # ===== HJ MODIFIED END =====
                else:
                    # All obstacles are behind, no obstacle in interest
                    self.obs_in_interest = None
            else:
                self.obs_in_interest = None
        else:
            # Using GB raceline: Check interference and find closest
            interfering_obs_gb = [obs for obs in all_static_obs
                                 if self._check_obstacle_interference(obs)]

            if interfering_obs_gb:
                # Filter forward obstacles only, then find closest
                forward_obs_gb = []
                for obs in interfering_obs_gb:
                    forward_dist = (obs.s_center - self.cur_s) % self.gb_max_s

                    # Skip obstacles behind (distance > half track length)
                    # Explicit check to handle wrap-around boundary correctly
                    if forward_dist > self.gb_max_s / 2:
                        continue  # Obstacle is behind, skip

                    forward_obs_gb.append((obs, forward_dist))

                if forward_obs_gb:
                    # Find closest among forward obstacles
                    closest_obs = min(forward_obs_gb, key=lambda x: x[1])[0]
                    self.obs_in_interest = closest_obs
                    mode_str = "POST-FIX (GB mode)" if self.fixed_path_generated else "PRE-FIX"
                    # rospy.loginfo_throttle(2.0,
                    #     f"[{self.name}] {mode_str}: {len(forward_obs_gb)} forward obstacles, "
                    #     f"closest at GB s={closest_obs.s_center:.2f}, d={closest_obs.d_center:.2f}")
                else:
                    # No forward obstacles
                    self.obs_in_interest = None
            else:
                self.obs_in_interest = None

        self.obstacles_updated = False
    # ===== HJ PHASE A END =====

    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object"""
        rospy.wait_for_message("/global_waypoints", WpntArray)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.get_logger().info(f"[{self.name}] initialized FrenetConverter object")

        return converter

    # ===== HJ ADDED: Safe grid filtering using existing functions =====
    def _create_conservative_map_with_obstacles(
        self,
        nearby_obs: List[Tuple[int, int, float, float]],
        resolution: float,
        origin_x: float,
        origin_y: float
    ) -> str:
        """
        Create conservative map with obstacles using existing _add_obstacles_to_occupancy_grid().
        Uses _for_spliner.png as base, adds obstacles, saves as _for_spliner_with_obs.png.

        Args:
            nearby_obs: List of (sector_id, obs_id, s, d) tuples (within 2.5m)
            resolution: Map resolution [m/px]
            origin_x, origin_y: Map origin [m]

        Returns:
            Path to _for_spliner_with_obs.png, or None if failed
        """
        # Paths
        conservative_base_path = os.path.join(self.map_dir, f'{self.map_name}_for_spliner.png')
        conservative_with_obs_path = os.path.join(self.map_dir, f'{self.map_name}_for_spliner_with_obs.png')

        # Load base conservative map
        if not os.path.exists(conservative_base_path):
            self.get_logger().warning(f"[{self.name}] Conservative base map not found: {conservative_base_path}")
            return None

        conservative_base = cv2.imread(conservative_base_path, cv2.IMREAD_GRAYSCALE)
        self.get_logger().info(f"[{self.name}] Loaded conservative base map: {conservative_base_path}")

        # Add obstacles using existing function (reuses all existing logic)
        _, conservative_with_obs = self._add_obstacles_to_occupancy_grid(
            conservative_base.copy(),
            nearby_obs,
            resolution,
            origin_x,
            origin_y
        )

        # Save result
        cv2.imwrite(conservative_with_obs_path, conservative_with_obs)
        self.get_logger().info(f"[{self.name}] Created conservative map with {len(nearby_obs)} obstacles: {conservative_with_obs_path}")

        return conservative_with_obs_path

    def _calculate_conservative_bounds_from_map(
        self,
        map_path: str,
        waypoints: List,
        mode: str = "GB"
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calculate conservative d_left/d_right for waypoints using existing functions.
        Reuses _extract_bounds_from_watershed() and _calculate_reftrack_distances().

        Args:
            map_path: Path to conservative map PNG (_for_spliner.png or _for_spliner_with_obs.png)
            waypoints: List of Wpnt messages (GB or Fixed path waypoints)
            mode: "GB" or "Smart" (for logging)

        Returns:
            (d_left_dict, d_right_dict): Dictionaries keyed by waypoint index
        """
        try:
            # Load conservative map
            if not os.path.exists(map_path):
                self.get_logger().warning(f"[{self.name}] Conservative map not found: {map_path}")
                return {}, {}

            conservative_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if conservative_img is None:
                self.get_logger().error(f"[{self.name}] Failed to load conservative map: {map_path}")
                return {}, {}

            # Flip image (same as global_planner_node)
            # conservative_bw = cv2.flip(conservative_img, 0)

            # Filtering with morphological opening (same as global_planner_node)
            kernel1 = np.ones((9, 9), np.uint8)
            opening = cv2.morphologyEx(conservative_img, cv2.MORPH_OPEN, kernel1, iterations=1)

            # Get morphological skeleton of the map (same as global_planner_node)
            skeleton = skeletonize(opening, method='lee')
            centerline_pixels = self._extract_centerline_from_skeleton(skeleton)

            if centerline_pixels is None:
                self.get_logger().error(f"[{self.name}] Failed to extract centerline from {map_path}")
                return {}, {}

            # Extract bounds using watershed (same as global_planner_node)
            # If watershed fails, fallback to distance transform
            use_watershed = True
            try:
                bound_r_pixels, bound_l_pixels = self._extract_bounds_from_watershed(
                    centerline_pixels, opening
                )
                dist_transform = None
                self.get_logger().info(f"[{self.name}] Using watershed for track bound extraction...")
            except IOError:
                self.get_logger().warning(f"[{self.name}] More than two track bounds detected with watershed algorithm")
                self.get_logger().info(f"[{self.name}] Trying with simple distance transform...")
                use_watershed = False
                bound_r_pixels = None
                bound_l_pixels = None
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            # Convert to meters
            yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')
            with open(yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)

            map_resolution = map_data['resolution']
            map_origin_x = map_data['origin'][0]
            map_origin_y = map_data['origin'][1]
            map_height = conservative_img.shape[0]

            # Convert waypoints to centerline format [x, y]
            centerline_m = np.array([[wp.x_m, wp.y_m] for wp in waypoints])

            # Calculate distances (same logic as global_planner_node.add_dist_to_cent)
            if dist_transform is not None:
                # Use distance transform (fallback)
                width_track = dist_transform[centerline_pixels[:, 1].astype(int),
                                             centerline_pixels[:, 0].astype(int)] * map_resolution
                # Interpolate if needed
                if len(width_track) != len(centerline_m):
                    width_track = np.interp(np.arange(0, len(centerline_m)), np.arange(0, len(width_track)), width_track)
                # For distance transform, left and right are the same
                reftrack = np.column_stack((centerline_m, width_track, width_track))
            else:
                # Use watershed bounds
                bound_r_m = self._pixels_to_meters(bound_r_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
                bound_l_m = self._pixels_to_meters(bound_l_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
                reftrack = self._calculate_reftrack_distances(centerline_m, bound_r_m, bound_l_m)

            # Extract d_right and d_left into dictionaries
            d_right_dict = {i: reftrack[i, 2] for i in range(len(waypoints))}
            d_left_dict = {i: reftrack[i, 3] for i in range(len(waypoints))}

            self.get_logger().info(f"[{self.name}] Calculated conservative bounds for {mode} mode: {len(d_right_dict)} waypoints")
            return d_left_dict, d_right_dict

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to calculate conservative bounds: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}
    # ===== HJ ADDED END =====


    # def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any], obs_s_idx: int) -> Tuple[str, float]:
    #     # ===== HJ MODIFIED: Use conservative bounds if USE_SAFE_GRID enabled =====
    #     if USE_SAFE_GRID:
    #         # Check if we're in GB mode or Smart mode based on use_fixed_path flag
    #         if not self.use_fixed_path:
    #             # GB mode: use GB conservative bounds from _for_spliner.png (no obstacles)
    #             if obs_s_idx in self.gb_conservative_d_left and obs_s_idx in self.gb_conservative_d_right:
    #                 wpnt_d_left = self.gb_conservative_d_left[obs_s_idx]
    #                 wpnt_d_right = self.gb_conservative_d_right[obs_s_idx]
    #             else:
    #                 # Fallback to original bounds if conservative bounds not available
    #                 wpnt_d_left = gb_wpnts[obs_s_idx].d_left
    #                 wpnt_d_right = gb_wpnts[obs_s_idx].d_right
    #         else:
    #             # Smart mode: use Smart conservative bounds from _for_spliner_with_obs.png
    #             if obs_s_idx in self.smart_conservative_d_left and obs_s_idx in self.smart_conservative_d_right:
    #                 wpnt_d_left = self.smart_conservative_d_left[obs_s_idx]
    #                 wpnt_d_right = self.smart_conservative_d_right[obs_s_idx]
    #             else:
    #                 # Fallback to original bounds if conservative bounds not available
    #                 wpnt_d_left = gb_wpnts[obs_s_idx].d_left
    #                 wpnt_d_right = gb_wpnts[obs_s_idx].d_right
    #     else:
    #         # USE_SAFE_GRID=False: use original GB waypoint bounds
    #         wpnt_d_left = gb_wpnts[obs_s_idx].d_left
    #         wpnt_d_right = gb_wpnts[obs_s_idx].d_right
    #     # ===== HJ MODIFIED END =====

    #     # rospy.logwarn_throttle(1.0,
    #     #     f"[{self.name}] _more_space: wpnt[{obs_s_idx}] d_left={wpnt_d_left:.2f}, d_right={wpnt_d_right:.2f}, "
    #     #     f"obs d_left={obstacle.d_left:.2f}, d_right={obstacle.d_right:.2f}")

    #     left_gap = abs(wpnt_d_left - obstacle.d_left)
    #     right_gap = abs(wpnt_d_right + obstacle.d_right)
    #     min_space = self.evasion_dist + self.spline_bound_mindist

    #     if right_gap > min_space and left_gap < min_space:
    #         # Compute apex distance to the right of the opponent
    #         d_apex_right = obstacle.d_right - self.evasion_dist
    #         # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
    #         if d_apex_right > 0:
    #             d_apex_right = 0
    #         return "right", d_apex_right

    #     elif left_gap > min_space and right_gap < min_space:
    #         # Compute apex distance to the left of the opponent
    #         d_apex_left = obstacle.d_left + self.evasion_dist
    #         # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
    #         if d_apex_left < 0:
    #             d_apex_left = 0
    #         return "left", d_apex_left
    #     else:
    #         # Both sides have insufficient space, choose the side with MORE available space
    #         # Calculate actual distance from obstacle center to each wall

    #         # Get obstacle center XY coordinates (convert from Frenet to Cartesian)
    #         # Use appropriate converter based on mode
    #         if self.use_fixed_path and self.fixed_converter is not None:
    #             # Smart mode: obstacle is in Fixed path Frenet coordinates
    #             obs_xy = self.fixed_converter.get_cartesian(
    #                 np.array([obstacle.s_center]),
    #                 np.array([obstacle.d_center])
    #             )
    #         else:
    #             # GB mode: obstacle is in GB Frenet coordinates
    #             obs_xy = self.converter.get_cartesian(
    #                 np.array([obstacle.s_center]),
    #                 np.array([obstacle.d_center])
    #             )
    #         obs_pos = np.array([obs_xy[0, 0], obs_xy[1, 0]])

    #         # Select appropriate bounds based on USE_SAFE_GRID and use_fixed_path flags
    #         if USE_SAFE_GRID:
    #             if not self.use_fixed_path:
    #                 # GB mode + USE_SAFE_GRID=True: Use conservative bounds (no obstacles)
    #                 if hasattr(self, 'bound_l_conservative') and hasattr(self, 'bound_r_conservative'):
    #                     bound_l = self.bound_l_conservative
    #                     bound_r = self.bound_r_conservative
    #                     bounds_source = "conservative (no obs)"
    #                 else:
    #                     rospy.logwarn_throttle(5.0, f"[{self.name}] Conservative bounds not available, using original")
    #                     bound_l = self.bound_l_original
    #                     bound_r = self.bound_r_original
    #                     bounds_source = "original (fallback)"
    #             else:
    #                 # Smart mode + USE_SAFE_GRID=True: Use conservative bounds with obstacles
    #                 if hasattr(self, 'bound_l_conservative_modified') and hasattr(self, 'bound_r_conservative_modified'):
    #                     bound_l = self.bound_l_conservative_modified
    #                     bound_r = self.bound_r_conservative_modified
    #                     bounds_source = "conservative + obs"
    #                 else:
    #                     rospy.logwarn_throttle(5.0, f"[{self.name}] Conservative modified bounds not available, using original modified")
    #                     bound_l = self.bound_l_modified
    #                     bound_r = self.bound_r_modified
    #                     bounds_source = "original + obs (fallback)"
    #         else:
    #             if not self.use_fixed_path:
    #                 # GB mode + USE_SAFE_GRID=False: Use original bounds (no obstacles)
    #                 if hasattr(self, 'bound_l_original') and hasattr(self, 'bound_r_original'):
    #                     bound_l = self.bound_l_original
    #                     bound_r = self.bound_r_original
    #                     bounds_source = "original (no obs)"
    #                 else:
    #                     rospy.logwarn_throttle(5.0, f"[{self.name}] Original bounds not available, extracting from GB waypoints")
    #                     # Extract bounds from GB waypoints as fallback
    #                     bound_l = np.array([[wp.x_m + wp.d_left * np.cos(wp.psi_rad + np.pi/2),
    #                                         wp.y_m + wp.d_left * np.sin(wp.psi_rad + np.pi/2)]
    #                                        for wp in gb_wpnts])
    #                     bound_r = np.array([[wp.x_m + wp.d_right * np.cos(wp.psi_rad - np.pi/2),
    #                                         wp.y_m + wp.d_right * np.sin(wp.psi_rad - np.pi/2)]
    #                                        for wp in gb_wpnts])
    #                     bounds_source = "GB wpnts (fallback)"
    #             else:
    #                 # Smart mode + USE_SAFE_GRID=False: Use original bounds with obstacles
    #                 if hasattr(self, 'bound_l_modified') and hasattr(self, 'bound_r_modified'):
    #                     bound_l = self.bound_l_modified
    #                     bound_r = self.bound_r_modified
    #                     bounds_source = "original + obs"
    #                 else:
    #                     rospy.logwarn_throttle(5.0, f"[{self.name}] Modified bounds not available, using original")
    #                     bound_l = self.bound_l_original
    #                     bound_r = self.bound_r_original
    #                     bounds_source = "original (fallback)"

    #         # Calculate minimum distance from obstacle center to each wall
    #         dist_to_left_wall = np.min(np.linalg.norm(bound_l - obs_pos, axis=1))
    #         dist_to_right_wall = np.min(np.linalg.norm(bound_r - obs_pos, axis=1))

    #         # Calculate available space (subtract obstacle radius)
    #         obstacle_radius = OBSTACLE_RADIUS
    #         space_to_left = dist_to_left_wall - obstacle_radius
    #         space_to_right = dist_to_right_wall - obstacle_radius

    #         rospy.logwarn_throttle(1.0,
    #             f"[{self.name}] _more_space: Both sides tight! bounds_source={bounds_source}, "
    #             f"dist_to_left_wall={dist_to_left_wall:.2f}m, dist_to_right_wall={dist_to_right_wall:.2f}m, "
    #             f"space_to_left={space_to_left:.2f}m, space_to_right={space_to_right:.2f}m")

    #         if space_to_left > space_to_right:
    #             # More space on the left side
    #             candidate_d_apex_left = obstacle.d_left + self.evasion_dist
    #             # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
    #             if candidate_d_apex_left < 0:
    #                 candidate_d_apex_left = 0
    #             rospy.logwarn_throttle(1.0, f"[{self.name}] _more_space: Choosing LEFT (space_diff={space_to_left - space_to_right:.2f}m)")
    #             return "left", candidate_d_apex_left
    #         else:
    #             # More space on the right side (or equal)
    #             candidate_d_apex_right = obstacle.d_right - self.evasion_dist
    #             # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
    #             if candidate_d_apex_right > 0:
    #                 candidate_d_apex_right = 0
    #             rospy.logwarn_throttle(1.0, f"[{self.name}] _more_space: Choosing RIGHT (space_diff={space_to_right - space_to_left:.2f}m)")
    #             return "right", candidate_d_apex_right
        

    def _more_space(self, obstacle: Obstacle, gb_wpnts: List[Any], obs_s_idx: int) -> Tuple[str, float]:
        """
        Determine evasion direction and d_apex for obstacle avoidance.

        Uses _decide_obstacle_strategy_gb_aware for direction (same as Global opt centerline shift),
        then calculates d_apex based on the chosen direction.
        """
        # Use _decide_obstacle_strategy_gb_aware for consistent direction with Global opt
        _, evasion_direction = self._decide_obstacle_strategy_gb_aware(
            obstacle.s_center, obstacle.d_center, obstacle.size / 2
        )

        # Calculate d_apex based on evasion direction
        if evasion_direction == "left":
            d_apex = obstacle.d_left + self.evasion_dist
            # If apex is to the right of the raceline, set to 0
            if d_apex < 0:
                d_apex = 0
            rospy.loginfo_throttle(1.0, f"[{self.name}] _more_space: evasion=LEFT, d_apex={d_apex:.2f}m")
        else:
            d_apex = obstacle.d_right - self.evasion_dist
            # If apex is to the left of the raceline, set to 0
            if d_apex > 0:
                d_apex = 0
            rospy.loginfo_throttle(1.0, f"[{self.name}] _more_space: evasion=RIGHT, d_apex={d_apex:.2f}m")

        return evasion_direction, d_apex

    def do_spline(self, obs: Obstacle, gb_wpnts: WpntArray, return_samples: bool = False) -> Tuple[WpntArray, MarkerArray]:
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
        wpnts = OTWpntArray()
        wpnts.header.stamp = self.get_clock().now().to_msg()
        wpnts.header.frame_id = "map"

        # ===== HJ MODIFIED: Calculate wpnt_dist using appropriate Frenet coordinate system =====
        ref_max_idx = len(gb_wpnts)  # Max index for current reference path (GB or Fixed)

        if self.use_fixed_path and self.fixed_converter is not None:
            # Use pre-calculated values from FrenetConverter (more accurate, avoids accumulation error)
            ref_max_s = self.fixed_converter.raceline_length

            # Calculate wpnt_dist from converter's waypoints_s
            wpnt_dist = self.fixed_converter.waypoints_s[1] - self.fixed_converter.waypoints_s[0]
            
        else:
            # GB mode: Use existing s_m from waypoints
            wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m
            ref_max_s = gb_wpnts[-1].s_m
        # ===== HJ MODIFIED END =====

        # CRITICAL: Convert current vehicle position to reference path Frenet coordinates
        if self.use_fixed_path and self.fixed_converter is not None:
            # Fixed path mode: Convert vehicle XY to Fixed Frenet
            result_ego = self.fixed_converter.get_frenet(np.array([self.cur_x]), np.array([self.cur_y]))
            cur_s_ref = float(result_ego[0]) % ref_max_s  # Normalize to [0, ref_max_s)
            cur_d_ref = float(result_ego[1])

            # Also normalize obstacle s to same range
            obs_s_normalized = obs.s_center % ref_max_s

            # rospy.loginfo_throttle(2.0,
            #     f"[{self.name}] DEBUG do_spline START (FIXED): ref_max_idx={ref_max_idx}, ref_max_s={ref_max_s:.2f}m, "
            #     f"wpnt_dist={wpnt_dist:.4f}m, cur_s_FIXED={cur_s_ref:.2f}m (raw={float(result_ego[0]):.2f}m, GB={self.cur_s:.2f}m), "
            #     f"obs.s={obs_s_normalized:.2f}m (raw={obs.s_center:.2f}m), obs.d={obs.d_center:.2f}m")

            # Update obs.s_center for this function scope
            obs = copy.deepcopy(obs)
            obs.s_center = obs_s_normalized
        else:
            # GB mode: Use existing GB Frenet coordinates
            cur_s_ref = self.cur_s
            cur_d_ref = self.cur_d
            # self.get_logger().error(
            #     f"[{self.name}] DEBUG do_spline START (GB): ref_max_idx={ref_max_idx}, ref_max_s={ref_max_s:.2f}m, "
            #     f"wpnt_dist={wpnt_dist:.4f}m, cur_s_GB={cur_s_ref:.2f}m, "
            #     f"obs.s={obs.s_center:.2f}m, obs.d={obs.d_center:.2f}m")
        # ===== HJ ADDED END =====

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        # close_obs = self._obs_filtering(obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if obs.is_static == True:
            # ===== HJ MODIFIED: Use cur_s_ref (reference path coordinates) instead of self.cur_s =====
            # Calculate raw distance (can be negative if obstacle is behind)
            pre_dist_raw = obs.s_center - cur_s_ref

            # Apply modulo for wrap-around
            pre_dist = pre_dist_raw % ref_max_s

            # Check if obstacle is behind (after wrap-around, distance > half track means it's actually behind)
            if pre_dist > ref_max_s / 2:
                # Obstacle is behind us, skip
                # self.get_logger().warning(
                #     f"[{self.name}] DEBUG ABORT #1-A: Obstacle behind! "
                #     f"pre_dist_raw={pre_dist_raw:.2f}m, pre_dist={pre_dist:.2f}m, ref_max_s/2={ref_max_s/2:.2f}m")
                wpnts.wpnts = []
                mrks.markers = []
                return wpnts, mrks

            if pre_dist < 0.5:
                # ===== HJ ADDED: Debug log for early return =====
                # self.get_logger().warning(
                #     f"[{self.name}] DEBUG ABORT #1-B: Obstacle too close! "
                #     f"pre_dist={pre_dist:.2f}m (threshold=0.5m), cur_s={cur_s_ref:.2f}m, obs.s={obs.s_center:.2f}m")
                # ===== HJ ADDED END =====
                wpnts.wpnts = []
                mrks.markers = []
                return wpnts, mrks

            obs_s_idx = int(obs.s_center / wpnt_dist) % ref_max_idx
            # ===== HJ MODIFIED END =====

            more_space, d_apex = self._more_space(obs, gb_wpnts, obs_s_idx)
            s_list = [obs.s_center]
            d_list = [d_apex]

            # ===== HJ MODIFIED: Use ref_max_s instead of self.gb_max_s =====
            post_dist = min(min(max(pre_dist, self.post_min_dist), self.post_max_dist), ref_max_s / 2)
            # ===== HJ MODIFIED END =====

            num_post_ref = int((post_dist // self.sampling_dist)) + 1

            for i in range(num_post_ref):
                s_list.append(obs.s_center + post_dist * ((i + 1)/ num_post_ref))
                d_list.append((d_apex * (1 - (i + 1)/ num_post_ref)))
                            
            # evasion_s = np.array([self.cur_s +1 ,self.cur_s +2])
            s_array = np.array(s_list)
            d_array = np.array(d_list)

            # ===== HJ MODIFIED: Use ref_max_s and ref_max_idx =====
            s_array = s_array % ref_max_s

            s_idx = np.round((s_array / wpnt_dist)).astype(int) % ref_max_idx
            # ===== HJ MODIFIED END =====
            # evasion2 = int(evasion_s[1] / wpnt_dist) % self.gb_max_idx
            
            # gb_idxs = [evasion1, evasion2]
            
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle

            # evasion_d = np.array([obs.d_center + (obs.size/2 + 0.5)  , 0])
            # evasion_d = np.array([d_apex  , 0])
            
            # Do frenet conversion via conversion service for spline and create markers and wpnts
            danger_flag = False
            # ===== HJ ADDED: Use appropriate Frenet converter =====
            # self.get_logger().error(
            #     f"[{self.name}] DEBUG Frenet→Cartesian: s_array={s_array}, d_array={d_array}, "
            #     f"Converter: {'FIXED' if (self.use_fixed_path and self.fixed_converter) else 'GB'}")

            if self.use_fixed_path and self.fixed_converter is not None:
                resp = self.fixed_converter.get_cartesian(s_array, d_array)
                # self.get_logger().error(f"[{self.name}] DEBUG: Using FIXED converter for Frenet→Cartesian")
            else:
                resp = self.converter.get_cartesian(s_array, d_array)
                # self.get_logger().error(f"[{self.name}] DEBUG: Using GB converter for Frenet→Cartesian")
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Debug log for Cartesian conversion results =====
            # self.get_logger().error(f"[{self.name}] DEBUG Cartesian points from Frenet conversion:")
            # for i in range(len(s_array)):
            #     self.get_logger().error(f"  Point {i}: s={s_array[i]:.2f}, d={d_array[i]:.2f} → x={resp[0,i]:.2f}, y={resp[1,i]:.2f}")
            # ===== HJ ADDED END =====

            points=[[self.cur_x,self.cur_y]]
            tangents=[[np.cos(self.cur_yaw), np.sin(self.cur_yaw)]]

            for i in range(len(s_idx)):
                points.append(resp[:, i])
                tangents.append(np.array([np.cos(gb_wpnts[s_idx[i]].psi_rad), np.sin(gb_wpnts[s_idx[i]].psi_rad)]))

            # points.append([self.cur_x,self.cur_y])
            # tangents.append(np.array([np.cos(self.cur_yaw), np.sin(self.cur_yaw)]))

            
            # points.append(resp[:,0])
            # points.append(resp[:,1])

            # tangents.append(np.array([np.cos(gb_wpnts[evasion1].psi_rad), np.sin(gb_wpnts[evasion1].psi_rad)]))
            # tangents.append(np.array([np.cos(gb_wpnts[evasion2].psi_rad), np.sin(gb_wpnts[evasion2].psi_rad)]))

            # tangents = np.dot(tangents, self.spline_scale*np.eye(len(points)))
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

            # ===== HJ ADDED: Apply Savitzky-Golay smoothing to reduce angular paths =====
            if SMOOTH_OTWPNTS and len(samples) >= SMOOTH_OTWPNTS_WINDOW:
                # Preserve start and end points
                start_pt = samples[0].copy()
                end_pt = samples[-1].copy()

                # Apply Savitzky-Golay filter to x and y separately
                samples[:, 0] = savgol_filter(samples[:, 0], SMOOTH_OTWPNTS_WINDOW, SMOOTH_OTWPNTS_POLYORDER)
                samples[:, 1] = savgol_filter(samples[:, 1], SMOOTH_OTWPNTS_WINDOW, SMOOTH_OTWPNTS_POLYORDER)

                # Restore start point (vehicle position - must not change)
                samples[0] = start_pt

                # Blend end points smoothly (last 5 points)
                blend_len = min(5, len(samples) - 1)
                for bi in range(blend_len):
                    idx = len(samples) - blend_len + bi
                    w = bi / blend_len  # 0 to almost 1
                    samples[idx] = samples[idx] * (1 - w) + end_pt * w
                samples[-1] = end_pt
            # ===== HJ ADDED END =====


            # if samples.shape[0] < self.n_loc_wpnts:
            n_additional = 100
            # ===== HJ MODIFIED: Use ref_max_idx instead of self.gb_max_idx =====
            xy_additional = np.array([
                (
                    gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].x_m,
                    gb_wpnts[(s_idx[-1] + i + 1) % ref_max_idx].y_m
                )
                for i in range(n_additional)
            ])


            # ===== HJ ADDED: Smooth transition between spline and GB path =====
            if SMOOTH_OTWPNTS and len(samples) > 0 and len(xy_additional) > 0:
                # Blend last N points of samples with corresponding GB points (no direction reversal)
                blend_to_gb_len = min(40, len(samples) - 1)  # 40 waypoints transition (stronger)

                # Get starting GB index for blending (go backwards from s_idx[-1])
                blend_start_gb_idx = max(0, s_idx[-1] - blend_to_gb_len)

                for bi in range(blend_to_gb_len):
                    idx = len(samples) - blend_to_gb_len + bi
                    # Use quadratic easing for smoother blend (stronger at end)
                    t = (bi + 1) / (blend_to_gb_len + 1)
                    w = t * t  # Quadratic: starts slow, accelerates toward GB

                    # Get corresponding GB point (moving forward along GB path)
                    gb_idx_for_blend = (s_idx[-1] - blend_to_gb_len + bi + 1) % ref_max_idx
                    target_pt = np.array([gb_wpnts[gb_idx_for_blend].x_m, gb_wpnts[gb_idx_for_blend].y_m])

                    samples[idx] = samples[idx] * (1 - w) + target_pt * w
            # ===== HJ ADDED END =====

            samples = np.vstack([samples, xy_additional])

            # ===== HJ ADDED: Resample to uniform spacing =====
            if SMOOTH_OTWPNTS and len(samples) > 2:
                # Calculate cumulative arc length
                diffs = np.diff(samples, axis=0)
                seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
                total_length = cum_length[-1]

                # Target spacing (use wpnt_dist from GB waypoints)
                target_spacing = wpnt_dist
                n_resampled = max(int(total_length / target_spacing), 2)

                # Create uniform parameter values
                uniform_s = np.linspace(0, total_length, n_resampled)

                # Interpolate x and y at uniform spacing
                samples_resampled = np.zeros((n_resampled, 2))
                samples_resampled[:, 0] = np.interp(uniform_s, cum_length, samples[:, 0])
                samples_resampled[:, 1] = np.interp(uniform_s, cum_length, samples[:, 1])
                samples = samples_resampled
            # ===== HJ ADDED END =====

            psi_, kappa_ = tph.calc_head_curv_num.\
                calc_head_curv_num(
                    path=samples,
                    el_lengths=wpnt_dist*np.ones(len(samples)-1),
                    is_closed=False
                )
            # ===== HJ MODIFIED: Calculate Frenet coordinates for gb_wpnt_i indexing =====
            # s_, d_ are used to find the correct waypoint index in the current reference path
            if self.use_fixed_path and self.fixed_converter is not None:
                s_, d_ = self.fixed_converter.get_frenet(samples[:, 0], samples[:, 1])
            else:
                s_, d_ = self.converter.get_frenet(samples[:, 0], samples[:, 1])
            # ===== HJ MODIFIED END =====

            # ===== HJ MODIFIED: Choose Frenet converter for waypoint output based on USE_FIXED_PATH_FRENET =====
            # Controlled by class variable at top of file (line ~57)
            if self.USE_FIXED_PATH_FRENET:
                # OPTION 1: Use Fixed path converter (if available and in Fixed mode)
                if self.use_fixed_path and self.fixed_converter is not None:
                    s_wpnt, d_wpnt = self.fixed_converter.get_frenet(samples[:, 0], samples[:, 1])
                    rospy.loginfo_throttle(5.0, f"[{self.name}] do_spline using FIXED PATH Frenet for waypoint output")
                else:
                    # Fallback to GB if not in Fixed mode
                    s_wpnt, d_wpnt = self.converter.get_frenet(samples[:, 0], samples[:, 1])
                    rospy.loginfo_throttle(5.0, f"[{self.name}] do_spline using GB Frenet (not in Fixed mode)")
            else:
                # OPTION 2: Always use GB Frenet (for debugging/comparison)
                s_wpnt, d_wpnt = self.converter.get_frenet(samples[:, 0], samples[:, 1])
                rospy.loginfo_throttle(5.0, f"[{self.name}] do_spline using GB Frenet (debugging mode)")
            # ===== HJ MODIFIED END =====

            # ===== HJ ADDED: Track bounds check results for marker visualization =====
            bounds_check_results = []
            # ===== HJ ADDED END =====

            danger_flag = False
            for i in range(samples.shape[0]):
                # ===== HJ MODIFIED: Use ref_max_idx (len of current reference path) =====
                gb_wpnt_i = int((s_[i] / wpnt_dist) % ref_max_idx)
                # ===== HJ MODIFIED END =====

                # ===== HJ MODIFIED: Use appropriate map filter =====
                # Priority: 1) Conservative maps (GB or Smart), 2) Fixed path map, 3) Original map
                if USE_SAFE_GRID:
                    # Check if we're in GB mode or Smart mode based on use_fixed_path flag
                    if not self.use_fixed_path:
                        # GB mode: use conservative base map (_for_spliner.png, no obstacles)
                        if self.map_filter_conservative_base is not None:
                            inside = self.map_filter_conservative_base.is_point_inside(samples[i, 0], samples[i, 1])
                        else:
                            # Fallback to original if conservative map not loaded
                            inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                    else:
                        # Smart mode: use conservative map with obstacles (_for_spliner_with_obs.png)
                        if self.map_filter_conservative_with_obs is not None:
                            inside = self.map_filter_conservative_with_obs.is_point_inside(samples[i, 0], samples[i, 1])
                        else:
                            # Fallback to original if conservative map not loaded
                            inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                elif self.use_fixed_path and self.map_filter_fixed is not None:
                    # Fixed path mode (without USE_SAFE_GRID): use obstacles_only map
                    inside = self.map_filter_fixed.is_point_inside(samples[i, 0], samples[i, 1])
                else:
                    # Default: use original map
                    inside = self.map_filter.is_point_inside(samples[i, 0], samples[i, 1])
                # ===== HJ MODIFIED END =====

                # ===== HJ ADDED: Track bounds check result =====
                bounds_check_results.append(inside)
                # ===== HJ ADDED END =====

                if not inside:
                    # ===== HJ MODIFIED: Enhanced debug logging =====
                    # self.get_logger().error(
                    #     f"[{self.name}] DEBUG ABORT #2: Point {i}/{samples.shape[0]} OUTSIDE bounds! "
                    #     f"xy=({samples[i, 0]:.2f}, {samples[i, 1]:.2f}), "
                    #     f"Filter: {'OBSTACLES_ONLY' if (self.use_fixed_path and self.map_filter_fixed) else 'ORIGINAL'}")
                    # ===== HJ MODIFIED END =====
                    danger_flag = True
                    break
                outside = True
                # Get V from gb wpnts and go slower if we are going through the inside
                vi = gb_wpnts[gb_wpnt_i].vx_mps if outside else gb_wpnts[gb_wpnt_i].vx_mps * 0.9 # TODO make speed scaling ros param

                # ===== HJ MODIFIED: Use appropriate Frenet coordinates based on USE_FIXED_PATH_FRENET =====
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=samples[i, 0], y=samples[i, 1], s=s_wpnt[i], d=d_wpnt[i], v=2, psi=psi_[i] + np.pi/2 , kappa= kappa_[i], wpnts=wpnts)
                )
                # ===== HJ MODIFIED END =====
                mrks.markers.append(self.xyv_to_markers(x=samples[i, 0], y=samples[i, 1], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts

            # ===== HJ ADDED: Publish spline sample markers for debugging =====
            self._publish_spline_samples_markers(samples, bounds_check_results)
            # ===== HJ ADDED END =====

            if danger_flag:
                # ===== HJ ADDED: Debug log for danger flag abort =====
                # self.get_logger().error(f"[{self.name}] DEBUG ABORT #3: danger_flag=True, returning empty waypoints")
                # ===== HJ ADDED END =====
                wpnts.wpnts = []
                mrks.markers = []
                if return_samples:
                    return wpnts, mrks, None
            else:
                # ===== HJ ADDED: Success log =====
                # self.get_logger().error(f"[{self.name}] DEBUG SUCCESS: Generated {len(wpnts.wpnts)} waypoints successfully!")
                # ===== HJ ADDED END =====
                pass  # All checks passed, wpnts already populated

        # Return with optional samples
        if return_samples:
            # Return samples if available (only when obstacle was processed)
            if 'samples' in dir():
                return wpnts, mrks, samples
            else:
                return wpnts, mrks, None
        return wpnts, mrks

    ######################
    # VIZ + MSG WRAPPING #
    ######################
    def xyv_to_markers(self, x:float, y:float, v:float, mrks: MarkerArray, is_fixed_path: bool = False) -> Marker:
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = v / self.gb_vmax
        mrk.color.a = 1.0

        # ===== HJ MODIFIED: Different colors for fixed path vs real-time spline =====
        if is_fixed_path:
            # Fixed path: Blue
            mrk.color.r = 0.0
            mrk.color.g = 0.0
            mrk.color.b = 1.0
        else:
            # Real-time spline: Purple/Pink
            mrk.color.r = 0.75
            mrk.color.g = 0.75 if self.from_bag else 0.0
            mrk.color.b = 0.75
        # ===== HJ MODIFIED END =====

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = v / self.gb_vmax / 2
        mrk.pose.orientation.w = 1

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
        mrk.color.r = 1 if opponent else 0
        mrk.color.g = 0.65

        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1

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

    ###########################
    # PHASE 3: GB OPTIMIZER   #
    ###########################
    def _generate_fixed_path_async(self, verified_obs: List[Tuple[int, int, float, float]], nearby_obs: List[Tuple[int, int, float, float]]):
        """
        Async wrapper for _generate_fixed_path. Runs in separate thread.

        This wrapper:
        1. Calls _generate_fixed_path (blocking GB optimizer)
        2. Updates self.fixed_path_generated flag
        3. Clears self.fixed_path_generating flag

        Args:
            verified_obs: List of (sector_id, obs_id, s, d) tuples for interfering obstacles (GB optimizer)
            nearby_obs: List of (sector_id, obs_id, s, d) tuples for nearby obstacles (spline occupancy grid)
        """
        try:
            self.get_logger().info(f"[{self.name}] [THREAD] Starting GB optimizer path generation...")
            success = self._generate_fixed_path(verified_obs, nearby_obs)

            if success:
                self.get_logger().info(f"[{self.name}] [THREAD] Fixed path generated successfully!")
                self.fixed_path_generated = True  # Static flag: generation completed
                self.fixed_path_generation_attempted = True  # Mark as attempted to prevent retry

                # ===== HJ MODIFIED: Publish fixed path IMMEDIATELY before flag =====
                # Critical: state_machine must receive waypoints BEFORE use_fixed_path flag
                # NOTE: Timestamp is already set at packaging (line 3398), don't update here
                # This keeps timestamp constant across republishes to avoid conflicts with global_velocity_planner

                # Publish waypoints FIRST
                self.fixed_path_pub.publish(self.fixed_path_wpnts)
                self.fixed_path_mrks_pub.publish(self.fixed_path_markers)
                self.fixed_path_last_pub_time = self.get_clock().now().to_msg()
                self.get_logger().info(f"[{self.name}] [THREAD] Published FIXED path ({len(self.fixed_path_wpnts.wpnts)} waypoints)")

                # Set internal flag - actual publish will happen in main loop after state check
                self.use_fixed_path = True  # Dynamic flag: start using fixed path
                self.get_logger().info(f"[{self.name}] [THREAD] Set use_fixed_path=True (will publish in main loop after state check)")
                # ===== HJ MODIFIED END =====

                # ===== HJ ADDED: Start adaptive timeout decay =====
                self.last_timeout_decay_time = self.get_clock().now().to_msg()
                self.get_logger().info(f"[{self.name}] Started adaptive memory timeout decay (initial={self.memory_timeout_sec}s)")
                # ===== HJ ADDED END =====
            else:
                self.get_logger().error(f"[{self.name}] [THREAD] Fixed path generation FAILED")
                # ===== HJ MODIFIED: Mark as attempted to prevent infinite retry =====
                self.fixed_path_generation_attempted = True
                self.get_logger().warning(f"[{self.name}] [THREAD] Marked generation as attempted. Will NOT retry.")
                self.get_logger().warning(f"[{self.name}] [THREAD] Continuing with GB mode for this session.")
                # ===== HJ MODIFIED END =====

        except Exception as e:
            self.get_logger().error(f"[{self.name}] [THREAD] Exception in path generation thread: {e}")
            import traceback
            traceback.print_exc()
            # ===== HJ ADDED: Also mark as attempted on exception =====
            self.fixed_path_generation_attempted = True
            # ===== HJ ADDED END =====

        finally:
            # Always mark as attempted and clear generating flag when done (success, fail, or any exception)
            self.fixed_path_generation_attempted = True
            self.fixed_path_generating = False
            self.get_logger().info(f"[{self.name}] [THREAD] Path generation thread finished.")

    def _generate_fixed_path(self, verified_obs: List[Tuple[int, int, float, float]], nearby_obs: List[Tuple[int, int, float, float]]) -> bool:
        """
        Phase 3: Generate fixed optimized path using GB optimizer.

        Uses same approach as mapping.launch + global_planner_node.py:
        - Creates CSV in current working directory (usually ~/.ros/ or ~/catkin_ws/)
        - Reads racecar_f110.ini from stack_master/config/{racecar_version}
        - Calls trajectory_optimizer with mincurv_iqp

        Args:
            verified_obs: List of (sector_id, obs_id, s, d) tuples for interfering obstacles (GB optimizer)
            nearby_obs: List of (sector_id, obs_id, s, d) tuples for nearby obstacles (spline occupancy grid)

        Returns:
            True if path generation successful, False otherwise
        """
        # Temporary CSV filename (relative path, created in current working directory)
        csv_filename = 'smart_static_avoidance_temp.csv'

        try:
            self.get_logger().info(f"[{self.name}] Phase 3: Starting GB optimizer path generation...")

            # Step 1: Prepare reftrack
            # Method selection: stable_centerline (new) vs occupancy_grid (old)
            use_stable_centerline = True  
            # use_stable_centerline = False # Not Good because of Multi Contours


            if use_stable_centerline:
                self.get_logger().info(f"[{self.name}] Using STABLE CENTERLINE approach (new method)...")

                # Convert obstacles from Frenet to XY coordinates
                # Format: (sector_id, obs_id, s, d) → (pos [x,y], radius)
                #
                # Two obstacle lists:
                # 1. obstacles: verified_obs + nearby_obs (for GLOBAL optimization boundaries)
                #    - GLOBAL opt uses these as boundary constraints, optimizer ignores if path doesn't intersect
                # 2. obstacles_interfering: verified_obs only (for LOCAL optimization)
                #    - LOCAL opt directly calculates shift for each obstacle, so only include interfering ones

                def convert_obs_to_xy(obs_list, list_name):
                    seen = set()
                    unique_obs = []
                    for sector_id, obs_id, s, d in obs_list:
                        if (sector_id, obs_id) not in seen:
                            seen.add((sector_id, obs_id))
                            unique_obs.append((sector_id, obs_id, s, d))

                    result = []
                    for sector_id, obs_id, s, d in unique_obs:
                        xy = self.converter.get_cartesian(np.array([s]), np.array([d]))
                        pos = np.array([xy[0, 0], xy[1, 0]])

                        key = (sector_id, obs_id)
                        if key in self.static_obs_memory and 'radius' in self.static_obs_memory[key]:
                            radius = self.static_obs_memory[key]['radius']
                        else:
                            radius = OBSTACLE_RADIUS

                        result.append((pos, radius))
                        self.get_logger().info(f"[{self.name}]   [{list_name}] Converted obstacle: sector={sector_id}, id={obs_id}, s={s:.2f}, d={d:.2f} → x={pos[0]:.2f}, y={pos[1]:.2f}, r={radius:.2f}")

                    return result, len(obs_list), len(unique_obs)

                # All obstacles (verified + nearby) for GLOBAL optimization boundaries
                all_obs_frenet = verified_obs + nearby_obs
                obstacles, total_all, unique_all = convert_obs_to_xy(all_obs_frenet, "ALL")
                self.get_logger().info(f"[{self.name}] All obstacles (for GLOBAL bounds): total={total_all}, unique={unique_all} (removed {total_all - unique_all} duplicates)")

                # Only interfering obstacles (verified_obs) for LOCAL optimization
                obstacles_interfering, total_interfering, unique_interfering = convert_obs_to_xy(verified_obs, "INTERFERING")
                self.get_logger().info(f"[{self.name}] Interfering obstacles (for LOCAL opt): total={total_interfering}, unique={unique_interfering}")

                self.get_logger().info(f"[{self.name}] Converted obstacles: {len(obstacles)} for GLOBAL bounds, {len(obstacles_interfering)} for LOCAL opt")

                # Load map for occupancy grid generation (needed for GridFilter)
                if not hasattr(self, 'original_bw'):
                    self.get_logger().info(f"[{self.name}] Loading map for occupancy grid generation...")
                    map_file = os.path.join(self.map_dir, f'{self.map_name}.yaml')
                    with open(map_file, 'r') as f:
                        map_data = yaml.safe_load(f)

                    og_img_path = os.path.join(self.map_dir, map_data['image'])
                    og_img = cv2.imread(og_img_path, cv2.IMREAD_GRAYSCALE)
                    map_resolution = map_data['resolution']
                    map_origin_x = map_data['origin'][0]
                    map_origin_y = map_data['origin'][1]

                    bw = np.where(og_img > 200, 255, 0).astype(np.uint8)
                    self.original_bw = bw.copy()
                    self.modified_resolution = map_resolution
                    self.map_origin = (map_origin_x, map_origin_y)
                    self.get_logger().info(f"[{self.name}] Loaded map: {og_img.shape}, res={map_resolution}, origin=({map_origin_x}, {map_origin_y})")

                reftrack = self._create_reftrack_with_stable_centerline(
                    obstacles=obstacles,
                    visualize=True  # Save debug visualization
                )
                if reftrack is None:
                    self.get_logger().error(f"[{self.name}] Failed to create reftrack with stable centerline")
                    return False
            else:
                self.get_logger().info(f"[{self.name}] Using OCCUPANCY GRID approach (old method)...")
                reftrack = self._prepare_reftrack_from_occupancy_grid(verified_obs)
                if reftrack is None:
                    self.get_logger().error(f"[{self.name}] Failed to prepare reftrack from occupancy grid")
                    return False

            self.get_logger().info(f"[{self.name}] Reftrack prepared: {reftrack.shape[0]} points")

            # ===== HJ ADDED: Switch between global and local optimization =====
            if ENABLE_LOCAL_OPT:
                # LOCAL OPTIMIZATION MODE
                # Use obstacles_interfering (verified_obs only) - LOCAL opt directly calculates shift for each obstacle
                self.get_logger().info(f"[{self.name}] ENABLE_LOCAL_OPT=True, using local optimization mode")
                try:
                    trajectory_opt, bound_r, bound_l = self._generate_local_optimized_path(
                        obstacles=obstacles_interfering,
                        reftrack=reftrack
                    )
                    # Local optimization doesn't return est_time, use placeholder
                    est_time = 0.0  # Local opt doesn't calculate lap time estimate
                    self.get_logger().info(f"[{self.name}] Local optimization SUCCESS! Trajectory: {trajectory_opt.shape[0]} points")
                except Exception as e:
                    self.get_logger().error(f"[{self.name}] Local optimization FAILED: {e}")
                    self.get_logger().warning(f"[{self.name}] Falling back to GB mode")
                    return False
            else:
                # GLOBAL OPTIMIZATION MODE (original behavior)
                self.get_logger().info(f"[{self.name}] ENABLE_LOCAL_OPT=False, using global optimization mode")

                # Step 2: Save to temporary CSV in current working directory (ONCE)
                self._save_reftrack_to_csv(reftrack, csv_filename)
                self.get_logger().info(f"[{self.name}] Reftrack saved to CSV")

                # Step 3: Retry with progressively reduced safety_width ratios if optimization fails
                # Reftrack is prepared ONCE above, only safety_width changes in retry loop
                # ===== HJ MODIFIED: Only try down to 0.9 ratio, then give up and use GB mode =====
                
                # safety_width_ratios = [1.0, 0.95, 0.9]  # Stop at 0.9
                safety_width_ratios = [1.0]

                trajectory_opt = None
                last_error = None
                used_local_opt_fallback = False  # Track if LOCAL_OPT fallback was used

                for idx, ratio in enumerate(safety_width_ratios):
                    try:
                        safety_width_adjusted = self.safety_width * ratio
                        self.get_logger().warning(
                            f"[{self.name}] Attempt {idx+1}/{len(safety_width_ratios)}: "
                            f"safety_width={safety_width_adjusted:.3f}m (ratio={ratio:.2f})"
                        )

                        # Call GB optimizer with adjusted safety_width
                        self.get_logger().info(f"[{self.name}] Calling GB optimizer (mincurv_iqp)...")
                        trajectory_opt, bound_r, bound_l, est_time = self._call_gb_optimizer(
                            csv_filename,
                            safety_width=safety_width_adjusted
                        )
                        self.get_logger().info(
                            f"[{self.name}] GB optimizer SUCCESS! "
                            f"Trajectory: {trajectory_opt.shape[0]} points, Est time: {est_time:.2f}s"
                        )
                        break  # Success! Exit retry loop

                    except (ValueError, RuntimeError) as e:
                        # Catch ValueError (constraints inconsistent), RuntimeError (track too small, timeout, IQP convergence)
                        error_msg = str(e)
                        if ("constraints are inconsistent" in error_msg or
                            "Problem not solvable" in error_msg or
                            "timeout" in error_msg.lower() or
                            "IQP failed to converge" in error_msg):
                            self.get_logger().error(
                                f"[{self.name}] Attempt {idx+1}/{len(safety_width_ratios)} failed: {error_msg}. "
                                f"Trying next configuration..."
                            )
                            last_error = e
                            continue  # Try next configuration
                        else:
                            raise  # Re-raise if different error

                # Check if all attempts failed
                if trajectory_opt is None:
                    self.get_logger().error(f"[{self.name}] ===== GLOBAL OPTIMIZATION FAILED =====")
                    self.get_logger().error(f"[{self.name}] All {len(safety_width_ratios)} optimization attempts failed (down to {safety_width_ratios[-1]} ratio)")

                    if LOCAL_OPT_FALLBACK:
                        # Fallback to LOCAL_OPT
                        # Use obstacles_interfering (verified_obs only) - LOCAL opt directly calculates shift for each obstacle
                        self.get_logger().warning(f"[{self.name}] LOCAL_OPT_FALLBACK=True, trying local optimization...")
                        try:
                            trajectory_opt, bound_r, bound_l = self._generate_local_optimized_path(
                                obstacles=obstacles_interfering,
                                reftrack=reftrack
                            )
                            est_time = 0.0
                            used_local_opt_fallback = True  # Mark that we used LOCAL_OPT fallback
                            self.get_logger().info(f"[{self.name}] Local optimization fallback SUCCESS! Trajectory: {trajectory_opt.shape[0]} points")
                        except Exception as e:
                            self.get_logger().error(f"[{self.name}] Local optimization fallback FAILED: {e}")
                            self.get_logger().warning(f"[{self.name}] Falling back to GB mode")
                            return False
                    else:
                        self.get_logger().warning(f"[{self.name}] LOCAL_OPT_FALLBACK=False, falling back to GB mode")
                        self.get_logger().warning(f"[{self.name}] Will use GB path with dynamic obstacle avoidance (do_spline)")
                        return False
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Step 3.5: Apply post-smoothing if enabled =====
            # Smoothing is applied only for GLOBAL optimization (not LOCAL_OPT)
            # LOCAL_OPT already produces smooth curves with BPoly transition smoothing
            # Skip smoothing if: ENABLE_LOCAL_OPT=True OR used_local_opt_fallback=True
            skip_smoothing = ENABLE_LOCAL_OPT or (not ENABLE_LOCAL_OPT and used_local_opt_fallback)

            if SMOOTH_OPT_OUTPUT and not skip_smoothing:
                self.get_logger().info(f"[{self.name}] SMOOTH_OPT_OUTPUT=True, applying Savitzky-Golay smoothing...")
                trajectory_opt = self._smooth_trajectory_output(trajectory_opt, target_stepsize=0.1)
            elif skip_smoothing:
                self.get_logger().info(f"[{self.name}] LOCAL_OPT mode (or fallback): skipping post-smoothing (already has BPoly smoothing)")
            else:
                self.get_logger().info(f"[{self.name}] SMOOTH_OPT_OUTPUT=False, using original optimizer output")
            # ===== HJ ADDED END =====

            # Step 4: Package into OTWpntArray with visualization markers
            self.fixed_path_wpnts, self.fixed_path_markers = self._package_to_otwpntarray(trajectory_opt)
            self.get_logger().info(f"[{self.name}] Packaged {len(self.fixed_path_wpnts.wpnts)} waypoints and {len(self.fixed_path_markers.markers)} markers")

            # Step 4.2: Calculate d_left, d_right for each waypoint using modified bounds (if stable centerline)
            if use_stable_centerline and hasattr(self, 'bound_r_modified') and hasattr(self, 'bound_l_modified'):
                self.get_logger().info(f"[{self.name}] Calculating d_left/d_right from modified bounds...")
                for wpnt in self.fixed_path_wpnts.wpnts:
                    pos = np.array([wpnt.x_m, wpnt.y_m])

                    # Distance to right bound
                    dists_r = np.linalg.norm(self.bound_r_modified - pos, axis=1)
                    wpnt.d_right = np.min(dists_r)

                    # Distance to left bound
                    dists_l = np.linalg.norm(self.bound_l_modified - pos, axis=1)
                    wpnt.d_left = np.min(dists_l)

                self.get_logger().info(f"[{self.name}] Calculated d_left/d_right for {len(self.fixed_path_wpnts.wpnts)} waypoints")

            self.get_logger().info(f"[{self.name}] Fixed path packaged: {len(self.fixed_path_wpnts.wpnts)} waypoints, {len(self.fixed_path_markers.markers)} markers")

            # ===== HJ REMOVED: No longer needed - smoothing doesn't add closing point anymore =====
            # Since smoothing (line 3354-3360) no longer adds a duplicate closing point,
            # we don't need to remove the last waypoint here
            # State machine will handle closed loop with modulo wrapping: (min_idx + i) % num_wpnts
            # if len(self.fixed_path_wpnts.wpnts) > 0:
            #     original_count = len(self.fixed_path_wpnts.wpnts)
            #     self.fixed_path_wpnts.wpnts = self.fixed_path_wpnts.wpnts[:-1]
            #     self.get_logger().info(f"[{self.name}] Removed last waypoint for open loop: {original_count} → {len(self.fixed_path_wpnts.wpnts)} waypoints")
            # ===== HJ REMOVED END =====

            # Step 4.5: Create FrenetConverter for fixed path (for interference checking)
            x_array = np.array([w.x_m for w in self.fixed_path_wpnts.wpnts])
            y_array = np.array([w.y_m for w in self.fixed_path_wpnts.wpnts])
            z_array = np.array([w.z_m for w in self.fixed_path_wpnts.wpnts])

            # ===== HJ ADDED: Check for NaN values =====
            if np.any(np.isnan(x_array)) or np.any(np.isnan(y_array)):
                self.get_logger().error(f"[{self.name}] NaN detected in fixed path waypoints! "
                            f"x_nan={np.sum(np.isnan(x_array))}, y_nan={np.sum(np.isnan(y_array))}")
                raise ValueError("Fixed path contains NaN waypoints")
            # ===== HJ ADDED END =====

            self.fixed_converter = FrenetConverter(x_array, y_array, z_array)
            self.get_logger().info(f"[{self.name}] Created FrenetConverter for fixed path ({len(x_array)} points)")

            # ===== HJ ADDED: Step 3.5: Regenerate occupancy grid with nearby_obs (for spline) =====
            # Define paths first (needed for GridFilter later)
            obstacles_only_png_path = os.path.join(self.map_dir, f'{self.map_name}_with_obs.png')
            modified_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')

            # ALWAYS generate occupancy grid (needed for GridFilter in do_spline)
            # Use nearby_obs instead of verified_obs for more accurate spline boundaries
            # nearby_obs includes ALL stable obstacles within 2.5m (not just interfering ones)
            self.get_logger().info(f"[{self.name}] Regenerating occupancy grid with nearby obstacles ({len(nearby_obs)} obs)...")
            _, obstacles_only_bw_nearby = self._add_obstacles_to_occupancy_grid(
                self.original_bw.copy(),
                nearby_obs,  # Use all nearby obstacles (not just interfering)
                self.modified_resolution,
                self.map_origin[0],
                self.map_origin[1]
            )

            # Save nearby obstacles-only map (for GridFilter and d_left/d_right calculation)
            cv2.imwrite(obstacles_only_png_path, obstacles_only_bw_nearby)
            self.get_logger().info(f"[{self.name}] Saved nearby obstacles-only grid to {obstacles_only_png_path}")

            # ===== HJ ADDED: Calculate Smart conservative d_left/d_right from stored bounds =====
            if USE_SAFE_GRID and hasattr(self, 'bound_r_conservative_modified') and hasattr(self, 'bound_l_conservative_modified'):
                self.get_logger().info(f"[{self.name}] USE_SAFE_GRID=True, calculating Smart conservative d_left/d_right...")

                # Calculate d_left/d_right for Fixed path waypoints using conservative bounds
                reftrack_conservative = self._calculate_reftrack_distances(
                    np.array([[wp.x_m, wp.y_m] for wp in self.fixed_path_wpnts.wpnts]),
                    self.bound_r_conservative_modified,
                    self.bound_l_conservative_modified
                )

                # Store in dictionaries
                self.smart_conservative_d_left = {i: reftrack_conservative[i, 3] for i in range(len(self.fixed_path_wpnts.wpnts))}
                self.smart_conservative_d_right = {i: reftrack_conservative[i, 2] for i in range(len(self.fixed_path_wpnts.wpnts))}

                self.get_logger().info(f"[{self.name}] Smart conservative bounds calculated: {len(self.smart_conservative_d_left)} waypoints")

                # Create Smart conservative GridFilter from _for_spliner_with_obs.png
                # Generate _for_spliner_with_obs.png by adding obstacles to _for_spliner.png
                conservative_with_obs_path = self._create_conservative_map_with_obstacles(
                    nearby_obs,
                    self.modified_resolution,
                    self.map_origin[0],
                    self.map_origin[1]
                )

                if conservative_with_obs_path:
                    self.map_filter_conservative_with_obs = GridFilter(map_topic=None, debug=False)
                    if self.map_filter_conservative_with_obs.load_from_file(conservative_with_obs_path, modified_yaml_path):
                        self.map_filter_conservative_with_obs.set_erosion_kernel_size(self.kernel_size)
                        self.get_logger().info(f"[{self.name}] Created Smart conservative GridFilter from {conservative_with_obs_path}")
                    else:
                        self.get_logger().warning(f"[{self.name}] Failed to load Smart conservative GridFilter")
                        self.map_filter_conservative_with_obs = None
                else:
                    self.get_logger().warning(f"[{self.name}] Failed to create conservative map with obstacles")
            # ===== HJ ADDED END =====

            # ===== HJ MODIFIED: Calculate d_left, d_right using nearby_obs =====
            # Step 4.5.5: Extract track boundaries and calculate d_left, d_right
            # Always calculate for LOCAL_OPT (needs Fixed path Frenet-based boundaries)
            # For GLOBAL mode with stable centerline, reftrack already has w_tr_right/left
            if ENABLE_LOCAL_OPT or not use_stable_centerline:
                self.get_logger().info(f"[{self.name}] Calculating d_left/d_right for Fixed path (LOCAL_OPT={ENABLE_LOCAL_OPT})...")
                self._calculate_fixed_path_boundaries(obstacles_only_png_path, modified_yaml_path, nearby_obs)
            else:
                self.get_logger().info(f"[{self.name}] Skipping d_left/d_right calculation (using stable centerline with reftrack distances)")
            # ===== HJ MODIFIED END =====

            # Step 4.6: Create GridFilter for obstacles-only map (for do_spline bounds checking)
            # ===== HJ MODIFIED: Use nearby obstacles-only map =====
            # ALWAYS create GridFilter (needed for do_spline collision checking)
            if os.path.exists(obstacles_only_png_path):
                self.map_filter_fixed = GridFilter(map_topic=None, debug=False)
                if self.map_filter_fixed.load_from_file(obstacles_only_png_path, modified_yaml_path):
                    self.map_filter_fixed.set_erosion_kernel_size(self.kernel_size)
                    self.get_logger().info(f"[{self.name}] Created GridFilter for nearby obstacles-only map: {obstacles_only_png_path}")

                    # ===== HJ ADDED: Publish map for visualization (ONE-TIME, latch=True) =====
                    # USE_SAFE_GRID=True: publish conservative map with obstacles
                    # USE_SAFE_GRID=False: publish original map with obstacles
                    if USE_SAFE_GRID and hasattr(self, 'map_filter_conservative_with_obs') and self.map_filter_conservative_with_obs is not None:
                        # Publish conservative map with obstacles (_for_spliner_with_obs.png)
                        conservative_with_obs_path = os.path.join(self.map_dir, f'{self.map_name}_for_spliner_with_obs.png')
                        if os.path.exists(conservative_with_obs_path):
                            self._publish_obstacles_only_map(conservative_with_obs_path, modified_yaml_path)
                            self.get_logger().info(f"[{self.name}] Published CONSERVATIVE obstacles map to /map_with_obs (latched)")
                        else:
                            self.get_logger().warning(f"[{self.name}] Conservative map with obstacles not found, publishing original")
                            self._publish_obstacles_only_map(obstacles_only_png_path, modified_yaml_path)
                    else:
                        # Publish original obstacles-only map
                        self._publish_obstacles_only_map(obstacles_only_png_path, modified_yaml_path)
                        self.get_logger().info(f"[{self.name}] Published original obstacles-only map to /map_with_obs (latched)")
                    # ===== HJ ADDED END =====
                else:
                    self.get_logger().warning(f"[{self.name}] Failed to load obstacles-only map, will use original map filter")
                    self.map_filter_fixed = None
            else:
                self.get_logger().warning(f"[{self.name}] Obstacles-only map not found at {obstacles_only_png_path}, will use original map filter")
                self.map_filter_fixed = None
            # ===== HJ MODIFIED END =====

            # Step 5: Save to JSON file (for future use)
            # Load ORIGINAL wall boundaries (from global_waypoints.json, without obstacles)
            bound_r_original, bound_l_original = self._get_original_wall_boundaries()
            if bound_r_original is not None and bound_l_original is not None:
                self._save_smart_global_waypoints_json(
                    trajectory_wpnts=self.fixed_path_wpnts,
                    trajectory_markers=self.fixed_path_markers,
                    bound_r_original=bound_r_original,
                    bound_l_original=bound_l_original,
                    est_time=est_time
                )
            else:
                self.get_logger().warning(f"[{self.name}] Failed to load original boundaries for JSON save")

            # Step 6: Clean up temporary CSV file
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
                self.get_logger().info(f"[{self.name}] Cleaned up temporary CSV file")

            return True

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Exception in _generate_fixed_path: {e}")
            import traceback
            traceback.print_exc()

            # Clean up temporary CSV file on error
            if os.path.exists(csv_filename):
                try:
                    os.remove(csv_filename)
                    self.get_logger().info(f"[{self.name}] Cleaned up temporary CSV file after error")
                except:
                    pass

            return False

    def _prepare_reftrack_from_occupancy_grid(
        self,
        verified_obs: List[Tuple[int, int, float, float]]
    ) -> np.ndarray:
        """
        Prepare reftrack using occupancy grid + contour extraction (like mapping.launch).
        This is more accurate than Frenet-based approach.

        Steps:
        1. Load map PNG and YAML
        2. Convert obstacles (s, d) to pixel coordinates
        3. Draw obstacles on occupancy grid (0.5m circle + 0.4m line to nearest wall)
        4. Extract centerline using skeletonize
        5. Extract bounds using watershed
        6. Calculate reftrack distances

        Returns:
            reftrack: [x_m, y_m, w_tr_right_m, w_tr_left_m]
        """
        try:
            # Step 1: Load map PNG and YAML
            map_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                f'{self.map_name}.png'
            )
            yaml_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                f'{self.map_name}.yaml'
            )

            if not os.path.exists(map_path) or not os.path.exists(yaml_path):
                self.get_logger().error(f"[{self.name}] Map files not found: {map_path}, {yaml_path}")
                return None

            # Load map image
            og_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if og_img is None:
                self.get_logger().error(f"[{self.name}] Failed to load map image: {map_path}")
                return None

            # Load map metadata
            with open(yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)

            map_resolution = map_data['resolution']  # m/pixel
            map_origin_x = map_data['origin'][0]     # meters
            map_origin_y = map_data['origin'][1]     # meters

            self.get_logger().info(f"[{self.name}] Loaded map: {og_img.shape}, res={map_resolution}, origin=({map_origin_x}, {map_origin_y})")

            # Create binary mask (white = free space, black = occupied)
            bw = np.where(og_img > 200, 255, 0).astype(np.uint8)

            # ===== HJ ADDED: Store original bw for later use (nearby_obs occupancy grid) =====
            self.original_bw = bw.copy()
            self.modified_resolution = map_resolution
            self.map_origin = (map_origin_x, map_origin_y)
            # ===== HJ ADDED END =====

            # Step 2 & 3: Add obstacles to map (GB optimizer용 - centerline 추출만)
            # ===== HJ MODIFIED: Only generate modified_bw for centerline extraction =====
            # obstacles_only version not needed here - will be generated later with nearby_obs
            modified_bw, _ = self._add_obstacles_to_occupancy_grid(
                bw, verified_obs, map_resolution, map_origin_x, map_origin_y
            )

            # DEBUG: Save modified occupancy grid (with wall lines - for skeletonize)
            debug_path = os.path.join(self.map_dir, f'{self.map_name}_modified_result.png')
            cv2.imwrite(debug_path, modified_bw)
            self.get_logger().info(f"[{self.name}] Saved modified occupancy grid (with wall lines) to {debug_path}")
            # Note: obstacles_only.png will be generated later with nearby_obs in Step 3.5
            # ===== HJ MODIFIED END =====

            # Step 4: Extract centerline using skeletonize (same as global_planner_node)
            # Filtering with morphological opening
            kernel1 = np.ones((9, 9), np.uint8)
            opening = cv2.morphologyEx(modified_bw, cv2.MORPH_OPEN, kernel1, iterations=1)

            # Get morphological skeleton of the map
            skeleton = skeletonize(opening, method='lee')

            # DEBUG: Save skeleton
            skeleton_path = os.path.join(self.map_dir, f'{self.map_name}_skeleton.png')
            cv2.imwrite(skeleton_path, skeleton)
            self.get_logger().info(f"[{self.name}] Saved skeleton to {skeleton_path}")

            centerline_pixels = self._extract_centerline_from_skeleton(skeleton)

            if centerline_pixels is None:
                self.get_logger().error(f"[{self.name}] Failed to extract centerline")
                return None

            # Step 5: Extract bounds using watershed (same as global_planner_node)
            # If watershed fails, fallback to distance transform
            try:
                bound_r_pixels, bound_l_pixels = self._extract_bounds_from_watershed(
                    centerline_pixels, opening
                )
                dist_transform = None
                self.get_logger().info(f"[{self.name}] Using watershed for track bound extraction...")
            except IOError:
                self.get_logger().warning(f"[{self.name}] More than two track bounds detected with watershed algorithm")
                self.get_logger().info(f"[{self.name}] Trying with simple distance transform...")
                bound_r_pixels = None
                bound_l_pixels = None
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            # Convert to meters
            map_height = og_img.shape[0]  # Image height in pixels
            centerline_m = self._pixels_to_meters(centerline_pixels, map_resolution, map_origin_x, map_origin_y, map_height)

            if dist_transform is not None:
                # Use distance transform (fallback) - left and right widths are the same
                width_track = dist_transform[centerline_pixels[:, 1].astype(int),
                                             centerline_pixels[:, 0].astype(int)] * map_resolution
                bound_r_m = None
                bound_l_m = None
            else:
                bound_r_m = self._pixels_to_meters(bound_r_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
                bound_l_m = self._pixels_to_meters(bound_l_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
                width_track = None

            # ===== HJ ADDED: Interpolate centerline for uniform spacing =====
            # Skeleton gives uneven spacing (~0.05m avg), interpolate to uniform stepsize
            # Adjust centerline_stepsize for speed vs accuracy tradeoff:
            #   0.1m = Global planner default (accurate, slower)
            #   0.3m = 3x faster (recommended for real-time)
            #   0.5m = 5x faster (speed priority)
            from global_racetrajectory_optimization import helper_funcs_glob

            centerline_stepsize = 0.1  # meters - ADJUST THIS VALUE

            original_count = centerline_m.shape[0]
            centerline_tmp = np.column_stack((centerline_m, np.zeros((centerline_m.shape[0], 2))))
            centerline_m_int = helper_funcs_glob.src.interp_track.interp_track(
                reftrack=centerline_tmp, stepsize_approx=centerline_stepsize
            )
            centerline_m = centerline_m_int[:, :2]  # Extract x, y only
            self.get_logger().info(f"[{self.name}] Centerline interpolated: {original_count} → {centerline_m.shape[0]} points (stepsize={centerline_stepsize}m)")
            # ===== HJ ADDED END =====

            # ===== HJ ADDED: Validate and correct centerline direction (same as global_planner_node.py) =====
            # Compare centerline direction with GB raceline direction at the same XY location
            # ===== HJ MODIFIED: Use signed area to determine direction (more robust for closed loops) =====
            # Calculate signed area to determine if centerline is CCW or CW
            is_ccw = self._is_polygon_ccw(centerline_m)

            self.get_logger().info(f"[{self.name}] Centerline direction: {'CCW (counter-clockwise)' if is_ccw else 'CW (clockwise)'}")

            # Determine desired direction based on reverse_mapping
            # reverse_mapping=False → want CCW (반시계)
            # reverse_mapping=True → want CW (시계)
            want_ccw = not self.reverse_mapping

            # Flip if current direction doesn't match desired direction
            if is_ccw != want_ccw:
                centerline_pixels = np.flip(centerline_pixels, axis=0)
                centerline_m = np.flip(centerline_m, axis=0)
                if bound_r_m is not None and bound_l_m is not None:
                    bound_r_m, bound_l_m = bound_l_m, bound_r_m  # Swap bounds when flipping
                self.get_logger().info(f"[{self.name}] Centerline FLIPPED to {'CCW' if want_ccw else 'CW'} (reverse_mapping={self.reverse_mapping})")
            else:
                self.get_logger().info(f"[{self.name}] Centerline direction correct ({'CCW' if want_ccw else 'CW'})")
            # ===== HJ MODIFIED END =====

            # Step 6: Calculate reftrack distances
            if width_track is not None:
                # Use distance transform (fallback) - left and right widths are the same
                # Interpolate width_track to match centerline_m after interpolation
                if len(width_track) != len(centerline_m):
                    width_track = np.interp(np.arange(0, len(centerline_m)), np.arange(0, len(width_track)), width_track)
                reftrack = np.column_stack((centerline_m, width_track, width_track))
            else:
                reftrack = self._calculate_reftrack_distances(centerline_m, bound_r_m, bound_l_m)

            self.get_logger().info(f"[{self.name}] Prepared reftrack from occupancy grid: {reftrack.shape[0]} points")
            return reftrack

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to prepare reftrack from occupancy grid: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_original_wall_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load original wall boundaries from global_waypoints.json (cached)."""
        # Return cached values if already loaded
        if hasattr(self, '_original_bound_r') and hasattr(self, '_original_bound_l') and \
           self._original_bound_r is not None and self._original_bound_l is not None:
            return self._original_bound_r, self._original_bound_l

        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                'global_waypoints.json'
            )

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract from markers array by color
            markers = data['trackbounds_markers']['markers']
            bound_right = []  # purple (0.5, 0.0, 0.5)
            bound_left = []   # green (0.5, 1.0, 0.0)

            for m in markers:
                pos = m['pose']['position']
                color = (m['color']['r'], m['color']['g'], m['color']['b'])

                if color == (0.5, 0.0, 0.5):  # Purple = Right
                    bound_right.append([pos['x'], pos['y']])
                elif color == (0.5, 1.0, 0.0):  # Green = Left
                    bound_left.append([pos['x'], pos['y']])

            bound_r = np.array(bound_right)
            bound_l = np.array(bound_left)

            # Cache the results
            self._original_bound_r = bound_r
            self._original_bound_l = bound_l

            self.get_logger().info(f"[{self.name}] Loaded wall boundaries: right={len(bound_r)}, left={len(bound_l)}")
            return bound_r, bound_l

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to load wall boundaries: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_conservative_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load conservative wall boundaries from _for_spliner.png by extracting bounds.

        Returns:
            (bound_r, bound_l): Right and left boundary arrays
        """
        try:
            # Load conservative map
            conservative_map_path = os.path.join(self.map_dir, f'{self.map_name}_for_spliner.png')
            if not os.path.exists(conservative_map_path):
                self.get_logger().error(f"[{self.name}] Conservative map not found: {conservative_map_path}")
                return None, None

            conservative_img = cv2.imread(conservative_map_path, cv2.IMREAD_GRAYSCALE)
            if conservative_img is None:
                self.get_logger().error(f"[{self.name}] Failed to load conservative map: {conservative_map_path}")
                return None, None

            # Flip image (same as global_planner_node)
            # conservative_bw = cv2.flip(conservative_img, 0)

            # Filtering with morphological opening (same as global_planner_node)
            kernel1 = np.ones((9, 9), np.uint8)
            opening = cv2.morphologyEx(conservative_img, cv2.MORPH_OPEN, kernel1, iterations=1)

            # Get morphological skeleton of the map (same as global_planner_node)
            skeleton = skeletonize(opening, method='lee')
            centerline_pixels = self._extract_centerline_from_skeleton(skeleton)

            if centerline_pixels is None:
                self.get_logger().error(f"[{self.name}] Failed to extract centerline from conservative map")
                return None, None

            # Extract bounds using watershed (same as global_planner_node)
            # If watershed fails, fallback to distance transform
            try:
                bound_r_pixels, bound_l_pixels = self._extract_bounds_from_watershed(
                    centerline_pixels, opening
                )
                self.get_logger().info(f"[{self.name}] Using watershed for conservative bound extraction...")
            except IOError:
                self.get_logger().warning(f"[{self.name}] More than two track bounds detected with watershed algorithm")
                self.get_logger().info(f"[{self.name}] Trying with simple distance transform...")
                # For _get_conservative_boundaries, we can't use distance transform as fallback
                # because we need actual boundary points. Return None to signal failure.
                return None, None

            # Convert to meters
            yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')
            with open(yaml_path, 'r') as f:
                map_data = yaml.safe_load(f)

            map_resolution = map_data['resolution']
            map_origin_x = map_data['origin'][0]
            map_origin_y = map_data['origin'][1]
            map_height = conservative_img.shape[0]

            bound_r_m = self._pixels_to_meters(bound_r_pixels, map_resolution, map_origin_x, map_origin_y, map_height)
            bound_l_m = self._pixels_to_meters(bound_l_pixels, map_resolution, map_origin_x, map_origin_y, map_height)

            self.get_logger().info(f"[{self.name}] Loaded conservative boundaries: right={len(bound_r_m)}, left={len(bound_l_m)}")
            return bound_r_m, bound_l_m

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to load conservative boundaries: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_original_centerline(self) -> np.ndarray:
        """
        Load original centerline from global_waypoints.json (cached).

        Returns:
            centerline: [[x, y], ...] numpy array
        """
        # Return cached value if already loaded
        if hasattr(self, '_original_centerline') and self._original_centerline is not None:
            return self._original_centerline

        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                '../../../stack_master/maps',
                self.map_name,
                'global_waypoints.json'
            )

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract centerline from centerline_markers, gb_raceline, or waypoints
            if 'centerline_markers' in data and 'markers' in data['centerline_markers']:
                markers = data['centerline_markers']['markers']
                centerline_points = []
                for m in markers:
                    pos = m['pose']['position']
                    centerline_points.append([pos['x'], pos['y']])
                centerline = np.array(centerline_points)
                self.get_logger().info(f"[{self.name}] Loaded centerline from 'centerline_markers'")
            elif 'gb_raceline' in data and 'markers' in data['gb_raceline']:
                markers = data['gb_raceline']['markers']
                centerline_points = []
                for m in markers:
                    pos = m['pose']['position']
                    centerline_points.append([pos['x'], pos['y']])
                centerline = np.array(centerline_points)
                self.get_logger().info(f"[{self.name}] Loaded centerline from 'gb_raceline'")
            elif 'waypoints' in data:
                # Fallback: use waypoints as centerline
                waypoints = data['waypoints']
                centerline_points = []
                for wp in waypoints:
                    centerline_points.append([wp['x_m'], wp['y_m']])
                centerline = np.array(centerline_points)
                self.get_logger().info(f"[{self.name}] Loaded centerline from 'waypoints'")
            else:
                self.get_logger().error(f"[{self.name}] No centerline data found in global_waypoints.json")
                return None

            # Cache the result
            self._original_centerline = centerline

            self.get_logger().info(f"[{self.name}] Loaded original centerline: {len(centerline)} points")
            return centerline

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to load centerline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_obstacle_to_bound(
        self,
        bound: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float,
        num_points: int = 200
    ) -> np.ndarray:
        """
        Add FULL circular obstacle (360 degrees) to bound with dense points.

        Args:
            bound: Original bound [[x, y], ...]
            obstacle_pos: [x, y] obstacle center
            obstacle_radius: Obstacle radius in meters
            num_points: Number of points for full circle (default: 100 for dense sampling)

        Returns:
            modified_bound: Bound with full obstacle circle inserted
        """
        # Find closest bound point to obstacle
        distances = np.linalg.norm(bound - obstacle_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Generate FULL circular points around obstacle (360 degrees, very dense)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        circle_points = []
        for angle in angles:
            x = obstacle_pos[0] + obstacle_radius * np.cos(angle)
            y = obstacle_pos[1] + obstacle_radius * np.sin(angle)
            circle_points.append([x, y])

        circle_points = np.array(circle_points)

        # Insert FULL circle into bound at closest point
        modified_bound = np.vstack([
            bound[:closest_idx],
            circle_points,
            bound[closest_idx:]
        ])

        return modified_bound

    def _shift_centerline_around_obstacle_spline(
        self,
        centerline: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float,
        obstacle_side: str = 'left'
    ) -> np.ndarray:
        """
        Shift centerline around obstacle using spline approach (similar to do_spline).

        Uses centerline-based Frenet coordinates and creates smooth spline with heading consideration.

        Args:
            centerline: Original centerline [[x, y], ...]
            obstacle_pos: [x, y] obstacle center
            obstacle_radius: Obstacle radius
            obstacle_side: 'left' or 'right' - which side obstacle is on

        Returns:
            shifted_centerline: Modified centerline with spline-based shift
        """
        from scipy.interpolate import BPoly

        # Step 1: Create Frenet converter for centerline
        centerline_converter = FrenetConverter(
            waypoints_x=centerline[:, 0],
            waypoints_y=centerline[:, 1],
            waypoints_z=centerline[:, 2] if centerline.shape[1] > 2 else None
        )

        # Step 2: Convert obstacle position to centerline Frenet coordinates
        obs_s_array, obs_d_array = centerline_converter.get_frenet(
            np.array([obstacle_pos[0]]),
            np.array([obstacle_pos[1]])
        )
        obs_s = float(obs_s_array[0])
        obs_d = float(obs_d_array[0])

        self.get_logger().info(f"[{self.name}]     Obstacle Frenet: s={obs_s:.2f}m, d={obs_d:.2f}m")

        # Step 3: Define shift parameters (similar to do_spline)
        # s direction: 앞뒤 1.5m
        pre_dist = 1.5
        post_dist = 1.5

        # d direction: obstacle_radius + evasion_dist
        d_apex = obs_d + (obstacle_radius + self.evasion_dist) if obstacle_side == 'left' else obs_d - (obstacle_radius + self.evasion_dist)

        self.get_logger().info(f"[{self.name}]     Shift params: pre={pre_dist:.2f}m, post={post_dist:.2f}m, d_apex={d_apex:.2f}m")

        # Step 4: Create spline control points (Frenet)
        # Point 0: Pre-apex (before obstacle)
        s_pre = obs_s - pre_dist
        d_pre = 0.0  # Return to centerline

        # Point 1: Apex (at obstacle)
        s_apex = obs_s
        d_shift = d_apex

        # Point 2: Post-apex (after obstacle)
        s_post = obs_s + post_dist
        d_post = 0.0  # Return to centerline

        # Convert control points to Cartesian
        s_control = np.array([s_pre, s_apex, s_post])
        d_control = np.array([d_pre, d_shift, d_post])

        xy_control = centerline_converter.get_cartesian(s_control, d_control)

        # Step 5: Calculate headings at control points
        # Get centerline indices for heading
        centerline_length = centerline_converter.raceline_length
        centerline_stepsize = centerline_length / len(centerline)

        idx_pre = int((s_pre % centerline_length) / centerline_stepsize) % len(centerline)
        idx_apex = int((s_apex % centerline_length) / centerline_stepsize) % len(centerline)
        idx_post = int((s_post % centerline_length) / centerline_stepsize) % len(centerline)

        # Calculate headings (psi) from centerline tangent
        def get_heading(idx):
            if idx >= len(centerline) - 1:
                dx = centerline[idx, 0] - centerline[idx - 1, 0]
                dy = centerline[idx, 1] - centerline[idx - 1, 1]
            else:
                dx = centerline[idx + 1, 0] - centerline[idx, 0]
                dy = centerline[idx + 1, 1] - centerline[idx, 1]
            return np.arctan2(dy, dx)

        psi_pre = get_heading(idx_pre)
        psi_apex = get_heading(idx_apex)
        psi_post = get_heading(idx_post)

        # Step 6: Create spline points and tangents (like do_spline)
        points = [
            [xy_control[0, 0], xy_control[1, 0]],  # Pre-apex
            [xy_control[0, 1], xy_control[1, 1]],  # Apex
            [xy_control[0, 2], xy_control[1, 2]]   # Post-apex
        ]

        tangents = [
            [np.cos(psi_pre), np.sin(psi_pre)],
            [np.cos(psi_apex), np.sin(psi_apex)],
            [np.cos(psi_post), np.sin(psi_post)]
        ]

        # Scale tangents (same as do_spline)
        tangents = np.dot(tangents, self.spline_scale * np.eye(2))
        points = np.asarray(points)

        # Step 7: Create Bernstein polynomial spline
        nPoints = len(points)
        c = np.zeros((2, nPoints * 2))

        for i in range(nPoints):
            c[:, 2*i] = points[i]
            if i < nPoints - 1:
                c[:, 2*i + 1] = points[i] + tangents[i] / 3.0
            if i > 0:
                c[:, 2*i] = (c[:, 2*i] + points[i] - tangents[i] / 3.0) / 2.0

        # Create piecewise polynomial
        pieces = []
        for i in range(nPoints - 1):
            coeff = c[:, 2*i:2*i + 4]
            pieces.append(BPoly(coeff[:, ::-1], [i, i + 1]))

        # Step 8: Sample spline and replace affected centerline section
        num_samples = int((pre_dist + post_dist) / 0.1)  # 0.1m sampling
        t_samples = np.linspace(0, nPoints - 1, num_samples)

        spline_points = []
        for t in t_samples:
            piece_idx = int(t)
            if piece_idx >= len(pieces):
                piece_idx = len(pieces) - 1
            t_local = t - piece_idx
            point = pieces[piece_idx](t_local)
            spline_points.append(point)

        spline_points = np.array(spline_points)

        # Step 9: Find affected centerline indices and replace
        s_start = s_pre
        s_end = s_post

        shifted_centerline = centerline.copy()

        # Find all centerline points in [s_start, s_end] range
        centerline_s, _ = centerline_converter.get_frenet(centerline[:, 0], centerline[:, 1])

        # Handle wrap-around
        s_start_normalized = s_start % centerline_length
        s_end_normalized = s_end % centerline_length

        if s_start_normalized <= s_end_normalized:
            # Normal case
            mask = (centerline_s >= s_start_normalized) & (centerline_s <= s_end_normalized)
        else:
            # Wrap-around case
            mask = (centerline_s >= s_start_normalized) | (centerline_s <= s_end_normalized)

        affected_indices = np.where(mask)[0]

        if len(affected_indices) > 0 and len(spline_points) > 0:
            # Interpolate spline points to match affected centerline points
            num_affected = len(affected_indices)
            sample_indices = np.linspace(0, len(spline_points) - 1, num_affected).astype(int)

            for i, idx in enumerate(affected_indices):
                shifted_centerline[idx] = spline_points[sample_indices[i]]

        self.get_logger().info(f"[{self.name}]     Replaced {len(affected_indices)} centerline points with spline")

        return shifted_centerline

    def _decide_obstacle_strategy_gb_aware(
        self,
        obs_s: float,
        obs_d: float,
        obs_radius: float
    ) -> Tuple[str, Optional[str]]:
        """
        Decide obstacle handling strategy based on GB raceline waypoint properties.

        Check opposite wall clearance:
        - If opposite side has space: add obstacle to original side boundary, shift centerline away
        - If opposite side no space: add obstacle to opposite boundary, shift centerline toward obstacle side

        Args:
            obs_s: Obstacle GB Frenet s coordinate
            obs_d: Obstacle GB Frenet d coordinate (positive=left, negative=right)
            obs_radius: Obstacle radius

        Returns:
            (bound_side, shift_direction) tuple
            - bound_side: 'left' or 'right' (which boundary to add obstacle)
            - shift_direction: 'left' or 'right' (centerline shift direction)
        """
        # 1. Find GB waypoint at obs_s
        if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            self.get_logger().warning(f"[{self.name}] GB waypoints not available, falling back to default")
            # Fallback: add to original side, shift to opposite
            if obs_d > 0:
                return ('left', 'right')
            else:
                return ('right', 'left')

        # Find waypoint index from s coordinate
        wpnt_dist = self.gb_wpnts.wpnts[1].s_m - self.gb_wpnts.wpnts[0].s_m
        obs_s_idx = int(obs_s / wpnt_dist) % len(self.gb_wpnts.wpnts)
        gb_wp = self.gb_wpnts.wpnts[obs_s_idx]

        # # 2. Handle based on obstacle side (left: d>0, right: d<0)
        # if obs_d > 0:  # Obstacle on left
        #     opposite_space = gb_wp.d_right  # Distance to right wall

        #     if opposite_space > OPPOSITE_SPACE_THRESHOLD:
        #         # Right side has space → add to left bound, shift centerline right
        #         self.get_logger().info(f"[{self.name}]   GB-aware: obs left, d_right={opposite_space:.2f}m > {OPPOSITE_SPACE_THRESHOLD}m → left bound, shift right")
        #         return ('left', 'right')
        #     else:
        #         # Right side no space → add to right bound, shift centerline left
        #         self.get_logger().info(f"[{self.name}]   GB-aware: obs left, d_right={opposite_space:.2f}m <= {OPPOSITE_SPACE_THRESHOLD}m → right bound, shift left")
        #         return ('right', 'left')
        # else:  # Obstacle on right
        #     opposite_space = gb_wp.d_left  # Distance to left wall

        #     if opposite_space > OPPOSITE_SPACE_THRESHOLD:
        #         # Left side has space → add to right bound, shift centerline left
        #         self.get_logger().info(f"[{self.name}]   GB-aware: obs right, d_left={opposite_space:.2f}m > {OPPOSITE_SPACE_THRESHOLD}m → right bound, shift left")
        #         return ('right', 'left')
        #     else:
        #         # Left side no space → add to left bound, shift centerline right
        #         self.get_logger().info(f"[{self.name}]   GB-aware: obs right, d_left={opposite_space:.2f}m <= {OPPOSITE_SPACE_THRESHOLD}m → left bound, shift right")
        #         return ('left', 'right')

        # 2. Prefer LEFT shift unless left side has no space
        # Check left side space first
        left_space = gb_wp.d_left  # Distance to left wall

        if left_space > OPPOSITE_SPACE_THRESHOLD:
            # Left side has space → shift left (add obstacle to right bound)
            self.get_logger().info(f"[{self.name}]   GB-aware: d_left={left_space:.2f}m > {OPPOSITE_SPACE_THRESHOLD}m → prefer LEFT, right bound, shift left")
            return ('right', 'left')
        else:
            # Left side no space → shift right (add obstacle to left bound)
            self.get_logger().info(f"[{self.name}]   GB-aware: d_left={left_space:.2f}m <= {OPPOSITE_SPACE_THRESHOLD}m → must go RIGHT, left bound, shift right")
            return ('left', 'right')

    def _generate_spline_for_obstacle(
        self,
        centerline_original: np.ndarray,
        centerline_converter: FrenetConverter,
        obs_pos: np.ndarray,
        obs_radius: float,
        obstacle_side: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spline for a single obstacle based on original centerline.

        Args:
            centerline_original: Original centerline [[x, y], ...]
            centerline_converter: Frenet converter for original centerline
            obs_pos: [x, y] obstacle position
            obs_radius: Obstacle radius
            obstacle_side: 'left' or 'right'

        Returns:
            affected_indices: Array of centerline indices affected by this obstacle
            spline_points: Array of new [x, y] points for those indices
        """
        from scipy.interpolate import CubicHermiteSpline

        # Step 1: Convert obstacle to Frenet coordinates
        obs_s_array, obs_d_array = centerline_converter.get_frenet(
            np.array([obs_pos[0]]),
            np.array([obs_pos[1]])
        )
        obs_s = float(obs_s_array[0])
        obs_d = float(obs_d_array[0])

        # Step 2: Define influence region
        pre_dist = 1.5  
        post_dist = 1.5 

        # d direction shift: OPPOSITE direction to avoid obstacle
        # If obstacle is on LEFT, shift centerline to RIGHT (negative d)
        # If obstacle is on RIGHT, shift centerline to LEFT (positive d)
        if obstacle_side == 'left':
            d_apex = obs_d - (obs_radius + self.evasion_dist)  # Shift RIGHT
        else:
            d_apex = obs_d + (obs_radius + self.evasion_dist)  # Shift LEFT

        self.get_logger().info(f"[{self.name}]     obs_s={obs_s:.2f}m, obs_d={obs_d:.2f}m, obs_radius={obs_radius:.2f}m, evasion_dist={self.evasion_dist:.2f}m → d_apex={d_apex:.2f}m")

        # Step 3: Define control points in XY space
        # Pre-apex: on original centerline
        s_pre = obs_s - pre_dist
        xy_pre = centerline_converter.get_cartesian(np.array([s_pre]), np.array([0.0]))
        pos_pre = np.array([xy_pre[0, 0], xy_pre[1, 0]])

        # Apex: shifted position
        s_apex = obs_s
        xy_apex = centerline_converter.get_cartesian(np.array([s_apex]), np.array([d_apex]))
        pos_apex = np.array([xy_apex[0, 0], xy_apex[1, 0]])

        # Post-apex: back on original centerline
        s_post = obs_s + post_dist
        xy_post = centerline_converter.get_cartesian(np.array([s_post]), np.array([0.0]))
        pos_post = np.array([xy_post[0, 0], xy_post[1, 0]])

        # Step 4: Get tangent vectors (heading) at pre and post points from ORIGINAL centerline
        centerline_length = centerline_converter.raceline_length
        def get_tangent_at_s(s):
            """Get unit tangent vector at s position on original centerline"""
            dx_ds, dy_ds = centerline_converter.get_derivative(s % centerline_length)
            tangent = np.array([dx_ds, dy_ds])
            return tangent / np.linalg.norm(tangent)

        tangent_pre = get_tangent_at_s(s_pre)
        tangent_post = get_tangent_at_s(s_post)

        # For apex, use average of pre and post tangents (smooth transition)
        tangent_apex = (tangent_pre + tangent_post) / 2.0
        tangent_apex = tangent_apex / np.linalg.norm(tangent_apex)

        self.get_logger().info(f"[{self.name}]     Control points: pre={pos_pre}, apex={pos_apex}, post={pos_post}")

        # Step 5: Create Hermite spline in XY space with heading constraints
        # Use parameter t: 0 (pre) → 1 (apex) → 2 (post)
        t_points = np.array([0.0, 1.0, 2.0])
        x_points = np.array([pos_pre[0], pos_apex[0], pos_post[0]])
        y_points = np.array([pos_pre[1], pos_apex[1], pos_post[1]])

        # Scale tangents for smooth spline (use spline_scale for continuity)
        scale = self.spline_scale * (pre_dist + post_dist) / 2.0
        dx_points = np.array([tangent_pre[0] * scale, tangent_apex[0] * scale, tangent_post[0] * scale])
        dy_points = np.array([tangent_pre[1] * scale, tangent_apex[1] * scale, tangent_post[1] * scale])

        # Create Hermite splines for x and y
        spline_x = CubicHermiteSpline(t_points, x_points, dx_points)
        spline_y = CubicHermiteSpline(t_points, y_points, dy_points)

        # Step 6: Sample the spline
        num_samples = int((pre_dist + post_dist) / 0.1)  # 0.1m spacing
        t_samples = np.linspace(0.0, 2.0, num_samples)

        x_samples = spline_x(t_samples)
        y_samples = spline_y(t_samples)
        spline_points_sampled = np.column_stack([x_samples, y_samples])

        # Step 5: Find affected centerline indices
        s_start = obs_s - pre_dist  # Start from pre-apex
        s_end = obs_s + post_dist   # End at post-apex

        # Convert all centerline points to Frenet
        centerline_s, _ = centerline_converter.get_frenet(
            centerline_original[:, 0],
            centerline_original[:, 1]
        )

        # Handle wrap-around
        centerline_length = centerline_converter.raceline_length
        s_start_normalized = s_start % centerline_length
        s_end_normalized = s_end % centerline_length

        if s_start_normalized <= s_end_normalized:
            # Normal case
            mask = (centerline_s >= s_start_normalized) & (centerline_s <= s_end_normalized)
        else:
            # Wrap-around case
            mask = (centerline_s >= s_start_normalized) | (centerline_s <= s_end_normalized)

        affected_indices = np.where(mask)[0]

        # Step 6: Match spline points to affected indices
        if len(affected_indices) > 0 and len(spline_points_sampled) > 0:
            num_affected = len(affected_indices)
            sample_indices = np.linspace(0, len(spline_points_sampled) - 1, num_affected).astype(int)
            spline_points = spline_points_sampled[sample_indices]
            self.get_logger().info(f"[{self.name}]     Matched {len(spline_points_sampled)} spline samples to {len(affected_indices)} affected centerline indices")
        else:
            spline_points = np.array([])
            self.get_logger().warning(f"[{self.name}]     No affected indices or spline points!")

        return affected_indices, spline_points

    def _create_reftrack_with_stable_centerline(
        self,
        obstacles: List[Tuple[np.ndarray, float]],
        visualize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Create reftrack using stable centerline approach instead of occupancy grid skeletonization.

        This approach:
        1. Loads verified centerline + bounds from global_waypoints.json
        2. Creates Frenet converter ONCE based on ORIGINAL centerline
        3. For each obstacle, generates spline independently (based on original centerline)
        4. Tracks which indices are affected by each obstacle
        5. Non-overlapping regions: use that obstacle's spline
        6. Overlapping regions: average the splines
        7. Returns reftrack in [x, y, w_right, w_left] format

        Args:
            obstacles: List of (position [x, y], radius) tuples
            visualize: Whether to save visualization for debugging

        Returns:
            reftrack: [x, y, w_right, w_left] numpy array, or None if failed
        """
        try:
            # Step 1: Load original centerline and bounds
            self.get_logger().info(f"[{self.name}] Loading original centerline and bounds...")
            centerline_original = self._get_original_centerline()
            bound_r_original, bound_l_original = self._get_original_wall_boundaries()

            if centerline_original is None or bound_r_original is None or bound_l_original is None:
                self.get_logger().error(f"[{self.name}] Failed to load original centerline/bounds")
                return None

            # Step 1.5: Load conservative bounds if USE_SAFE_GRID enabled
            if USE_SAFE_GRID:
                self.get_logger().info(f"[{self.name}] USE_SAFE_GRID=True, loading conservative bounds...")
                bound_r_conservative, bound_l_conservative = self._get_conservative_boundaries()
                if bound_r_conservative is None or bound_l_conservative is None:
                    self.get_logger().warning(f"[{self.name}] Failed to load conservative bounds, will use original bounds")
                    bound_r_conservative = bound_r_original.copy()
                    bound_l_conservative = bound_l_original.copy()
            else:
                bound_r_conservative = None
                bound_l_conservative = None

            # Step 2: Create Frenet converter ONCE for original centerline
            self.get_logger().info(f"[{self.name}] Creating Frenet converter for original centerline...")
            centerline_converter = FrenetConverter(
                waypoints_x=centerline_original[:, 0],
                waypoints_y=centerline_original[:, 1],
                waypoints_z=centerline_original[:, 2] if centerline_original.shape[1] > 2 else None
            )

            # Initialize modified centerline and bounds (for Fixed path - original based)
            centerline_modified = centerline_original.copy()
            bound_r_modified = bound_r_original.copy()
            bound_l_modified = bound_l_original.copy()

            # Initialize conservative bounds (for Smart mode - conservative based)
            if USE_SAFE_GRID:
                bound_r_conservative_modified = bound_r_conservative.copy()
                bound_l_conservative_modified = bound_l_conservative.copy()

            # # JUST FOR TEST
            # # SAVE BOUNDS BEFORE OBS ADD
            # self.bound_r_modified = bound_r_modified
            # self.bound_l_modified = bound_l_modified 

            # Track spline contributions for each centerline index
            # spline_sum[i] = sum of all spline points at index i
            # spline_count[i] = number of obstacles affecting index i
            spline_sum = np.zeros_like(centerline_original)
            spline_count = np.zeros(len(centerline_original), dtype=int)

            # Counters for logging
            num_shift_obstacles = 0
            num_bound_only_obstacles = 0

            # Step 3: Process each obstacle independently
            for i, (obs_pos, obs_radius) in enumerate(obstacles):
                self.get_logger().info(f"[{self.name}] Processing obstacle {i+1}/{len(obstacles)}: pos={obs_pos}, r={obs_radius:.2f}m")

                # Convert obstacle to centerline Frenet coordinates
                obs_s_array, obs_d_array = centerline_converter.get_frenet(
                    np.array([obs_pos[0]]),
                    np.array([obs_pos[1]])
                )
                obs_s_centerline = float(obs_s_array[0])
                obs_d_centerline = float(obs_d_array[0])

                # ===== HJ MODIFIED: GB-aware strategy (check opposite wall space) =====
                # Convert obstacle to GB Frenet coordinates
                obs_s_gb_array, obs_d_gb_array = self.converter.get_frenet(
                    np.array([obs_pos[0]]),
                    np.array([obs_pos[1]])
                )
                obs_s_gb = float(obs_s_gb_array[0])
                obs_d_gb = float(obs_d_gb_array[0])

                self.get_logger().info(f"[{self.name}]   GB Frenet: s={obs_s_gb:.2f}m, d={obs_d_gb:.2f}m")
                self.get_logger().info(f"[{self.name}]   Centerline Frenet: s={obs_s_centerline:.2f}m, d={obs_d_centerline:.2f}m")

                # Decide strategy based on GB waypoint properties
                bound_side, shift_direction = self._decide_obstacle_strategy_gb_aware(
                    obs_s_gb, obs_d_gb, obs_radius
                )
                # ===== HJ MODIFIED END =====

                # Add obstacle to appropriate bound (original - for Fixed path)
                if bound_side == 'left':
                    bound_l_modified = self._add_obstacle_to_bound(
                        bound_l_modified, obs_pos, obs_radius
                    )
                    self.get_logger().info(f"[{self.name}]   Added obstacle to LEFT bound (original)")
                else:
                    bound_r_modified = self._add_obstacle_to_bound(
                        bound_r_modified, obs_pos, obs_radius
                    )
                    self.get_logger().info(f"[{self.name}]   Added obstacle to RIGHT bound (original)")

                # Add obstacle to conservative bound (for Smart mode)
                if USE_SAFE_GRID:
                    if bound_side == 'left':
                        bound_l_conservative_modified = self._add_obstacle_to_bound(
                            bound_l_conservative_modified, obs_pos, obs_radius
                        )
                        self.get_logger().info(f"[{self.name}]   Added obstacle to LEFT bound (conservative)")
                    else:
                        bound_r_conservative_modified = self._add_obstacle_to_bound(
                            bound_r_conservative_modified, obs_pos, obs_radius
                        )
                        self.get_logger().info(f"[{self.name}]   Added obstacle to RIGHT bound (conservative)")

                # Generate spline if shift_direction is set
                if shift_direction is not None:
                    self.get_logger().info(f"[{self.name}]   Generating spline for centerline shift (direction={shift_direction})")

                    # Convert shift_direction to obstacle_side for spline generation
                    # shift_direction='right' means obstacle on left → obstacle_side='left'
                    # shift_direction='left' means obstacle on right → obstacle_side='right'
                    obstacle_side_for_spline = 'left' if shift_direction == 'right' else 'right'

                    # Generate spline for this obstacle (based on ORIGINAL centerline)
                    affected_indices, spline_points = self._generate_spline_for_obstacle(
                        centerline_original=centerline_original,
                        centerline_converter=centerline_converter,
                        obs_pos=obs_pos,
                        obs_radius=obs_radius,
                        obstacle_side=obstacle_side_for_spline
                    )

                    if len(affected_indices) > 0 and len(spline_points) > 0:
                        # Accumulate spline contributions
                        for idx, point in zip(affected_indices, spline_points):
                            spline_sum[idx] += point
                            spline_count[idx] += 1

                        self.get_logger().info(f"[{self.name}]   Generated spline affecting {len(affected_indices)} indices")
                        num_shift_obstacles += 1
                    else:
                        self.get_logger().warning(f"[{self.name}]   Failed to generate spline (no affected indices)")
                else:
                    self.get_logger().info(f"[{self.name}]   No centerline shift (bound only)")
                    num_bound_only_obstacles += 1

            # Step 4: Apply averaged splines to centerline
            for i in range(len(centerline_modified)):
                if spline_count[i] > 0:
                    # Average all spline contributions at this index
                    centerline_modified[i] = spline_sum[i] / spline_count[i]

            num_modified = np.sum(spline_count > 0)
            self.get_logger().info(
                f"[{self.name}] Obstacle processing summary: "
                f"Total={len(obstacles)}, Shift+Bound={num_shift_obstacles}, Bound-only={num_bound_only_obstacles}"
            )
            self.get_logger().info(f"[{self.name}] Modified {num_modified} centerline points (averaged where overlapping)")

            # Step 5: Calculate distances from modified centerline to modified bounds (Fixed path)
            # Choose bounds for optimization based on USE_CONSERVATIVE_BOUNDS_FOR_OPT flag
            # Store original centerline for visualization comparison
            self._viz_original_centerline = centerline_original

            if USE_SAFE_GRID and USE_CONSERVATIVE_BOUNDS_FOR_OPT:
                self.get_logger().info(f"[{self.name}] Calculating distances from centerline to bounds (conservative)...")
                reftrack = self._calculate_reftrack_distances(
                    centerline_modified,
                    bound_r_conservative_modified,
                    bound_l_conservative_modified,
                    update_viz=visualize,
                    obstacles=obstacles
                )
            else:
                self.get_logger().info(f"[{self.name}] Calculating distances from centerline to bounds (original)...")
                reftrack = self._calculate_reftrack_distances(
                    centerline_modified,
                    bound_r_modified,
                    bound_l_modified,
                    update_viz=visualize,
                    obstacles=obstacles
                )
            self.get_logger().info(f"[{self.name}] Created reftrack: {reftrack.shape[0]} points")

            # Step 5.5: Store bounds for later use (_more_space and d_left/d_right calculation)
            # Original bounds WITHOUT obstacles (for GB mode)
            self.bound_r_original = bound_r_original
            self.bound_l_original = bound_l_original
            self.get_logger().info(f"[{self.name}] Stored original bounds (no obstacles) for GB mode")

            # Original bounds WITH obstacles (for Smart mode USE_SAFE_GRID=False)
            self.bound_r_modified = bound_r_modified
            self.bound_l_modified = bound_l_modified
            self.get_logger().info(f"[{self.name}] Stored original bounds with obstacles for Smart mode")

            # Conservative bounds (for GB/Smart mode USE_SAFE_GRID=True)
            if USE_SAFE_GRID:
                # Conservative bounds WITHOUT obstacles (for GB mode USE_SAFE_GRID=True)
                self.bound_r_conservative = bound_r_conservative
                self.bound_l_conservative = bound_l_conservative
                self.get_logger().info(f"[{self.name}] Stored conservative bounds (no obstacles) for GB mode")

                # Conservative bounds WITH obstacles (for Smart mode USE_SAFE_GRID=True)
                self.bound_r_conservative_modified = bound_r_conservative_modified
                self.bound_l_conservative_modified = bound_l_conservative_modified
                self.get_logger().info(f"[{self.name}] Stored conservative bounds with obstacles for Smart mode")

            # Step 6: Visualization is now handled inside _calculate_reftrack_distances
            # when update_viz=True, showing the actual bounds used for optimization

            return reftrack

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to create reftrack with stable centerline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_reftrack_to_csv(self, reftrack: np.ndarray, csv_filename: str):
        """
        Save reftrack array to CSV file for GB optimizer input.

        Args:
            reftrack: [x_m, y_m, w_tr_right_m, w_tr_left_m]
            csv_filename: Relative filename (will be saved in current working directory)
        """
        np.savetxt(
            csv_filename,
            reftrack,
            delimiter=',',
            fmt='%.6f'
        )
        self.get_logger().info(f"[{self.name}] Saved reftrack to {os.path.abspath(csv_filename)}: {reftrack.shape[0]} points")

    def _call_gb_optimizer(
        self,
        csv_filename: str,
        safety_width: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Call GB optimizer trajectory_optimizer with mincurv_iqp.
        Uses same approach as mapping.launch + global_planner_node.py.

        Args:
            csv_filename: Relative filename in current working directory (e.g., 'smart_static_avoidance_temp.csv')
            safety_width: Override safety_width parameter (uses self.safety_width if None)

        Returns:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2]
            bound_r: Right boundary XY coordinates
            bound_l: Left boundary XY coordinates
            est_time: Estimated lap time (s)
        """
        if safety_width is None:
            safety_width = self.safety_width
        # Get input_path for racecar_f110.ini location (same as global_planner_node.py)
        rospack = RosPack()
        stack_master_path = rospack.get_path('stack_master')
        input_path = os.path.join(stack_master_path, 'config', self.racecar_version)

        # Create temporary directory with modified ini file (tighter convergence)
        import tempfile
        import shutil
        import configparser

        temp_dir = tempfile.mkdtemp(prefix='smart_avoidance_')
        temp_input_path = temp_dir

        try:
            # Copy entire config directory structure (ini + veh_dyn_info/ + devices/, etc.)
            # trajectory_optimizer needs veh_dyn_info/ggv.csv and other files
            shutil.copytree(input_path, temp_input_path, dirs_exist_ok=True)

            # Modify convergence criteria (tighter tolerance for faster exit after min iterations)
            temp_ini = os.path.join(temp_input_path, 'racecar_f110.ini')

            # Read entire ini file as text (can't use configparser - values are JSON)
            with open(temp_ini, 'r') as f:
                ini_content = f.read()

            # Find and replace the entire optim_opts_mincurv section
            import re

            # Log original
            original_match = re.search(r'optim_opts_mincurv=\{[^}]+\}', ini_content, re.DOTALL)
            if original_match:
                self.get_logger().info(f"[{self.name}] BEFORE: {original_match.group()}")

            # Replace the entire optim_opts_mincurv dictionary with tighter settings
            # This is more robust than trying to replace individual values
            # Use global parameters for easy tuning

            replacement = f'optim_opts_mincurv={{"width_opt": {OPT_WIDTH_OPT},\n                    "iqp_iters_min": {OPT_IQP_ITERS_MIN},\n                    "iqp_iters_max": {OPT_IQP_ITERS_MAX},\n                    "iqp_curverror_allowed": {OPT_IQP_CURVERROR_ALLOWED}}}'

            ini_content_modified = re.sub(
                r'optim_opts_mincurv=\{[^}]+\}',
                replacement,
                ini_content,
                flags=re.DOTALL
            )

            # Verify the replacement worked
            modified_match = re.search(r'optim_opts_mincurv=\{[^}]+\}', ini_content_modified, re.DOTALL)
            if modified_match:
                self.get_logger().info(f"[{self.name}] AFTER: {modified_match.group()}")

            # ===== HJ ADDED: Modify additional parameters for faster optimization =====
            # Modify stepsize_opts for faster internal processing
            # Use global parameters for easy tuning
            stepsize_replacement = f'stepsize_opts={{"stepsize_prep": {OPT_STEPSIZE_PREP},\n               "stepsize_reg": {OPT_STEPSIZE_REG},\n               "stepsize_interp_after_opt": {OPT_STEPSIZE_INTERP}}}'
            ini_content_modified = re.sub(
                r'stepsize_opts=\{[^}]+\}',
                stepsize_replacement,
                ini_content_modified,
                flags=re.DOTALL
            )

            # NOTE: curv_calc_opts modification not needed - only used in mintime optimization
            # We use mincurv_iqp which uses analytical curvature calculation (calc_head_curv_an)
            # curv_replacement = 'curv_calc_opts = {"d_preview_curv": 1.0,\n                  "d_review_curv": 1.0,\n                  "d_preview_head": 0.5,\n                  "d_review_head": 0.5}'
            # ini_content_modified = re.sub(
            #     r'curv_calc_opts\s*=\s*\{[^}]+\}',
            #     curv_replacement,
            #     ini_content_modified,
            #     flags=re.DOTALL
            # )

            self.get_logger().info(f"[{self.name}] Modified stepsize_opts for faster optimization")
            # ===== HJ ADDED END =====

            # Write modified content back
            with open(temp_ini, 'w') as f:
                f.write(ini_content_modified)

            self.get_logger().info(f"[{self.name}] Wrote modified ini to: {temp_ini}")

        except Exception as e:
            self.get_logger().warning(f"[{self.name}] Failed to create temp config: {e}, using original")
            temp_input_path = input_path

        # Extract track_name from filename (without .csv extension)
        track_name = os.path.splitext(csv_filename)[0]

        self.get_logger().info(
            f"[{self.name}] Calling trajectory_optimizer: "
            f"input_path={temp_input_path}, "
            f"track_name={track_name}, "
            f"curv_opt_type=mincurv_iqp, "
            f"safety_width={safety_width}, "
            f"plot={self.gb_optimizer_plot}"
        )

        # Call trajectory_optimizer
        trajectory_opt, bound_r, bound_l, est_time = trajectory_optimizer(
            input_path=temp_input_path,
            track_name=track_name,
            curv_opt_type='mincurv_iqp',
            safety_width=safety_width,
            plot=self.gb_optimizer_plot
        )

        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        self.get_logger().info(
            f"[{self.name}] GB optimizer completed: "
            f"trajectory={trajectory_opt.shape[0]} points, "
            f"bound_r={bound_r.shape[0]}, bound_l={bound_l.shape[0]}, "
            f"est_time={est_time:.2f}s"
        )

        return trajectory_opt, bound_r, bound_l, est_time

    def _generate_local_optimized_path(
        self,
        obstacles: List[Tuple[np.ndarray, float]],
        reftrack: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate path using do_spline logic for each obstacle.

        Fakes vehicle position to be before each obstacle, then calls do_spline
        to generate avoidance trajectory. Merges all spline segments with GB raceline.

        Args:
            obstacles: List of (position [x, y], radius) tuples
            reftrack: Reference track [x, y, w_tr_right, w_tr_left] (for bounds info)

        Returns:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2]
            bound_r: Right boundary XY
            bound_l: Left boundary XY
        """
        import trajectory_planning_helpers as tph
        from f110_msgs.msg import Obstacle

        self.get_logger().info(f"[{self.name}] ===== LOCAL OPT: Using do_spline with fake position =====")
        self.get_logger().info(f"[{self.name}] Obstacles: {len(obstacles)}")

        # Get GB waypoints
        if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
            raise ValueError("GB waypoints not available")

        gb_wpnts_list = self.gb_wpnts.wpnts
        n_gb = len(gb_wpnts_list)
        wpnt_dist = gb_wpnts_list[1].s_m - gb_wpnts_list[0].s_m if n_gb > 1 else 0.1
        ref_max_s = gb_wpnts_list[-1].s_m + wpnt_dist

        # Save original vehicle state
        orig_cur_x = self.cur_x
        orig_cur_y = self.cur_y
        orig_cur_s = self.cur_s
        orig_cur_d = self.cur_d
        orig_cur_yaw = self.cur_yaw

        # Start with GB raceline as base
        gb_xy = np.array([[w.x_m, w.y_m] for w in gb_wpnts_list])
        gb_s = np.array([w.s_m for w in gb_wpnts_list])
        gb_psi = np.array([w.psi_rad for w in gb_wpnts_list])
        gb_kappa = np.array([w.kappa_radpm for w in gb_wpnts_list])
        gb_vx = np.array([w.vx_mps for w in gb_wpnts_list])

        # Result starts as copy of GB (will only modify avoidance regions)
        result_xy = gb_xy.copy()
        result_psi = gb_psi.copy()
        result_kappa = gb_kappa.copy()

        # Track which indices are modified
        merged_indices = set()

        # Track transition points (start/end of each avoidance region) for final smoothing
        transition_points = []

        from scipy.interpolate import CubicSpline

        # Process each obstacle: create smooth avoidance spline with GB boundary conditions
        for obs_idx, (obs_pos, obs_radius) in enumerate(obstacles):
            self.get_logger().info(f"[{self.name}] Processing obstacle {obs_idx+1}: pos={obs_pos}, r={obs_radius:.2f}m")

            # Convert to Frenet coordinates
            obs_s_arr, obs_d_arr = self.converter.get_frenet(
                np.array([obs_pos[0]]), np.array([obs_pos[1]])
            )
            obs_s = float(obs_s_arr[0])
            obs_d = float(obs_d_arr[0])

            self.get_logger().info(f"[{self.name}]   Frenet: s={obs_s:.2f}m, d={obs_d:.2f}m")

            # Determine evasion side using _more_space logic
            obs_s_idx = int(obs_s / wpnt_dist) % n_gb

            # Get wall bounds from GB waypoint
            gb_wpnt = gb_wpnts_list[obs_s_idx]
            w_left = gb_wpnt.d_left
            w_right = gb_wpnt.d_right

            # Create Obstacle message for _more_space
            obs_msg = Obstacle()
            obs_msg.is_static = True
            obs_msg.s_center = obs_s
            obs_msg.d_center = obs_d
            obs_msg.size = obs_radius * 2
            obs_msg.d_left = obs_d + obs_radius
            obs_msg.d_right = obs_d - obs_radius

            # Get evasion direction from _decide_obstacle_strategy_gb_aware (same as Global opt centerline shift)
            # This ensures Local opt and Global opt use the same evasion direction
            _, shift_direction = self._decide_obstacle_strategy_gb_aware(obs_s, obs_d, obs_radius)
            # shift_direction is the direction to shift centerline, which is the evasion direction
            evasion_side = shift_direction  # 'left' or 'right'

            # Calculate d_apex using LOCAL_OPT_EVASION_DIST
            if evasion_side == "left":
                d_apex = obs_d + obs_radius + LOCAL_OPT_EVASION_DIST
                # Clamp to wall margin
                max_d = w_left - LOCAL_OPT_WALL_MARGIN
                d_apex = min(d_apex, max_d)
            else:
                d_apex = obs_d - obs_radius - LOCAL_OPT_EVASION_DIST
                # Clamp to wall margin
                min_d = -(w_right - LOCAL_OPT_WALL_MARGIN)
                d_apex = max(d_apex, min_d)

            # self.get_logger().info(f"[{self.name}]   Evasion: side={more_space}, d_apex={d_apex:.2f}m (evasion_dist={LOCAL_OPT_EVASION_DIST}m)")
            self.get_logger().info(f"[{self.name}]   Evasion: side={evasion_side}, d_apex={d_apex:.2f}m (evasion_dist={LOCAL_OPT_EVASION_DIST}m)")

            # Define avoidance region: start_idx to end_idx
            start_idx = int((obs_s - LOCAL_OPT_RANGE) / wpnt_dist) % n_gb
            end_idx = int((obs_s + LOCAL_OPT_RANGE) / wpnt_dist) % n_gb

            # Build index list for this avoidance region
            if end_idx > start_idx:
                region_indices = list(range(start_idx, end_idx + 1))
            else:  # wrap-around
                region_indices = list(range(start_idx, n_gb)) + list(range(0, end_idx + 1))

            n_region = len(region_indices)
            if n_region < 5:
                self.get_logger().warning(f"[{self.name}]   Region too small ({n_region}), skipping")
                continue

            # Build control points for cubic spline with wall margin constraints:
            # 5 control points: start, 1/4, apex (1/2), 3/4, end
            # Each intermediate point is clamped to wall margin

            def get_wall_clamped_d(s_val, target_d):
                """Clamp target_d to maintain wall margin at given s position"""
                idx = int(s_val / wpnt_dist) % n_gb
                wpnt = gb_wpnts_list[idx]
                w_l = wpnt.d_left
                w_r = wpnt.d_right
                # Clamp to wall margin
                max_d = w_l - LOCAL_OPT_WALL_MARGIN
                min_d = -(w_r - LOCAL_OPT_WALL_MARGIN)
                return np.clip(target_d, min_d, max_d)

            # Calculate s positions for intermediate control points (5 points: start, 1/4, apex, 3/4, end)
            s_quarter = (obs_s - LOCAL_OPT_RANGE * 0.5) % ref_max_s
            s_three_quarter = (obs_s + LOCAL_OPT_RANGE * 0.5) % ref_max_s

            # d values: use LOCAL_OPT_MID_D_RATIO for 1/4 and 3/4 points
            # With wall margin clamping
            d_quarter = get_wall_clamped_d(s_quarter, d_apex * LOCAL_OPT_MID_D_RATIO)
            d_apex_clamped = get_wall_clamped_d(obs_s, d_apex)
            d_three_quarter = get_wall_clamped_d(s_three_quarter, d_apex * LOCAL_OPT_MID_D_RATIO)

            # Convert to XY
            quarter_xy = self.converter.get_cartesian(np.array([s_quarter]), np.array([d_quarter]))
            apex_xy_clamped = self.converter.get_cartesian(np.array([obs_s]), np.array([d_apex_clamped]))
            three_quarter_xy = self.converter.get_cartesian(np.array([s_three_quarter]), np.array([d_three_quarter]))

            # Build control points array
            ctrl_pts = []
            ctrl_s_param = []
            cumul_dist = 0.0

            # Point 0: Start (GB)
            ctrl_pts.append([gb_xy[start_idx, 0], gb_xy[start_idx, 1]])
            ctrl_s_param.append(cumul_dist)

            # Point 1: 1/4 point (wall-clamped)
            pt1 = [quarter_xy[0, 0], quarter_xy[1, 0]]
            cumul_dist += np.sqrt((pt1[0] - ctrl_pts[-1][0])**2 + (pt1[1] - ctrl_pts[-1][1])**2)
            ctrl_pts.append(pt1)
            ctrl_s_param.append(cumul_dist)

            # Point 2: Apex (wall-clamped)
            pt2 = [apex_xy_clamped[0, 0], apex_xy_clamped[1, 0]]
            cumul_dist += np.sqrt((pt2[0] - ctrl_pts[-1][0])**2 + (pt2[1] - ctrl_pts[-1][1])**2)
            ctrl_pts.append(pt2)
            ctrl_s_param.append(cumul_dist)

            # Point 3: 3/4 point (wall-clamped)
            pt3 = [three_quarter_xy[0, 0], three_quarter_xy[1, 0]]
            cumul_dist += np.sqrt((pt3[0] - ctrl_pts[-1][0])**2 + (pt3[1] - ctrl_pts[-1][1])**2)
            ctrl_pts.append(pt3)
            ctrl_s_param.append(cumul_dist)

            # Point 4: End (GB)
            pt4 = [gb_xy[end_idx, 0], gb_xy[end_idx, 1]]
            cumul_dist += np.sqrt((pt4[0] - ctrl_pts[-1][0])**2 + (pt4[1] - ctrl_pts[-1][1])**2)
            ctrl_pts.append(pt4)
            ctrl_s_param.append(cumul_dist)

            ctrl_pts = np.array(ctrl_pts)
            ctrl_s_param = np.array(ctrl_s_param)

            # Fit cubic spline with clamped boundary conditions (match GB tangent)
            start_psi = gb_psi[start_idx]
            end_psi = gb_psi[end_idx]

            try:
                spline_x = CubicSpline(ctrl_s_param, ctrl_pts[:, 0],
                                       bc_type=((1, np.cos(start_psi)), (1, np.cos(end_psi))))
                spline_y = CubicSpline(ctrl_s_param, ctrl_pts[:, 1],
                                       bc_type=((1, np.sin(start_psi)), (1, np.sin(end_psi))))
            except Exception as e:
                self.get_logger().warning(f"[{self.name}]   CubicSpline failed: {e}, skipping")
                continue

            self.get_logger().info(f"[{self.name}]   Control points: d_quarter={d_quarter:.2f}, d_apex={d_apex_clamped:.2f}, d_3quarter={d_three_quarter:.2f} (mid_ratio={LOCAL_OPT_MID_D_RATIO})")

            # Sample spline at each GB index in the region
            total_arc = ctrl_s_param[-1]
            for i, idx in enumerate(region_indices):
                t = i / (n_region - 1) * total_arc

                result_xy[idx, 0] = spline_x(t)
                result_xy[idx, 1] = spline_y(t)
                merged_indices.add(idx)

                # Calculate psi from spline derivative
                dx_dt = spline_x(t, 1)
                dy_dt = spline_y(t, 1)
                result_psi[idx] = np.arctan2(dy_dt, dx_dt)

                # Calculate kappa from spline second derivative
                d2x_dt2 = spline_x(t, 2)
                d2y_dt2 = spline_y(t, 2)
                speed_sq = dx_dt**2 + dy_dt**2
                if speed_sq > 1e-6:
                    result_kappa[idx] = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed_sq ** 1.5)

            # Record transition points for later smoothing
            transition_points.append(start_idx)
            transition_points.append(end_idx)

            self.get_logger().info(f"[{self.name}]   Avoidance spline: indices {start_idx}-{end_idx}, {n_region} points")

        # Restore original vehicle state
        self.cur_x = orig_cur_x
        self.cur_y = orig_cur_y
        self.cur_s = orig_cur_s
        self.cur_d = orig_cur_d
        self.cur_yaw = orig_cur_yaw

        self.get_logger().info(f"[{self.name}] Modified {len(merged_indices)}/{n_gb} points, rest is original GB")

        # Final smoothing at transition points (where avoidance meets GB)
        # Using BPoly.from_derivatives for smooth curves with tangent control
        from scipy.interpolate import BPoly

        for trans_idx in transition_points:
            # Generate control point indices evenly distributed within ±SMOOTH_TRANS_MARGIN
            n_ctrl = SMOOTH_TRANS_NUM_CTRL_PTS
            ctrl_indices = []
            for i in range(n_ctrl):
                # -SMOOTH_TRANS_MARGIN to +SMOOTH_TRANS_MARGIN, evenly spaced
                offset = -SMOOTH_TRANS_MARGIN + (2 * SMOOTH_TRANS_MARGIN) * i / (n_ctrl - 1)
                idx = (trans_idx + int(round(offset))) % n_gb
                ctrl_indices.append(idx)

            # Get XY and psi at control points
            ctrl_xy = np.array([result_xy[i] for i in ctrl_indices])
            ctrl_psi = np.array([result_psi[i] for i in ctrl_indices])

            # Calculate tangent vectors (unit direction * scale)
            tangents = np.array([
                [np.cos(ctrl_psi[i]) * SMOOTH_TRANS_SCALE, np.sin(ctrl_psi[i]) * SMOOTH_TRANS_SCALE]
                for i in range(n_ctrl)
            ])

            # Calculate cumulative arc length between control points
            ctrl_s = [0.0]
            for i in range(1, n_ctrl):
                d = np.linalg.norm(ctrl_xy[i] - ctrl_xy[i-1])
                ctrl_s.append(ctrl_s[-1] + d)
            ctrl_s = np.array(ctrl_s)

            # Skip if points are too close
            if ctrl_s[-1] < 0.5:
                self.get_logger().warning(f"[{self.name}]   Transition {trans_idx}: points too close, skipping")
                continue

            try:
                # Build derivatives array for BPoly: [(point, tangent), ...]
                derivs_x = np.array([[ctrl_xy[i, 0], tangents[i, 0]] for i in range(n_ctrl)])
                derivs_y = np.array([[ctrl_xy[i, 1], tangents[i, 1]] for i in range(n_ctrl)])

                poly_x = BPoly.from_derivatives(ctrl_s, derivs_x)
                poly_y = BPoly.from_derivatives(ctrl_s, derivs_y)
            except Exception as e:
                self.get_logger().warning(f"[{self.name}]   Transition {trans_idx}: BPoly failed: {e}")
                continue

            # Build list of indices to smooth (from first to last control point)
            idx_start = ctrl_indices[0]
            idx_end = ctrl_indices[-1]
            if idx_end > idx_start:
                smooth_indices = list(range(idx_start, idx_end + 1))
            else:  # wrap-around
                smooth_indices = list(range(idx_start, n_gb)) + list(range(0, idx_end + 1))

            n_smooth = len(smooth_indices)
            total_arc = ctrl_s[-1]

            # Apply smoothed values
            for i, idx in enumerate(smooth_indices):
                t = i / (n_smooth - 1) * total_arc

                result_xy[idx, 0] = poly_x(t)
                result_xy[idx, 1] = poly_y(t)

                # Calculate psi from derivative
                dx_dt = poly_x(t, 1)
                dy_dt = poly_y(t, 1)
                result_psi[idx] = np.arctan2(dy_dt, dx_dt)

                # Calculate kappa from second derivative
                d2x_dt2 = poly_x(t, 2)
                d2y_dt2 = poly_y(t, 2)
                speed_sq = dx_dt**2 + dy_dt**2
                if speed_sq > 1e-6:
                    result_kappa[idx] = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed_sq ** 1.5)

            self.get_logger().info(f"[{self.name}]   Smoothed transition at {trans_idx}: {n_smooth} points ({n_ctrl} ctrl pts, scale={SMOOTH_TRANS_SCALE})")

        # Recalculate s from XY (need to redo after restoring XY)
        diffs = np.diff(np.vstack([result_xy, result_xy[0:1]]), axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        s_new = np.zeros(n_gb)
        s_new[1:] = np.cumsum(seg_lengths[:-1])

        # ===== Velocity calculation =====
        if RECOMPUTE_LOCAL_OPT_SPEED:
            try:
                # Get kappa-based velocity
                vx_kappa = self._recalculate_velocity_from_curvature(result_kappa, seg_lengths)

                if USE_VEL_BLENDING:
                    # Blend kappa velocity with GB velocity near obstacles
                    self.get_logger().info(f"[{self.name}] Computing blended velocity (kappa + GB)...")

                    # Build obstacle distance map for blending
                    obs_s_list = []
                    for obs_pos, obs_radius in obstacles:
                        obs_s_arr, _ = self.converter.get_frenet(
                            np.array([obs_pos[0]]), np.array([obs_pos[1]])
                        )
                        obs_s_list.append(float(obs_s_arr[0]))

                    # Blend velocity: near obstacle = avg(kappa, GB), far = GB only
                    vx_profile = np.zeros(n_gb)
                    blend_range = VEL_BLEND_RANGE
                    blend_transition = VEL_BLEND_TRANSITION

                    for i in range(n_gb):
                        s_i = i * wpnt_dist

                        # Find minimum distance to any obstacle
                        min_dist = float('inf')
                        for obs_s in obs_s_list:
                            dist = min(
                                abs(s_i - obs_s),
                                abs(s_i - obs_s + ref_max_s),
                                abs(s_i - obs_s - ref_max_s)
                            )
                            min_dist = min(min_dist, dist)

                        if min_dist < blend_range:
                            # Inside blend zone: weighted blend of kappa and GB
                            # VEL_BLEND_GB_WEIGHT: 0=100% kappa, 0.5=average, 1=100% GB
                            vx_profile[i] = vx_kappa[i] * (1 - VEL_BLEND_GB_WEIGHT) + gb_vx[i] * VEL_BLEND_GB_WEIGHT
                        elif min_dist < blend_range + blend_transition:
                            # Transition zone: blend → GB
                            t = (min_dist - blend_range) / blend_transition
                            v_blended = vx_kappa[i] * (1 - VEL_BLEND_GB_WEIGHT) + gb_vx[i] * VEL_BLEND_GB_WEIGHT
                            vx_profile[i] = v_blended * (1 - t) + gb_vx[i] * t
                        else:
                            # Outside: use GB velocity
                            vx_profile[i] = gb_vx[i]

                    self.get_logger().info(f"[{self.name}] Velocity blended (GB_weight={VEL_BLEND_GB_WEIGHT}): "
                                 f"kappa min={np.min(vx_kappa):.2f}, result min={np.min(vx_profile):.2f}, max={np.max(vx_profile):.2f} m/s")
                else:
                    # Use kappa velocity only (no blending)
                    vx_profile = vx_kappa
                    self.get_logger().info(f"[{self.name}] Velocity from kappa: min={np.min(vx_profile):.2f}, max={np.max(vx_profile):.2f} m/s")

            except Exception as e:
                self.get_logger().warning(f"[{self.name}] Velocity calculation failed: {e}, using GB speed")
                vx_profile = gb_vx
        else:
            self.get_logger().info(f"[{self.name}] RECOMPUTE_LOCAL_OPT_SPEED=False, using GB speed")
            vx_profile = gb_vx

        trajectory_opt = np.column_stack([
            s_new,
            result_xy[:, 0],
            result_xy[:, 1],
            result_psi,
            result_kappa,
            vx_profile,
            np.zeros(n_gb)
        ])

        # Calculate bounds from reftrack
        reftrack_xy = reftrack[:, :2]
        n_ref = len(reftrack_xy)
        psi_ref = np.zeros(n_ref)
        for i in range(n_ref):
            i_next = (i + 1) % n_ref
            dx = reftrack_xy[i_next, 0] - reftrack_xy[i, 0]
            dy = reftrack_xy[i_next, 1] - reftrack_xy[i, 1]
            psi_ref[i] = np.arctan2(dy, dx)

        normvecs = tph.calc_normal_vectors.calc_normal_vectors(psi_ref)
        w_right = reftrack[:, 2:3]
        w_left = reftrack[:, 3:4]
        bound_r = reftrack_xy - w_right * normvecs
        bound_l = reftrack_xy + w_left * normvecs
        # Add obstacles to boundary (for _more_space debug visualization)
        # Use same criterion as Global opt: bound_side from _decide_obstacle_strategy_gb_aware
        for obs_idx, (obs_pos, obs_radius) in enumerate(obstacles):
            # Get the shift direction (same as evasion direction used above)
            obs_s_arr, obs_d_arr = self.converter.get_frenet(
                np.array([obs_pos[0]]), np.array([obs_pos[1]])
            )
            obs_s = float(obs_s_arr[0])
            obs_d = float(obs_d_arr[0])

            _, bound_side = self._decide_obstacle_strategy_gb_aware(obs_s, obs_d, obs_radius)

            # Add obstacle to OPPOSITE bound (we shift away from obstacle, so obstacle is on opposite side)
            if bound_side == 'left':
                # Shift left means obstacle is on right side
                bound_r = self._add_obstacle_to_bound(bound_r, obs_pos, obs_radius)
                self.get_logger().info(f"[{self.name}] LOCAL_OPT: Added obstacle {obs_idx+1} to RIGHT bound (shift={bound_side})")
            else:
                # Shift right means obstacle is on left side
                bound_l = self._add_obstacle_to_bound(bound_l, obs_pos, obs_radius)
                self.get_logger().info(f"[{self.name}] LOCAL_OPT: Added obstacle {obs_idx+1} to LEFT bound (shift={bound_side})")

        self.get_logger().info(f"[{self.name}] Local opt complete: {trajectory_opt.shape[0]} points (full track)")

        return trajectory_opt, bound_r, bound_l

    def _smooth_trajectory_output(
        self,
        trajectory_opt: np.ndarray,
        target_stepsize: float = 0.1
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to GB optimizer output trajectory.

        Smooths x, y coordinates while maintaining constant arc length spacing,
        then recalculates heading (psi) and curvature (kappa) analytically in ROS convention.

        Args:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2] from GB optimizer
            target_stepsize: Desired waypoint spacing in meters (default 0.1m)

        Returns:
            Smoothed trajectory [s_m, x_m, y_m, psi_rad_ROS, kappa, vx_mps, ax_mps2]
            Note: psi_rad is already in ROS convention (no +90deg conversion needed)
        """
        if len(trajectory_opt) < SMOOTH_WINDOW:
            self.get_logger().warning(f"[{self.name}] Trajectory too short ({len(trajectory_opt)} < {SMOOTH_WINDOW}) for smoothing, skipping")
            return trajectory_opt

        self.get_logger().info(f"[{self.name}] Applying Savitzky-Golay smoothing (window={SMOOTH_WINDOW}, polyorder={SMOOTH_POLYORDER})...")

        # Extract x, y coordinates
        xy = trajectory_opt[:, 1:3].copy()  # columns 1, 2 are x, y

        # Apply Savitzky-Golay filter with circular padding for closed loop
        pad_size = SMOOTH_WINDOW // 2
        xy_padded = np.vstack([
            xy[-pad_size:],  # Add last points to beginning
            xy,              # Original data
            xy[:pad_size]    # Add first points to end
        ])

        # Smooth padded array
        xy_smooth_padded = savgol_filter(xy_padded, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER, axis=0)

        # Remove padding to get smoothed closed loop
        xy_smooth = xy_smooth_padded[pad_size:-pad_size]

        self.get_logger().info(f"[{self.name}] Applied circular padding (pad_size={pad_size}) for closed loop smoothing")

        # Re-interpolate to maintain constant arc length spacing
        dxy = np.diff(xy_smooth, axis=0)
        segment_lengths = np.sqrt(np.sum(dxy**2, axis=1))
        arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)

        # Add closing segment (last point -> first point) to total length
        closing_segment_length = np.linalg.norm(xy_smooth[0] - xy_smooth[-1])
        total_length = arc_lengths[-1] + closing_segment_length
        self.get_logger().info(f"[{self.name}] Path length: open={arc_lengths[-1]:.2f}m, closing={closing_segment_length:.2f}m, total={total_length:.2f}m")

        # Create uniform arc length array
        num_points = int(total_length / target_stepsize)
        if num_points < 10:
            self.get_logger().warning(f"[{self.name}] Smoothed trajectory too short ({num_points} points), using original")
            return trajectory_opt

        s_uniform = np.linspace(0, total_length, num_points, endpoint=False)

        # Interpolate smoothed x, y and velocity
        from scipy.interpolate import interp1d
        interp_x = interp1d(arc_lengths, xy_smooth[:, 0], kind='cubic', fill_value='extrapolate')
        interp_y = interp1d(arc_lengths, xy_smooth[:, 1], kind='cubic', fill_value='extrapolate')
        interp_vx = interp1d(trajectory_opt[:, 0], trajectory_opt[:, 5], kind='linear', fill_value='extrapolate')

        xy_uniform = np.column_stack([interp_x(s_uniform), interp_y(s_uniform)])
        vx_uniform = interp_vx(s_uniform)

        # Recalculate segment lengths (closed loop: include last->first segment)
        dxy_uniform = np.diff(xy_uniform, axis=0)
        segment_lengths_uniform = np.sqrt(np.sum(dxy_uniform**2, axis=1))

        # Add closing segment (last point -> first point)
        closing_segment = np.linalg.norm(xy_uniform[0] - xy_uniform[-1])
        segment_lengths_closed = np.append(segment_lengths_uniform, closing_segment)

        # ===== HJ ADDED: Debug closing segment =====
        self.get_logger().warning(f"[{self.name}] Closing segment: {closing_segment:.4f}m, "
                     f"avg other segments: {np.mean(segment_lengths_uniform):.4f}m, "
                     f"min: {np.min(segment_lengths_uniform):.4f}m, max: {np.max(segment_lengths_uniform):.4f}m")
        # ===== HJ ADDED END =====

        # Calculate heading in ROS convention (0 = east/x-axis)
        psi_uniform = np.zeros(len(xy_uniform))
        psi_uniform[:-1] = np.arctan2(dxy_uniform[:, 1], dxy_uniform[:, 0])
        # Last heading: from last point to first point (closing)
        psi_uniform[-1] = np.arctan2(xy_uniform[0, 1] - xy_uniform[-1, 1],
                                      xy_uniform[0, 0] - xy_uniform[-1, 0])

        # Calculate curvature from heading changes (closed loop)
        psi_closed = np.append(psi_uniform, psi_uniform[0])  # Close heading
        dpsi = np.diff(psi_closed)
        dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))  # Handle wrapping
        kappa_uniform = dpsi / segment_lengths_closed

        # ===== HJ ADDED: Debug curvature at connection =====
        self.get_logger().warning(f"[{self.name}] Curvature at connection: kappa[-1]={kappa_uniform[-1]:.4f}, "
                     f"kappa[0]={kappa_uniform[0]:.4f}, "
                     f"avg kappa: {np.mean(np.abs(kappa_uniform)):.4f}")
        # ===== HJ ADDED END =====

        # Recalculate velocity profile based on smoothed curvature
        self.get_logger().info(f"[{self.name}] Recalculating velocity from smoothed curvature...")
        vx_profile_new = self._recalculate_velocity_from_curvature(kappa_uniform, segment_lengths_closed)

        # ===== HJ ADDED: Debug velocity at connection =====
        self.get_logger().warning(f"[{self.name}] Velocity at connection: vx[-1]={vx_profile_new[-1]:.2f}, "
                     f"vx[0]={vx_profile_new[0]:.2f}, vx[1]={vx_profile_new[1]:.2f}, "
                     f"avg vx: {np.mean(vx_profile_new):.2f}m/s")
        # ===== HJ ADDED END =====

        # Calculate acceleration profile (closed loop)
        vx_profile_cl = np.append(vx_profile_new, vx_profile_new[0])  # Close velocity loop
        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_profile_cl,
            el_lengths=segment_lengths_closed,  # Use closed segment lengths
            eq_length_output=False
        )

        # ===== HJ ADDED: Debug acceleration at connection =====
        self.get_logger().warning(f"[{self.name}] Acceleration at connection: ax[-1]={ax_profile[-1]:.2f}, "
                     f"ax[0]={ax_profile[0]:.2f}, ax[1]={ax_profile[1]:.2f}, "
                     f"avg |ax|: {np.mean(np.abs(ax_profile)):.2f}m/s^2")
        # ===== HJ ADDED END =====

        # Assemble smoothed trajectory
        # Note: endpoint=False means we don't have a duplicate closing point, so no need to remove last point
        trajectory_smooth = np.column_stack([
            s_uniform,           # s_m (placeholder, will be recalculated in packaging)
            xy_uniform[:, 0],    # x_m
            xy_uniform[:, 1],    # y_m
            psi_uniform,         # psi_rad (ROS convention)
            kappa_uniform,       # kappa (recalculated, closed)
            vx_profile_new,      # vx_mps (recalculated from curvature)
            ax_profile           # ax_mps2 (recalculated, closed)
        ])

        # ===== HJ REMOVED: Don't add closing point - state_machine removes it anyway =====
        # State machine (like GB callback) removes last point: data.wpnts[:-1]
        # So we should NOT add a closing point here - just keep N points without duplication
        # State machine uses modulo wrapping for closed loop: index % num_waypoints
        # closing_row = trajectory_smooth[0].copy()
        # trajectory_smooth = np.vstack([trajectory_smooth, closing_row])
        # ===== HJ REMOVED END =====

        self.get_logger().info(f"[{self.name}] Smoothing complete: {len(trajectory_opt)} → {len(trajectory_smooth)} waypoints, "
                     f"avg spacing={np.mean(segment_lengths_closed):.3f}m, "
                     f"avg velocity={np.mean(vx_profile_new):.2f}m/s")

        return trajectory_smooth

    def _recalculate_velocity_from_curvature(
        self,
        kappa: np.ndarray,
        el_lengths: np.ndarray
    ) -> np.ndarray:
        """
        Recalculate velocity profile from curvature using vehicle dynamics.

        Uses same method as state_machine's update_velocity and GB optimizer.
        Applies Savgol filter to kappa for smoother velocity profile.

        Args:
            kappa: Curvature array [rad/m]
            el_lengths: Element lengths array [m]

        Returns:
            vx_profile: Velocity array [m/s]
        """
        from vel_planner.vel_planner import calc_vel_profile

        # Smooth kappa to reduce velocity spikes from noisy curvature
        kappa_for_vel = kappa.copy()
        if KAPPA_SMOOTH_WINDOW > 0 and len(kappa) > KAPPA_SMOOTH_WINDOW:
            from scipy.signal import savgol_filter
            # Savgol filter: window must be odd, polyorder < window
            window = KAPPA_SMOOTH_WINDOW if KAPPA_SMOOTH_WINDOW % 2 == 1 else KAPPA_SMOOTH_WINDOW + 1
            polyorder = min(3, window - 1)
            kappa_for_vel = savgol_filter(kappa, window, polyorder, mode='wrap')
            self.get_logger().info(f"[{self.name}] Kappa smoothed: window={window}, polyorder={polyorder}")

        # Use same parameters as GB optimizer (closed track)
        vx_profile = calc_vel_profile(
            ggv=self.ggv,
            ax_max_machines=self.ax_max_machines,
            b_ax_max_machines=self.b_ax_max_machines,
            v_max=self.pars["veh_params"]["v_max"],
            kappa=kappa_for_vel,
            el_lengths=el_lengths,
            closed=True,  # Fixed path is closed loop
            filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
            drag_coeff=self.pars["veh_params"]["dragcoeff"],
            m_veh=self.pars["veh_params"]["mass"]
        )

        return vx_profile

    def _package_to_otwpntarray(
        self,
        trajectory_opt: np.ndarray
    ) -> Tuple[OTWpntArray, MarkerArray]:
        """
        Package optimized trajectory into OTWpntArray message with visualization markers.

        IMPORTANT: GB optimizer trajectory uses shifted reference line,
        so we need to convert (x, y) back to ORIGINAL GB raceline Frenet coordinates (s, d).

        Uses Wpnt format (same as xyv_to_wpnts) for consistency with existing code.

        Args:
            trajectory_opt: [s_m, x_m, y_m, psi_rad, kappa, vx_mps, ax_mps2] from GB optimizer

        Returns:
            Tuple of (OTWpntArray with packaged waypoints, MarkerArray for visualization)
        """
        wpnt_array = OTWpntArray()
        wpnt_array.header.stamp = self.get_clock().now().to_msg()
        wpnt_array.header.frame_id = "map"

        mrks = MarkerArray()

        # ===== HJ MODIFIED: Toggle between GB Frenet and Fixed path coordinates =====
        # Use class variable defined at top of file (line ~57)
        xy = trajectory_opt[:, 1:3]  # Extract x, y

        if self.USE_FIXED_PATH_FRENET:
            # OPTION 1: Fixed path arc length coordinates (proper implementation)
            # Calculate arc length along fixed path XY trajectory
            arc_lengths = np.zeros(len(xy))
            arc_lengths[0] = 0.0

            for i in range(1, len(xy)):
                dx = xy[i, 0] - xy[i-1, 0]
                dy = xy[i, 1] - xy[i-1, 1]
                segment_length = np.sqrt(dx**2 + dy**2)
                arc_lengths[i] = arc_lengths[i-1] + segment_length

            # Create temporary converter for d coordinate calculation
            temp_fixed_converter = FrenetConverter(xy[:, 0], xy[:, 1], xy[:, 2] if xy.shape[1] > 2 else None)
            s_coord, d_coord = temp_fixed_converter.get_frenet(xy[:, 0], xy[:, 1])

            self.get_logger().info(f"[{self.name}] Using FIXED PATH coordinates: total_arc_length={arc_lengths[-1]:.2f}m")
        else:
            # OPTION 2: GB Frenet coordinates (for debugging/comparison with old behavior)
            arc_lengths, d_coord = self.converter.get_frenet(xy[:, 0], xy[:, 1])
            self.get_logger().info(f"[{self.name}] Using GB FRENET coordinates (old behavior)")
        # ===== HJ MODIFIED END =====

        for i in range(trajectory_opt.shape[0]):
            wpnt = Wpnt()
            wpnt.id = i
            wpnt.s_m = arc_lengths[i]  # ===== HJ MODIFIED: Arc length (Fixed) or GB s =====
            wpnt.x_m = trajectory_opt[i, 1]  # Absolute X
            wpnt.y_m = trajectory_opt[i, 2]  # Absolute Y
            wpnt.d_m = d_coord[i]  # ===== HJ MODIFIED: Fixed path d or GB d =====

            # ===== HJ MODIFIED: Conditional psi conversion based on SMOOTH_OPT_OUTPUT =====
            if SMOOTH_OPT_OUTPUT:
                # Smoothed trajectory already has psi in ROS convention (no conversion needed)
                wpnt.psi_rad = trajectory_opt[i, 3]
            else:
                # Original optimizer output: convert from tph convention to ROS convention
                # tph: 0 = north/y-axis, ROS: 0 = east/x-axis → add π/2
                psi_tph = trajectory_opt[i, 3]
                psi_ros = psi_tph + np.pi / 2
                if psi_ros > np.pi:
                    psi_ros = psi_ros - 2 * np.pi
                wpnt.psi_rad = psi_ros
            # ===== HJ MODIFIED END =====

            wpnt.kappa_radpm = trajectory_opt[i, 4]  # Curvature
            wpnt.vx_mps = trajectory_opt[i, 5]  # Velocity

            wpnt_array.wpnts.append(wpnt)

            # ===== HJ MODIFIED: Mark as fixed path for green color =====
            # Create visualization marker (same style as xyv_to_markers)
            mrk = self.xyv_to_markers(
                x=trajectory_opt[i, 1],
                y=trajectory_opt[i, 2],
                v=trajectory_opt[i, 5],
                mrks=mrks,
                is_fixed_path=True  # Green color for fixed path
            )
            mrks.markers.append(mrk)
            # ===== HJ MODIFIED END =====

        # ===== HJ ADDED: Debug first and last waypoints =====
        if len(wpnt_array.wpnts) > 0:
            first_wp = wpnt_array.wpnts[0]
            last_wp = wpnt_array.wpnts[-1]
            dist = np.sqrt((first_wp.x_m - last_wp.x_m)**2 + (first_wp.y_m - last_wp.y_m)**2)
            self.get_logger().warning(f"[{self.name}] First wpnt: s={first_wp.s_m:.2f}, x={first_wp.x_m:.2f}, y={first_wp.y_m:.2f}, vx={first_wp.vx_mps:.2f}")
            self.get_logger().warning(f"[{self.name}] Last wpnt:  s={last_wp.s_m:.2f}, x={last_wp.x_m:.2f}, y={last_wp.y_m:.2f}, vx={last_wp.vx_mps:.2f}")
            self.get_logger().warning(f"[{self.name}] Distance between first and last: {dist:.4f}m")
        # ===== HJ ADDED END =====

        self.get_logger().info(f"[{self.name}] Packaged {len(wpnt_array.wpnts)} waypoints and {len(mrks.markers)} markers")
        return wpnt_array, mrks

    def _add_obstacles_to_occupancy_grid(
        self,
        bw: np.ndarray,
        verified_obs: List[Tuple[int, int, float, float]],
        map_resolution: float,
        map_origin_x: float,
        map_origin_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add obstacles to occupancy grid by:
        1. Drawing 0.5m diameter circle at obstacle position
        2. Drawing 0.4m thick line from circle to nearest wall (black pixel)

        NEW LOGIC (HJ):
        - If nearest wall distance < 2.0m: Connect to nearest wall (any direction)
        - If nearest wall distance >= 2.0m: Connect to closer GB boundary (left or right)

        Args:
            bw: Binary image (255=free, 0=occupied)
            verified_obs: List of (sector_id, obs_id, s, d)
            map_resolution: meters/pixel
            map_origin_x, map_origin_y: Map origin in meters

        Returns:
            Tuple of (modified_bw, obstacles_only_bw):
            - modified_bw: With obstacles + wall lines (for skeletonize)
            - obstacles_only_bw: With obstacles only (for GridFilter)
        """
        modified_bw = bw.copy()
        # ===== HJ ADDED: Create obstacles-only version for GridFilter =====
        obstacles_only_bw = bw.copy()
        # ===== HJ ADDED END =====
        obstacle_radius_m = OBSTACLE_RADIUS
        line_thickness_m = 0.4    # 0.4m thick line

        obstacle_radius_px = int(obstacle_radius_m / map_resolution)
        line_thickness_px = max(1, int(line_thickness_m / map_resolution))

        # ===== HJ ADDED: Load GB raceline boundaries for smart connection =====
        WALL_CONNECTION_THRESHOLD = 2.0  # meters - threshold for smart boundary connection
        bound_r_original, bound_l_original = self._get_original_wall_boundaries()
        use_gb_boundaries = (bound_r_original is not None and bound_l_original is not None)

        if not use_gb_boundaries:
            self.get_logger().warning(f"[{self.name}] GB boundaries not available, using nearest wall for all obstacles")
        else:
            self.get_logger().info(f"[{self.name}] Using GB boundaries for smart obstacle connection (threshold={WALL_CONNECTION_THRESHOLD}m)")
        # ===== HJ ADDED END =====

        for sector_id, obs_id, obs_s, obs_d in verified_obs:
            # Convert Frenet (s, d) to Cartesian (x, y)
            obs_xy = self.converter.get_cartesian(np.array([obs_s]), np.array([obs_d]))
            obs_x_m = obs_xy[0][0]
            obs_y_m = obs_xy[1][0]

            # Convert meters to pixels
            obs_x_px = int((obs_x_m - map_origin_x) / map_resolution)
            obs_y_px = int((obs_y_m - map_origin_y) / map_resolution)

            # Flip y because image coordinates are top-down
            obs_y_px = modified_bw.shape[0] - obs_y_px

            # Step 1: Draw obstacle circle (black = occupied)
            # ===== HJ MODIFIED: Draw on both versions =====
            cv2.circle(modified_bw, (obs_x_px, obs_y_px), obstacle_radius_px, 0, -1)
            cv2.circle(obstacles_only_bw, (obs_x_px, obs_y_px), obstacle_radius_px, 0, -1)  # Obstacles only
            # ===== HJ MODIFIED END =====

            # Step 2: Find nearest wall pixel (existing black pixel)
            # Create mask of existing walls (before adding this obstacle)
            wall_mask = (bw == 0).astype(np.uint8)

            # Find all wall pixels
            wall_pixels = np.column_stack(np.where(wall_mask > 0))

            if len(wall_pixels) == 0:
                self.get_logger().warning(f"[{self.name}] No wall pixels found for obstacle at ({obs_x_m:.2f}, {obs_y_m:.2f})")
                continue

            # Calculate distances to all wall pixels
            distances = np.sqrt((wall_pixels[:, 0] - obs_y_px)**2 + (wall_pixels[:, 1] - obs_x_px)**2)
            nearest_idx = np.argmin(distances)
            nearest_distance_m = distances[nearest_idx] * map_resolution

            # ===== HJ MODIFIED: Smart connection logic based on gap size =====
            connection_type = ""
            target_wall_x_px = None
            target_wall_y_px = None

            # ===== DEBUG: Log decision criteria =====
            # self.get_logger().error(f"[{self.name}] ========== OBSTACLE CONNECTION DEBUG ==========")
            # self.get_logger().error(f"[{self.name}] Obstacle {obs_id} at s={obs_s:.2f}, d={obs_d:.2f}, xy=({obs_x_m:.2f}, {obs_y_m:.2f})")
            # self.get_logger().error(f"[{self.name}] Nearest wall distance: {nearest_distance_m:.2f}m")
            # self.get_logger().error(f"[{self.name}] Threshold: {WALL_CONNECTION_THRESHOLD}m")
            # self.get_logger().error(f"[{self.name}] GB boundaries available: {use_gb_boundaries}")
            # ===== DEBUG END =====

            if nearest_distance_m < WALL_CONNECTION_THRESHOLD or not use_gb_boundaries:
                # Small gap OR no GB boundaries: Use nearest wall (original method)
                nearest_wall_y_px = wall_pixels[nearest_idx, 0]
                nearest_wall_x_px = wall_pixels[nearest_idx, 1]
                target_wall_x_px = nearest_wall_x_px
                target_wall_y_px = nearest_wall_y_px
                connection_type = f"nearest_wall ({nearest_distance_m:.2f}m < {WALL_CONNECTION_THRESHOLD}m)"

                # ===== DEBUG =====
                # self.get_logger().error(f"[{self.name}] DECISION: Using NEAREST WALL (gap too small or no GB boundaries)")
                # self.get_logger().error(f"[{self.name}] Target: pixel ({nearest_wall_x_px}, {nearest_wall_y_px})")
                # ===== DEBUG END =====

            else:
                # Large gap: Use GB boundary with normal direction (smart method)
                # self.get_logger().error(f"[{self.name}] DECISION: Using GB BOUNDARY (gap >= {WALL_CONNECTION_THRESHOLD}m)")

                # Get GB normal directions at obstacle s position
                left_normal, right_normal = self._get_gb_normals_at_s(obs_s)
                # self.get_logger().error(f"[{self.name}] GB normals: left={left_normal}, right={right_normal}")

                obs_pos = np.array([obs_x_m, obs_y_m])

                # Find intersections in both normal directions
                left_intersection, left_distance = self._find_boundary_intersection(
                    obs_pos, left_normal, bound_l_original
                )
                right_intersection, right_distance = self._find_boundary_intersection(
                    obs_pos, right_normal, bound_r_original
                )

                # ===== DEBUG =====
                # self.get_logger().error(f"[{self.name}] Left boundary: intersection={left_intersection is not None}, distance={left_distance if left_distance else 'N/A'}")
                # self.get_logger().error(f"[{self.name}] Right boundary: intersection={right_intersection is not None}, distance={right_distance if right_distance else 'N/A'}")
                # ===== DEBUG END =====

                # Handle 4 cases:
                # 1. Both intersections found: Choose closer one
                # 2. Only left found: Use left
                # 3. Only right found: Use right
                # 4. Neither found: Fallback to nearest wall

                if left_intersection is not None and right_intersection is not None:
                    # Both found: choose closer
                    if left_distance < right_distance:
                        target_x_m, target_y_m = left_intersection
                        connection_type = f"left_normal ({left_distance:.2f}m < {right_distance:.2f}m)"
                        # self.get_logger().error(f"[{self.name}] RESULT: Both boundaries found, chose LEFT (closer)")
                    else:
                        target_x_m, target_y_m = right_intersection
                        connection_type = f"right_normal ({right_distance:.2f}m < {left_distance:.2f}m)"
                        # self.get_logger().error(f"[{self.name}] RESULT: Both boundaries found, chose RIGHT (closer)")

                elif left_intersection is not None:
                    # Only left found
                    target_x_m, target_y_m = left_intersection
                    connection_type = f"left_normal_only ({left_distance:.2f}m)"
                    # self.get_logger().error(f"[{self.name}] RESULT: Only LEFT boundary found")

                elif right_intersection is not None:
                    # Only right found
                    target_x_m, target_y_m = right_intersection
                    connection_type = f"right_normal_only ({right_distance:.2f}m)"
                    # self.get_logger().error(f"[{self.name}] RESULT: Only RIGHT boundary found")

                else:
                    # Neither found: fallback to nearest wall
                    # self.get_logger().error(f"[{self.name}] RESULT: NO GB boundaries found, fallback to NEAREST WALL")
                    nearest_wall_y_px = wall_pixels[nearest_idx, 0]
                    nearest_wall_x_px = wall_pixels[nearest_idx, 1]
                    target_wall_x_px = nearest_wall_x_px
                    target_wall_y_px = nearest_wall_y_px
                    connection_type = f"nearest_wall_fallback ({nearest_distance_m:.2f}m)"

                # Convert target position to pixels (if we used GB boundary)
                if target_wall_x_px is None:
                    target_x_px = int((target_x_m - map_origin_x) / map_resolution)
                    target_y_px = int((target_y_m - map_origin_y) / map_resolution)
                    # Flip Y for image coordinates
                    target_y_px = modified_bw.shape[0] - target_y_px

                    # Validate pixel coordinates
                    target_x_px = max(0, min(target_x_px, modified_bw.shape[1] - 1))
                    target_y_px = max(0, min(target_y_px, modified_bw.shape[0] - 1))

                    target_wall_x_px = target_x_px
                    target_wall_y_px = target_y_px

                    # self.get_logger().error(f"[{self.name}] Target: meters ({target_x_m:.2f}, {target_y_m:.2f}) -> pixel ({target_wall_x_px}, {target_wall_y_px})")

            # Step 3: Draw thick line from obstacle to target wall (ONLY on modified_bw, NOT on obstacles_only_bw)
            cv2.line(
                modified_bw,
                (obs_x_px, obs_y_px),
                (target_wall_x_px, target_wall_y_px),
                0,  # Black (occupied)
                line_thickness_px
            )
            # ===== HJ MODIFIED END =====

            # self.get_logger().error(f"[{self.name}] ========== FINAL: Connected via {connection_type} ==========")
            self.get_logger().info(
                f"[{self.name}] Added obstacle {obs_id} at pixel ({obs_x_px}, {obs_y_px}), "
                f"connected via {connection_type}"
            )

        # ===== HJ MODIFIED: Return both versions =====
        return modified_bw, obstacles_only_bw
        # ===== HJ MODIFIED END =====

    def _extract_centerline_from_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Extract centerline from skeleton using contour detection.
        Based on global_planner_node.py lines 698-741.

        Returns:
            centerline_pixels: [[x1, y1], [x2, y2], ...] in pixel coordinates
        """
        # Get contours from skeleton
        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Save all closed contours
        closed_contours = []
        for i, elem in enumerate(contours):
            opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
            if not opened:
                closed_contours.append(elem)

        if len(closed_contours) == 0:
            self.get_logger().error(f"[{self.name}] No closed contours found in skeleton")
            return None

        # # ===== HJ ADDED: Alert if multiple contours (obstacle connection may have failed) =====
        # if len(closed_contours) > 1:
        #     self.get_logger().error(f"[{self.name}] ========== WARNING: MULTIPLE CONTOURS DETECTED ==========")
        #     self.get_logger().error(f"[{self.name}] Found {len(closed_contours)} closed contours in skeleton!")
        #     self.get_logger().error(f"[{self.name}] This may indicate obstacle-wall connection failed.")
        # else:
        #     self.get_logger().info(f"[{self.name}] Found {len(closed_contours)} closed contour in skeleton (OK)")
        # # ===== HJ ADDED END =====

        # Calculate the line length of every contour to get the real centerline
        # (Same logic as global_planner_node.extract_centerline)
        line_lengths = [math.inf] * len(closed_contours)
        for i, cont in enumerate(closed_contours):
            line_length = 0
            for k, pnt in enumerate(cont):
                line_length += np.sqrt((pnt[0][0] - cont[k - 1][0][0]) ** 2 +
                                       (pnt[0][1] - cont[k - 1][0][1]) ** 2)
            # Accept all contours (same as global_planner_node with test_on_car=False)
            line_lengths[i] = line_length

        # Take the shortest line (same as global_planner_node)
        min_line_length = min(line_lengths)

        if min_line_length == math.inf:
            self.get_logger().error(f"[{self.name}] Only invalid closed contour line lengths")
            return None

        min_length_index = line_lengths.index(min_line_length)
        smallest_contour = np.array(closed_contours[min_length_index]).flatten()
        self.get_logger().info(f"[{self.name}] Selected contour {min_length_index} as centerline (length={line_lengths[min_length_index]:.1f} pixels)")

        # Reshape from [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(smallest_contour) / 2)
        centerline = smallest_contour.reshape(len_reshape, 2)

        # Check if centerline is closed (first point should be close to last point)
        first_point = centerline[0]
        last_point = centerline[-1]
        closure_distance = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)

        if closure_distance > 10.0:  # More than 10 pixels apart
            self.get_logger().warning(
                f"[{self.name}] WARNING: Centerline may not be closed! "
                f"First-to-last distance: {closure_distance:.1f} pixels"
            )

        # Smooth centerline (like global_planner_node.py lines 771-788)
        centerline_smooth = self._smooth_centerline(centerline)

        # Direction validation is done in _prepare_reftrack_from_occupancy_grid() after converting to meters

        self.get_logger().info(f"[{self.name}] Extracted centerline: {len(centerline_smooth)} points")
        return centerline_smooth

    def _smooth_centerline(self, centerline: np.ndarray) -> np.ndarray:
        """Smooth centerline with Savitzky-Golay filter (from global_planner_node.py)."""
        centerline_length = len(centerline)

        if centerline_length > 2000:
            filter_length = int(centerline_length / 200) * 10 + 1
        elif centerline_length > 1000:
            filter_length = 81
        elif centerline_length > 500:
            filter_length = 41
        else:
            filter_length = 21

        centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # Apply second smoothing for end/start transition
        cen_len = int(len(centerline) / 2)
        centerline2 = np.append(centerline[cen_len:], centerline[0:cen_len], axis=0)
        centerline_smooth2 = savgol_filter(centerline2, filter_length, 3, axis=0)

        # Take points from second centerline for first centerline
        centerline_smooth[0:filter_length] = centerline_smooth2[cen_len:(cen_len + filter_length)]
        centerline_smooth[-filter_length:] = centerline_smooth2[(cen_len - filter_length):cen_len]

        return centerline_smooth

    def _extract_bounds_from_watershed(
        self,
        centerline_pixels: np.ndarray,
        filtered_bw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract track bounds using watershed algorithm.
        Based on global_planner_node.py lines 817-938.

        Returns:
            bound_r_pixels, bound_l_pixels: Boundary points in pixel coordinates
        """
        # Create centerline image
        cent_img = np.zeros((filtered_bw.shape[0], filtered_bw.shape[1]), dtype=np.uint8)
        cv2.drawContours(cent_img, [centerline_pixels.astype(int)], 0, 255, 2, cv2.LINE_8)

        # Create markers for watershed
        _, cent_markers = cv2.connectedComponents(cent_img)

        # Apply watershed algorithm
        dist_transform = cv2.distanceTransform(filtered_bw, cv2.DIST_L2, 5)
        labels = watershed(-dist_transform, cent_markers, mask=filtered_bw)

        closed_contours = []

        for label in np.unique(labels):
            if label == 0:
                continue

            # Create mask for this label
            mask = np.zeros(filtered_bw.shape, dtype="uint8")
            mask[labels == label] = 255

            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # Save all closed contours
            for i, cont in enumerate(contours):
                opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
                if not opened:
                    closed_contours.append(cont)

        # there must not be more (or less) than two closed contour (same as global_planner_node)
        if len(closed_contours) != 2:
            raise IOError(f"More than two track bounds detected! Found {len(closed_contours)}")

        # Longest contour is outer boundary, shortest is inner
        bound_long = max(closed_contours, key=len)
        bound_long = np.array(bound_long).flatten()
        len_reshape = int(len(bound_long) / 2)
        bound_long = bound_long.reshape(len_reshape, 2)

        bound_short = min(closed_contours, key=len)
        bound_short = np.array(bound_short).flatten()
        len_reshape = int(len(bound_short) / 2)
        bound_short = bound_short.reshape(len_reshape, 2)

        self.get_logger().info(f"[{self.name}] Extracted bounds: long={len(bound_long)}, short={len(bound_short)}")

        # Determine which is right/left based on initial position
        # For now, just return as-is (we'll fix this later if needed)
        return bound_long, bound_short

    def _pixels_to_meters(
        self,
        pixels: np.ndarray,
        map_resolution: float,
        map_origin_x: float,
        map_origin_y: float,
        map_height: int
    ) -> np.ndarray:
        """Convert pixel coordinates to meters."""
        meters = np.zeros(np.shape(pixels))
        meters[:, 0] = pixels[:, 0] * map_resolution + map_origin_x
        # Flip y back (image coordinates are top-down, map coordinates are bottom-up)
        meters[:, 1] = (map_height - pixels[:, 1]) * map_resolution + map_origin_y
        return meters

    def _calculate_reftrack_distances(
        self,
        centerline_m: np.ndarray,
        bound_r_m: np.ndarray,
        bound_l_m: np.ndarray,
        update_viz: bool = False,
        obstacles: List[Tuple[np.ndarray, float]] = None
    ) -> np.ndarray:
        """
        Calculate reftrack [x, y, w_tr_right, w_tr_left] from centerline and bounds.
        Uses XY Euclidean minimum distance (like global_planner_node.py lines 968-990).

        IMPORTANT: Interpolates bounds to 0.1m stepsize before distance calculation.

        Args:
            update_viz: If True, update centerline_modification_viz with actual bounds used
            obstacles: List of obstacles for visualization (required if update_viz=True)
        """
        # Interpolate bounds to 0.1m stepsize (like global_planner_node.py line 972-975)
        # Add dummy columns for w_tr_right, w_tr_left (required by interp_track)
        from global_racetrajectory_optimization import helper_funcs_glob

        bound_r_tmp = np.column_stack((bound_r_m, np.zeros((bound_r_m.shape[0], 2))))
        bound_l_tmp = np.column_stack((bound_l_m, np.zeros((bound_l_m.shape[0], 2))))

        bound_r_int = helper_funcs_glob.src.interp_track.interp_track(
            reftrack=bound_r_tmp, stepsize_approx=0.1
        )
        bound_l_int = helper_funcs_glob.src.interp_track.interp_track(
            reftrack=bound_l_tmp, stepsize_approx=0.1
        )

        # Calculate distances
        N = centerline_m.shape[0]
        reftrack = np.zeros((N, 4))
        reftrack[:, 0:2] = centerline_m

        for i in range(N):
            wpnt = centerline_m[i]

            # Distance to right boundary (minimum XY Euclidean distance)
            dists_bound_right = np.sqrt(
                np.power(bound_r_int[:, 0] - wpnt[0], 2) +
                np.power(bound_r_int[:, 1] - wpnt[1], 2)
            )
            reftrack[i, 2] = np.amin(dists_bound_right)

            # Distance to left boundary
            dists_bound_left = np.sqrt(
                np.power(bound_l_int[:, 0] - wpnt[0], 2) +
                np.power(bound_l_int[:, 1] - wpnt[1], 2)
            )
            reftrack[i, 3] = np.amin(dists_bound_left)

        self.get_logger().info(
            f"[{self.name}] Calculated reftrack distances: "
            f"avg_right={np.mean(reftrack[:, 2]):.2f}m, avg_left={np.mean(reftrack[:, 3]):.2f}m"
        )

        # Update visualization with actual bounds used for optimization
        if update_viz and obstacles is not None:
            # Get original centerline for comparison (use modified as both if not available)
            original_centerline = getattr(self, '_viz_original_centerline', centerline_m)
            self._visualize_centerline_modification(
                original_centerline=original_centerline,
                modified_centerline=centerline_m,
                original_bound_r=bound_r_m,  # Show actual bounds used
                original_bound_l=bound_l_m,  # Show actual bounds used
                modified_bound_r=bound_r_m,  # Same (already modified with obstacles)
                modified_bound_l=bound_l_m,  # Same (already modified with obstacles)
                obstacles=obstacles,
                save_path=None
            )
            self.get_logger().info(f"[{self.name}] Updated visualization with actual optimization bounds")

        return reftrack

    def _save_smart_global_waypoints_json(
        self,
        trajectory_wpnts: OTWpntArray,
        trajectory_markers: MarkerArray,
        bound_r_original: np.ndarray,
        bound_l_original: np.ndarray,
        est_time: float
    ):
        """
        Save optimized trajectory and original track bounds to JSON file.
        Follows the same structure as global_planner_node.py for compatibility.

        Args:
            trajectory_wpnts: OTWpntArray with waypoints (converted from trajectory_opt)
            trajectory_markers: MarkerArray for visualization
            bound_r_original: Original right boundary XY (without obstacles)
            bound_l_original: Original left boundary XY (without obstacles)
            est_time: Estimated lap time
        """
        try:
            from rospy_message_converter import message_converter
            from visualization_msgs.msg import Marker, MarkerArray
            from f110_msgs.msg import WpntArray

            # Convert OTWpntArray to WpntArray (same format as global_planner)
            trajectory_wpnts_array = WpntArray()
            trajectory_wpnts_array.header = trajectory_wpnts.header
            for wpnt in trajectory_wpnts.wpnts:
                trajectory_wpnts_array.wpnts.append(wpnt)

            # Create markers for original track bounds
            trackbounds_markers = MarkerArray()
            marker_id = 0

            # Right bound (purple)
            for i in range(0, len(bound_r_original), 10):  # Downsample for file size
                mk = Marker()
                mk.header.frame_id = "map"
                mk.ns = "bound_right"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_r_original[i, 0]
                mk.pose.position.y = bound_r_original[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.05
                mk.color.r = 0.5
                mk.color.g = 0.0
                mk.color.b = 0.5
                mk.color.a = 0.8
                trackbounds_markers.markers.append(mk)

            # Left bound (green)
            for i in range(0, len(bound_l_original), 10):
                mk = Marker()
                mk.header.frame_id = "map"
                mk.ns = "bound_left"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = bound_l_original[i, 0]
                mk.pose.position.y = bound_l_original[i, 1]
                mk.pose.position.z = 0.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.05
                mk.color.r = 0.5
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.color.a = 0.8
                trackbounds_markers.markers.append(mk)

            # Build JSON dictionary (same structure as global_planner_node)
            json_data = {
                'map_info_str': {'data': f'Smart static avoidance path for {self.map_name}'},
                'est_lap_time': {'data': est_time},
                'global_traj_markers_iqp': message_converter.convert_ros_message_to_dictionary(trajectory_markers),
                'global_traj_wpnts_iqp': message_converter.convert_ros_message_to_dictionary(trajectory_wpnts_array),
                'trackbounds_markers': message_converter.convert_ros_message_to_dictionary(trackbounds_markers)
            }

            # Save to file
            json_path = os.path.join(self.map_dir, 'smart_global_waypoints.json')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            self.get_logger().info(
                f"[{self.name}] Saved smart global waypoints to {json_path}\n"
                f"  - Trajectory waypoints: {len(trajectory_wpnts_array.wpnts)} points\n"
                f"  - Trajectory markers: {len(trajectory_markers.markers)} markers\n"
                f"  - Right bound: {len(bound_r_original)} points\n"
                f"  - Left bound: {len(bound_l_original)} points\n"
                f"  - Est time: {est_time:.2f}s"
            )

        except Exception as e:
            self.get_logger().warning(f"[{self.name}] Failed to save JSON: {e}")
            import traceback
            traceback.print_exc()

    # ===== HJ ADDED: Publish obstacles-only map for visualization =====
    def _publish_obstacles_only_map(self, png_path: str, yaml_path: str):
        """
        Load obstacles-only PNG and publish as OccupancyGrid for RViz visualization.

        Args:
            png_path: Path to obstacles-only PNG file
            yaml_path: Path to YAML metadata file
        """
        try:
            import cv2
            import yaml
            from nav_msgs.msg import OccupancyGrid

            # Load PNG image
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().error(f"[{self.name}] Failed to load PNG: {png_path}")
                return

            # Load YAML metadata
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Create OccupancyGrid message
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.header.frame_id = "map"

            # Set map metadata
            grid_msg.info.resolution = yaml_data['resolution']
            grid_msg.info.width = img.shape[1]
            grid_msg.info.height = img.shape[0]
            grid_msg.info.origin.position.x = yaml_data['origin'][0]
            grid_msg.info.origin.position.y = yaml_data['origin'][1]
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0

            # Convert image to occupancy grid data
            # PNG: 255=free (white), 0=occupied (black)
            # OccupancyGrid: 0=free, 100=occupied, -1=unknown
            flipped_img = np.flipud(img)  # Flip vertically (image origin is top-left, map origin is bottom-left)
            occupancy_data = np.zeros(img.shape, dtype=np.int8)
            occupancy_data[flipped_img == 0] = 100  # Black pixels -> occupied
            occupancy_data[flipped_img == 255] = 0  # White pixels -> free
            grid_msg.data = occupancy_data.flatten().tolist()

            # Publish (latched)
            self.map_with_obs_pub.publish(grid_msg)
            self.get_logger().info(f"[{self.name}] Published obstacles-only map: {img.shape[1]}x{img.shape[0]} @ {yaml_data['resolution']}m/px")

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to publish obstacles-only map: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

    # ===== HJ ADDED: Publish do_spline samples as markers =====
    def _publish_spline_samples_markers(self, samples: np.ndarray, bounds_check_results: list):
        """
        Publish spline sample points as markers for debugging.

        Args:
            samples: Nx2 array of (x, y) coordinates
            bounds_check_results: List of booleans, True if point passed bounds check
        """
        try:
            markers = MarkerArray()

            for i in range(samples.shape[0]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "spline_samples"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Position
                marker.pose.position.x = samples[i, 0]
                marker.pose.position.y = samples[i, 1]
                marker.pose.position.z = 0.1
                marker.pose.orientation.w = 1.0

                # Size
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                # Color: Green if passed bounds check, Red if failed
                if i < len(bounds_check_results):
                    if bounds_check_results[i]:
                        # Green - passed
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 0.8
                    else:
                        # Red - failed
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                else:
                    # Blue - not checked yet (additional points)
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 0.5

                marker.lifetime = rospy.Duration(1.0)
                markers.markers.append(marker)

            self.temp_spline_markers_pub.publish(markers)
            # self.get_logger().info(f"[{self.name}] Published {len(markers.markers)} spline sample markers")

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to publish spline sample markers: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

    def _calculate_fixed_path_boundaries(self, obstacles_only_png_path: str, yaml_path: str, verified_obs: List[Tuple[int, int, float, float]]):
        """
        Calculate d_left and d_right for Fixed path waypoints.

        Uses original wall boundaries from global_waypoints.json and adds obstacle
        circular boundaries to the closer wall.

        Args:
            obstacles_only_png_path: Not used (kept for compatibility)
            yaml_path: Not used (kept for compatibility)
            verified_obs: List of (sector_id, obs_id, s, d) tuples for obstacles
        """
        try:
            self.get_logger().info(f"[{self.name}] _calculate_fixed_path_boundaries CALLED with {len(verified_obs)} obstacles")

            # Get original wall boundaries (cached)
            bound_r_original, bound_l_original = self._get_original_wall_boundaries()

            if bound_r_original is None or bound_l_original is None:
                self.get_logger().error(f"[{self.name}] Failed to load original wall boundaries - ABORTING boundary calculation")
                return

            # Start with original wall boundaries
            boundary_left = bound_l_original.copy()
            boundary_right = bound_r_original.copy()

            self.get_logger().info(f"[{self.name}] Loaded original boundaries: LEFT={len(boundary_left)}, RIGHT={len(boundary_right)}")

            # Add obstacle circular boundaries from verified_obs
            # Use Fixed path Frenet to determine which side the obstacle is on
            if verified_obs is not None and len(verified_obs) > 0:
                for sector_id, obs_id, obs_s, obs_d in verified_obs:
                    # Convert GB Frenet (s, d) to XY
                    obs_xy = self.converter.get_cartesian([obs_s], [obs_d])
                    obs_x = obs_xy[0][0]
                    obs_y = obs_xy[1][0]
                    obs_pos = np.array([obs_x, obs_y])

                    # Use default obstacle size from global parameter
                    obs_radius = OBSTACLE_RADIUS
                    offset = 0.0  # ~m offset from obstacle edge
                    num_points = 100  # Number of points around circle

                    # Determine which side obstacle is on using Fixed path Frenet
                    # This ensures consistency with the avoidance direction
                    obs_s_fixed, obs_d_fixed = self.fixed_converter.get_frenet(
                        np.array([obs_x]), np.array([obs_y])
                    )
                    obs_d_fixed = float(obs_d_fixed[0])

                    # Generate circular boundary points around obstacle
                    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                    circle_radius = obs_radius + offset
                    circle_points = []
                    for angle in angles:
                        x = obs_pos[0] + circle_radius * np.cos(angle)
                        y = obs_pos[1] + circle_radius * np.sin(angle)
                        circle_points.append([x, y])

                    circle_points = np.array(circle_points)

                    # Add to boundary based on Fixed path Frenet d value
                    # d > 0: obstacle is on LEFT of Fixed path → add to LEFT boundary
                    # d < 0: obstacle is on RIGHT of Fixed path → add to RIGHT boundary
                    if obs_d_fixed > 0:
                        boundary_left = np.vstack([boundary_left, circle_points])
                        self.get_logger().info(f"[{self.name}] Added obstacle #{obs_id} (sector={sector_id}) to LEFT boundary (fixed_d={obs_d_fixed:.2f}m, xy={obs_x:.2f},{obs_y:.2f})")
                    else:
                        boundary_right = np.vstack([boundary_right, circle_points])
                        self.get_logger().info(f"[{self.name}] Added obstacle #{obs_id} (sector={sector_id}) to RIGHT boundary (fixed_d={obs_d_fixed:.2f}m, xy={obs_x:.2f},{obs_y:.2f})")

            self.get_logger().info(f"[{self.name}] Final boundaries: LEFT={len(boundary_left)} points, RIGHT={len(boundary_right)} points")

            # Calculate d_left, d_right for each Fixed path waypoint
            # Pure minimum distance (GB optimizer style - no normal filtering!)
            self.fixed_boundary_left_xy = []   # List of [x, y] for left boundary
            self.fixed_boundary_right_xy = []  # List of [x, y] for right boundary

            for i, wpnt in enumerate(self.fixed_path_wpnts.wpnts):
                wpnt_pos = np.array([wpnt.x_m, wpnt.y_m])

                # Find minimum distance to LEFT boundary
                dists_left = np.linalg.norm(boundary_left - wpnt_pos, axis=1)
                min_idx_left = np.argmin(dists_left)
                min_dist_left = dists_left[min_idx_left]
                closest_left_xy = boundary_left[min_idx_left]

                # Find minimum distance to RIGHT boundary
                dists_right = np.linalg.norm(boundary_right - wpnt_pos, axis=1)
                min_idx_right = np.argmin(dists_right)
                min_dist_right = dists_right[min_idx_right]
                closest_right_xy = boundary_right[min_idx_right]

                # Set boundary distances
                wpnt.d_left = min_dist_left
                wpnt.d_right = min_dist_right

                # Store closest boundary XY
                self.fixed_boundary_left_xy.append(closest_left_xy.copy())
                self.fixed_boundary_right_xy.append(closest_right_xy.copy())

            self.get_logger().info(f"[{self.name}] Calculated d_left/d_right for {len(self.fixed_path_wpnts.wpnts)} Fixed path waypoints")

            # Debug: Log first few waypoints
            for i in range(min(5, len(self.fixed_path_wpnts.wpnts))):
                wpnt = self.fixed_path_wpnts.wpnts[i]
                self.get_logger().info(f"  Wpnt {i}: xy=({wpnt.x_m:.2f}, {wpnt.y_m:.2f}), d_left={wpnt.d_left:.2f}, d_right={wpnt.d_right:.2f}")

            # Visualize Fixed path boundaries in RViz
            self._visualize_fixed_path_boundaries(boundary_left, boundary_right, verified_obs)

        except Exception as e:
            self.get_logger().error(f"[{self.name}] Failed to calculate Fixed path boundaries: {e}")
            import traceback
            traceback.print_exc()
    # ===== HJ ADDED END =====

    def _visualize_fixed_path_boundaries(
        self,
        boundary_left: np.ndarray,
        boundary_right: np.ndarray,
        verified_obs: List[Tuple[int, int, float, float]]
    ):
        """
        Visualize Fixed path boundaries with obstacles in RViz.
        Similar to _visualize_centerline_modification but for LOCAL_OPT mode.

        Publishes to /fixed_path_boundaries_viz
        """
        try:
            # Create publisher if not exists
            if not hasattr(self, 'fixed_boundary_viz_pub'):
                self.fixed_boundary_viz_pub = rospy.Publisher(
                    '/fixed_path_boundaries_viz', MarkerArray, queue_size=1, latch=True
                )
                self.get_logger().info(f"[{self.name}] Created RViz marker publisher: /fixed_path_boundaries_viz")

            marker_array = MarkerArray()
            marker_id = 0

            # 1. Left boundary (cyan points)
            left_marker = Marker()
            left_marker.header.frame_id = "map"
            left_marker.header.stamp = self.get_clock().now().to_msg()
            left_marker.ns = "fixed_bound_left"
            left_marker.id = marker_id
            marker_id += 1
            left_marker.type = Marker.POINTS
            left_marker.action = Marker.ADD
            left_marker.scale.x = 0.05  # point size
            left_marker.scale.y = 0.05
            left_marker.color.r = 0.0
            left_marker.color.g = 1.0
            left_marker.color.b = 1.0
            left_marker.color.a = 0.8
            left_marker.pose.orientation.w = 1.0

            for pt in boundary_left:
                from geometry_msgs.msg import Point
                p = Point()
                p.x = pt[0]
                p.y = pt[1]
                p.z = 0.1
                left_marker.points.append(p)

            marker_array.markers.append(left_marker)

            # 2. Right boundary (magenta points)
            right_marker = Marker()
            right_marker.header.frame_id = "map"
            right_marker.header.stamp = self.get_clock().now().to_msg()
            right_marker.ns = "fixed_bound_right"
            right_marker.id = marker_id
            marker_id += 1
            right_marker.type = Marker.POINTS
            right_marker.action = Marker.ADD
            right_marker.scale.x = 0.05  # point size
            right_marker.scale.y = 0.05
            right_marker.color.r = 1.0
            right_marker.color.g = 0.0
            right_marker.color.b = 1.0
            right_marker.color.a = 0.8
            right_marker.pose.orientation.w = 1.0

            for pt in boundary_right:
                from geometry_msgs.msg import Point
                p = Point()
                p.x = pt[0]
                p.y = pt[1]
                p.z = 0.1
                right_marker.points.append(p)

            marker_array.markers.append(right_marker)

            # 3. Fixed path (green line)
            if hasattr(self, 'fixed_path_wpnts') and self.fixed_path_wpnts is not None:
                path_marker = Marker()
                path_marker.header.frame_id = "map"
                path_marker.header.stamp = self.get_clock().now().to_msg()
                path_marker.ns = "fixed_path"
                path_marker.id = marker_id
                marker_id += 1
                path_marker.type = Marker.LINE_STRIP
                path_marker.action = Marker.ADD
                path_marker.scale.x = 0.08
                path_marker.color.r = 0.0
                path_marker.color.g = 1.0
                path_marker.color.b = 0.0
                path_marker.color.a = 1.0
                path_marker.pose.orientation.w = 1.0

                for wpnt in self.fixed_path_wpnts.wpnts:
                    from geometry_msgs.msg import Point
                    p = Point()
                    p.x = wpnt.x_m
                    p.y = wpnt.y_m
                    p.z = 0.15
                    path_marker.points.append(p)

                marker_array.markers.append(path_marker)

            # 4. Obstacles (red spheres)
            if verified_obs is not None:
                for sector_id, obs_id, obs_s, obs_d in verified_obs:
                    obs_xy = self.converter.get_cartesian([obs_s], [obs_d])
                    obs_x = obs_xy[0][0]
                    obs_y = obs_xy[1][0]

                    obs_marker = Marker()
                    obs_marker.header.frame_id = "map"
                    obs_marker.header.stamp = self.get_clock().now().to_msg()
                    obs_marker.ns = "fixed_obstacles"
                    obs_marker.id = marker_id
                    marker_id += 1
                    obs_marker.type = Marker.CYLINDER
                    obs_marker.action = Marker.ADD
                    obs_marker.pose.position.x = obs_x
                    obs_marker.pose.position.y = obs_y
                    obs_marker.pose.position.z = 0.2
                    obs_marker.pose.orientation.w = 1.0
                    obs_marker.scale.x = OBSTACLE_RADIUS * 2
                    obs_marker.scale.y = OBSTACLE_RADIUS * 2
                    obs_marker.scale.z = 0.4
                    obs_marker.color.r = 1.0
                    obs_marker.color.g = 0.0
                    obs_marker.color.b = 0.0
                    obs_marker.color.a = 0.8

                    marker_array.markers.append(obs_marker)

            self.fixed_boundary_viz_pub.publish(marker_array)
            self.get_logger().info(f"[{self.name}] Published Fixed path boundaries visualization ({len(marker_array.markers)} markers)")

        except Exception as e:
            self.get_logger().warning(f"[{self.name}] Failed to visualize Fixed path boundaries: {e}")

    def _is_polygon_ccw(self, points: np.ndarray) -> bool:
        """
        Determine if a closed polygon is oriented counter-clockwise (CCW) using signed area.

        Uses the Shoelace formula to calculate signed area.
        Since points are in ROS coordinates (X right, Y up), standard formula applies:
        - Positive signed area → CCW
        - Negative signed area → CW

        Args:
            points: Nx2 array of (x, y) coordinates in ROS coordinate system

        Returns:
            True if counter-clockwise (CCW), False if clockwise (CW)
        """
        n = len(points)
        if n < 3:
            self.get_logger().warning(f"[{self.name}] Not enough points ({n}) to determine polygon direction")
            return True  # Default to CCW

        # Calculate signed area using Shoelace formula
        # Area = 0.5 * sum((x_i * y_{i+1}) - (x_{i+1} * y_i))
        signed_area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]  # Next point (wrap around)
            signed_area += x1 * y2 - x2 * y1

        # In ROS coordinates (right-handed): positive area = CCW
        return signed_area > 0

    def _get_gb_normals_at_s(self, obs_s: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get left and right normal direction vectors at given s position on GB raceline.

        Args:
            obs_s: s position on GB raceline

        Returns:
            (left_normal, right_normal): Tuple of (dx, dy) direction vectors in ROS coordinates (meters)
        """
        if len(self.gb_wpnts.wpnts) < 2:
            self.get_logger().warning(f"[{self.name}] Not enough GB waypoints for heading interpolation")
            # Default: horizontal normals
            return (0.0, 1.0), (0.0, -1.0)

        # Find two waypoints bracketing obs_s for interpolation
        s_values = np.array([wp.s_m for wp in self.gb_wpnts.wpnts])
        psi_values = np.array([wp.psi_rad for wp in self.gb_wpnts.wpnts])

        # Handle wraparound for closed loop
        max_s = s_values[-1]
        obs_s_wrapped = obs_s % max_s if max_s > 0 else obs_s

        # Find bracketing indices
        idx_after = np.searchsorted(s_values, obs_s_wrapped)

        if idx_after == 0:
            # Before first waypoint: use first waypoint's heading
            psi_ros = psi_values[0]
        elif idx_after >= len(s_values):
            # After last waypoint: use last waypoint's heading
            psi_ros = psi_values[-1]
        else:
            # Interpolate between idx_after-1 and idx_after
            idx_before = idx_after - 1
            s_before = s_values[idx_before]
            s_after = s_values[idx_after]
            psi_before = psi_values[idx_before]
            psi_after = psi_values[idx_after]

            # Handle angle wraparound for interpolation
            delta_psi = psi_after - psi_before
            if delta_psi > np.pi:
                delta_psi -= 2 * np.pi
            elif delta_psi < -np.pi:
                delta_psi += 2 * np.pi

            # Linear interpolation
            t = (obs_s_wrapped - s_before) / (s_after - s_before) if s_after != s_before else 0.0
            psi_ros = psi_before + t * delta_psi

            # Normalize to [-π, π]
            if psi_ros > np.pi:
                psi_ros -= 2 * np.pi
            elif psi_ros < -np.pi:
                psi_ros += 2 * np.pi

        # Calculate normal vectors in ROS coordinates (X right, Y up)
        # psi_ros: 0 = east/X-axis, π/2 = north/Y-axis
        # Left normal: psi + π/2 (CCW 90 degrees)
        # Right normal: psi - π/2 (CW 90 degrees)
        psi_left = psi_ros + np.pi / 2
        psi_right = psi_ros - np.pi / 2

        left_normal = (np.cos(psi_left), np.sin(psi_left))
        right_normal = (np.cos(psi_right), np.sin(psi_right))

        return left_normal, right_normal

    def _find_boundary_intersection(
        self,
        obs_pos: np.ndarray,
        direction: Tuple[float, float],
        boundary_points: np.ndarray,
        max_distance_m: float = 10.0
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Find intersection of bidirectional ray with boundary segments.

        Ray: starts at obstacle, goes in normal direction (bidirectional - both + and - direction)
        Boundary: consecutive points form line segments

        Returns the closest intersection point to the obstacle.

        Args:
            obs_pos: (x, y) obstacle position in meters (ROS coordinates)
            direction: (dx, dy) normal direction vector (bidirectional ray)
            boundary_points: Nx2 array of boundary XY positions (consecutive points form segments)
            max_distance_m: Maximum distance from obstacle to consider

        Returns:
            (intersection_point, distance): Tuple of intersection (x, y) and distance from obstacle,
                                           or (None, None) if no intersection found
        """
        if len(boundary_points) < 2:
            return None, None

        dir_x, dir_y = direction

        # Normalize direction
        dir_magnitude = np.sqrt(dir_x**2 + dir_y**2)
        if dir_magnitude < 1e-6:
            self.get_logger().warning(f"[{self.name}] Direction vector too small: {direction}")
            return None, None

        dir_x /= dir_magnitude
        dir_y /= dir_magnitude

        closest_intersection = None
        closest_distance = float('inf')

        # Check intersection with each boundary segment
        for i in range(len(boundary_points) - 1):
            p1 = boundary_points[i]      # Segment start
            p2 = boundary_points[i + 1]  # Segment end

            # Ray-Segment intersection
            # Ray: P = obs_pos + t * direction (t can be positive or negative - bidirectional)
            # Segment: Q = p1 + s * (p2 - p1), where 0 <= s <= 1

            seg_dir = p2 - p1  # Segment direction

            # Solve: obs_pos + t * dir = p1 + s * seg_dir
            # Rearrange: t * dir - s * seg_dir = p1 - obs_pos
            # In matrix form:
            # [dir_x, -seg_dir_x] [t]   [p1_x - obs_pos_x]
            # [dir_y, -seg_dir_y] [s] = [p1_y - obs_pos_y]

            denom = dir_x * (-seg_dir[1]) - dir_y * (-seg_dir[0])
            denom = dir_x * seg_dir[1] - dir_y * seg_dir[0]

            if abs(denom) < 1e-10:
                # Parallel or collinear - no unique intersection
                continue

            diff = p1 - obs_pos
            t = (diff[0] * seg_dir[1] - diff[1] * seg_dir[0]) / denom
            s = (diff[0] * dir_y - diff[1] * dir_x) / denom

            # Check if intersection is valid:
            # 1. s in [0, 1]: intersection is on the segment (not just the extended line)
            # 2. Distance within max_distance_m
            if 0 <= s <= 1:
                distance = abs(t)  # Distance along ray (bidirectional, so absolute value)

                if distance <= max_distance_m and distance < closest_distance:
                    # Calculate intersection point
                    intersection = obs_pos + t * np.array([dir_x, dir_y])
                    closest_intersection = intersection
                    closest_distance = distance

        if closest_intersection is not None:
            return closest_intersection, closest_distance
        else:
            return None, None

    def _visualize_centerline_modification(
        self,
        original_centerline: np.ndarray,
        modified_centerline: np.ndarray,
        original_bound_r: np.ndarray,
        original_bound_l: np.ndarray,
        modified_bound_r: np.ndarray,
        modified_bound_l: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
        save_path: str = None
    ):
        """
        Visualize centerline and bounds modification using RViz markers.
        Publishes at 1Hz for continuous visualization.

        Args:
            original_centerline: Original centerline [[x, y], ...]
            modified_centerline: Modified centerline [[x, y], ...]
            original_bound_r: Original right bound [[x, y], ...]
            original_bound_l: Original left bound [[x, y], ...]
            modified_bound_r: Modified right bound [[x, y], ...]
            modified_bound_l: Modified left bound [[x, y], ...]
            obstacles: List of (position [x, y], radius) tuples
            save_path: Unused (kept for compatibility)
        """
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        # Store data for periodic publishing
        self.viz_data = {
            'original_centerline': original_centerline,
            'modified_centerline': modified_centerline,
            'original_bound_r': original_bound_r,
            'original_bound_l': original_bound_l,
            'modified_bound_r': modified_bound_r,
            'modified_bound_l': modified_bound_l,
            'obstacles': obstacles
        }

        # Create publisher if not exists
        if not hasattr(self, 'viz_marker_pub'):
            self.viz_marker_pub = rospy.Publisher('/centerline_modification_viz', MarkerArray, queue_size=1, latch=True)
            self.get_logger().info(f"[{self.name}] Created RViz marker publisher: /centerline_modification_viz")

        # Create timer for 1Hz publishing if not exists
        if not hasattr(self, 'viz_timer'):
            self.viz_timer = rospy.Timer(rospy.Duration(1.0), self._publish_centerline_viz_markers)
            self.get_logger().info(f"[{self.name}] Started RViz visualization timer at 1Hz")

        # Publish immediately
        self._publish_centerline_viz_markers(None)

    def _publish_centerline_viz_markers(self, event):
        """Publish centerline visualization markers to RViz."""
        if not hasattr(self, 'viz_data') or not hasattr(self, 'viz_marker_pub'):
            return

        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        markers = MarkerArray()
        marker_id = 0

        # Helper function to create sphere list marker
        def create_sphere_list_marker(points, color, scale, ns):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = 0
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color = color
            for pt in points:
                p = Point()
                p.x, p.y, p.z = pt[0], pt[1], 0.0
                marker.points.append(p)
            return marker

        # Original centerline (gray, semi-transparent)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['original_centerline'],
            ColorRGBA(0.5, 0.5, 0.5, 0.5),
            0.04,
            "original_centerline"
        ))

        # Modified centerline (light black/dark gray)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['modified_centerline'],
            ColorRGBA(0.3, 0.3, 0.3, 1.0),
            0.06,
            "modified_centerline"
        ))

        # Original right bound (light red, semi-transparent)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['original_bound_r'],
            ColorRGBA(1.0, 0.7, 0.7, 0.5),
            0.03,
            "original_bound_r"
        ))

        # Original left bound (light green, semi-transparent)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['original_bound_l'],
            ColorRGBA(0.7, 1.0, 0.7, 0.5),
            0.03,
            "original_bound_l"
        ))

        # Modified right bound (red)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['modified_bound_r'],
            ColorRGBA(1.0, 0.0, 0.0, 1.0),
            0.05,
            "modified_bound_r"
        ))

        # Modified left bound (green)
        markers.markers.append(create_sphere_list_marker(
            self.viz_data['modified_bound_l'],
            ColorRGBA(0.0, 1.0, 0.0, 1.0),
            0.05,
            "modified_bound_l"
        ))

        # Obstacles (cylinders)
        for i, (obs_pos, obs_radius) in enumerate(self.viz_data['obstacles']):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = obs_pos[0]
            marker.pose.position.y = obs_pos[1]
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = obs_radius * 2.0
            marker.scale.y = obs_radius * 2.0
            marker.scale.z = 0.5
            marker.color = ColorRGBA(1.0, 0.5, 0.0, 0.7)
            markers.markers.append(marker)
            marker_id += 1

        self.viz_marker_pub.publish(markers)
        # rospy.loginfo_throttle(5.0, f"[{self.name}] Publishing centerline visualization markers (1Hz)")

    def _publish_smart_active_marker(self):
        """
        Publish SMART_ACTIVE marker above STATE MARKER.
        Shows rectangular background (blue for GB, cyan for SMART) with white text.
        """
        if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) < 2:
            return

        # Calculate marker position (same logic as state_machine STATE MARKER)
        # Position on left side of track, 125% of left track boundary from first waypoint
        x0 = self.gb_wpnts.wpnts[0].x_m
        y0 = self.gb_wpnts.wpnts[0].y_m
        x1 = self.gb_wpnts.wpnts[1].x_m
        y1 = self.gb_wpnts.wpnts[1].y_m

        # Compute normal vector pointing left
        xy_norm = (
            -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1]) * 1.25 * self.gb_wpnts.wpnts[0].d_left
        )

        x_viz = x0 + xy_norm[0]
        y_viz = y0 + xy_norm[1]

        # Publish as MarkerArray with background rectangle + text
        from visualization_msgs.msg import MarkerArray
        marker_array = MarkerArray()

        # Background rectangle (CUBE marker) - horizontal rectangle above STATE MARKER
        bg_marker = Marker()
        bg_marker.type = Marker.CUBE
        bg_marker.id = 1
        bg_marker.header.frame_id = "map"
        bg_marker.header.stamp = self.get_clock().now().to_msg()
        bg_marker.pose.position.x = x_viz + 0.8
        bg_marker.pose.position.y = y_viz 
        bg_marker.pose.position.z = 0  # Above STATE MARKER sphere (STATE is at z=0)
        bg_marker.pose.orientation.w = 1
        bg_marker.scale.x = 0.4  # Width (horizontal - long)
        bg_marker.scale.y = 1.0  # Depth
        bg_marker.scale.z = 0.0  # Height (short - horizontal)
        bg_marker.color.a = 0.9

        # Text marker
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.id = 2
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.pose.position.x = x_viz + 0.8
        text_marker.pose.position.y = y_viz
        text_marker.pose.position.z = 0  # Same as background
        text_marker.pose.orientation.w = 1
        text_marker.scale.z = 0.3  # Text height
        text_marker.color.r = 0.0  # Black text
        text_marker.color.g = 0.0
        text_marker.color.b = 0.0
        text_marker.color.a = 1.0

        # Set color and text based on published_flag (actual published value)
        if self.published_flag:
            # SMART: Cyan background
            bg_marker.color.r = 0.0
            bg_marker.color.g = 1.0
            bg_marker.color.b = 1.0
            text_marker.text = "SMART"
        else:
            # GB: Blue background
            bg_marker.color.r = 0.0
            bg_marker.color.g = 0.0
            bg_marker.color.b = 1.0
            text_marker.text = "GB"

        marker_array.markers.append(bg_marker)
        marker_array.markers.append(text_marker)

        self.smart_active_marker_pub.publish(marker_array)

    def _publish_more_space_debug(self):
        """
        Visualize d_left/d_right endpoints for debugging _more_space() function.
        - GB mode (use_fixed_path=False): Blue spheres
        - Smart mode (use_fixed_path=True): Cyan spheres
        Shows where the bounds are in d direction from each waypoint.
        """
        marker_array = MarkerArray()
        marker_id = 0

        # Determine which waypoints and bounds to use
        if not self.use_fixed_path:
            # GB mode: use GB waypoints
            if self.gb_wpnts.wpnts is None or len(self.gb_wpnts.wpnts) == 0:
                return

            waypoints = self.gb_wpnts.wpnts
            color_r, color_g, color_b = 0.0, 0.0, 1.0  # Blue

            # Get d_left/d_right (conservative if USE_SAFE_GRID, else original)
            if USE_SAFE_GRID and hasattr(self, 'gb_conservative_d_left') and len(self.gb_conservative_d_left) > 0:
                d_left_dict = self.gb_conservative_d_left
                d_right_dict = self.gb_conservative_d_right
            else:
                d_left_dict = None
                d_right_dict = None
        else:
            # Smart mode: use Fixed path waypoints
            if self.fixed_path_wpnts is None or len(self.fixed_path_wpnts.wpnts) == 0:
                return

            waypoints = self.fixed_path_wpnts.wpnts
            color_r, color_g, color_b = 0.0, 1.0, 1.0  # Cyan

            # Get d_left/d_right (conservative if USE_SAFE_GRID, else original)
            if USE_SAFE_GRID and hasattr(self, 'smart_conservative_d_left') and len(self.smart_conservative_d_left) > 0:
                d_left_dict = self.smart_conservative_d_left
                d_right_dict = self.smart_conservative_d_right
            else:
                d_left_dict = None
                d_right_dict = None

        # Visualize every Nth waypoint to avoid clutter (skip=10 → every 10th point)
        skip = 1
        for i, wpnt in enumerate(waypoints[::skip]):
            idx = i * skip

            # Get d_left and d_right
            if d_left_dict is not None and idx in d_left_dict:
                d_left = d_left_dict[idx]
                d_right = d_right_dict[idx]
            else:
                d_left = wpnt.d_left
                d_right = wpnt.d_right

            # Calculate psi (heading) - use psi_rad if available, else calculate from next waypoint
            if hasattr(wpnt, 'psi_rad'):
                psi = wpnt.psi_rad
            else:
                # Fallback: calculate from next waypoint
                if idx + 1 < len(waypoints):
                    next_wpnt = waypoints[idx + 1]
                    psi = np.arctan2(next_wpnt.y_m - wpnt.y_m, next_wpnt.x_m - wpnt.x_m)
                else:
                    psi = 0

            # Calculate left endpoint (90 degrees CCW from heading)
            left_x = wpnt.x_m + d_left * np.cos(psi + np.pi / 2)
            left_y = wpnt.y_m + d_left * np.sin(psi + np.pi / 2)

            # Calculate right endpoint (90 degrees CW from heading)
            right_x = wpnt.x_m + d_right * np.cos(psi - np.pi / 2)
            right_y = wpnt.y_m + d_right * np.sin(psi - np.pi / 2)

            # Left sphere marker
            left_marker = Marker()
            left_marker.header.frame_id = "map"
            left_marker.header.stamp = self.get_clock().now().to_msg()
            left_marker.ns = "more_space_debug_left"
            left_marker.id = marker_id
            marker_id += 1
            left_marker.type = Marker.SPHERE
            left_marker.action = Marker.ADD
            left_marker.pose.position.x = left_x
            left_marker.pose.position.y = left_y
            left_marker.pose.position.z = 0.1
            left_marker.pose.orientation.w = 1.0
            left_marker.scale.x = 0.1
            left_marker.scale.y = 0.1
            left_marker.scale.z = 0.1
            left_marker.color.r = color_r
            left_marker.color.g = color_g
            left_marker.color.b = color_b
            left_marker.color.a = 0.8
            marker_array.markers.append(left_marker)

            # Right sphere marker
            right_marker = Marker()
            right_marker.header.frame_id = "map"
            right_marker.header.stamp = self.get_clock().now().to_msg()
            right_marker.ns = "more_space_debug_right"
            right_marker.id = marker_id
            marker_id += 1
            right_marker.type = Marker.SPHERE
            right_marker.action = Marker.ADD
            right_marker.pose.position.x = right_x
            right_marker.pose.position.y = right_y
            right_marker.pose.position.z = 0.1
            right_marker.pose.orientation.w = 1.0
            right_marker.scale.x = 0.1
            right_marker.scale.y = 0.1
            right_marker.scale.z = 0.1
            right_marker.color.r = color_r
            right_marker.color.g = color_g
            right_marker.color.b = color_b
            right_marker.color.a = 0.8
            marker_array.markers.append(right_marker)

        self.more_space_debug_pub.publish(marker_array)

    def _publish_local_opt_debug_viz(self, xy_before: np.ndarray, xy_after: np.ndarray, obstacles: list):
        """
        Publish visualization for local optimization debugging.
        - GREEN: Path before smoothing (raw spline samples merged with GB)
        - RED: Path after smoothing
        - YELLOW spheres: Obstacle positions
        """
        from visualization_msgs.msg import MarkerArray
        from geometry_msgs.msg import Point

        # Create publisher if not exists
        if not hasattr(self, 'local_opt_debug_pub'):
            self.local_opt_debug_pub = rospy.Publisher(
                '/local_opt_debug', MarkerArray, queue_size=1, latch=True
            )

        marker_array = MarkerArray()

        # Before smoothing path (GREEN) - LINE_STRIP
        before_marker = Marker()
        before_marker.header.frame_id = "map"
        before_marker.header.stamp = self.get_clock().now().to_msg()
        before_marker.ns = "before_smoothing"
        before_marker.id = 0
        before_marker.type = Marker.LINE_STRIP
        before_marker.action = Marker.ADD
        before_marker.scale.x = 0.05  # Line width
        before_marker.color.r = 0.0
        before_marker.color.g = 1.0
        before_marker.color.b = 0.0
        before_marker.color.a = 1.0
        before_marker.pose.orientation.w = 1.0

        for i in range(len(xy_before)):
            p = Point()
            p.x = xy_before[i, 0]
            p.y = xy_before[i, 1]
            p.z = 0.15
            before_marker.points.append(p)
        marker_array.markers.append(before_marker)

        # After smoothing path (RED) - LINE_STRIP
        after_marker = Marker()
        after_marker.header.frame_id = "map"
        after_marker.header.stamp = self.get_clock().now().to_msg()
        after_marker.ns = "after_smoothing"
        after_marker.id = 1
        after_marker.type = Marker.LINE_STRIP
        after_marker.action = Marker.ADD
        after_marker.scale.x = 0.05
        after_marker.color.r = 1.0
        after_marker.color.g = 0.0
        after_marker.color.b = 0.0
        after_marker.color.a = 1.0
        after_marker.pose.orientation.w = 1.0

        for i in range(len(xy_after)):
            p = Point()
            p.x = xy_after[i, 0]
            p.y = xy_after[i, 1]
            p.z = 0.1
            after_marker.points.append(p)
        marker_array.markers.append(after_marker)

        # Obstacle positions (YELLOW spheres)
        for obs_idx, (obs_pos, obs_radius) in enumerate(obstacles):
            obs_marker = Marker()
            obs_marker.header.frame_id = "map"
            obs_marker.header.stamp = self.get_clock().now().to_msg()
            obs_marker.ns = "obstacles"
            obs_marker.id = obs_idx + 10
            obs_marker.type = Marker.CYLINDER
            obs_marker.action = Marker.ADD
            obs_marker.pose.position.x = obs_pos[0]
            obs_marker.pose.position.y = obs_pos[1]
            obs_marker.pose.position.z = 0.1
            obs_marker.pose.orientation.w = 1.0
            obs_marker.scale.x = obs_radius * 2
            obs_marker.scale.y = obs_radius * 2
            obs_marker.scale.z = 0.3
            obs_marker.color.r = 1.0
            obs_marker.color.g = 1.0
            obs_marker.color.b = 0.0
            obs_marker.color.a = 0.8
            marker_array.markers.append(obs_marker)

        self.local_opt_debug_pub.publish(marker_array)
        self.get_logger().info(f"[{self.name}] Published local_opt_debug: GREEN=before smoothing, RED=after smoothing")

if __name__ == "__main__":
    spliner = ObstacleSpliner()
    spliner.loop()
