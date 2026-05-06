"""Init mixin — state_machine 노드 초기화 헬퍼 모음.

이 mixin은 6개 헬퍼를 제공한다 — `_load_rosparams`, `_load_vehicle_dynamics`,
`_load_vel_planner_params`, `_init_state_attributes`, `_setup_ros_subscribers`,
`_setup_ros_publishers`. StateMachine `__init__` 가 이 순서로 호출하므로 의존성은
호출 순서로 보장된다 (rosparam → vehicle dynamics → vel_planner → state attrs → IO).

이 mixin이 사용하는 외부 모듈 / 본체 attribute:
    - `WaypointData`, `states`, `state_transitions` (모듈) — 본체 import 그대로 의존
    - 본체 callback 메서드들 (`odom_cb`, `glb_wpnts_cb` 등) — Subscriber 등록 시 참조
    - 본체 helper 메서드 `_apply_vel_planner_params`, `_vel_planner_3d_param_cb`
"""
import configparser
import json
import os
import threading

# trajectory_planning_helpers 는 ROS1 docker container 안에만 설치되어 있고 ROS2 ws
# 검증 환경에는 없다. C-3 import 통과를 위해 conditional. 실제 ggv / ax_max 로드는
# C-4 메인 노드 검증 시점에 pip install 결정 — 그때까지는 _load_vehicle_dynamics
# 가 호출되면 ImportError 명시적으로 raise.
try:
    import trajectory_planning_helpers as tph
except ImportError:
    tph = None
from f110_msgs.msg import (
    BehaviorStrategy,
    ObstacleArray,
    OTWpntArray,
    PredictionArray,
    WpntArray,
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from state_machine import states
from state_machine import state_transitions
from state_machine.states_types import StateType
from state_machine.waypoint_data import WaypointData

# VESC msg는 sim 이 아닐 때만 필요 — import 자체는 통과해야 하므로 fallback
try:
    from vesc_msgs.msg import VescStateStamped
except ImportError:
    VescStateStamped = None


class InitMixin:
    """StateMachine `__init__` 의 책임을 6개 헬퍼로 분리한 mixin."""

    def _load_rosparams(self):
        """모든 ROS 파라미터 로드 + sectors_params 등 derived 값 계산.

        다른 헬퍼들이 self.racecar_version / self.ot_planner 등에 의존하므로 가장 먼저 호출.
        """
        # 노드 기본
        self.rate_hz = self._get_param_or_default("state_machine/rate")
        self.n_loc_wpnts = self._get_param_or_default("state_machine/n_loc_wpnts")
        self.measuring = self._get_param_or_default("/measure", default=False)

        # Racecar / sectors
        self.racecar_version = self._get_param_or_default("/racecar_version")
        self.sectors_params = self._get_param_or_default("/map_params")
        self.timetrials_only = self._get_param_or_default("state_machine/timetrials_only", False)
        self.n_sectors = self.sectors_params["n_sectors"]

        # OT sectors / planner
        self.ot_sectors_params = self._get_param_or_default("/ot_map_params")
        self.n_ot_sectors = self.ot_sectors_params["n_sectors"]
        self.volt_threshold = self._get_param_or_default("state_machine/volt_threshold", default=10)
        self.ot_planner = self._get_param_or_default("state_machine/ot_planner", default="predictive_spliner")

        # Waypoint dimensions
        self.gb_ego_width_m = self._get_param_or_default("state_machine/gb_ego_width_m")
        self.lateral_width_gb_m = self._get_param_or_default("state_machine/lateral_width_gb_m", 0.3)
        self.gb_horizon_m = self._get_param_or_default("state_machine/gb_horizon_m")
        self.interest_horizon_m = self._get_param_or_default("state_machine/interest_horizon_m", 20.0)

        # Spliner / overtaking
        self.use_force_trailing = not self._get_param_or_default("state_machine/use_force_trailing", False)
        if self.ot_planner == "spliner":
            self.splini_ttl = self._get_param_or_default("state_machine/splini_ttl", 2.0)
        else:
            self.splini_ttl = self._get_param_or_default("state_machine/pred_splini_ttl", 0.2)
        self.overtaking_horizon_m = self._get_param_or_default("state_machine/overtaking_horizon_m", 6.9)
        self.lateral_width_ot_m = self._get_param_or_default("state_machine/lateral_width_ot_m", 0.3)
        self.splini_hyst_timer_sec = self._get_param_or_default("state_machine/splini_hyst_timer_sec", 0.75)
        self.emergency_break_horizon = self._get_param_or_default("state_machine/emergency_break_horizon", 1.1)

        # Track / FTG / force GBTRACK / overtaking TTL
        self.track_length = self._get_param_or_default("/global_republisher/track_length")
        self.ftg_speed_mps = self._get_param_or_default("state_machine/ftg_speed_mps", 1.0)
        self.ftg_timer_sec = self._get_param_or_default("state_machine/ftg_timer_sec", 3.0)
        self.ftg_disabled = not self._get_param_or_default("state_machine/ftg_active", False)
        self.force_gbtrack_state = self._get_param_or_default("state_machine/force_GBTRACK", False)
        self.overtaking_ttl_sec = self._get_param_or_default("state_machine/overtaking_ttl_sec", 3.0)

    def _load_vehicle_dynamics(self):
        """racecar_f110.ini 의 차량 파라미터 + GGV / ax_max / b_ax_max csv 로드."""
        if tph is None:
            raise ImportError(
                "trajectory_planning_helpers 가 설치되지 않음. "
                "C-4 메인 노드 검증 시 `pip install trajectory_planning_helpers` 필요."
            )
        config_dir = os.path.join(
            self._resolve_stack_master_path("config", self.racecar_version)
        )
        ini_path = os.path.join(config_dir, 'racecar_f110.ini')

        parser = configparser.ConfigParser()
        self.pars = {}
        if not parser.read(ini_path):
            raise ValueError('Specified config file does not exist or is empty!')
        self.pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
        self.pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

        veh_dyn_dir = os.path.join(config_dir, "veh_dyn_info")
        ggv_path = os.path.join(veh_dyn_dir, "ggv.csv")
        ax_max_path = os.path.join(veh_dyn_dir, "ax_max_machines.csv")
        b_ax_max_path = os.path.join(veh_dyn_dir, "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=ax_max_path,
        )
        _, self.b_ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=b_ax_max_path,
        )

    def _load_vel_planner_params(self):
        """3D vel planner 파라미터 로드 (vel_planner.yaml) + dyn_reconfigure 구독.

        (1) default 값 → (2) yaml 에서 덮어쓰기 → (3) rqt 실시간 변경 구독
        """
        import yaml as _yaml

        # (1) default
        self._h_cog = self.pars["veh_params"].get("cog_z", 0.074)
        self._slope_correction = 1.0
        self._slope_brake_margin = 0.0
        self._slope_brake_vmax = 5.0
        self._grip_scale_exp = 0.7

        # (2) yaml 에서 덮어쓰기
        yaml_path = self._resolve_stack_master_path(
            "config", self.racecar_version, "vel_planner.yaml"
        )
        try:
            with open(yaml_path) as f:
                params = _yaml.safe_load(f)
            self._apply_vel_planner_params(params)
            self.get_logger().info("[StateMachine3D] vel_planner.yaml loaded")
        except Exception as e:
            self.get_logger().warning(f"[StateMachine3D] vel_planner.yaml not found ({e}), using defaults")

        # (3) rqt 실시간 변경 구독
        # TODO C-5: ROS2 native parameter callback 으로 활성화. 원본은 dynamic_reconfigure
        # /global_velplanner_3d/parameter_updates (Config) 구독하여 self._vel_planner_3d_param_cb
        # 호출. ROS2 에서는 add_on_set_parameters_callback + ros2 param set 으로 같은 효과.
        pass

    def _init_state_attributes(self):
        """모든 인스턴스 변수 default + WaypointData 6개 + states/state_transitions 딕셔너리.

        rosparam이 모두 로드된 후 호출 (일부 변수가 rate_hz / overtaking_ttl_sec 등에 의존).
        """
        # 노드 기본
        self.local_wpnts = WpntArray()
        self.waypoints_dist = 0.1  # [m]
        self.lock = threading.Lock()

        # FTG
        self.only_ftg_zones = []
        self.ftg_counter = 0

        # 자차 위치
        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0

        # Overtaking 상태
        self.overtake_wpnts = None
        self.overtake_zones = []
        self.ot_begin_margin = 0.5
        self.cur_volt = 11.69  # default value for sim
        self.static_overtaking_mode = False

        # Waypoint 메타 / 카운터
        self.cur_id_ot = 1
        self.max_speed = -1
        self.max_s = 0
        self.current_position = None
        self.gb_wpnts = None
        self.recovery_wpnts = None
        self.smart_static_wpnts = None  # Smart static avoidance waypoints from spliner
        self.smart_static_active = False  # Flag from spliner — is smart static mode active?
        self.gb_max_idx = None
        self.wpnt_dist = self.waypoints_dist
        self.num_glb_wpnts = 0
        self.num_ot_points = 0
        self.previous_index = 0
        self.last_recovery_update_time = None

        # WaypointData 인스턴스 6개 + 메타
        self.cur_gb_wpnts = WaypointData('global_tracking', True)
        self.cur_recovery_wpnts = WaypointData('recovery_planner', False)
        self.cur_avoidance_wpnts = WaypointData('dynamic_avoidance_planner', False)
        self.cur_static_avoidance_wpnts = WaypointData('static_avoidance_planner', False)
        self.cur_start_wpnts = WaypointData('start_planner', False)
        # smart_static은 static_avoidance_planner 파라미터를 공유하되 closed=True
        self.cur_smart_static_avoidance_wpnts = WaypointData('static_avoidance_planner', True)
        self.smart_track_length = None
        self.smart_wpnt_dist = None

        # WaypointData 속성 설정
        self.cur_avoidance_wpnts.is_ot_wpnts = True
        self.cur_static_avoidance_wpnts.is_ot_wpnts = True
        self.cur_gb_wpnts.is_gb_track_wpnts = True
        self.cur_recovery_wpnts.vel_planner_safety_factor = 0.5

        # closest target / gap (visualization 캐시)
        self.gb_closest_target = None
        self.gb_closest_gap = None
        self.recovery_closest_target = None
        self.recovery_closest_gap = None
        self.ot_closest_target = None
        self.ot_closest_gap = None

        # behavior strategy 메시지
        self.behavior_strategy = BehaviorStrategy()

        # Splines (mincurv + ot)
        self.mincurv_spline_x = None
        self.mincurv_spline_y = None
        self.ot_spline_x = None
        self.ot_spline_y = None
        self.ot_spline_d = None
        self.recompute_ot_spline = True

        # 장애물 회피 변수
        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []
        self.obstacles_perception = []
        self.obstacles_prediction_id = None
        self.obstacles_prediction = []
        self.ego_prediction = []
        self.obstacle_was_here = True
        self.side_by_side_threshold = 0.6
        self.merger = None
        self.force_trailing = False

        # Spliner 변수
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
        self.avoidance_wpnts = None
        self.static_avoidance_wpnts = None
        self.start_wpnts = None
        self.start_wpnts_array = None
        self.last_valid_avoidance_wpnts = None
        self.last_valid_avoidance_array = None
        self.last_valid_static_avoidance_wpnts = None
        self.emergency_break_d = 0.12  # [m]

        # Graph based + Frenet
        self.graph_based_wpts = None
        self.gb_wpnts_arr = None
        self.frenet_wpnts = WpntArray()

        # Overtaking TTL counter
        self.overtaking_ttl_count = 0
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)

        # Start trajectory 상태
        self.save_start_traj = False
        self.cur_start_wpnts_candidate = OTWpntArray()
        self.need_start_traj = False

        # Visualization 보조
        self.first_visualization = True
        self.x_viz = 0
        self.y_viz = 0

        # State 변수
        self.cur_state = StateType.GB_TRACK
        self.local_wpnts_src = StateType.GB_TRACK
        self.static_avoid = False
        self.fail_trailing = False

        # State -> wpnt 생성 함수 매핑
        self.states = {
            StateType.GB_TRACK: states.GlobalTracking,
            StateType.OVERTAKE: states.Overtaking,
            StateType.FTGONLY: states.FTGOnly,
            StateType.RECOVERY: states.RECOVERY,
            StateType.START: states.START,
            StateType.SMART_STATIC: states.SmartStatic,
        }

        # State -> 다음 state 결정 함수 매핑
        self.state_transitions = {
            StateType.GB_TRACK: state_transitions.GlobalTrackingTransition,
            StateType.RECOVERY: state_transitions.RecoveryTransition,
            StateType.TRAILING: state_transitions.TrailingTransition,
            StateType.ATTACK: state_transitions.TrailingTransition,
            StateType.OVERTAKE: state_transitions.OvertakingTransition,
            StateType.FTGONLY: state_transitions.FTGOnlyTransition,
            StateType.START: state_transitions.StartTransition,
            StateType.SMART_STATIC: state_transitions.SmartStaticTransition,
        }

    def _setup_ros_subscribers(self):
        """모든 ROS Subscriber 등록 + 필수 메시지 대기.

        ot_planner 종류에 따라 일부 토픽만 선택적으로 구독.

        ROS2 포팅 (C-3):
        - rospy.wait_for_message → rclpy.wait_for_message (Jazzy 헬퍼)
        - dynamic_reconfigure (Config) sub 4개 → 주석 (C-5 에서 ROS2 native parameter
          callback 으로 활성화)
        - vesc sub → 주석 (vesc 패키지 미포팅)
        """
        from rclpy.wait_for_message import wait_for_message

        self.opponent = ObstacleArray()

        # Localization / global track
        self.create_subscription(Odometry, "/car_state/odom", self.odom_cb, 10)
        wait_for_message(Odometry, self, "/car_state/odom", time_to_wait=10.0)
        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.glb_wpnts_cb, 10)
        self.create_subscription(WpntArray, "/planner/recovery/wpnts", self.recovery_wpnts_cb, 10)
        self.create_subscription(WpntArray, "/global_waypoints/overtaking", self.overtake_cb, 10)
        wait_for_message(WpntArray, self, "/global_waypoints_scaled", time_to_wait=10.0)
        wait_for_message(WpntArray, self, "/global_waypoints/overtaking", time_to_wait=10.0)
        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_pose_cb, 10)
        wait_for_message(Odometry, self, "/car_state/odom_frenet", time_to_wait=10.0)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_wpnts_og_cb, 10)

        # TODO C-5: dynamic_reconfigure (Config) sub 4 개 — ROS2 native parameter callback 으로 활성화
        # self.create_subscription(Config, "/dyn_statemachine/parameter_updates", self.dyn_param_cb, 10)
        # self.create_subscription(Config, "/dyn_sector_tuner/speed/parameter_updates", self.sector_dyn_param_cb, 10)
        # self.create_subscription(Config, "/dyn_sector_tuner/overtake/parameter_updates", self.ot_dyn_param_cb, 10)

        # Perception / prediction
        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacle_perception_cb, 10)
        self.create_subscription(PredictionArray, "/opponent_prediction/obstacles_pred", self.obstacle_prediction_cb, 10)
        self.create_subscription(PredictionArray, "/mpc_controller/ego_prediction", self.ego_prediction_cb, 10)

        # Planner-specific (ot_planner 에 따라)
        if self.ot_planner in ("spliner", "predictive_spliner"):
            self.create_subscription(OTWpntArray, "/planner/avoidance/otwpnts", self.avoidance_cb, 10)
            # Smart Static 모드 (HJ 추가)
            self.create_subscription(OTWpntArray, "/planner/avoidance/smart_static_otwpnts", self.smart_static_avoidance_cb, 10)
            self.create_subscription(Bool, "/planner/avoidance/smart_static_active", self.smart_static_active_cb, 10)
            if self.ot_planner == "predictive_spliner":
                self.create_subscription(OTWpntArray, "/planner/avoidance/static_otwpnts", self.static_avoidance_cb, 10)
        if self.ot_planner == "predictive_spliner":
            self.create_subscription(Float32MultiArray, "/planner/avoidance/merger", self.merger_cb, 10)
            self.create_subscription(Bool, "collision_prediction/force_trailing", self.force_trailing_cb, 10)
            self.create_subscription(Bool, "planner/avoidance/fail_trailing", self.fail_trailing_cb, 10)

        # TODO: vesc 패키지 ROS2 포팅 안 됨 — voltage 모니터링 비활성화. C-4 검증 필수 아님.
        # if not self._get_param_or_default("/sim"):
        #     self.create_subscription(VescStateStamped, "/vesc/sensors/core", self.vesc_state_cb, 10)

        # Start trajectory 저장 트리거
        self.create_subscription(OTWpntArray, "/planner/start_wpnts", self.start_wpnts_cb, 10)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, 10)

    def _setup_ros_publishers(self):
        """모든 ROS Publisher 등록."""
        self.behavior_strategy_pub = self.create_publisher(BehaviorStrategy, "behavior_strategy", 1)
        self.trailing_marker_pub = self.create_publisher(Marker, "/state_machine/trailing_target", 10)
        self.overtaking_marker_pub = self.create_publisher(Marker, "/state_machine/overtaking_target", 10)
        self.obstacles_in_interest_marker_pub = self.create_publisher(
            MarkerArray, "/state_machine/obstacles_in_interest", 10
        )

        self.loc_wpnt_pub = self.create_publisher(WpntArray, "local_waypoints", 1)
        self.vis_loc_wpnt_pub = self.create_publisher(MarkerArray, "local_waypoints/markers", 10)
        self.vis_loc_vel_pub = self.create_publisher(MarkerArray, "local_waypoints/vel_markers", 10)
        self.state_pub = self.create_publisher(String, "state_machine", 1)
        self.state_mrk = self.create_publisher(Marker, "/state_marker", 10)
        self.state_wpnts_src_marker = self.create_publisher(Marker, "/state_wpnts_src_marker", 10)
        self.emergency_pub = self.create_publisher(Marker, "/emergency_marker", 5)  # for low voltage
        self.ot_section_check_pub = self.create_publisher(Bool, "/ot_section_check", 1)

        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/state_machine/latency", 10)

    # ------------------------------------------------------------------
    # ROS2 호환 헬퍼 (rospy.get_param + rospkg fallback)
    # ------------------------------------------------------------------

    def _get_param_or_default(self, name, default=None):
        """ROS1 의 rospy.get_param 호환 helper.

        Node.__init__ 의 automatically_declare_parameters_from_overrides=True 덕분에
        launch 에서 set 한 파라미터는 자동 declare 되어 있다. 여기서는 그 외의
        경우만 처리:
        - default 가 있으면 declare 후 반환
        - default 가 없으면 None 반환 (rospy 는 KeyError 였으나 ROS2 는 silent None)
        """
        try:
            return self.get_parameter(name).value
        except Exception:
            if default is not None:
                try:
                    self.declare_parameter(name, default)
                    return self.get_parameter(name).value
                except Exception:
                    return default
            return None

    def _resolve_stack_master_path(self, *parts) -> str:
        """stack_master 경로 해결 — 우리 ws 에 stack_master 미포팅이라
        ROS1 ws 의 stack_master 디렉터리로 fallback (B-1 / B-6 패턴)."""
        return os.path.expanduser(
            os.path.join("~/unicorn_ws/ICRA2026_HJ/stack_master", *parts)
        )
