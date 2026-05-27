"""fast_livo localization + PP 자동주행 통합 launch (middle level).

map argument 하나로 모든 데이터 로드 — 기본값 my_seat. fast_livo2/map/<map>/
디렉터리에 prior_map(.pcd/.fmap) + global_waypoints.json 이 같이 들어있는 구조.

low_level_mac 포함, 한 줄로 모든 체인 기동:
  low_level_mac (vesc/livox/cam/joy)
    └─ /livox/lidar, /odom, /joy
  fast_livo localization (mid360_localization.yaml, prior_map_dir=<map>)
    └─ /aft_mapped_to_init  →remap→  /car_state/odom (~10Hz)
  static TF camera_init→map (identity; fast_livo frame ↔ race-stack frame)
  global_republisher (<map>/global_waypoints.json)
    └─ /global_waypoints + /global_waypoints/vel_markers_tuned
  frenet_odom_republisher (cartesian ↔ frenet)
    └─ /car_state/odom_frenet
  fake_topic_relay (state_machine 의존 토픽 stub)
    └─ /global_waypoints_scaled, /global_waypoints/overtaking, /car_state/pose 등
  state_machine (timetrial_only + force_GBTRACK)
    └─ /local_waypoints, /behavior_strategy
  controller_manager (L1 = pure pursuit)
    └─ /vesc/high_level/ackermann_cmd_mux/input/nav_1
  simple_mux (joy mux: RB=autodrive, LB=humandrive)
    └─ /ackermann_cmd (out_topic) → ackermann_to_vesc → motor/servo
  rviz (relocalization.rviz, global path 포함, 옵션)

전제:
  - sb 로 워크스페이스 source (DYLD_LIBRARY_PATH 채워짐)
  - cb 후 ros_fix_install 동작 (conda env/lib symlink, rpath, xattr strip)
  - VESC USB 연결, livox/camera 정상
  - <map> prior map (cloudGlobal.pcd, prior_map.fmap, global_waypoints.json)
    가 fast_livo2/map/<map>/ 에 존재

사용:
  ros2 launch stack_master middle_level_mac.launch.py
  ros2 launch stack_master middle_level_mac.launch.py map:=<other_map>
  ros2 launch stack_master middle_level_mac.launch.py use_low_level:=false   # low_level_mac 따로
  ros2 launch stack_master middle_level_mac.launch.py use_rviz_livo:=false   # rviz 없이

조작:
  - RB (right bumper): autodrive ON — controller 명령이 motor/servo 로 흐름
  - LB (left bumper) + sticks: humandrive — joystick 직접 조종
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Args ───────────────────────────────────────────────────────────
    map_arg          = DeclareLaunchArgument("map", default_value="f")
    low_level_arg    = DeclareLaunchArgument("use_low_level", default_value="true",
                                             description="low_level_mac (vesc/livox/cam/joy) 포함 여부")
    rviz_low_arg     = DeclareLaunchArgument("use_rviz_low_level", default_value="false")
    rviz_livo_arg    = DeclareLaunchArgument("use_rviz_livo", default_value="true")
    joy_arg          = DeclareLaunchArgument("use_joy", default_value="true")

    map_name        = LaunchConfiguration("map")
    use_low_level   = LaunchConfiguration("use_low_level")
    use_rviz_low    = LaunchConfiguration("use_rviz_low_level")
    use_rviz_livo   = LaunchConfiguration("use_rviz_livo")
    use_joy         = LaunchConfiguration("use_joy")

    # ── Resolved paths ─────────────────────────────────────────────────
    sm_share        = get_package_share_directory("stack_master")
    livo_share      = get_package_share_directory("fast_livo")
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )
    livo_main       = os.path.join(livo_share, "config", "mid360_localization.yaml")
    livo_cam        = os.path.join(livo_share, "config", "camera_see3cam.yaml")
    livo_rviz_cfg   = os.path.join(livo_share, "rviz_cfg", "relocalization.rviz")
    # Maps live alongside the fast_livo prior map (cloudGlobal.pcd, prior_map.fmap, ...)
    # under fast_livo2/map/<name>/. global_waypoints.json sits in the same dir so a
    # single `map:=<name>` argument resolves both localization and waypoints.
    # (New global_planner pipeline — centerline_extractor / trajectory_optimizer —
    # still hardcodes stack_master/maps/<name>/; create that as a symlink to this
    # directory when you start using it.)
    src_maps_root = "/Users/mini/ros2_ws/src/IFAC2026_SH/src/slam/fast_livo2/map"

    # ── 1. low_level_mac (vesc + livox + camera + joy + DYLD_LIBRARY_PATH) ──
    low_level = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sm_share, "launch", "low_level_mac.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz_low,
            "joy": use_joy,
        }.items(),
        condition=IfCondition(use_low_level),
    )

    # ── 2. fast_livo localization (prior_map_dir 은 `map` 인자로 override) ─
    # /aft_mapped_to_init → /car_state/odom remap 으로 frenet/state_machine 체인 진입.
    from launch.substitutions import PathJoinSubstitution
    livo_prior_map_dir = PathJoinSubstitution([src_maps_root, map_name])
    livo_node = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        parameters=[
            livo_main,
            livo_cam,
            {"relocalization.prior_map_dir": livo_prior_map_dir},
        ],
        remappings=[("/aft_mapped_to_init", "/car_state/odom")],
        output="screen",
    )
    livo_rviz = Node(
        condition=IfCondition(use_rviz_livo),
        package="rviz2",
        executable="rviz2",
        name="rviz2_livo",
        arguments=["-d", livo_rviz_cfg],
        output="screen",
    )

    # ── 2b. global init (KISS-Matcher) ─────────────────────────────────
    # auto_start=false: trigger 대기. zsh alias `pose` 로 호출하면
    # ~/livox/lidar 누적 → KISS-Matcher → /initialpose 발행 → fast_livo 초기화.
    gi_share = get_package_share_directory("fast_livo_global_init")
    gi_yaml = os.path.join(gi_share, "config", "global_init.yaml")
    gi_pcd_path = PathJoinSubstitution([src_maps_root, map_name, "cloudGlobal.pcd"])
    global_init_node = Node(
        package="fast_livo_global_init",
        executable="global_init_node",
        name="fast_livo_global_init",
        parameters=[
            gi_yaml,
            {
                "prior_map.pcd_path": gi_pcd_path,
                "control.auto_start": False,
            },
        ],
        output="screen",
    )

    # ── 3. static TF camera_init → map (identity) ──────────────────────
    # fast_livo 는 world/camera_init frame, race-stack 의 path 마커는 map frame.
    # 둘을 동일 frame 으로 묶어주기 위한 static transform.
    static_tf_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_camera_init_to_map",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "camera_init", "--child-frame-id", "map",
        ],
        output="log",
    )

    # ── 4. global_republisher: <src>/maps/<map>/global_waypoints.json ──
    # PathJoinSubstitution lets us use the launch-arg `map` at runtime so
    # `ros2 launch ... map:=garage` just works without rebuilding.
    from launch.substitutions import PathJoinSubstitution
    waypoints_path = PathJoinSubstitution([src_maps_root, map_name, "global_waypoints.json"])
    global_repub = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{
            "map": map_name,
            "map_path": waypoints_path,
        }],
        output="screen",
    )

    # ── 5. frenet_odom_republisher (TimerAction: localization init 후) ─
    frenet_odom_repub = TimerAction(period=4.0, actions=[Node(
        package="frenet_odom_republisher",
        executable="frenet_odom_republisher",
        name="frenet_odom_republisher",
        remappings=[
            ("/odom", "/car_state/odom"),
            ("/odom_frenet", "/car_state/odom_frenet"),
            ("/odom_frenet_fixed", "/car_state/odom_frenet_fixed"),
        ],
        output="screen",
    )])

    # ── 6. fake_topic_relay (state_machine init 직전에 stub 발행 시작) ──
    fake_relay = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])

    # ── 7. state_machine (timetrial_only + force_GBTRACK) ──────────────
    sm_node = TimerAction(period=6.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": "SIM",
            "map": map_name,
            "state_machine.rate": 50.0,
            "state_machine.n_loc_wpnts": 80,
            "state_machine.ot_planner": "",
            "state_machine.timetrials_only": True,
            "state_machine.force_GBTRACK": True,
            "state_machine.gb_ego_width_m": 0.3,
            "state_machine.gb_horizon_m": 5.0,
            "state_machine.lateral_width_gb_m": 0.3,
            "state_machine.interest_horizon_m": 20.0,
            "state_machine.use_force_trailing": False,
            "state_machine.splini_ttl": 5.0,
            "state_machine.pred_splini_ttl": 0.2,
            "state_machine.overtaking_horizon_m": 6.9,
            "state_machine.lateral_width_ot_m": 0.3,
            "state_machine.splini_hyst_timer_sec": 3.0,
            "state_machine.emergency_break_horizon": 1.1,
            "state_machine.ftg_speed_mps": 1.0,
            "state_machine.ftg_timer_sec": 3.0,
            "state_machine.ftg_active": False,
            "state_machine.overtaking_ttl_sec": 10.0,
            "state_machine.volt_threshold": 10.0,
            "measure": False,
            "sim": False,
        }],
        output="screen",
    )])

    # ── 8. simple_pp (minimal pure-pursuit, vx_mps 그대로) ─────────────
    # 기존 controller_manager (L1 + lat_err/accel_lim 후처리) 가 vx_mps 를
    # 깎는 문제 디버깅을 위해 순수 PP 로 교체. speed_scale 만 살려서
    # 전체 비례 조정 가능.
    controller_node = TimerAction(period=7.0, actions=[Node(
        package="controller",
        executable="simple_pp",
        name="simple_pp",
        parameters=[{
            "lookahead_distance": 1.2,
            "wheelbase": 0.33,
            "max_steering_rad": 0.4,
            "max_speed_mps": 8.0,
            "speed_scale": 1.0,
            "control_rate_hz": 50.0,
            "drive_topic": "/vesc/high_level/ackermann_cmd_mux/input/nav_1",
        }],
        output="screen",
    )])

    # simple_mux 는 low_level_mac 에서 띄움 (low 만으로도 joy 직접 조작 가능하도록).
    # middle_level 은 controller 명령(/vesc/.../nav_1) 을 추가하기만 함.

    return LaunchDescription([
        map_arg, low_level_arg, rviz_low_arg, rviz_livo_arg, joy_arg,
        low_level,
        livo_node, livo_rviz,
        global_init_node,
        static_tf_map,
        global_repub,
        frenet_odom_repub,
        fake_relay,
        sm_node,
        controller_node,
    ])
