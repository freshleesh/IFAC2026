"""C-6 통합 smoke launch.

state_machine 노드와 그 의존을 모두 한 번에 띄움 — startup 진입 검증용.

기동 순서 (TimerAction 으로 staggered):
  t=0:    global_republisher (트랙 sticky 발행)
  t=1:    frenet_conversion_server (service)
  t=1:    fake_odom_publisher (raceline 따라 odom 발행 → /car_state/odom)
  t=2:    frenet_odom_republisher (/car_state/odom → /car_state/odom_frenet)
  t=2:    fake_topic_relay (/global_waypoints alias 두 개 + recovery 빈 메시지)
  t=2:    random_obstacle_publisher (/obstacles → 그대로 /tracking/obstacles 으로 사용)
  t=4:    state_machine (모든 의존 ready 후 startup)

map=gazebo_wall_2 default. trajectory_planning_helpers (tph) 가 필요한 노드는
state_machine 의 _load_vehicle_dynamics 안. 검증 시점에 pip install 결정.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    map_arg = DeclareLaunchArgument("map", default_value="gazebo_wall_2")
    racecar_arg = DeclareLaunchArgument("racecar_version", default_value="SIM")
    ot_planner_arg = DeclareLaunchArgument(
        "ot_planner",
        default_value="",
        description="\"\" 또는 \"spliner\" 또는 \"predictive_spliner\". 빈 값이면 OT sub 분기 skip (smoke 검증 단순화)",
    )

    # 1) global_republisher
    global_repub = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{"map": LaunchConfiguration("map")}],
        output="screen",
    )

    # 2) frenet_conversion_server
    frenet_server = TimerAction(period=1.0, actions=[Node(
        package="frenet_conversion",
        executable="frenet_conversion_server",
        name="frenet_conversion_server",
        output="screen",
    )])

    # 3) fake_odom_publisher (raceline 기반 odom — /car_state/odom 으로 remap)
    fake_odom = TimerAction(period=1.0, actions=[Node(
        package="fake_odom_publisher",
        executable="fake_odom_publisher",
        name="fake_odom_publisher",
        parameters=[{"map": LaunchConfiguration("map"), "rate": 50.0, "speed_scale": 1.0}],
        remappings=[("/glim_ros/base_odom", "/car_state/odom")],
        output="screen",
    )])

    # 4) frenet_odom_republisher (/odom → /odom_frenet, launch 안 remap 그대로)
    frenet_odom_repub = TimerAction(period=2.0, actions=[Node(
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

    # 5) fake_topic_relay (/scaled, /overtaking, /recovery)
    fake_relay = TimerAction(period=2.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])

    # 6) random_obstacle_publisher (/obstacles — state_machine 가 /tracking/obstacles 받음)
    random_obs = TimerAction(period=2.0, actions=[Node(
        package="random_obstacle_publisher",
        executable="random_obstacle_publisher",
        name="random_obstacle_publisher",
        parameters=[{"n_obstacles": 4, "rate_hz": 20.0}],
        remappings=[("/obstacles", "/tracking/obstacles")],
        output="screen",
    )])

    # 7) state_machine — 모든 의존 ready 후 startup
    sm_node = TimerAction(period=4.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            # state_machine/* (init mixin 이 _load_rosparams 에서 read)
            "racecar_version": LaunchConfiguration("racecar_version"),
            "state_machine/rate": 50.0,
            "state_machine/n_loc_wpnts": 80,
            "state_machine/ot_planner": LaunchConfiguration("ot_planner"),
            "state_machine/timetrials_only": False,
            "state_machine/gb_ego_width_m": 0.3,
            "state_machine/gb_horizon_m": 1.0,
            "state_machine/lateral_width_gb_m": 0.3,
            "state_machine/interest_horizon_m": 20.0,
            "state_machine/use_force_trailing": False,
            "state_machine/splini_ttl": 2.0,
            "state_machine/pred_splini_ttl": 0.2,
            "state_machine/overtaking_horizon_m": 6.9,
            "state_machine/lateral_width_ot_m": 0.3,
            "state_machine/splini_hyst_timer_sec": 0.75,
            "state_machine/emergency_break_horizon": 1.1,
            "state_machine/ftg_speed_mps": 1.0,
            "state_machine/ftg_timer_sec": 3.0,
            "state_machine/ftg_active": False,
            "state_machine/force_GBTRACK": False,
            "state_machine/overtaking_ttl_sec": 3.0,
            "state_machine/volt_threshold": 10.0,
            # /global_republisher 가 set 한 track_length 도 우리 노드는 못 읽음
            # (별도 노드의 parameter — ROS1 처럼 global rosparam 가 ROS2 에는 없음)
            "/global_republisher/track_length": 85.64,  # gazebo_wall_2 default
            # /map_params, /ot_map_params (nested dict) 은 launch 에서 못 줌 — init mixin 이
            # None fallback 처리. 실 동작 시엔 yaml 파일로 load 필요 (C-6 검증 범위 외).
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])

    return LaunchDescription([
        map_arg, racecar_arg, ot_planner_arg,
        global_repub,
        frenet_server, fake_odom,
        frenet_odom_repub, fake_relay, random_obs,
        sm_node,
    ])
