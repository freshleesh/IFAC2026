"""HJ 풀 racing stack + f1tenth_gym sim 통합 launch.

체인:
  gym_bridge (sim) → /car_state/odom + /scan + /car_state/pose
    → frenet_odom_republisher → /car_state/odom_frenet
    → state_machine (FSM) → /local_waypoints + /behavior_strategy
    → controller_manager (L1) → /vesc/high_level/ackermann_cmd
    → simple_mux → /vesc/ackermann_cmd → gym_bridge (driving)

LaunchArgs:
  map (f): 맵 이름 — stack_master/maps/<name>/{<name>.{png,yaml}, global_waypoints.json}
  racecar_version (SIM): 차량 설정 이름
  mode (timetrial | overtake): 운영 모드
    - timetrial: GB_TRACK 만, 추월 분기 비활성 (n_obstacles=0). 검증된 기본 모드.
    - overtake : OVERTAKE 분기 + spliner 정적 회피 + 가짜 장애물 (n_obstacles=4).
  n_obstacles (auto): 명시 시 mode 와 무관하게 강제. 0=정적 장애물 발생 안 함.

기동 순서 (TimerAction):
  t=0:  global_republisher + low_level (gym_bridge + simple_mux + obstacle + rviz)
  t=2:  frenet_conversion_server + frenet_odom_republisher
  t=3:  fake_topic_relay + random_obstacle_publisher
  t=4:  spliner (overtake 모드일 때만)
  t=5:  state_machine
  t=6:  controller_manager
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _build(context: LaunchContext, *_args, **_kwargs):
    map_name = LaunchConfiguration("map").perform(context)
    racecar_version = LaunchConfiguration("racecar_version").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    n_obstacles_str = LaunchConfiguration("n_obstacles").perform(context)

    if mode not in ("timetrial", "overtake", "avoid"):
        raise ValueError(f"mode must be 'timetrial' / 'overtake' / 'avoid', got {mode!r}")

    # 'avoid' = 'overtake' alias — 정적 장애물 회피 의도 명시. 코드 동작 동일.
    if mode == "avoid":
        mode = "overtake"

    # mode 에 따른 default. n_obstacles="auto" 면 mode 기준.
    if n_obstacles_str == "auto":
        n_obstacles = 4 if mode == "overtake" else 0
    else:
        n_obstacles = int(n_obstacles_str)

    timetrials_only = (mode == "timetrial")
    force_gbtrack = (mode == "timetrial")
    ot_planner = "spliner" if mode == "overtake" else ""

    sm_share = get_package_share_directory("stack_master")
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )

    # ── low_level ──
    low_level = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(sm_share, "launch", "low_level.launch.xml")),
        launch_arguments={"sim": "true", "map": map_name}.items(),
    )

    # ── global_republisher ──
    global_repub = Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{
            "map": map_name,
            "map_path": os.path.join(sm_share, "maps", map_name, "global_waypoints.json"),
        }],
        output="screen",
    )

    # ── frenet ──
    frenet_server = TimerAction(period=2.0, actions=[Node(
        package="frenet_conversion",
        executable="frenet_conversion_server",
        name="frenet_conversion_server",
        output="screen",
    )])
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

    # ── fake topic relay + obstacle pub ──
    fake_relay = TimerAction(period=3.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])
    random_obs = TimerAction(period=3.0, actions=[Node(
        package="random_obstacle_publisher",
        executable="random_obstacle_publisher",
        name="random_obstacle_publisher",
        parameters=[{"n_obstacles": n_obstacles, "rate_hz": 20.0}],
        remappings=[("/obstacles", "/tracking/obstacles")],
        output="screen",
    )])

    # ── overtake 분기 노드 (spliner) ──
    actions = [
        low_level,
        global_repub,
        frenet_server, frenet_odom_repub,
        fake_relay, random_obs,
    ]
    if mode == "overtake":
        spliner_node = TimerAction(period=4.0, actions=[Node(
            package="spliner",
            executable="static_avoidance_node",  # 정적 + 동적 회피 spliner
            name="spliner",
            output="screen",
        )])
        actions.append(spliner_node)

    # ── state_machine ──
    sm_node = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": racecar_version,
            "map": map_name,    # ROS2: state_machine init 의 ot_sectors.yaml fallback 용
            "state_machine/rate": 50.0,
            "state_machine/n_loc_wpnts": 80,
            "state_machine/ot_planner": ot_planner,
            "state_machine/timetrials_only": timetrials_only,
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
            "state_machine/force_GBTRACK": force_gbtrack,
            "state_machine/overtaking_ttl_sec": 3.0,
            "state_machine/volt_threshold": 10.0,
            "/global_republisher/track_length": 25.0,
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])
    actions.append(sm_node)

    # ── controller_manager ──
    controller_node = TimerAction(period=6.0, actions=[Node(
        package="controller",
        executable="controller_manager",
        name="control_node",
        parameters=[controller_yaml, {"~drive_topic": "/vesc/high_level/ackermann_cmd"}],
        output="screen",
    )])
    actions.append(controller_node)

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="f"),
        DeclareLaunchArgument("racecar_version", default_value="SIM"),
        DeclareLaunchArgument("mode", default_value="timetrial",
                              description="timetrial | overtake"),
        DeclareLaunchArgument("n_obstacles", default_value="auto",
                              description="0=강제 비활성, auto=mode 기준, 정수=강제"),
        OpaqueFunction(function=_build),
    ])
