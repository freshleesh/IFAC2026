"""HJ 풀 racing stack + f1tenth_gym sim 통합 launch.

체인:
  gym_bridge (sim) → /car_state/odom + /scan + /car_state/pose
    → frenet_odom_republisher → /car_state/odom_frenet
    → state_machine (FSM) → /local_waypoints + /behavior_strategy
    → controller_manager (L1) → /vesc/high_level/ackermann_cmd
    → simple_mux → /vesc/ackermann_cmd → gym_bridge (driving)

기동 순서 (TimerAction):
  t=0:  global_republisher + low_level (gym_bridge + simple_mux + obstacle)
  t=2:  frenet_conversion_server + frenet_odom_republisher
  t=3:  fake_topic_relay + random_obstacle_publisher (state_machine 의 sub 만족)
  t=5:  state_machine
  t=6:  controller_manager (drive_topic remap → /vesc/high_level/ackermann_cmd)

map: f (HJ 의 f110-simulator 표준 맵 — global_waypoints.json + raceline 포함).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    map_arg = DeclareLaunchArgument("map", default_value="f")
    racecar_arg = DeclareLaunchArgument("racecar_version", default_value="SIM")

    sm_share = get_package_share_directory("stack_master")
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )

    # 0) low_level (gym_bridge + simple_mux + obstacle_publisher + rviz)
    low_level = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(sm_share, "launch", "low_level.launch.xml")),
        launch_arguments={"sim": "true", "map": LaunchConfiguration("map")}.items(),
    )

    # 1) global_republisher (트랙 sticky 발행)
    global_repub = TimerAction(period=0.0, actions=[Node(
        package="global_republisher",
        executable="global_republisher",
        name="global_republisher",
        parameters=[{
            "map": LaunchConfiguration("map"),
            "map_path": PathJoinSubstitution([sm_share, "maps", LaunchConfiguration("map"), "global_waypoints.json"]),
        }],
        output="screen",
    )])

    # 2) frenet_conversion_server
    frenet_server = TimerAction(period=2.0, actions=[Node(
        package="frenet_conversion",
        executable="frenet_conversion_server",
        name="frenet_conversion_server",
        output="screen",
    )])

    # 3) frenet_odom_republisher (gym_bridge 의 /car_state/odom → /car_state/odom_frenet)
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

    # 4) fake_topic_relay (/scaled, /overtaking, /recovery 분기 토픽)
    fake_relay = TimerAction(period=3.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])

    # 5) random_obstacle_publisher (/obstacles → /tracking/obstacles, state_machine 의 sub 만족)
    random_obs = TimerAction(period=3.0, actions=[Node(
        package="random_obstacle_publisher",
        executable="random_obstacle_publisher",
        name="random_obstacle_publisher",
        parameters=[{"n_obstacles": 0, "rate_hz": 20.0}],   # 0 = 비어있는 ObstacleArray (clean run)
        remappings=[("/obstacles", "/tracking/obstacles")],
        output="screen",
    )])

    # 6) state_machine
    sm_node = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": LaunchConfiguration("racecar_version"),
            "state_machine/rate": 50.0,
            "state_machine/n_loc_wpnts": 80,
            "state_machine/ot_planner": "",
            "state_machine/timetrials_only": True,    # 추월/회피 분기 비활성 — raceline 따라가기만
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
            "state_machine/force_GBTRACK": True,    # 강제 GB_TRACK (timetrial)
            "state_machine/overtaking_ttl_sec": 3.0,
            "state_machine/volt_threshold": 10.0,
            "/global_republisher/track_length": 25.0,    # f 맵 raceline 길이 (대략)
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])

    # 7) controller_manager — t=6
    # drive_topic 을 /vesc/high_level/ackermann_cmd 로 remap (simple_mux in_topic 매치)
    controller_node = TimerAction(period=6.0, actions=[Node(
        package="controller",
        executable="controller_manager",
        name="control_node",
        parameters=[controller_yaml, {"~drive_topic": "/vesc/high_level/ackermann_cmd"}],
        output="screen",
    )])

    return LaunchDescription([
        map_arg, racecar_arg,
        low_level,
        global_repub,
        frenet_server, frenet_odom_repub,
        fake_relay, random_obs,
        sm_node,
        controller_node,
    ])
