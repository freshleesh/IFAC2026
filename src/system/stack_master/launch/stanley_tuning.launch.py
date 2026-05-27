"""Stanley 파라미터 튜닝 통합 launch.

프랙티스 모드와 경기 모드 두 가지를 지원한다.

LaunchArgs:
  map             — 맵 이름 (stack_master/maps/<name>/...)
  racecar_version — 차량 설정 (기본 SIM)
  tuner_mode      — practice | race
                    practice : stanley + tuner_node (학습 + param_profile 발행)
                    race     : stanley + param_mapper_node (YAML 로드 + param_profile 발행)
  pp_mode         — rule_based | bayes  (practice 모드에서 사용, 기본 rule_based)

기동 순서:
  t=0:  global_republisher + low_level (gym_bridge + simple_mux + rviz)
  t=2:  frenet_conversion_server + frenet_odom_republisher
  t=3:  fake_topic_relay
  t=5:  state_machine
  t=7:  stanley (controller)
  t=8:  tuner_node 또는 param_mapper_node
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _build(context: LaunchContext, *_args, **_kwargs):
    map_name        = LaunchConfiguration("map").perform(context)
    racecar_version = LaunchConfiguration("racecar_version").perform(context)
    tuner_mode      = LaunchConfiguration("tuner_mode").perform(context)
    pp_mode         = LaunchConfiguration("pp_mode").perform(context)

    if tuner_mode not in ("practice", "race"):
        raise ValueError(f"tuner_mode must be 'practice' or 'race', got {tuner_mode!r}")
    if pp_mode not in ("rule_based", "bayes"):
        raise ValueError(f"pp_mode must be 'rule_based' or 'bayes', got {pp_mode!r}")

    _sm_install = get_package_share_directory("stack_master")
    _ws = os.path.normpath(os.path.join(_sm_install, '..', '..', '..', '..'))
    sm_share = os.path.join(_ws, 'src', 'system', 'stack_master')

    stanley_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "stanley_params.yaml"
    )
    tuner_yaml = os.path.join(
        get_package_share_directory("stanley_tuner"), "config", "tuner_params.yaml"
    )

    # ── low_level (gym_bridge + simple_mux + obstacle) ──
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
            "map_path": os.path.join(
                get_package_share_directory("stack_master"),
                "maps", map_name, "global_waypoints.json"
            ),
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
            ("/odom",              "/car_state/odom"),
            ("/odom_frenet",       "/car_state/odom_frenet"),
            ("/odom_frenet_fixed", "/car_state/odom_frenet_fixed"),
        ],
        output="screen",
    )])

    # ── fake_topic_relay ──
    fake_relay = TimerAction(period=3.0, actions=[Node(
        package="state_machine",
        executable="fake_topic_relay",
        name="fake_topic_relay",
        output="screen",
    )])

    # ── state_machine (GB_TRACK 강제: timetrial 순수 주행) ──
    sm_node = TimerAction(period=5.0, actions=[Node(
        package="state_machine",
        executable="state_machine",
        name="state_machine",
        parameters=[{
            "racecar_version": racecar_version,
            "map": map_name,
            "state_machine.rate": 50.0,
            "state_machine.n_loc_wpnts": 80,
            "state_machine.ot_planner": "",
            "state_machine.timetrials_only": True,
            "state_machine.gb_ego_width_m": 0.3,
            "state_machine.force_GBTRACK": True,
            "state_machine.ftg_active": False,
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])

    # ── stanley controller ──
    stanley_node = TimerAction(period=7.0, actions=[Node(
        package="controller",
        executable="stanley",
        name="stanley",
        parameters=[stanley_yaml],
        output="screen",
    )])

    # ── 튜너 노드 (모드에 따라 선택) ──
    # tuner_params.yaml을 기본으로 로드하고, map/tuner_mode만 launch 인자로 덮어쓰기
    if tuner_mode == "practice":
        tuner_node = TimerAction(period=8.0, actions=[Node(
            package="stanley_tuner",
            executable="tuner_node",
            name="stanley_tuner",
            parameters=[tuner_yaml, {"map": map_name, "tuner_mode": pp_mode}],
            output="screen",
        )])
    else:  # race
        tuner_node = TimerAction(period=8.0, actions=[Node(
            package="stanley_tuner",
            executable="param_mapper_node",
            name="stanley_param_mapper",
            parameters=[tuner_yaml, {"map": map_name}],
            output="screen",
        )])

    return [
        low_level,
        global_repub,
        frenet_server,
        frenet_odom_repub,
        fake_relay,
        sm_node,
        stanley_node,
        tuner_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map",             default_value="",         description="Map name"),
        DeclareLaunchArgument("racecar_version", default_value="SIM",      description="Racecar config name"),
        DeclareLaunchArgument("tuner_mode",      default_value="practice", description="practice | race"),
        DeclareLaunchArgument("pp_mode",         default_value="rule_based", description="rule_based | bayes"),
        OpaqueFunction(function=_build),
    ])
