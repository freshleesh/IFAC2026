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
  mode (timetrial | overtake | mpcc): 운영 모드
    - timetrial: GB_TRACK 만, 추월 분기 비활성 (n_obstacles=0). 검증된 기본 모드.
    - overtake : OVERTAKE 분기 + spliner 정적 회피 + 가짜 장애물 (n_obstacles=4).
    - mpcc     : controller_manager 대신 nonlinear_mpc_acados (MPCC) 사용.
                 timetrial 인프라 + mpc_node + joy_node + auto-engage helper.
                 IFAC 데모용. 자체 reference 추종, state_machine은 GB_TRACK 강제.
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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _build(context: LaunchContext, *_args, **_kwargs):
    map_name = LaunchConfiguration("map").perform(context)
    racecar_version = LaunchConfiguration("racecar_version").perform(context)
    mode = LaunchConfiguration("mode").perform(context)

    if mode not in ("timetrial", "overtake", "avoid", "mpcc"):
        raise ValueError(f"mode must be 'timetrial' / 'overtake' / 'avoid' / 'mpcc', got {mode!r}")

    # 'avoid' = 'overtake' alias — 정적 장애물 회피 의도 명시. 코드 동작 동일.
    if mode == "avoid":
        mode = "overtake"

    is_mpcc = (mode == "mpcc")
    # MPCC mode: state_machine 은 GB_TRACK 강제 (mpc 가 자체 reference 추종).
    timetrials_only = (mode == "timetrial") or is_mpcc
    force_gbtrack = (mode == "timetrial") or is_mpcc
    ot_planner = "spliner" if mode == "overtake" else ""

    sm_share = get_package_share_directory("stack_master")
    controller_yaml = os.path.join(
        get_package_share_directory("controller"), "config", "sim_controller_params.yaml"
    )

    # ── low_level ──
    low_level = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(sm_share, "launch", "low_level.launch.xml")),
        launch_arguments={
            "sim": "true", "map": map_name,
            "gym_mu_scale": LaunchConfiguration("gym_mu_scale").perform(context),
        }.items(),
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
    # random_obs = TimerAction(period=3.0, actions=[Node(
    #     package="random_obstacle_publisher",
    #     executable="random_obstacle_publisher",
    #     name="random_obstacle_publisher",
    #     parameters=[{"n_obstacles": n_obstacles, "rate_hz": 20.0}],
    #     remappings=[("/obstacles", "/tracking/obstacles")],
    #     output="screen",
    # )])

    # ── overtake 분기 노드 (spliner) ──
    actions = [
        low_level,
        global_repub,
        frenet_server, frenet_odom_repub,
        fake_relay,
        #random_obs,
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
            # OVERTAKE ↔ GB_TRACK 진동 방지 — sim hysteresis 강화
            "state_machine/gb_horizon_m": 5.0,         # 1.0 → 5.0 (enemy_in_front 더 길게 True)
            "state_machine/lateral_width_gb_m": 0.3,
            "state_machine/interest_horizon_m": 20.0,
            "state_machine/use_force_trailing": False,
            "state_machine/splini_ttl": 5.0,            # 2.0 → 5.0 (회피 wpnts freshness)
            "state_machine/pred_splini_ttl": 0.2,
            "state_machine/overtaking_horizon_m": 6.9,
            "state_machine/lateral_width_ot_m": 0.3,
            "state_machine/splini_hyst_timer_sec": 3.0,  # 0.75 → 3.0
            "state_machine/emergency_break_horizon": 1.1,
            "state_machine/ftg_speed_mps": 1.0,
            "state_machine/ftg_timer_sec": 3.0,
            "state_machine/ftg_active": False,
            "state_machine/force_GBTRACK": force_gbtrack,
            "state_machine/overtaking_ttl_sec": 10.0,    # 3.0 → 10.0 (OVERTAKE 종료 지연)
            "state_machine/volt_threshold": 10.0,
            "/global_republisher/track_length": 25.0,
            "measure": False,
            "sim": True,
        }],
        output="screen",
    )])
    actions.append(sm_node)

    if not is_mpcc:
        # ── controller_manager (timetrial / overtake) ──
        controller_node = TimerAction(period=6.0, actions=[Node(
            package="controller",
            executable="controller_manager",
            name="control_node",
            parameters=[controller_yaml, {"~drive_topic": "/vesc/high_level/ackermann_cmd"}],
            output="screen",
        )])
        actions.append(controller_node)
    else:
        # ── MPCC 모드: nonlinear_mpc_acados 가 controller 자리 대체 ──
        mpc_share = get_package_share_directory("nonlinear_mpc_acados")
        mpc_params = os.path.join(mpc_share, "config", "ddrx_unified_params.yaml")
        # ACADOS env (libacados.so / Tera renderer / generated solver dlopen).
        # 사용자가 export 했다면 그 값 우선, 없으면 ~/acados.
        acados_dir = os.environ.get("ACADOS_SOURCE_DIR") or os.path.expanduser("~/acados")
        ld_extra = os.path.join(acados_dir, "lib")
        actions.append(SetEnvironmentVariable("ACADOS_SOURCE_DIR", acados_dir))
        actions.append(SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            ld_extra + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        ))

        # Optional model toggles — override the yaml WITHOUT editing it (avoids the
        # config-clobber regression class). Empty default → keep ddrx_unified_params.
        def _tri(name):
            v = LaunchConfiguration(name).perform(context).strip().lower()
            if v in ("true", "1", "yes"):
                return True
            if v in ("false", "0", "no"):
                return False
            return None  # unset → don't override
        _model_overrides = {}
        for _k in ("use_dynamic", "use_lmpc"):
            _v = _tri(_k)
            if _v is not None:
                _model_overrides[_k] = _v

        # mpc_disable=true 면 mpc_node 비활성 → mux 가 pp_fallback 사용 (PP baseline 측정용).
        mpc_disable_str = LaunchConfiguration("mpc_disable").perform(context).lower()
        if mpc_disable_str not in ("true", "1", "yes"):
            mpc_node = TimerAction(period=6.0, actions=[Node(
                package="nonlinear_mpc_acados",
                executable="mpc_node",
                name="mpc_node",
                parameters=[
                    mpc_params,
                    {
                        "mpc_backend": "acados",
                        # simple_mux in_topic 과 매칭: mpc → /vesc/high_level/ackermann_cmd
                        "cmd_vel_topic_name": "/vesc/high_level/ackermann_cmd",
                        # CSV fallback 이 올바른 track 을 로드하도록 — map 이름이
                        # nonlinear_mpc_acados/share/tracks/track<name>/ 와 매칭.
                        "track_name": map_name,
                        # IQP raceline json for LMPC apex seed (grip-clamped). Same
                        # file global_republisher uses. Empty → centerline seed.
                        "lmpc_raceline_json": os.path.join(
                            sm_share, "maps", map_name, "global_waypoints.json"),
                        # teleport-rescue toggle (검증 시 false 로 raw wedging 노출)
                        "enable_sim_reset": (
                            LaunchConfiguration("enable_sim_reset").perform(context).lower()
                            in ("true", "1", "yes")
                        ),
                        # use_dynamic / use_lmpc launch-arg overrides (empty → yaml).
                        **_model_overrides,
                    },
                ],
                output="screen",
            )])
            actions.append(mpc_node)

            # ── mpc_debug_logger: 매 cycle CSV (~/mpc_logs/) + 죽는 순간 자동
            # event dump (~/mpc_logs/events/event_<reason>_*.csv).
            debug_logger = TimerAction(period=6.0, actions=[Node(
                package="nonlinear_mpc_acados",
                executable="mpc_debug_logger",
                name="mpc_debug_logger",
                output="log",
            )])
            actions.append(debug_logger)

        # FTG reactive fallback — kept as secondary safety net (현재 mux 가 안 씀,
        # 필요 시 fallback_topic 을 /vesc/ftg_fallback 으로 되돌리면 활성).
        ftg_fallback = Node(
            package="nonlinear_mpc_acados",
            executable="ftg_fallback_node",
            name="ftg_fallback",
            parameters=[{
                "scan_topic": "/scan",
                "cmd_topic":  "/vesc/ftg_fallback",
                "rate_hz":    20.0,
                "max_speed":  2.5,
            }],
            output="log",
        )
        actions.append(ftg_fallback)

        # PP fallback — MPCC 죽으면 mux 가 자동으로 switch. PP baseline 측정 시
        # wpnts_topic=/global_waypoints (raceline) + max_speed=BO 의 v.
        pp_wpnts_topic = LaunchConfiguration("pp_wpnts_topic").perform(context)
        pp_max_speed   = float(LaunchConfiguration("pp_max_speed").perform(context))
        pp_fallback = Node(
            package="nonlinear_mpc_acados",
            executable="pp_fallback_node",
            name="pp_fallback",
            parameters=[{
                "wpnts_topic": pp_wpnts_topic,
                "odom_topic":  "/car_state/odom",
                "cmd_topic":   "/vesc/pp_fallback",
                "rate_hz":     20.0,
                "max_speed":   pp_max_speed,
                "lookahead":   1.5,
                "wheelbase":   0.307,
                "s_max":       0.4,
            }],
            output="log",
        )
        actions.append(pp_fallback)

        # ── joy (수동/자동 토글). USB joystick 없으면 idle. ──
        joy_node = Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[{"deadzone": 0.05, "autorepeat_rate": 20.0}],
            output="log",
        )
        actions.append(joy_node)

        # ── auto-engage helper: mpc 가 solver codegen 끝낼 충분한 시간 (~40s)
        # 후에 joy RB(buttons[5])=1 한 번 publish → simple_mux 의 autodrive_latched
        # rising-edge 트리거. 이후엔 joy LB 로 수동 takeover, RB 로 다시 autodrive.
        # mpc_disable=true (PP baseline 측정) 시도 발행 — 짧은 timer (5s, codegen 없음).
        engage_period = 5.0 if mpc_disable_str in ("true", "1", "yes") else 40.0
        auto_engage = TimerAction(period=engage_period, actions=[ExecuteProcess(
            cmd=[
                "ros2", "topic", "pub", "--once", "/joy",
                "sensor_msgs/msg/Joy",
                "{header: {frame_id: 'auto_engage'}, "
                "axes: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "
                "buttons: [0, 0, 0, 0, 0, 1, 0, 0]}",
            ],
            output="log",
        )])
        actions.append(auto_engage)

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="f"),
        DeclareLaunchArgument("racecar_version", default_value="SIM"),
        DeclareLaunchArgument("mode", default_value="timetrial",
                              description="timetrial | overtake"),
        DeclareLaunchArgument("mpc_disable", default_value="false",
                              description="true → mpc_node 안 띄움. mux 가 pp_fallback 사용 (PP baseline 측정용)"),
        DeclareLaunchArgument("pp_wpnts_topic", default_value="/global_waypoints",
                              description="PP fallback reference. raceline(global) 기본 — 시작/폴백을 레이싱라인으로 (2026-06-07 사용자). centerline 원하면 /centerline_waypoints"),
        DeclareLaunchArgument("pp_max_speed", default_value="4.0",
                              description="PP fallback max speed. baseline 측정 시 yaml max_speed 와 맞춤"),
        DeclareLaunchArgument("enable_sim_reset", default_value="true",
                              description="true → STUCK 시 /initialpose teleport-rescue. false → 실차/검증 (teleport off)"),
        DeclareLaunchArgument("gym_mu_scale", default_value="1.0",
                              description="B4' known-mismatch: scale gym TRUE tire friction (1.0 = off)"),
        DeclareLaunchArgument("use_dynamic", default_value="",
                              description="override yaml: true=dynamic 8-state, false=kinematic 5-state. empty=use yaml"),
        DeclareLaunchArgument("use_lmpc", default_value="",
                              description="override yaml: true/false. (kinematic auto-forces off). empty=use yaml"),
        OpaqueFunction(function=_build),
    ])
