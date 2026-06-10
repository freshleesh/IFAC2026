"""BO training launch wrapper (calls scripts/run_bo.sh).

Usage:
  ros2 launch nonlinear_mpc_acados bo_train.launch.py map:=rand_a
  ros2 launch nonlinear_mpc_acados bo_train.launch.py map:=rand_a n_calls:=30
  ros2 launch nonlinear_mpc_acados bo_train.launch.py maps:="rand_a rand_b"
  ros2 launch nonlinear_mpc_acados bo_train.launch.py x0:="4.0,2.0,1.0,1.5,1.0,3.0,1.5"
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    declared = [
        DeclareLaunchArgument("map",           default_value="rand_a"),
        DeclareLaunchArgument("maps",          default_value=""),
        DeclareLaunchArgument("map_mode",      default_value="alternate"),
        DeclareLaunchArgument("n_calls",       default_value="20"),
        DeclareLaunchArgument("n_initial",     default_value="5"),
        DeclareLaunchArgument("n_laps",        default_value="3"),
        DeclareLaunchArgument("wall_timeout",  default_value="180"),
        DeclareLaunchArgument("stuck_timeout", default_value="60"),
        DeclareLaunchArgument("mode",          default_value="bucketed"),
        DeclareLaunchArgument("x0",            default_value=""),
    ]

    script = "/home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/scripts/run_bo.sh"

    run_bo = ExecuteProcess(
        cmd=[
            "bash", script,
            LaunchConfiguration("map"),
            LaunchConfiguration("n_calls"),
            LaunchConfiguration("n_initial"),
            LaunchConfiguration("n_laps"),
            LaunchConfiguration("wall_timeout"),
            LaunchConfiguration("stuck_timeout"),
            LaunchConfiguration("maps"),
            LaunchConfiguration("map_mode"),
            LaunchConfiguration("mode"),
            LaunchConfiguration("x0"),
        ],
        output="screen",
    )

    return LaunchDescription(declared + [run_bo])
