#!/usr/bin/python3
#
# Usage:
#   ros2 launch fast_livo_global_init global_init.launch.py                # yaml default
#   ros2 launch fast_livo_global_init global_init.launch.py map:=my_seat   # cloudGlobal from fast_livo2/map/my_seat/
#
# fast_livo `localization.launch.py` 와 동일한 `map:=<name>` 인자 패턴.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


MAPS_ROOT = "/Users/mini/ros2_ws/src/IFAC2026_SH/src/slam/fast_livo2/map"


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("fast_livo_global_init")
    params = os.path.join(share, "config", "global_init.yaml")

    map_name = LaunchConfiguration("map").perform(context)
    auto_start_str = LaunchConfiguration("auto_start").perform(context)
    extra_params = []
    if map_name:
        pcd_path = os.path.join(MAPS_ROOT, map_name, "cloudGlobal.pcd")
        extra_params.append({"prior_map.pcd_path": pcd_path})
    if auto_start_str:
        extra_params.append(
            {"control.auto_start": auto_start_str.lower() == "true"})

    node = Node(
        package="fast_livo_global_init",
        executable="global_init_node",
        name="fast_livo_global_init",
        parameters=[params, *extra_params],
        output="screen",
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "map", default_value="",
            description="Prior map name under fast_livo2/map/<name>/. "
                        "Empty = yaml default (prior_map.pcd_path).",
        ),
        DeclareLaunchArgument(
            "auto_start", default_value="",
            description="true: 노드 시작 시 한 번 자동. false: ~/trigger 대기. "
                        "Empty = yaml default (control.auto_start, true).",
        ),
        OpaqueFunction(function=launch_setup),
    ])
