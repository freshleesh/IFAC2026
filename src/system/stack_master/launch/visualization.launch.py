"""middle_level_mac 와 동일한 RViz 단독 launch.

config 파일을 src/ 디렉터리에서 직접 읽으므로 rviz_cfg 를 수정하면 다음 launch
실행 시 cb 없이 바로 반영됨. 같은 머신에서 middle_level_mac 가 use_rviz_livo:=true
로 이미 rviz 를 띄우고 있다면 이 launch 를 더 띄우면 중복 — 둘 중 하나만.

사용:
  ros2 launch stack_master visualization.launch.py
  ros2 launch stack_master visualization.launch.py rviz_config:=<absolute path>

전제:
  - sb 로 워크스페이스 source
  - middle_level_mac 또는 fast_livo localization 이 따로 떠 있어야 /tf, /global_waypoints
    등이 publish 되어 RViz 가 의미 있는 내용을 그림
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Source-tree path so edits to the rviz config show up without cb.
DEFAULT_RVIZ_CFG = "/Users/mini/ros2_ws/src/fast_livo2/rviz_cfg/relocalization.rviz"


def generate_launch_description() -> LaunchDescription:
    rviz_cfg_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=DEFAULT_RVIZ_CFG,
        description="RViz config file path (default reads from src/, no rebuild "
                    "needed to pick up edits).",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_livo",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    return LaunchDescription([
        rviz_cfg_arg,
        rviz_node,
    ])
