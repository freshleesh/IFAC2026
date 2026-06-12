"""elrs_joy_ft232_node — FT232RL @ /dev/ELRS_FT232, CRSF native 420000 baud, CRC-8 enabled.

ls /dev/tty.*      to find the correct port for your FT232-based receiver. The default is set to /dev/tty.usbserial-310, but it may be different on your system.

CLI overrides:
    ros2 launch elrs_joy elrs_joy_ft232.launch.py enable_crc:=false
    ros2 launch elrs_joy elrs_joy_ft232.launch.py port:=/dev/tty.usbserial-FT12345

Watch CRC effectiveness:
    ros2 topic echo /elrs_joy_node/debug_stats
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("enable_crc", default_value="true"),
        DeclareLaunchArgument("port", default_value="/dev/tty.usbserial-310"),
        DeclareLaunchArgument("baud_rate", default_value="420000"),
        DeclareLaunchArgument("debug_stats_hz", default_value="1.0"),
        DeclareLaunchArgument("rb_stability_frames", default_value="10"),
        DeclareLaunchArgument("rb_released_min", default_value="992"),
        DeclareLaunchArgument("rb_jerk_max", default_value="200"),
    ]

    static_params = {
        "publish_rate": 100,

        "num_axes": 8,
        "num_buttons": 11,

        "axes_joy_indices": [1, 3],
        "axes_crsf_channels": [2, 0],
        "axes_invert": [1.0, -1.0],

        # Per-axis CRSF calibration (carried over from legacy elrs_joy.launch)
        "axes_cal_min": [174, 174],
        "axes_cal_mid": [1007, 992],
        "axes_cal_max": [1811, 1773],

        "button_joy_indices": [0, 4, 5],
        "button_crsf_channels": [7, 5, 6],
        "button_invert": [1, 0, 0],
        "button_threshold": 992,

        "deadzone": 0.05,
        "failsafe_timeout": 0.5,

        "lb_pressed_max": 350,
        "lb_idle_min": 700,
        "lb_idle_max": 1300,
        "lb_released_min": 1600,
        "lb_debounce_frames": 5,

        "a_debounce_frames": 5,

        "frame_id": "elrs_joy",
    }

    cli_params = {
        "port": LaunchConfiguration("port"),
        "baud_rate": PythonExpression(["int(\"", LaunchConfiguration("baud_rate"), "\")"]),
        "enable_crc": PythonExpression(["\"", LaunchConfiguration("enable_crc"), "\".lower() == \"true\""]),
        "debug_stats_hz": PythonExpression(["float(\"", LaunchConfiguration("debug_stats_hz"), "\")"]),
        "rb_stability_frames": PythonExpression(["int(\"", LaunchConfiguration("rb_stability_frames"), "\")"]),
        "rb_released_min": PythonExpression(["int(\"", LaunchConfiguration("rb_released_min"), "\")"]),
        "rb_jerk_max": PythonExpression(["int(\"", LaunchConfiguration("rb_jerk_max"), "\")"]),
    }

    return LaunchDescription(args + [
        Node(
            package="elrs_joy",
            executable="elrs_joy_ft232_node",
            name="elrs_joy_node",
            output="screen",
            parameters=[static_params, cli_params],
        ),
    ])
