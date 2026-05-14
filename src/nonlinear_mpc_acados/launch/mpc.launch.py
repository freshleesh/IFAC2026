"""ROS2 launch — IFAC MPC node + parameter file.

Portability notes:
- `acados_dir` default falls back through env (`$ACADOS_SOURCE_DIR`) then
  `~/acados`. Override at launch with `acados_dir:=/your/path` if either
  is wrong.
- `LD_LIBRARY_PATH` is extended to include `$ACADOS_SOURCE_DIR/lib` so the
  acados Python bindings can dlopen `libacados.so` / `libblasfeo.so` etc.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('nonlinear_mpc_acados')
    default_params = os.path.join(pkg_share, 'config', 'ddrx_unified_params.yaml')

    # Default acados location:
    #   1) $ACADOS_SOURCE_DIR if exported
    #   2) ~/acados as the conventional checkout location
    # User can still override with `acados_dir:=...` on the command line.
    default_acados = os.environ.get('ACADOS_SOURCE_DIR') \
        or os.path.expanduser('~/acados')

    return LaunchDescription([
        DeclareLaunchArgument('acados_dir', default_value=default_acados,
                              description='Path to the acados install (C source + lib/).'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('mpc_backend', default_value='acados'),

        # ACADOS_SOURCE_DIR is consulted by acados_template when generating
        # the OCP C code and dlopen-ing the resulting shared object.
        SetEnvironmentVariable('ACADOS_SOURCE_DIR',
                               LaunchConfiguration('acados_dir')),
        # Prepend $ACADOS_SOURCE_DIR/lib to LD_LIBRARY_PATH so the loader
        # finds libacados.so / libblasfeo.so / libhpipm.so at runtime.
        SetEnvironmentVariable(
            'LD_LIBRARY_PATH',
            PythonExpression([
                "'", LaunchConfiguration('acados_dir'), "' + '/lib:' + '",
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
                "'",
            ]),
        ),

        Node(
            package='nonlinear_mpc_acados',
            executable='mpc_node',
            name='mpc_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'mpc_backend': LaunchConfiguration('mpc_backend')},
            ],
        ),
    ])
