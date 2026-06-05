"""ament_python setup for nonlinear_mpc_acados."""
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nonlinear_mpc_acados'

# Dynamic tracks discovery — every subdir under share/tracks/ gets installed
# intact. Lets gen_random_track.py drop new tracks in without editing this file.
_track_entries = [
    (f'share/{package_name}/tracks/{os.path.basename(d)}',
     [f for f in glob(os.path.join(d, '*')) if os.path.isfile(f)])
    for d in glob('share/tracks/*') if os.path.isdir(d)
]

setup(
    name=package_name,
    version='0.1.0',
    # find_packages picks up `nonlinear_mpc_acados.mpc_core` (the bundled
    # ROS-free MPCC core) along with the top-level package.
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py') + glob('launch/*.launch.xml')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/mpc', glob('config/mpc/*.json')),
        ('share/' + package_name + '/config/tire', glob('config/tire/*.yaml')),
        *_track_entries,
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hmcl',
    maintainer_email='dnrwls04@naver.com',
    description='ROS2 wrapper for EVO-MPCC acados/IPOPT core',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_node = nonlinear_mpc_acados.mpc_node:main',
            'mpc_debug_logger = nonlinear_mpc_acados.mpc_debug_logger:main',
            'ftg_fallback_node = nonlinear_mpc_acados.ftg_fallback_node:main',
            'pp_fallback_node = nonlinear_mpc_acados.pp_fallback_node:main',
        ],
    },
)
