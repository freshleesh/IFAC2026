"""ament_python setup for nonlinear_mpc_acados."""
from glob import glob
from setuptools import find_packages, setup

package_name = 'nonlinear_mpc_acados'

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
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/mpc', glob('config/mpc/*.json')),
        ('share/' + package_name + '/config/tire', glob('config/tire/*.yaml')),
        # Track CSVs — one entry per track so colcon copies the per-track
        # subdirectory intact (track_loader expects share/tracks/track<NAME>/).
        ('share/' + package_name + '/tracks/trackf',
         glob('share/tracks/trackf/*')),
        ('share/' + package_name + '/tracks/trackicra',
         glob('share/tracks/trackicra/*')),
        ('share/' + package_name + '/tracks/trackwheV1racing',
         glob('share/tracks/trackwheV1racing/*')),
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
        ],
    },
)
