from glob import glob

from setuptools import find_packages, setup

package_name = 'global_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/create_path.launch.py']),
        ('share/' + package_name + '/config', glob('config/*.ini')),
        ('share/' + package_name + '/config/inputs/veh_dyn_info', 
        glob('config/inputs/veh_dyn_info/*.csv')),
        ('share/' + package_name + '/config/inputs/tracks', 
        ['config/inputs/tracks/.gitkeep']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc5',
    maintainer_email='jeongsangryu@gmail.com',
    description='Global planner: centerline extraction and trajectory optimization for F1TENTH',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'centerline_extractor = global_planner.centerline_extractor:main',
            'trajectory_optimizer = global_planner.trajectory_optimizer:main',
        ],
    },
)
