from setuptools import find_packages, setup
from glob import glob
import os

package_name = "obstacle_publisher"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py") + glob("launch/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="Dynamic obstacle publisher (ROS2 Jazzy port)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "obstacle_publisher = obstacle_publisher.obstacle_publisher:main",
            "static_obstacle_manager = obstacle_publisher.static_obstacle_manager:main",
            "dynamic_obstacle_publisher = obstacle_publisher.dynamic_obstacle_publisher:main",
            "collision_detector = obstacle_publisher.collision_detector:main",
            "gazebo_static_obstacle_publisher = obstacle_publisher.gazebo_static_obstacle_publisher:main",
        ],
    },
)
