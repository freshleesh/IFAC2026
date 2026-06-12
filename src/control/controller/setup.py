from setuptools import find_packages, setup
from glob import glob
import os

package_name = "controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="Controller manager (ROS2 Jazzy port)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_manager = controller.controller_manager:main",
            "simple_pp = controller.simple_pp:main",
            "stanley = controller.stanley:main",
            "stanley_plot = controller.stanley_plot:main",
            "gap_follow_node = controller.gapfollow:main",
            "wall_follow_node = controller.wallfollow:main",
            "lqr = controller.lqr:main",
            'fc_node = controller.friction_circle_controller:main',
            'friction_test = controller.friction_test:main',
        ],
    },
)
