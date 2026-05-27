from setuptools import find_packages, setup
from glob import glob
import os

package_name = "spliner_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="spliner_planner (ROS2 Jazzy port)",
    license="Apache-2.0",
    entry_points={"console_scripts": [
            "dynamic_avoidance_node = spliner_planner.dynamic_avoidance_node:main",
        ]},
)
