from setuptools import find_packages, setup
from glob import glob
import os

package_name = "spliner"

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
    description="spliner (ROS2 Jazzy port)",
    license="Apache-2.0",
    entry_points={"console_scripts": [
            "static_avoidance_node = spliner.static_avoidance_node:main",
            "start_spline_node = spliner.start_spline_node:main",
            "start_spline_node_v2 = spliner.start_spline_node_v2:main",
            "static_avoidance_node_3d = spliner.static_avoidance_node_3d:main",
            "smart_static_avoidance_node = spliner.smart_static_avoidance_node:main",
        ]},
)
