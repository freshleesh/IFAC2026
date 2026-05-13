from setuptools import find_packages, setup
from glob import glob
import os

package_name = "stack_master"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py") + glob("launch/*.xml")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "SIM"),
         glob("config/SIM/*.yaml") + glob("config/SIM/*.ini") + glob("config/SIM/*.rviz")),
        (os.path.join("share", package_name, "config", "SIM", "veh_dyn_info"),
         glob("config/SIM/veh_dyn_info/*.csv")),
        (os.path.join("share", package_name, "maps", "midterm"), glob("maps/midterm/*")),
        (os.path.join("share", package_name, "maps", "f"), glob("maps/f/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="stack_master (ROS2 Jazzy port)",
    license="Apache-2.0",
    entry_points={"console_scripts": [
            "simple_mux_node.py = stack_master.simple_mux_node:main",
        ]},
)
