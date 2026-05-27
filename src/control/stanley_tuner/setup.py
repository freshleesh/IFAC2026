from setuptools import find_packages, setup
from glob import glob
import os

package_name = "stanley_tuner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="Per-corner Stanley parameter tuner",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tuner_node = stanley_tuner.tuner_node:main",
            "param_mapper_node = stanley_tuner.param_mapper_node:main",
            "sector_visualizer = stanley_tuner.sector_visualizer:main",
        ],
    },
)
