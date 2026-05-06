from setuptools import find_packages, setup
from glob import glob
import os

package_name = "fake_odom_publisher"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch / 샘플 maps 설치 (테스트 편의)
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="3D fake odom publisher — ROS2 Jazzy port",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fake_odom_publisher = fake_odom_publisher.fake_odom_publisher:main",
        ],
    },
)
