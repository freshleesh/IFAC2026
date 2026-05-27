from setuptools import find_packages, setup

package_name = "elrs_joy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="ELRS Joy node (ROS2 Jazzy port)",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "elrs_joy_node = elrs_joy.elrs_joy_node:main",
    ]},
)
