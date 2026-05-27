from setuptools import find_packages, setup
package_name = "fast_sqp_planner"
setup(
    name=package_name, version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="SH", maintainer_email="freshleesh@gmail.com",
    description="fast_sqp_planner", license="Apache-2.0",
    entry_points={"console_scripts": [
            "fast_sqp_planner_node = fast_sqp_planner.fast_sqp_planner_node:main",
        ]},
)
