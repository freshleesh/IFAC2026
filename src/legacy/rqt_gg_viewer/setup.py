from setuptools import find_packages, setup

package_name = "rqt_gg_viewer"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "plugin.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="RQT plugin: GGV diamond viewer.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
