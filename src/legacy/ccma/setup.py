from setuptools import setup

package_name = "ccma"

setup(
    name=package_name,
    version="0.0.1",
    # 원본 소스 트리: src/ccma/<name>.py.
    # ament_python 으로 install 할 때 site-packages/ccma/<name>.py 로 매핑.
    packages=[package_name],
    package_dir={package_name: "src/ccma"},
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="SH",
    maintainer_email="freshleesh@gmail.com",
    description="Curvature Corrected Moving Average (ament_python wrapper).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
