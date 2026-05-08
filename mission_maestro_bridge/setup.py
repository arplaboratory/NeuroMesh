from setuptools import find_packages, setup

package_name = "mission_maestro_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/mission_maestro_bridge.launch.py"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Long Quang",
    maintainer_email="longquang@nyu.edu",
    description="Bridge generic NeuroMesh mission topics to arl_mission_maestro services.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_maestro_bridge = mission_maestro_bridge.bridge_node:main",
        ],
    },
)
