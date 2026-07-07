import os
from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = "helhest_stack_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ales Kucera",
    maintainer_email="kuceral4@fel.cvut.cz",
    description="ROS 2 Kilted wrapper for the helhest.perception library.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "single_scan_terrain_node = helhest_stack_ros.single_scan_terrain_node:main",
            "terrain_accumulator_node = helhest_stack_ros.terrain_accumulator_node:main",
            "elevation_node = helhest_stack_ros.elevation_node:main",
        ],
    },
)
