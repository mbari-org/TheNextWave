#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("the_next_wave")
    params_file = os.path.join(pkg_share, "config", "config.yaml")

    return LaunchDescription(
        [
            Node(
                package="the_next_wave",
                executable="the_next_wave_sim_node",
                name="the_next_wave_node",
                output="screen",
                parameters=[params_file],
            )
        ]
    )