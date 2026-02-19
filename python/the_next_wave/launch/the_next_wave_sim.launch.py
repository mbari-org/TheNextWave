#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("the_next_wave")
    default_params_file = os.path.join(pkg_share, "config", "config.yaml")
    params_file = LaunchConfiguration("params_file")
    enable_plotter = LaunchConfiguration("enable_plotter")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Path to the YAML parameter file.",
            ),
            DeclareLaunchArgument(
                "enable_plotter",
                default_value="false",
                description="If true, also run the standalone wave prediction plotter node.",
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_sim_node",
                name="the_next_wave_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_plotter_node",
                name="the_next_wave_plotter_node",
                output="screen",
                parameters=[params_file],
                condition=IfCondition(enable_plotter),
            ),
        ]
    )