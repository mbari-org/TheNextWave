#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("the_next_wave")
    default_params_file = os.path.join(pkg_share, "config", "config.yaml")
    params_file = LaunchConfiguration("params_file")
    enable_plotter = LaunchConfiguration("enable_plotter")
    enable_sbg_bridge = LaunchConfiguration("enable_sbg_bridge")
    enable_sbg_tcp_replay = LaunchConfiguration("enable_sbg_tcp_replay")
    sbg_bridge_host = LaunchConfiguration("sbg_bridge_host")
    sbg_bridge_port_base = LaunchConfiguration("sbg_bridge_port_base")

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
            DeclareLaunchArgument(
                "enable_sbg_bridge",
                default_value="false",
                description="If true, enable the SBG TCP bridge server inside the predictor node (for ingesting raw SBG Ethernet-bridge streams).",
            ),
            DeclareLaunchArgument(
                "enable_sbg_tcp_replay",
                default_value="false",
                description="If true, run 4 SBG TCP replay nodes (SWIFT22-25) and enable the TCP SBG bridge server in the predictor node.",
            ),
            DeclareLaunchArgument(
                "sbg_bridge_host",
                default_value="127.0.0.1",
                description="Host the replay nodes connect to (TCP server lives inside the predictor node).",
            ),
            DeclareLaunchArgument(
                "sbg_bridge_port_base",
                default_value="3001",
                description="Base port for SWIFT22 TCP server; swiftN uses port_base+(N-22).",
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_sim_node",
                name="the_next_wave_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        # Replay requires the bridge server; allow bridge-only mode via enable_sbg_bridge.
                        "sbg_bridge_enable": ParameterValue(
                            PythonExpression([
                                "('",
                                enable_sbg_bridge,
                                "' == 'true') or ('",
                                enable_sbg_tcp_replay,
                                "' == 'true')",
                            ]),
                            value_type=bool,
                        ),
                        "sbg_bridge_bind": "0.0.0.0",
                        "sbg_bridge_port_base": ParameterValue(sbg_bridge_port_base, value_type=int),
                    },
                ],
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_plotter_node",
                name="the_next_wave_plotter_node",
                output="screen",
                parameters=[params_file],
                condition=IfCondition(enable_plotter),
            ),

            # SBG TCP replay nodes (one per buoy) for local testing of the bridge ingest.
            Node(
                package="the_next_wave",
                executable="the_next_wave_sbg_tcp_replay_node",
                name="sbg_tcp_replay_swift22",
                output="screen",
                parameters=[
                    {
                        "host": sbg_bridge_host,
                        "port": ParameterValue(sbg_bridge_port_base, value_type=int),
                        "swift_num": 22,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_sbg_tcp_replay_node",
                name="sbg_tcp_replay_swift23",
                output="screen",
                parameters=[
                    {
                        "host": sbg_bridge_host,
                        "port": ParameterValue(PythonExpression([sbg_bridge_port_base, " + 1"]), value_type=int),
                        "swift_num": 23,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_sbg_tcp_replay_node",
                name="sbg_tcp_replay_swift24",
                output="screen",
                parameters=[
                    {
                        "host": sbg_bridge_host,
                        "port": ParameterValue(PythonExpression([sbg_bridge_port_base, " + 2"]), value_type=int),
                        "swift_num": 24,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package="the_next_wave",
                executable="the_next_wave_sbg_tcp_replay_node",
                name="sbg_tcp_replay_swift25",
                output="screen",
                parameters=[
                    {
                        "host": sbg_bridge_host,
                        "port": ParameterValue(PythonExpression([sbg_bridge_port_base, " + 3"]), value_type=int),
                        "swift_num": 25,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
        ]
    )