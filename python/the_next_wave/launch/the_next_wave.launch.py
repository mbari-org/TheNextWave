#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    NotEqualsSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    enable_plotter = LaunchConfiguration('enable_plotter')
    enable_sbg_bridge = LaunchConfiguration('enable_sbg_bridge')
    enable_sbg_tcp_replay = LaunchConfiguration('enable_sbg_tcp_replay')
    params_file_override = LaunchConfiguration('params_file')
    selected_params_file = LaunchConfiguration('selected_params_file')
    selected_params_reason = LaunchConfiguration('selected_params_reason')
    effective_params_file = LaunchConfiguration('effective_params_file')
    sbg_bridge_enable_effective = LaunchConfiguration('sbg_bridge_enable_effective')

    pkg_share = FindPackageShare('the_next_wave')
    regular_params_file = PathJoinSubstitution([pkg_share, 'config', 'config.yaml'])
    deployment_params_file = PathJoinSubstitution(
        [pkg_share, 'config', 'config_deployment.yaml']
    )
    tcp_replay_params_file = PathJoinSubstitution(
        [pkg_share, 'config', 'config_sbg_tcp_replay.yaml']
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'enable_plotter',
                default_value='false',
                description='If true, also run the standalone wave prediction plotter node.',
            ),
            DeclareLaunchArgument(
                'enable_sbg_bridge',
                default_value='false',
                description=(
                    'If true, enable the SBG TCP bridge server inside the '
                    'predictor node (for ingesting raw SBG Ethernet-bridge streams).'
                ),
            ),
            DeclareLaunchArgument(
                'enable_sbg_tcp_replay',
                default_value='false',
                description=(
                    'If true, run 4 SBG TCP replay nodes (SWIFT22-25) and '
                    'enable the TCP SBG bridge server in the predictor node.'
                ),
            ),
            DeclareLaunchArgument(
                'params_file',
                default_value='',
                description=(
                    'Optional override path to YAML parameter file. When empty, '
                    'defaults automatically to '
                    'config_sbg_tcp_replay.yaml when enable_sbg_tcp_replay:=true, '
                    'config_deployment.yaml when enable_sbg_bridge:=true (and replay '
                    'is false), otherwise config.yaml.'
                ),
            ),

            # default config selection
            SetLaunchConfiguration(
                'selected_params_file',
                regular_params_file,
            ),
            SetLaunchConfiguration(
                'selected_params_reason',
                'default',
            ),
            # if bridge enabled, select deployment config
            # (will override below if replay also enabled)
            SetLaunchConfiguration(
                'selected_params_file',
                deployment_params_file,
                condition=IfCondition(enable_sbg_bridge),
            ),
            SetLaunchConfiguration(
                'selected_params_reason',
                'bridge',
                condition=IfCondition(enable_sbg_bridge),
            ),
            # if replay enabled, select replay config
            SetLaunchConfiguration(
                'selected_params_file',
                tcp_replay_params_file,
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            SetLaunchConfiguration(
                'selected_params_reason',
                'replay',
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            # if no params file override, use selected config
            SetLaunchConfiguration(
                'effective_params_file',
                selected_params_file,
                condition=IfCondition(
                    EqualsSubstitution(LaunchConfiguration('params_file'), '')
                ),
            ),
            # if params file was provided, override selected config
            SetLaunchConfiguration(
                'effective_params_file',
                params_file_override,
                condition=IfCondition(
                    NotEqualsSubstitution(LaunchConfiguration('params_file'), '')
                ),
            ),
            SetLaunchConfiguration(
                'selected_params_reason',
                'override',
                condition=IfCondition(
                    NotEqualsSubstitution(LaunchConfiguration('params_file'), '')
                ),
            ),
            # don't use bridge by default
            SetLaunchConfiguration(
                'sbg_bridge_enable_effective',
                'false',
            ),
            # override to enable bridge if either bridge or replay enabled
            # (replay requires bridge, so replay implies bridge)
            SetLaunchConfiguration(
                'sbg_bridge_enable_effective',
                'true',
                condition=IfCondition(enable_sbg_bridge),
            ),
            # override to enable bridge if either bridge or replay enabled
            # (replay requires bridge, so replay implies bridge)
            SetLaunchConfiguration(
                'sbg_bridge_enable_effective',
                'true',
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            LogInfo(
                msg=[
                    'the_next_wave: params_file=',
                    effective_params_file,
                    ' (source=',
                    selected_params_reason,
                    ')',
                ]
            ),

            # main predictor node
            Node(
                package='the_next_wave',
                executable='the_next_wave_node',
                name='the_next_wave_node',
                output='screen',
                parameters=[
                    effective_params_file,
                    {
                        # Replay requires the bridge server; allow bridge-only
                        # mode via enable_sbg_bridge.
                        'sbg_bridge_enable': ParameterValue(
                            sbg_bridge_enable_effective,
                            value_type=bool,
                        ),
                    },
                ],
            ),

            # optional plotter node
            Node(
                package='the_next_wave',
                executable='the_next_wave_plotter_node',
                name='the_next_wave_plotter_node',
                output='screen',
                parameters=[effective_params_file],
                condition=IfCondition(enable_plotter),
            ),

            # SBG TCP replay nodes (one per buoy) for local testing of the bridge ingest.
            Node(
                package='the_next_wave',
                executable='the_next_wave_sbg_tcp_replay_node',
                name='sbg_tcp_replay_swift22',
                output='screen',
                parameters=[
                    effective_params_file,
                    {
                        'swift_num': 22,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package='the_next_wave',
                executable='the_next_wave_sbg_tcp_replay_node',
                name='sbg_tcp_replay_swift23',
                output='screen',
                parameters=[
                    effective_params_file,
                    {
                        'swift_num': 23,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package='the_next_wave',
                executable='the_next_wave_sbg_tcp_replay_node',
                name='sbg_tcp_replay_swift24',
                output='screen',
                parameters=[
                    effective_params_file,
                    {
                        'swift_num': 24,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            Node(
                package='the_next_wave',
                executable='the_next_wave_sbg_tcp_replay_node',
                name='sbg_tcp_replay_swift25',
                output='screen',
                parameters=[
                    effective_params_file,
                    {
                        'swift_num': 25,
                    }
                ],
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
        ]
    )
