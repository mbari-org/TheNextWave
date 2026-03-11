#!/usr/bin/env python3

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.actions import LogInfo, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    NotEqualsSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_incwave_config() -> tuple[str, str, bool]:
    inc_wave_dir = 'None'
    inc_wave_height_points = 'None'
    has_overrides = False

    try:
        pkg_share = get_package_share_directory('the_next_wave')
        config_path = f'{pkg_share}/config/config.yaml'
        with open(config_path, 'r', encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream) or {}

        params = cfg.get('/the_next_wave_node', {}).get('ros__parameters', {})

        wave_dir = params.get('wave_dir', None)
        if wave_dir is not None:
            inc_wave_dir = str(float(wave_dir))

        swift_coords = params.get('swift_coords', {}) or {}
        x_vals = swift_coords.get('x', None)
        y_vals = swift_coords.get('y', None)
        if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) == len(y_vals):
            points = []
            for x_val, y_val in zip(x_vals, y_vals):
                points.append(f'{float(x_val)}:{float(y_val)}')
            if points:
                inc_wave_height_points = ';'.join(points)
        has_overrides = (inc_wave_dir != 'None') and (inc_wave_height_points != 'None')
    except Exception:
        # Keep launch robust: if YAML is missing/malformed, fall back to simulator defaults.
        pass

    return inc_wave_dir, inc_wave_height_points, has_overrides


def resolve_gz_extra_args(context):
    headless = str(LaunchConfiguration('gzsim_headless').perform(context)).lower() == 'true'
    verbose = str(LaunchConfiguration('gzsim_verbose').perform(context)).lower() == 'true'

    if headless and verbose:
        value = '-rsv4'
    elif headless:
        value = '-rs'
    elif verbose:
        value = '-rv4'
    else:
        value = '-r'

    return [SetLaunchConfiguration('gzsim_extra_gz_args', value)]


def generate_launch_description() -> LaunchDescription:
    enable_plotter = LaunchConfiguration('enable_plotter')
    enable_mbari_wec = LaunchConfiguration('enable_mbari_wec')
    enable_sbg_bridge = LaunchConfiguration('enable_sbg_bridge')
    enable_sbg_tcp_replay = LaunchConfiguration('enable_sbg_tcp_replay')
    params_file_override = LaunchConfiguration('params_file')
    selected_params_file = LaunchConfiguration('selected_params_file')
    selected_params_reason = LaunchConfiguration('selected_params_reason')
    effective_params_file = LaunchConfiguration('effective_params_file')
    sbg_bridge_enable_effective = LaunchConfiguration('sbg_bridge_enable_effective')
    mbari_wec_enable_effective = LaunchConfiguration('mbari_wec_enable_effective')
    gzsim_extra_gz_args = LaunchConfiguration('gzsim_extra_gz_args')

    regular_inc_wave_dir, regular_inc_wave_height_points, has_incwave_overrides = \
        load_incwave_config()

    pkg_share = FindPackageShare('the_next_wave')
    regular_params_file = PathJoinSubstitution([pkg_share, 'config', 'config.yaml'])
    deployment_params_file = PathJoinSubstitution(
        [pkg_share, 'config', 'config_deployment.yaml']
    )
    tcp_replay_params_file = PathJoinSubstitution(
        [pkg_share, 'config', 'config_sbg_tcp_replay.yaml']
    )

    buoy_gazebo_pkg_share = FindPackageShare('buoy_gazebo')
    mbari_wec_launch = PathJoinSubstitution(
        [buoy_gazebo_pkg_share, 'launch', 'mbari_wec.launch.py']
    )

    if has_incwave_overrides:
        mbari_wec_include = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mbari_wec_launch),
            launch_arguments={
                'extra_gz_args': gzsim_extra_gz_args,
                'inc_wave_dir': regular_inc_wave_dir,
                'inc_wave_height_points': regular_inc_wave_height_points,
            }.items(),
            condition=IfCondition(mbari_wec_enable_effective),
        )
    else:
        mbari_wec_include = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mbari_wec_launch),
            launch_arguments={
                'extra_gz_args': gzsim_extra_gz_args,
            }.items(),
            condition=IfCondition(mbari_wec_enable_effective),
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'enable_plotter',
                default_value='false',
                description='If true, also run the standalone wave prediction plotter node.',
            ),
            DeclareLaunchArgument(
                'enable_mbari_wec',
                default_value='true',
                description=(
                    'If true, include mbari_wec.launch.py. '
                    'Set false when running MBARI WEC simulator separately.'
                ),
            ),
            DeclareLaunchArgument(
                'gzsim_headless',
                default_value='false',
                description='Run Gazebo sim headless (no GUI).',
            ),
            DeclareLaunchArgument(
                'gzsim_verbose',
                default_value='true',
                description='Enable verbose Gazebo output (-v4).',
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
            SetLaunchConfiguration('gzsim_extra_gz_args', '-r'),
            OpaqueFunction(function=resolve_gz_extra_args),
            # MBARI WEC launch: enabled by default unless explicitly disabled,
            # but always disabled in bridge/replay modes.
            SetLaunchConfiguration('mbari_wec_enable_effective', 'false'),
            SetLaunchConfiguration(
                'mbari_wec_enable_effective',
                'true',
                condition=IfCondition(enable_mbari_wec),
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
            SetLaunchConfiguration(
                'mbari_wec_enable_effective',
                'false',
                condition=IfCondition(enable_sbg_bridge),
            ),
            SetLaunchConfiguration(
                'mbari_wec_enable_effective',
                'false',
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
            # Launch MBARI WEC simulator only when neither bridge nor TCP replay is enabled.
            mbari_wec_include,
        ]
    )
