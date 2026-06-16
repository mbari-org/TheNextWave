#!/usr/bin/env python3

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.actions import LogInfo, SetLaunchConfiguration
from launch.actions import SetEnvironmentVariable
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


def load_incwave_config() -> tuple[str, str, str, bool]:
    def stringify_number(value) -> str:
        return str(float(value))

    def encode_scalar_or_list(name: str, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return f"{name}:" + ':'.join(stringify_number(v) for v in value)
        return f'{name}:{value}'

    def encode_points_from_xy(x_vals, y_vals) -> str:
        points = []
        for x_val, y_val in zip(x_vals, y_vals):
            points.append(f'{float(x_val)}:{float(y_val)}')
        return ';'.join(points)

    def parse_inc_wave_height_points(params: dict) -> str:
        swift_coords = params.get('swift_coords', {}) or {}
        x_vals = swift_coords.get('x', None)
        y_vals = swift_coords.get('y', None)
        if (
            isinstance(x_vals, list)
            and isinstance(y_vals, list)
            and len(x_vals) == len(y_vals)
            and x_vals
        ):
            return encode_points_from_xy(x_vals, y_vals)

        return 'None'

    def parse_inc_wave_spectrum(params: dict) -> str:
        spectrum_cfg = params.get('bretschneider', None)
        if spectrum_cfg is None:
            spectrum_cfg = params.get('Bretschneider', None)

        if not isinstance(spectrum_cfg, dict) or not spectrum_cfg:
            return 'None'

        parts = ['inc_wave_spectrum_type:Bretschneider']
        for name, value in spectrum_cfg.items():
            encoded = encode_scalar_or_list(name, value)
            if encoded is not None:
                parts.append(encoded)

        return ';'.join(parts)

    inc_wave_dir = 'None'
    inc_wave_height_points = 'None'
    inc_wave_spectrum = 'None'
    has_overrides = False

    try:
        pkg_share = get_package_share_directory('the_next_wave')
        config_path = f'{pkg_share}/config/config.yaml'
        with open(config_path, 'r', encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream) or {}

        params = cfg.get('/the_next_wave_node', {}).get('ros__parameters', {})

        wave_dir = params.get('wave_dir', None)
        if wave_dir is not None:
            inc_wave_dir = stringify_number(wave_dir)

        inc_wave_height_points = parse_inc_wave_height_points(params)
        inc_wave_spectrum = parse_inc_wave_spectrum(params)
        has_overrides = any(
            value != 'None'
            for value in (inc_wave_dir, inc_wave_height_points, inc_wave_spectrum)
        )
    except Exception:
        # Keep launch robust: if YAML is missing/malformed, fall back to simulator defaults.
        pass

    return inc_wave_dir, inc_wave_height_points, inc_wave_spectrum, has_overrides


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


def resolve_run_mode(context):
    def to_bool(value: str) -> bool:
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

    replay_enabled = to_bool(LaunchConfiguration('enable_sbg_tcp_replay').perform(context))
    bridge_enabled = to_bool(LaunchConfiguration('enable_sbg_bridge').perform(context))
    nextwave_enabled = to_bool(LaunchConfiguration('enable_nextwave_node').perform(context))

    if replay_enabled and not nextwave_enabled:
        mode = 'replay_only'
    elif bridge_enabled and (not replay_enabled) and nextwave_enabled:
        mode = 'bridge_only'
    elif (not replay_enabled) and (not bridge_enabled) and nextwave_enabled:
        mode = 'full'
    else:
        mode = 'custom'

    return [SetLaunchConfiguration('run_mode', mode)]


def generate_launch_description() -> LaunchDescription:
    enable_plotter = LaunchConfiguration('enable_plotter')
    enable_nextwave_node = LaunchConfiguration('enable_nextwave_node')
    enable_mbari_wec = LaunchConfiguration('enable_mbari_wec')
    enable_sbg_bridge = LaunchConfiguration('enable_sbg_bridge')
    enable_sbg_tcp_replay = LaunchConfiguration('enable_sbg_tcp_replay')
    params_file_override = LaunchConfiguration('params_file')
    selected_params_file = LaunchConfiguration('selected_params_file')
    selected_params_reason = LaunchConfiguration('selected_params_reason')
    run_mode = LaunchConfiguration('run_mode')
    effective_params_file = LaunchConfiguration('effective_params_file')
    sbg_bridge_enable_effective = LaunchConfiguration('sbg_bridge_enable_effective')
    mbari_wec_enable_effective = LaunchConfiguration('mbari_wec_enable_effective')
    gzsim_extra_gz_args = LaunchConfiguration('gzsim_extra_gz_args')

    regular_inc_wave_dir, regular_inc_wave_height_points, regular_inc_wave_spectrum, \
        has_incwave_overrides = \
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
        mbari_wec_launch_arguments = {
            'extra_gz_args': gzsim_extra_gz_args,
        }
        if regular_inc_wave_dir != 'None':
            mbari_wec_launch_arguments['inc_wave_dir'] = regular_inc_wave_dir
        if regular_inc_wave_height_points != 'None':
            mbari_wec_launch_arguments['inc_wave_height_points'] = regular_inc_wave_height_points
        if regular_inc_wave_spectrum != 'None':
            mbari_wec_launch_arguments['inc_wave_spectrum'] = regular_inc_wave_spectrum

        mbari_wec_include = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mbari_wec_launch),
            launch_arguments=mbari_wec_launch_arguments.items(),
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
                'enable_nextwave_node',
                default_value='true',
                description=(
                    'If true, run the main the_next_wave_node. Set false when this machine '
                    'should run only replay nodes.'
                ),
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
            # Networked bridge/replay modes need inter-host DDS visibility.
            SetEnvironmentVariable(
                name='RMW_IMPLEMENTATION',
                value='rmw_fastrtps_cpp',
                condition=IfCondition(enable_sbg_bridge),
            ),
            SetEnvironmentVariable(
                name='RMW_IMPLEMENTATION',
                value='rmw_fastrtps_cpp',
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            SetEnvironmentVariable(
                name='ROS_LOCALHOST_ONLY',
                value='0',
                condition=IfCondition(enable_sbg_bridge),
            ),
            SetEnvironmentVariable(
                name='ROS_LOCALHOST_ONLY',
                value='0',
                condition=IfCondition(enable_sbg_tcp_replay),
            ),
            # Derived launch role for operator visibility.
            SetLaunchConfiguration('run_mode', 'custom'),
            OpaqueFunction(function=resolve_run_mode),
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
            LogInfo(
                msg=[
                    'the_next_wave: run_mode=',
                    run_mode,
                    ' (enable_nextwave_node=',
                    enable_nextwave_node,
                    ', sbg_bridge_enable_effective=',
                    sbg_bridge_enable_effective,
                    ', enable_sbg_tcp_replay=',
                    enable_sbg_tcp_replay,
                    ')',
                ]
            ),
            LogInfo(
                msg='the_next_wave: forcing DDS env for network mode: '
                    'RMW_IMPLEMENTATION=rmw_fastrtps_cpp, ROS_LOCALHOST_ONLY=0',
                condition=IfCondition(enable_sbg_bridge),
            ),
            LogInfo(
                msg='the_next_wave: forcing DDS env for network mode: '
                    'RMW_IMPLEMENTATION=rmw_fastrtps_cpp, ROS_LOCALHOST_ONLY=0',
                condition=IfCondition(enable_sbg_tcp_replay),
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
                condition=IfCondition(enable_nextwave_node),
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
