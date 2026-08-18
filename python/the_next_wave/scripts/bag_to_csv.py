#!/usr/bin/env python3

"""
Convert a TheNextWave simulation rosbag into flat CSV files.

Produces two CSVs per bag:

* ``<prefix>_input.csv``  -- from ``/latent_data``: the simulator's incident-wave
  ground truth sampled at the target WEC and at each SWIFT location. This is
  exactly what the predictor node ingests (before its optional latent noise).
* ``<prefix>_output.csv`` -- from ``/wave_predictions``: one row per predicted
  future sample at the target, with the per-window bulk wavespec parameters and
  solver diagnostics repeated on each row.

Optional extra CSVs (``--dense``, ``--spectrum``) dump the dense in-window model
projection and the per-window 1D energy spectrum.

Usage::

    python3 scripts/bag_to_csv.py <bag_dir> --prefix out/nospread \
        --config config/config_sim_nospread.yaml
"""

import argparse
import csv
import os
import sys

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml


INPUT_TOPIC_DEFAULT = '/latent_data'
OUTPUT_TOPIC_DEFAULT = '/wave_predictions'


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def load_point_labels(config_path: str | None) -> dict[int, str]:
    """Map ``inc_wave_heights`` array index -> human label, from the node config."""
    # Index 0 is always the incident wave at the target WEC (never used as solver input).
    labels = {0: 'target'}
    if not config_path:
        return labels
    try:
        with open(config_path, 'r', encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream) or {}
        swifts = cfg.get('/the_next_wave_node', {}).get('ros__parameters', {}).get('swifts', {})
        for name, idx in (swifts or {}).items():
            labels[int(idx)] = str(name)
    except Exception as exc:  # noqa: B902 - labels are cosmetic; never fail the export
        print(f'warning: could not read swift labels from {config_path}: {exc}', file=sys.stderr)
    return labels


def open_bag(bag_path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=''),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        ),
    )
    type_by_topic = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_by_topic


def read_messages(bag_path: str, topics: list[str]):
    """
    Yield ``(topic, msg, bag_time_ns)`` for the requested topics.

    A topic that is absent from the bag is warned about and skipped rather than
    raising, so a partially-recorded run still exports the topics it does have
    (and still produces a header-only CSV for the ones it doesn't).
    """
    reader, type_by_topic = open_bag(bag_path)
    present = [t for t in topics if t in type_by_topic]
    for topic in topics:
        if topic not in type_by_topic:
            print(
                f'warning: bag {bag_path} has no topic {topic}; '
                f'available: {sorted(type_by_topic)}',
                file=sys.stderr,
            )
    if not present:
        return

    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    msg_class = {t: get_message(type_by_topic[t]) for t in present}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, deserialize_message(data, msg_class[topic]), t_ns


INPUT_COLUMNS = [
    'bag_time_s',
    'msg_stamp_s',
    'sample_time_s',
    'point_index',
    'label',
    'x_m',
    'y_m',
    'z_m',
    'u_east_mps',
    'v_north_mps',
    'etadot_mps',
    'gps_ref_lat_deg',
    'gps_ref_lon_deg',
]


def write_input_csv(bag_path: str, topic: str, out_path: str, labels: dict[int, str]) -> int:
    rows = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(INPUT_COLUMNS)
        for _topic, msg, t_ns in read_messages(bag_path, [topic]):
            msg_stamp = stamp_to_sec(msg.header.stamp)
            for idx, inc in enumerate(msg.inc_wave_heights):
                pose_stamp = stamp_to_sec(inc.pose.header.stamp)
                writer.writerow([
                    f'{t_ns * 1e-9:.9f}',
                    f'{msg_stamp:.9f}',
                    f'{pose_stamp + float(inc.relative_time):.9f}',
                    idx,
                    labels.get(idx, f'idx{idx}'),
                    f'{inc.pose.pose.position.x:.6f}',
                    f'{inc.pose.pose.position.y:.6f}',
                    f'{inc.pose.pose.position.z:.6f}',
                    f'{inc.velocities.x:.6f}',
                    f'{inc.velocities.y:.6f}',
                    f'{inc.velocities.z:.6f}',
                    f'{inc.gps_ref.latitude:.9f}',
                    f'{inc.gps_ref.longitude:.9f}',
                ])
                rows += 1
    return rows


# Per-window fields repeated on every prediction row so the CSV stands alone.
WINDOW_COLUMNS = [
    'bag_time_s',
    'msg_stamp_s',
    'window_start_time',
    'window_end_time',
    'n_measurements',
    'x_target_m',
    'y_target_m',
    'solve_time_s',
    'num_wavelengths',
    'centroid_period_s',
    'has_wavespec_bulk',
    'wavespec_hs_m',
    'wavespec_tp_s',
    'wavespec_tm01_s',
    'wavespec_tm02_s',
    'wavespec_dp_deg_from',
    'wavespec_dm_deg_from',
    'wavespec_spreadp_deg',
]

OUTPUT_COLUMNS = WINDOW_COLUMNS + [
    'pred_time_s',
    'pred_x_m',
    'pred_y_m',
    'pred_elevation_m',
    'pred_vel_east_mps',
    'pred_vel_north_mps',
]


def window_row(msg, t_ns) -> list:
    return [
        f'{t_ns * 1e-9:.9f}',
        f'{stamp_to_sec(msg.header.stamp):.9f}',
        f'{msg.window_start_time:.9f}',
        f'{msg.window_end_time:.9f}',
        int(msg.n_measurements),
        f'{msg.x_target:.6f}',
        f'{msg.y_target:.6f}',
        f'{msg.solve_time:.6f}',
        int(msg.num_wavelengths),
        f'{msg.centroid_period:.6f}',
        int(bool(msg.has_wavespec_bulk)),
        f'{msg.wavespec_hs:.6f}',
        f'{msg.wavespec_tp:.6f}',
        f'{msg.wavespec_tm01:.6f}',
        f'{msg.wavespec_tm02:.6f}',
        f'{msg.wavespec_dp:.6f}',
        f'{msg.wavespec_dm:.6f}',
        f'{msg.wavespec_spreadp:.6f}',
    ]


def write_output_csvs(
    bag_path: str,
    topic: str,
    out_path: str,
    dense_path: str | None,
    spectrum_path: str | None,
) -> tuple[int, int, int, int]:
    n_msgs = n_pred = n_dense = n_spec = 0

    dense_fh = open(dense_path, 'w', newline='', encoding='utf-8') if dense_path else None
    spec_fh = open(spectrum_path, 'w', newline='', encoding='utf-8') if spectrum_path else None
    try:
        dense_writer = csv.writer(dense_fh) if dense_fh else None
        spec_writer = csv.writer(spec_fh) if spec_fh else None
        if dense_writer:
            dense_writer.writerow(
                WINDOW_COLUMNS
                + ['dense_time_s', 'dense_z_m', 'dense_u_east_mps', 'dense_v_north_mps']
            )
        if spec_writer:
            spec_writer.writerow(
                WINDOW_COLUMNS + ['frequency_hz', 'energy_m2_per_hz']
            )

        with open(out_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(OUTPUT_COLUMNS)
            for _topic, msg, t_ns in read_messages(bag_path, [topic]):
                n_msgs += 1
                base = window_row(msg, t_ns)
                for point in msg.predictions:
                    writer.writerow(base + [
                        f'{point.time:.9f}',
                        f'{point.x:.6f}',
                        f'{point.y:.6f}',
                        f'{point.elevation:.6f}',
                        f'{point.vel_east:.6f}',
                        f'{point.vel_north:.6f}',
                    ])
                    n_pred += 1

                if dense_writer and msg.has_dense_predictions:
                    for t_s, z, u, v in zip(
                        msg.dense_predictions_time,
                        msg.dense_predictions_z,
                        msg.dense_predictions_u,
                        msg.dense_predictions_v,
                    ):
                        dense_writer.writerow(
                            base + [f'{t_s:.9f}', f'{z:.6f}', f'{u:.6f}', f'{v:.6f}']
                        )
                        n_dense += 1

                if spec_writer:
                    for freq, energy in zip(msg.frequencies, msg.energy_by_freq):
                        spec_writer.writerow(base + [f'{freq:.9f}', f'{energy:.9f}'])
                        n_spec += 1
    finally:
        if dense_fh:
            dense_fh.close()
        if spec_fh:
            spec_fh.close()

    return n_msgs, n_pred, n_dense, n_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='rosbag2 directory (or .mcap/.db3 file)')
    parser.add_argument('--prefix', required=True,
                        help='output path prefix; writes <prefix>_input.csv / <prefix>_output.csv')
    parser.add_argument('--config', default=None,
                        help='node YAML config used to label SWIFT indices')
    parser.add_argument('--input-topic', default=INPUT_TOPIC_DEFAULT)
    parser.add_argument('--output-topic', default=OUTPUT_TOPIC_DEFAULT)
    parser.add_argument('--dense', action='store_true',
                        help='also write <prefix>_dense.csv (in-window model projection)')
    parser.add_argument('--spectrum', action='store_true',
                        help='also write <prefix>_spectrum.csv (1D energy spectrum per window)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_csv = f'{args.prefix}_input.csv'
    output_csv = f'{args.prefix}_output.csv'
    dense_csv = f'{args.prefix}_dense.csv' if args.dense else None
    spectrum_csv = f'{args.prefix}_spectrum.csv' if args.spectrum else None

    labels = load_point_labels(args.config)
    n_in = write_input_csv(args.bag, args.input_topic, input_csv, labels)
    print(f'{input_csv}: {n_in} rows from {args.input_topic}')

    n_msgs, n_pred, n_dense, n_spec = write_output_csvs(
        args.bag, args.output_topic, output_csv, dense_csv, spectrum_csv
    )
    print(f'{output_csv}: {n_pred} rows from {n_msgs} {args.output_topic} messages')
    if dense_csv:
        print(f'{dense_csv}: {n_dense} rows')
    if spectrum_csv:
        print(f'{spectrum_csv}: {n_spec} rows')

    if n_in == 0 or n_msgs == 0:
        print('warning: one or both topics were empty in this bag', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
