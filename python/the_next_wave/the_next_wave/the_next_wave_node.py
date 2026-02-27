#!/usr/bin/env python3

"""
ROS 2 node wrapper for TheNextWave near-realtime predictions.

All ROS 2 concerns live here (subscriptions, publishers, message packaging).
The core algorithmic pipeline is implemented in `the_next_wave.the_next_wave.TheNextWave`.
"""

from collections import deque, OrderedDict
from dataclasses import dataclass, field
import threading
import time
import traceback

from builtin_interfaces.msg import Time as TimeMsg
from buoy_api.interface import Interface
from buoy_interfaces.msg import WavePredictionOutput, WavePredictionPoint
from buoy_interfaces.msg import XBRecord
import numpy as np
import rclpy
from std_msgs.msg import Header
from the_next_wave import TheNextWave, TheNextWaveConfig
from the_next_wave.sbg_bridge_service import SbgBridgeService
from the_next_wave.swift import SBGData, SWIFTArray
from the_next_wave.utilities import (
    bulk_wave_params_from_wavespec,
    generic_coordinate_transform_inverse,
)


@dataclass
class TheNextWaveNodeParams:
    window_duration_sec: float = 256.0
    processing_interval_sec: float = 2.0
    expected_fs: float = 5.0
    # Rotation applied in the lat/lon <-> local x/y projection (clockwise-positive).
    # For sim / incident-wave latent inputs, x/y and u/v are already world ENU
    # (x=East, y=North; u=East, v=North), so `rotation_deg` should typically be 0.
    # If you set `rotation_deg != 0`, you must rotate u/v consistently (the current
    # pipeline does not do that automatically).
    rotation_deg: float = 0.0
    flip_z_sign: bool = True
    n_te: float = 10.0

    downsample_to_hz: float = 0.0
    wavespec_update_period_sec: float = 0.0
    wec_actual_history_sec: float = 256.0
    dense_prediction_window: float = 0.0
    enable_dense_history_projection: bool = False
    sbg_bridge_enable: bool = False
    sbg_bridge_bind: str = '0.0.0.0'
    sbg_bridge_socket_timeout_sec: float = 1.0
    sbg_use_example_frame: bool = False
    example_latorigin: float = 41.6878
    example_lonorigin: float = -9.0545
    example_rotation_deg: float = 0.0
    example_xtarget: float = 200.0
    example_ytarget: float = 200.0
    swift_idx: OrderedDict[str, int] = field(default_factory=OrderedDict)
    sbg_bridge_port_by_swift: dict[int, int] = field(default_factory=dict)

    # Latent (Gazebo) ingest noise floor.
    # The MATLAB MEM estimator can become ill-conditioned for perfectly coherent
    # plane waves (e.g., |c2|==1). Real sensors have noise, so add a tiny,
    # configurable noise floor to sim latent measurements to avoid singular cases.
    latent_noise_std_z_m: float = 1e-4
    latent_noise_std_uv_mps: float = 1e-4
    latent_noise_seed: int = 0

    # Least-squares solver controls
    lsq_ridge: float = 1e-6
    lsq_max_iter: int = 60
    lsq_use_spectrum_weighted_ridge: bool = True
    lsq_spectrum_ridge_floor: float = 1e-6
    lsq_diagnostics_enable: bool = False
    lsq_near_bound_ratio: float = 0.95

    # Optional stabilization for MEM directional estimation (disabled by default).
    mem_moment_cap_enable: bool = False
    mem_moment_cap: float = 0.999

    def __str__(self) -> str:
        swift_idx_str = dict(self.swift_idx)
        port_map_str = dict(sorted(self.sbg_bridge_port_by_swift.items()))
        return '\n'.join(
            [
                'TheNextWaveNodeParams(',
                f'  window_duration_sec={self.window_duration_sec},',
                f'  processing_interval_sec={self.processing_interval_sec},',
                f'  expected_fs={self.expected_fs},',
                f'  rotation_deg={self.rotation_deg},',
                f'  n_te={self.n_te},',
                f'  downsample_to_hz={self.downsample_to_hz},',
                f'  wavespec_update_period_sec={self.wavespec_update_period_sec},',
                f'  wec_actual_history_sec={self.wec_actual_history_sec},',
                f'  dense_prediction_window={self.dense_prediction_window},',
                f'  enable_dense_history_projection={self.enable_dense_history_projection},',
                f'  sbg_bridge_enable={self.sbg_bridge_enable},',
                f"  sbg_bridge_bind='{self.sbg_bridge_bind}',",
                f'  sbg_bridge_socket_timeout_sec={self.sbg_bridge_socket_timeout_sec},',
                f'  latent_noise_std_z_m={self.latent_noise_std_z_m},',
                f'  latent_noise_std_uv_mps={self.latent_noise_std_uv_mps},',
                f'  latent_noise_seed={self.latent_noise_seed},',
                f'  sbg_use_example_frame={self.sbg_use_example_frame},',
                f'  example_latorigin={self.example_latorigin},',
                f'  example_lonorigin={self.example_lonorigin},',
                f'  example_rotation_deg={self.example_rotation_deg},',
                f'  example_xtarget={self.example_xtarget},',
                f'  example_ytarget={self.example_ytarget},',
                f'  swift_idx={swift_idx_str},',
                f'  sbg_bridge_port_by_swift={port_map_str}',
                f'  lsq_ridge={self.lsq_ridge},',
                f'  lsq_max_iter={self.lsq_max_iter},',
                f'  lsq_use_spectrum_weighted_ridge={self.lsq_use_spectrum_weighted_ridge},',
                f'  lsq_spectrum_ridge_floor={self.lsq_spectrum_ridge_floor},',
                f'  lsq_diagnostics_enable={self.lsq_diagnostics_enable},',
                f'  lsq_near_bound_ratio={self.lsq_near_bound_ratio},',
                f'  mem_moment_cap_enable={self.mem_moment_cap_enable},'
                f'  mem_moment_cap={self.mem_moment_cap},'
                ')',
            ]
        )


class TheNextWaveNode(Interface):
    """
    ROS 2 orchestration layer for TheNextWave prediction processing.

    Sequential runtime outline:
    1) Initialization
        - Build `self.params` from ROS parameters via `set_params()`.
        - Initialize rolling SBG windows (`self.swifts`) and readiness state.
        - Construct `TheNextWave` predictor and start periodic processing timer.
    2) Data ingress path A (Gazebo / latent topics)
        - `latent_callback()` receives `inc_wave_heights` samples.
        - For each configured SWIFT index, convert local x/y to lat/lon.
        - Forward standardized sample fields to `ingest_swift_sample_locked()`.
    3) Data ingress path B (SBG TCP bridge)
        - `SbgBridgeService` accepts TCP connections and parses SBG messages.
        - Partial ShipMotion/GpsVel/GpsPos pieces are assembled per timestamp.
        - Completed records are forwarded to `ingest_swift_sample_locked()`.
    4) Shared preprocessing (both paths converge here)
        - `ingest_swift_sample_locked()` appends sample data to SWIFT deques.
        - Applies downsampling and time-jump reset protections.
        - Calls `maintain_sbg_window()` to enforce rolling window and readiness.
    5) Shared prediction pipeline
        - `process_timer_callback()` starts background processing when ready.
        - `run_wave_processing()` snapshots SWIFT buffers and resolves target lat/lon.
        - `self.predictor.process(...)` computes wave predictions.
        - `publish_prediction()` emits `WavePredictionOutput`.
    """

    def __init__(self, node_name='the_next_wave_node', **kwargs):
        # TCP replay / lightweight runs may not have the buoy controller service stack.
        # We don't need any services in this node anyways, so disable Interface's
        # service availability checks here.
        super().__init__(
            node_name=node_name,
            check_for_services=False,
            wait_for_services=False,
            **kwargs,
        )
        self.params = TheNextWaveNodeParams()
        self.swifts = SWIFTArray()
        # Initialize SBG objects with empty lists for 256s windowing.
        self.init_sbg_windows()

        self.last_process_time_us = None
        self.window_ready = False
        self.window_ready_by_swift = {}
        self.last_accept_t_us_by_swift: dict[int, float] = {}
        self.last_seen_t_us_by_swift: dict[int, float] = {}

        self.set_params()

        # RNG for sim latent noise floor (used only in latent_callback).
        try:
            self._latent_rng = np.random.default_rng(int(self.params.latent_noise_seed))
        except Exception:
            self._latent_rng = np.random.default_rng(0)

        # WEC "actual" sample (packaged into WavePredictionOutput for external tools)
        self.wec_actual_latest = None  # dict with keys: t_s, z, u, v

        # Higher-rate history of WEC actual samples for frequency/phase comparisons.
        self.wec_actual_hist = deque()  # (t_s, z, u, v)

        self.predictor = TheNextWave(
            config=TheNextWaveConfig(
                expected_fs=self.params.expected_fs,
                rotation_deg=self.params.rotation_deg,
                flip_z_sign=self.params.flip_z_sign,
                n_te=self.params.n_te,
                wavespec_update_period_sec=self.params.wavespec_update_period_sec,
                dense_prediction_window=self.params.dense_prediction_window,
                enable_dense_history_projection=self.params.enable_dense_history_projection,
                lsq_ridge=self.params.lsq_ridge,
                lsq_max_iter=self.params.lsq_max_iter,
                lsq_use_spectrum_weighted_ridge=self.params.lsq_use_spectrum_weighted_ridge,
                lsq_spectrum_ridge_floor=self.params.lsq_spectrum_ridge_floor,
                lsq_diagnostics_enable=self.params.lsq_diagnostics_enable,
                lsq_near_bound_ratio=self.params.lsq_near_bound_ratio,
                mem_moment_cap_enable=self.params.mem_moment_cap_enable,
                mem_moment_cap=self.params.mem_moment_cap,
            ),
            logger=self.get_logger(),
        )

        self.data_lock = threading.Lock()
        self.processing = False
        self.last_wavespec_warn_walltime = 0.0
        self.sbg_bridge_service: SbgBridgeService | None = None

        # WEC buoy pose for prediction target (set by ahrs_callback)
        self.wec_lat = None
        self.wec_lon = None

        if not self.params.sbg_bridge_enable:
            self.use_sim_time()

        if self.params.sbg_bridge_enable and self.params.sbg_use_example_frame:
            self.predictor.lat_origin = float(self.params.example_latorigin)
            self.predictor.lon_origin = float(self.params.example_lonorigin)
            self.predictor.config.rotation_deg = float(self.params.example_rotation_deg)
            self.get_logger().info(
                'SBG bridge example-frame override enabled: '
                f'origin=({self.params.example_latorigin:.6f},'
                f'{self.params.example_lonorigin:.6f}) '
                f'rotation_deg={self.params.example_rotation_deg:.3f} '
                f'target_xy=({self.params.example_xtarget:.1f},{self.params.example_ytarget:.1f})'
            )

        # Track window readiness only for configured buoys
        configured_swift_nums = [int(name[-2:]) for name in self.params.swift_idx.keys()]
        if configured_swift_nums:
            self.window_ready_by_swift = {sid: False for sid in configured_swift_nums}
        else:
            # Safe default if no params set (won't trigger unless data arrives)
            self.window_ready_by_swift = {sid: False for sid in range(22, 26)}

        self.pred_publisher = self.create_publisher(WavePredictionOutput, 'wave_predictions', 10)

        if self.params.sbg_bridge_enable:
            swift_nums = [int(name[-2:]) for name in self.params.swift_idx.keys()]
            if not swift_nums:
                swift_nums = list(range(22, 26))
            self.sbg_bridge_service = SbgBridgeService(
                bind=self.params.sbg_bridge_bind,
                socket_timeout_sec=self.params.sbg_bridge_socket_timeout_sec,
                port_by_swift=self.params.sbg_bridge_port_by_swift,
                logger=self.get_logger(),
                data_lock=self.data_lock,
                ingest_swift_sample_locked=self.ingest_swift_sample_locked,
            )
            self.sbg_bridge_service.start(swift_nums)

        self.process_timer = self.create_timer(
            self.params.processing_interval_sec,
            self.process_timer_callback,
        )

    def init_sbg_windows(self):
        for sid in range(22, 26):
            sbg = SBGData()
            sbg.ShipMotion.time_stamp = deque()
            sbg.ShipMotion.heave = deque()
            sbg.GpsVel.time_stamp = deque()
            sbg.GpsVel.vel_e = deque()
            sbg.GpsVel.vel_n = deque()
            sbg.GpsPos.time_stamp = deque()
            sbg.GpsPos.lat = deque()
            sbg.GpsPos.long = deque()
            setattr(self.swifts, f'sbg{sid}', sbg)

    def process_timer_callback(self):
        t_last = (
            self.swifts.sbg22.ShipMotion.time_stamp[-1]
            if self.swifts.sbg22.ShipMotion.time_stamp
            else float('nan')
        )
        t_first = (
            self.swifts.sbg22.ShipMotion.time_stamp[0]
            if self.swifts.sbg22.ShipMotion.time_stamp
            else float('nan')
        )
        time_span_s = (t_last / 1e6) - (t_first / 1e6)
        self.get_logger().info(
            f'window ready? {self.window_ready}'
            f' :: n={len(self.swifts.sbg22.ShipMotion.time_stamp)} samples'
            f' :: t={time_span_s:.3f} s'
        )
        with self.data_lock:
            if self.processing or not self.window_ready:
                return
            # Set this here (before starting the thread) to avoid a race where
            # multiple timer callbacks can launch multiple processing threads.
            self.processing = True
        self.get_logger().info('Triggering wave processing...')

        thread = threading.Thread(target=self.run_wave_processing, daemon=True)
        thread.start()

    def run_wave_processing(self):
        try:
            with self.data_lock:
                swifts_snapshot = SWIFTArray()
                for sid in range(22, 26):
                    sbg_src = getattr(self.swifts, f'sbg{sid}')
                    sbg_dst = SBGData()
                    sbg_dst.ShipMotion.time_stamp = np.array(
                        sbg_src.ShipMotion.time_stamp,
                        dtype=np.float64,
                    )
                    sbg_dst.ShipMotion.heave = np.array(sbg_src.ShipMotion.heave, dtype=np.float64)
                    sbg_dst.GpsVel.time_stamp = np.array(
                        sbg_src.GpsVel.time_stamp,
                        dtype=np.float64,
                    )
                    sbg_dst.GpsVel.vel_e = np.array(sbg_src.GpsVel.vel_e, dtype=np.float64)
                    sbg_dst.GpsVel.vel_n = np.array(sbg_src.GpsVel.vel_n, dtype=np.float64)
                    sbg_dst.GpsPos.time_stamp = np.array(
                        sbg_src.GpsPos.time_stamp,
                        dtype=np.float64,
                    )
                    sbg_dst.GpsPos.lat = np.array(sbg_src.GpsPos.lat, dtype=np.float64)
                    sbg_dst.GpsPos.long = np.array(sbg_src.GpsPos.long, dtype=np.float64)
                    setattr(swifts_snapshot, f'sbg{sid}', sbg_dst)

                wec_lat = self.wec_lat
                wec_lon = self.wec_lon
                wec_actual_latest = (
                    dict(self.wec_actual_latest) if self.wec_actual_latest is not None else None
                )
                wec_actual_hist = list(self.wec_actual_hist) if self.wec_actual_hist else []

            if (
                (wec_lat is None or wec_lon is None)
                and self.params.sbg_bridge_enable
                and self.params.sbg_use_example_frame
            ):
                try:
                    lat, lon = generic_coordinate_transform_inverse(
                        self.params.example_xtarget,
                        self.params.example_ytarget,
                        self.params.example_latorigin,
                        self.params.example_lonorigin,
                        self.params.example_rotation_deg,
                    )
                    wec_lat = float(np.asarray(lat).reshape((-1,))[0])
                    wec_lon = float(np.asarray(lon).reshape((-1,))[0])
                except Exception:
                    # Best-effort; predictor will fall back to (0,0) target.
                    wec_lat = None
                    wec_lon = None

            dense_eval_time_s = None
            if self.params.enable_dense_history_projection and wec_actual_hist:
                dense_eval_time_s = np.asarray([float(p[0]) for p in wec_actual_hist], dtype=float)

            results = self.predictor.process(
                swifts_snapshot,
                wec_lat=wec_lat,
                wec_lon=wec_lon,
                dense_eval_time_s=dense_eval_time_s,
            )

            window_end_time = float(results.get('window_end_time', float('nan')))
            wec_hist_filtered = []
            if wec_actual_hist:
                for p in wec_actual_hist:
                    t_s = float(p[0])
                    if np.isfinite(t_s) and (not np.isfinite(window_end_time) or t_s <= window_end_time):
                        wec_hist_filtered.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))

            wec_at_window_end = None
            if wec_hist_filtered:
                t_end_local = window_end_time
                if np.isfinite(t_end_local):
                    wec_at_window_end = min(
                        wec_hist_filtered,
                        key=lambda p: abs(float(p[0]) - t_end_local),
                    )
                else:
                    wec_at_window_end = wec_hist_filtered[-1]
            elif wec_actual_latest is not None:
                wec_at_window_end = (
                    float(wec_actual_latest.get('t_s', 0.0)),
                    float(wec_actual_latest.get('z', 0.0)),
                    float(wec_actual_latest.get('u', 0.0)),
                    float(wec_actual_latest.get('v', 0.0)),
                )

            results['wec_actual'] = wec_at_window_end
            results['wec_actual_series'] = wec_hist_filtered
            self.publish_prediction(results)

        except ValueError as e:
            # Warmup/transient condition: wave spectra may not be usable yet.
            msg = str(e)
            if 'No usable wavespec available yet' in msg:
                now = time.monotonic()
                if now - self.last_wavespec_warn_walltime > 5.0:
                    self.last_wavespec_warn_walltime = now
                    self.get_logger().warn(msg)
            else:
                self.get_logger().error(f'Background wave processing failed: {e}')
                self.get_logger().error(traceback.format_exc())
        except Exception as e:
            self.get_logger().error(f'Background wave processing failed: {e}')
            self.get_logger().error(traceback.format_exc())
        finally:
            self.processing = False

    def publish_prediction(self, results: dict) -> None:
        wavespec = results.get('wavespec')
        params = results.get('params')

        msg = WavePredictionOutput()
        # PlotJuggler (and most ROS tooling) aligns streams by `header.stamp`.
        # Use the *measurement* time base (end of window) rather than publish time,
        # otherwise predictions appear delayed/misaligned relative to incoming data.
        stamp = self.get_clock().now().to_msg()
        t_end_s = results.get('window_end_time')
        try:
            t_end_s = float(t_end_s)
        except (TypeError, ValueError):
            t_end_s = float('nan')

        if np.isfinite(t_end_s) and t_end_s >= 0.0:
            sec = int(np.floor(t_end_s))
            nsec = int(np.round((t_end_s - sec) * 1e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            if nsec < 0:
                nsec = 0
            stamp = TimeMsg(sec=sec, nanosec=nsec)

        msg.header = Header(frame_id='wec_buoy', stamp=stamp)

        msg.window_start_time = float(results['window_start_time'])
        msg.window_end_time = float(results['window_end_time'])
        msg.n_measurements = int(results['n_samples'])

        # If the message definition supports bulk wavespec fields, initialize them.
        if hasattr(msg, 'has_wavespec_bulk'):
            msg.has_wavespec_bulk = False
            msg.wavespec_hs = float('nan')
            msg.wavespec_tp = float('nan')
            msg.wavespec_tm01 = float('nan')
            msg.wavespec_tm02 = float('nan')
            msg.wavespec_dp = float('nan')
            msg.wavespec_dm = float('nan')
            msg.wavespec_spreadp = float('nan')

        if wavespec is not None:
            msg.frequencies = np.asarray(wavespec.f, dtype=float).flatten().tolist()
            msg.directions = np.asarray(wavespec.theta, dtype=float).flatten().tolist()

            Etheta = np.asarray(wavespec.Etheta, dtype=float)
            f = np.asarray(wavespec.f, dtype=float).flatten()
            theta = np.asarray(wavespec.theta, dtype=float).flatten()
            dtheta = 1.0
            if theta.size > 1:
                dtheta_deg = float(np.nanmedian(np.diff(np.sort(theta))))
                dtheta = float(dtheta_deg * (np.pi / 180.0))
            if Etheta.ndim == 2:
                if Etheta.shape == (theta.size, f.size):
                    energy_by_freq = np.sum(Etheta * dtheta, axis=0)
                elif Etheta.shape == (f.size, theta.size):
                    energy_by_freq = np.sum(Etheta * dtheta, axis=1)
                else:
                    energy_by_freq = np.sum(Etheta * dtheta, axis=0)
            else:
                energy_by_freq = np.array([])

            msg.energy_by_freq = np.asarray(energy_by_freq, dtype=float).flatten().tolist()

            # Optional bulk parameters from the directional wavespec used for prediction.
            # These fields may not exist on older message definitions; guard with hasattr.
            if hasattr(msg, 'has_wavespec_bulk'):
                bulk = bulk_wave_params_from_wavespec(wavespec)
                msg.has_wavespec_bulk = True
                msg.wavespec_hs = float(bulk.get('Hs_m', float('nan')))
                msg.wavespec_tp = float(bulk.get('Tp_s', float('nan')))
                msg.wavespec_tm01 = float(bulk.get('Tm01_s', float('nan')))
                msg.wavespec_tm02 = float(bulk.get('Tm02_s', float('nan')))
                msg.wavespec_dp = float(bulk.get('Dp_deg', float('nan')))
                msg.wavespec_dm = float(bulk.get('Dm_deg', float('nan')))
                msg.wavespec_spreadp = float(bulk.get('spreadp_deg', float('nan')))

        msg.centroid_period = float(results.get('Te', 0.0))
        msg.solve_time = float(results.get('solve_time', 0.0))
        msg.num_wavelengths = (
            int(getattr(params, 'kx', np.array([])).size) if params is not None else 0
        )

        x_target = float(results.get('x_target', 0.0))
        y_target = float(results.get('y_target', 0.0))
        msg.x_target = x_target
        msg.y_target = y_target

        # Raw measurements (for external plotter/debug tools)
        t_meas = np.asarray(results.get('t_meas', []), dtype=float)
        x_meas = np.asarray(results.get('x_meas', []), dtype=float)
        y_meas = np.asarray(results.get('y_meas', []), dtype=float)
        z_meas = np.asarray(results.get('z_meas', []), dtype=float)
        u_meas = np.asarray(results.get('u_meas', []), dtype=float)
        v_meas = np.asarray(results.get('v_meas', []), dtype=float)

        if t_meas.ndim == 2 and t_meas.shape[1] > 0:
            msg.measurements.time = t_meas[:, 0].flatten().tolist()
        else:
            msg.measurements.time = t_meas.flatten().tolist()

        msg.measurements.x_meas = x_meas.flatten(order='F').tolist() if x_meas.size else []
        msg.measurements.y_meas = y_meas.flatten(order='F').tolist() if y_meas.size else []
        msg.measurements.z_meas = z_meas.flatten(order='F').tolist() if z_meas.size else []
        msg.measurements.u_meas = u_meas.flatten(order='F').tolist() if u_meas.size else []
        msg.measurements.v_meas = v_meas.flatten(order='F').tolist() if v_meas.size else []
        msg.measurements.n_samples = int(results.get('n_samples', 0))
        msg.measurements.n_buoys = int(results.get('n_buoys', 0))

        # Actual-at-target sample/history captured from the same processing snapshot.
        wec = results.get('wec_actual')
        wec_series = results.get('wec_actual_series')
        if wec is not None:
            msg.has_wec_actual = True
            msg.wec_time = float(wec[0])
            msg.wec_z = float(wec[1])
            msg.wec_u = float(wec[2])
            msg.wec_v = float(wec[3])
        else:
            msg.has_wec_actual = False
            msg.wec_time = 0.0
            msg.wec_z = 0.0
            msg.wec_u = 0.0
            msg.wec_v = 0.0

        if wec_series is not None and len(wec_series) > 0:
            msg.has_wec_actual_series = True
            msg.wec_series_time = [float(p[0]) for p in wec_series]
            msg.wec_series_z = [float(p[1]) for p in wec_series]
            msg.wec_series_u = [float(p[2]) for p in wec_series]
            msg.wec_series_v = [float(p[3]) for p in wec_series]
        else:
            msg.has_wec_actual_series = False
            msg.wec_series_time = []
            msg.wec_series_z = []
            msg.wec_series_u = []
            msg.wec_series_v = []

        # Dense model predictions at target over measurement timestamps
        # (high-rate comparison series; future predictions remain unchanged).
        if hasattr(msg, 'has_dense_predictions'):
            dense_predictions_time = np.asarray(
                results.get('dense_predictions_time', []), dtype=float
            ).reshape((-1,))
            dense_predictions_z = np.asarray(
                results.get('dense_predictions_z', []), dtype=float
            ).reshape((-1,))
            dense_predictions_u = np.asarray(
                results.get('dense_predictions_u', []), dtype=float
            ).reshape((-1,))
            dense_predictions_v = np.asarray(
                results.get('dense_predictions_v', []), dtype=float
            ).reshape((-1,))

            n_dp = int(
                min(
                    dense_predictions_time.size,
                    dense_predictions_z.size,
                    dense_predictions_u.size,
                    dense_predictions_v.size,
                )
            )
            if n_dp > 0:
                msg.has_dense_predictions = True
                msg.dense_predictions_time = dense_predictions_time[:n_dp].tolist()
                msg.dense_predictions_z = dense_predictions_z[:n_dp].tolist()
                msg.dense_predictions_u = dense_predictions_u[:n_dp].tolist()
                msg.dense_predictions_v = dense_predictions_v[:n_dp].tolist()
            else:
                msg.has_dense_predictions = False
                msg.dense_predictions_time = []
                msg.dense_predictions_z = []
                msg.dense_predictions_u = []
                msg.dense_predictions_v = []

        t_pred = np.asarray(results.get('t_pred', []), dtype=float)
        z_pred = np.asarray(results.get('z_pred', []), dtype=float)
        u_pred = np.asarray(results.get('u_pred', []), dtype=float)
        v_pred = np.asarray(results.get('v_pred', []), dtype=float)

        first_t_pred = float(t_pred[0]) if t_pred.size > 0 else float('nan')
        wec_time = float(msg.wec_time) if bool(msg.has_wec_actual) else float('nan')
        window_end_time = float(msg.window_end_time)
        self.get_logger().debug(
            'Timing check: '
            f'window_end={window_end_time:.6f}s '
            f'wec_time={wec_time:.6f}s '
            f'first_t_pred={first_t_pred:.6f}s '
            f'pred_minus_window_end={first_t_pred - window_end_time:.6f}s '
            f'pred_minus_wec={first_t_pred - wec_time:.6f}s'
        )

        for i in range(int(t_pred.size)):
            pred_point = WavePredictionPoint()
            # Per-point absolute timestamp for PlotJuggler alignment.
            # `t_pred` is in the same time base as the incoming SBG timestamps
            # (ROS/sim time in seconds).
            t_s = float(t_pred[i])
            sec = int(np.floor(t_s)) if np.isfinite(t_s) and t_s >= 0.0 else 0
            nsec = int(np.round((t_s - sec) * 1e9)) if np.isfinite(t_s) and t_s >= 0.0 else 0
            if nsec >= int(1e9):
                sec += 1
                nsec -= int(1e9)
            if nsec < 0:
                nsec = 0
            pred_point.header = Header(frame_id='wec_buoy', stamp=TimeMsg(sec=sec, nanosec=nsec))

            pred_point.time = t_s
            pred_point.x = x_target
            pred_point.y = y_target
            pred_point.elevation = float(z_pred[i]) if i < z_pred.size else 0.0
            pred_point.vel_east = float(u_pred[i]) if i < u_pred.size else 0.0
            pred_point.vel_north = float(v_pred[i]) if i < v_pred.size else 0.0
            msg.predictions.append(pred_point)

        z_recon = np.asarray(results.get('z_recon', []), dtype=float)
        u_recon = np.asarray(results.get('u_recon', []), dtype=float)
        v_recon = np.asarray(results.get('v_recon', []), dtype=float)

        if t_meas.ndim == 2 and t_meas.shape[1] > 0:
            msg.reconstruction.time = t_meas[:, 0].flatten().tolist()
        else:
            msg.reconstruction.time = t_meas.flatten().tolist()

        msg.reconstruction.z_recon = z_recon.flatten(order='F').tolist() if z_recon.size else []
        msg.reconstruction.u_recon = u_recon.flatten(order='F').tolist() if u_recon.size else []
        msg.reconstruction.v_recon = v_recon.flatten(order='F').tolist() if v_recon.size else []
        msg.reconstruction.n_samples = int(results.get('n_samples', 0))
        msg.reconstruction.n_buoys = int(results.get('n_buoys', 0))

        self.pred_publisher.publish(msg)

    def ahrs_callback(self, data: XBRecord):
        with self.data_lock:
            self.wec_lat = data.gps.latitude
            self.wec_lon = data.gps.longitude

    def latent_callback(self, data):
        if not data.inc_wave_heights:
            return

        with self.data_lock:
            # Capture WEC "actual" at the target site from inc_wave_heights[0]
            # so downstream tools can subscribe to only the prediction topic.
            #
            # Frame note (mbari_wec_gz IncWaveHeight):
            # - pose.position.x/y are world coordinates (meters)
            # - pose.position.z is eta (height above waterplane, up-positive)
            # - velocities.x/y are Eulerian surface velocities u/v in East/North (m/s)
            # - velocities.z is etadot (m/s)
            try:
                inc0 = data.inc_wave_heights[0]
                t0_us = inc0.pose.header.stamp.sec * 1e6 + inc0.pose.header.stamp.nanosec / 1e3
                t0_s = float(t0_us) / 1e6
                wec_sample = {
                    't_s': t0_s,
                    'z': float(inc0.pose.pose.position.z),
                    'u': float(inc0.velocities.x),
                    'v': float(inc0.velocities.y),
                }
                self.wec_actual_latest = wec_sample

                # Maintain a higher-rate rolling history for plotting / spectral comparison.
                self.wec_actual_hist.append(
                    (wec_sample['t_s'], wec_sample['z'], wec_sample['u'], wec_sample['v'])
                )
                t_min = wec_sample['t_s'] - float(self.params.wec_actual_history_sec)
                while self.wec_actual_hist and self.wec_actual_hist[0][0] < t_min:
                    self.wec_actual_hist.popleft()
            except Exception:
                # Best-effort; don't break the main ingest path.
                pass

            for swift_name, swift_idx in self.params.swift_idx.items():
                swift_num = int(swift_name[-2:])
                try:
                    idx = int(swift_idx)
                except Exception:
                    continue

                # In deployment / TCP-bridge configs, swifts.* may be used as TCP ports.
                # Only treat it as an index if it falls within the message array.
                if idx < 0 or idx >= len(data.inc_wave_heights):
                    continue

                inc = data.inc_wave_heights[idx]
                t_us = inc.pose.header.stamp.sec * 1e6 + inc.pose.header.stamp.nanosec / 1e3

                lat_ref = inc.gps_ref.latitude
                lon_ref = inc.gps_ref.longitude
                x = inc.pose.pose.position.x
                y = inc.pose.pose.position.y

                # IncWaveHeight supplies x/y in world ENU (x=East, y=North) relative to gps_ref.
                # To interpret those x/y as the same local frame used by UTM deltas, keep
                # rotation_deg = 0. Nonzero rotation here changes the implied axes and will
                # desynchronize x/y from the published u/v (which remain East/North).
                lat, lon = generic_coordinate_transform_inverse(
                    x, y, lat_ref, lon_ref, self.params.rotation_deg
                )

                z = float(inc.pose.pose.position.z)
                u = float(inc.velocities.x)
                v = float(inc.velocities.y)

                # Apply a tiny sim-only noise floor to avoid perfectly coherent
                # wave inputs that can make MEM directional estimation singular.
                std_z = float(self.params.latent_noise_std_z_m)
                std_uv = float(self.params.latent_noise_std_uv_mps)
                if (std_z > 0.0 or std_uv > 0.0) and self._latent_rng is not None:
                    if std_z > 0.0:
                        z = z + float(self._latent_rng.normal(0.0, std_z))
                    if std_uv > 0.0:
                        u = u + float(self._latent_rng.normal(0.0, std_uv))
                        v = v + float(self._latent_rng.normal(0.0, std_uv))

                self.ingest_swift_sample_locked(
                    swift_num=swift_num,
                    t_us=float(t_us),
                    z=z,
                    u=u,
                    v=v,
                    lat=float(lat),
                    lon=float(lon),
                )

    def ingest_swift_sample_locked(
        self,
        *,
        swift_num: int,
        t_us: float,
        z: float,
        u: float,
        v: float,
        lat: float,
        lon: float,
    ) -> None:
        sbg = getattr(self.swifts, f'sbg{swift_num}')

        last_seen_t_us = self.last_seen_t_us_by_swift.get(swift_num)
        # Detect timestamp regressions (e.g., sim-time reset / /clock jump).
        # Use a small tolerance to avoid false positives from float rounding.
        if last_seen_t_us is not None and t_us < (last_seen_t_us - 1.0):
            dt_us = t_us - last_seen_t_us
            self.get_logger().warn(
                f'Time jumped backwards for swift{swift_num}: '
                f'prev={last_seen_t_us:.3f} us new={t_us:.3f} us (dt={dt_us:.3f} us)'
            )
            # Drop all pre-jump samples so timestamps remain monotonic.
            self.reset_swift_window(swift_num)
            sbg = getattr(self.swifts, f'sbg{swift_num}')

        # Detect large forward jumps (e.g., sim pause/resume or clock discontinuity)
        # that will effectively invalidate the rolling 256s window.
        if last_seen_t_us is not None and (
            t_us - last_seen_t_us
        ) > (self.params.window_duration_sec * 1e6):
            dt_us = t_us - last_seen_t_us
            self.get_logger().warn(
                f'Time jumped forwards for swift{swift_num}: '
                f'prev={last_seen_t_us:.3f} us new={t_us:.3f} us (dt={dt_us:.3f} us)'
            )
            self.reset_swift_window(swift_num)
            sbg = getattr(self.swifts, f'sbg{swift_num}')

        self.last_seen_t_us_by_swift[swift_num] = t_us

        if self.params.downsample_to_hz > 0.0:
            min_dt_us = 1e6 / self.params.downsample_to_hz
            last_t_us = self.last_accept_t_us_by_swift.get(swift_num)
            if last_t_us is not None and (t_us - last_t_us) < min_dt_us:
                return
            self.last_accept_t_us_by_swift[swift_num] = t_us

        sbg.ShipMotion.time_stamp.append(t_us)
        sbg.ShipMotion.heave.append(z)

        sbg.GpsVel.time_stamp.append(t_us)
        sbg.GpsVel.vel_e.append(u)
        sbg.GpsVel.vel_n.append(v)

        sbg.GpsPos.time_stamp.append(t_us)
        sbg.GpsPos.lat.append(float(lat))
        sbg.GpsPos.long.append(float(lon))

        self.maintain_sbg_window(sbg, t_us, swift_num)

        if self.last_process_time_us is None:
            self.last_process_time_us = t_us

    def reset_swift_window(self, swift_num: int) -> None:
        """
        Drop accumulated samples for a SWIFT and reset readiness state.

        This is used when we detect a backwards time jump (sim-time reset / clock jump)
        so we don't keep a non-monotonic timestamp history.
        """
        sbg = getattr(self.swifts, f'sbg{swift_num}', None)
        if sbg is None:
            return

        sbg.ShipMotion.time_stamp = deque()
        sbg.ShipMotion.heave = deque()
        sbg.GpsVel.time_stamp = deque()
        sbg.GpsVel.vel_e = deque()
        sbg.GpsVel.vel_n = deque()
        sbg.GpsPos.time_stamp = deque()
        sbg.GpsPos.lat = deque()
        sbg.GpsPos.long = deque()

        self.window_ready_by_swift[swift_num] = False
        self.window_ready = bool(self.window_ready_by_swift) and all(
            self.window_ready_by_swift.values()
        )

        # Also reset downsampling state so the first post-jump sample is accepted.
        self.last_accept_t_us_by_swift.pop(swift_num, None)

    def maintain_sbg_window(self, sbg: SBGData, current_t_us: float, swift_num: int):
        window_us = self.params.window_duration_sec * 1e6
        cutoff_t_us = current_t_us - window_us

        # Deque-based rolling window: pop from the left until we're within cutoff.
        # Assumes timestamps are monotonic; we reset the window on detected time jumps.
        try:
            # IMPORTANT: Err slightly on the side of a *too-long* window.
            #
            # If we always pop until the oldest sample is >= cutoff, the resulting
            # span will often be just under window_duration_sec by ~one sample period
            # (e.g., 255.8s for 5 Hz) due to discrete sampling. Keeping a single
            # sample just before the cutoff makes the span slightly >= 256s.
            while (
                len(sbg.ShipMotion.time_stamp) >= 2
                and sbg.ShipMotion.time_stamp[0] < cutoff_t_us
                and sbg.ShipMotion.time_stamp[1] < cutoff_t_us
            ):
                sbg.ShipMotion.time_stamp.popleft()
                sbg.ShipMotion.heave.popleft()
                sbg.GpsVel.time_stamp.popleft()
                sbg.GpsVel.vel_e.popleft()
                sbg.GpsVel.vel_n.popleft()
                sbg.GpsPos.time_stamp.popleft()
                sbg.GpsPos.lat.popleft()
                sbg.GpsPos.long.popleft()
        except IndexError:
            # If any field gets out of sync (shouldn't happen), reset to recover.
            self.get_logger().warn(f'swift{swift_num} window deques out of sync; resetting')
            self.reset_swift_window(swift_num)
            sbg = getattr(self.swifts, f'sbg{swift_num}')

        if sbg.ShipMotion.time_stamp:
            time_span_s = (sbg.ShipMotion.time_stamp[-1] - sbg.ShipMotion.time_stamp[0]) / 1e6
        else:
            time_span_s = 0.0

        if swift_num not in self.window_ready_by_swift:
            self.window_ready_by_swift[swift_num] = False

        was_ready = bool(self.window_ready_by_swift.get(swift_num, False))
        # Use a small tolerance when determining readiness.
        # With discrete sampling, a perfectly healthy 256 s rolling window can report
        # slightly less than 256.0 s span due to sample spacing/quantization.
        fs_ready = (
            self.params.downsample_to_hz
            if self.params.downsample_to_hz > 0.0
            else self.params.expected_fs
        )
        ready_epsilon_s = (1.0 / fs_ready) if fs_ready and fs_ready > 0.0 else 0.0
        is_ready = time_span_s >= (self.params.window_duration_sec - ready_epsilon_s)
        self.window_ready_by_swift[swift_num] = is_ready
        if was_ready and not is_ready:
            n = len(sbg.ShipMotion.time_stamp)
            first = sbg.ShipMotion.time_stamp[0] if n else float('nan')
            last = sbg.ShipMotion.time_stamp[-1] if n else float('nan')
            self.get_logger().warn(
                f'swift{swift_num} window_ready dropped False: time_span_s={time_span_s:.3f} '
                f'(threshold={self.params.window_duration_sec - ready_epsilon_s:.3f}) n={n} '
                f'first={first:.3f} us last={last:.3f} us'
            )
        self.window_ready = bool(self.window_ready_by_swift) and all(
            self.window_ready_by_swift.values()
        )

    def set_params(self):
        params = self.params
        defaults = TheNextWaveNodeParams()

        # Optional: downsample incoming latent data to mimic field SBG rate.
        # Set to 0.0 to disable (use all incoming samples).
        self.declare_parameter('downsample_to_hz', defaults.downsample_to_hz)

        # Coordinate frame convention.
        # IMPORTANT: if u/v are treated as East/North velocities, then either:
        # - set rotation_deg=0 so x/y are East/North, OR
        # - rotate u/v consistently with x/y upstream.
        self.declare_parameter('rotation_deg', defaults.rotation_deg)

        # Measurement sign convention.
        # Real SWIFT SBG heave is upside-down and needs a sign flip.
        # In gz sim, z is typically already up-positive, so set False.
        self.declare_parameter('flip_z_sign', defaults.flip_z_sign)

        # If > 0, recompute the averaged directional spectrum only at this interval
        # (in measurement time), reusing the last valid wavespec in between.
        self.declare_parameter(
            'wavespec_update_period_sec',
            defaults.wavespec_update_period_sec,
        )

        # Higher-rate history of WEC actual samples for frequency/phase comparisons.
        self.declare_parameter('wec_actual_history_sec', defaults.wec_actual_history_sec)

        # Dense model predictions at target are computed over the measurement window.
        # This parameter optionally clips them to the last N seconds.
        self.declare_parameter('dense_prediction_window', defaults.dense_prediction_window)
        self.declare_parameter(
            'enable_dense_history_projection',
            defaults.enable_dense_history_projection,
        )

        # Least-squares solver controls
        self.declare_parameter('lsq_ridge', defaults.lsq_ridge)
        self.declare_parameter('lsq_max_iter', defaults.lsq_max_iter)
        self.declare_parameter(
            'lsq_use_spectrum_weighted_ridge',
            defaults.lsq_use_spectrum_weighted_ridge,
        )
        self.declare_parameter('lsq_spectrum_ridge_floor', defaults.lsq_spectrum_ridge_floor)
        self.declare_parameter('lsq_diagnostics_enable', defaults.lsq_diagnostics_enable)
        self.declare_parameter('lsq_near_bound_ratio', defaults.lsq_near_bound_ratio)

        # Optional: stabilize MEM by capping directional moments (disabled by default).
        self.declare_parameter('mem_moment_cap_enable', defaults.mem_moment_cap_enable)
        self.declare_parameter('mem_moment_cap', defaults.mem_moment_cap)

        for idx, sid in enumerate(range(22, 26)):
            self.declare_parameter(f'swifts.swift{sid}', idx + 1)

        # Optional: ingest raw SBG data via TCP (Ethernet bridge).
        # This runs in parallel with any ROS subscriptions.
        self.declare_parameter('sbg_bridge_enable', defaults.sbg_bridge_enable)
        self.declare_parameter('sbg_bridge_bind', defaults.sbg_bridge_bind)
        self.declare_parameter(
            'sbg_bridge_socket_timeout_sec',
            defaults.sbg_bridge_socket_timeout_sec,
        )

        # Optional: sim latent noise floor to avoid MEM singularities.
        self.declare_parameter('latent_noise_std_z_m', defaults.latent_noise_std_z_m)
        self.declare_parameter('latent_noise_std_uv_mps', defaults.latent_noise_std_uv_mps)
        self.declare_parameter('latent_noise_seed', defaults.latent_noise_seed)

        # Optional: when ingesting raw SBG via the TCP bridge without Gazebo/WEC topics,
        # replicate the coordinate frame and target location from example.py.
        self.declare_parameter('sbg_bridge_use_example_frame', defaults.sbg_use_example_frame)
        self.declare_parameter('example_latorigin', defaults.example_latorigin)
        self.declare_parameter('example_lonorigin', defaults.example_lonorigin)
        self.declare_parameter('example_rotation_deg', defaults.example_rotation_deg)
        self.declare_parameter('example_xtarget', defaults.example_xtarget)
        self.declare_parameter('example_ytarget', defaults.example_ytarget)

        params.sbg_bridge_enable = bool(self.get_parameter('sbg_bridge_enable').value)
        params.sbg_bridge_bind = str(self.get_parameter('sbg_bridge_bind').value)
        params.sbg_bridge_socket_timeout_sec = float(
            self.get_parameter('sbg_bridge_socket_timeout_sec').value
        )

        params.latent_noise_std_z_m = float(self.get_parameter('latent_noise_std_z_m').value)
        params.latent_noise_std_uv_mps = float(
            self.get_parameter('latent_noise_std_uv_mps').value
        )
        params.latent_noise_seed = int(self.get_parameter('latent_noise_seed').value)

        params.sbg_use_example_frame = bool(
            self.get_parameter('sbg_bridge_use_example_frame').value
        )
        params.example_latorigin = float(self.get_parameter('example_latorigin').value)
        params.example_lonorigin = float(self.get_parameter('example_lonorigin').value)
        params.example_rotation_deg = float(self.get_parameter('example_rotation_deg').value)
        params.example_xtarget = float(self.get_parameter('example_xtarget').value)
        params.example_ytarget = float(self.get_parameter('example_ytarget').value)
        params.downsample_to_hz = float(self.get_parameter('downsample_to_hz').value)

        params.rotation_deg = float(self.get_parameter('rotation_deg').value)
        params.flip_z_sign = bool(self.get_parameter('flip_z_sign').value)
        params.wavespec_update_period_sec = float(
            self.get_parameter('wavespec_update_period_sec').value
        )
        params.wec_actual_history_sec = float(self.get_parameter('wec_actual_history_sec').value)
        params.dense_prediction_window = float(self.get_parameter('dense_prediction_window').value)
        params.enable_dense_history_projection = bool(
            self.get_parameter('enable_dense_history_projection').value
        )

        params.lsq_ridge = float(self.get_parameter('lsq_ridge').value)
        params.lsq_max_iter = int(self.get_parameter('lsq_max_iter').value)
        params.lsq_use_spectrum_weighted_ridge = bool(
            self.get_parameter('lsq_use_spectrum_weighted_ridge').value
        )
        params.lsq_spectrum_ridge_floor = float(
            self.get_parameter('lsq_spectrum_ridge_floor').value
        )
        params.lsq_diagnostics_enable = bool(
            self.get_parameter('lsq_diagnostics_enable').value
        )
        params.lsq_near_bound_ratio = float(self.get_parameter('lsq_near_bound_ratio').value)

        params.mem_moment_cap_enable = bool(self.get_parameter('mem_moment_cap_enable').value)
        params.mem_moment_cap = float(self.get_parameter('mem_moment_cap').value)

        swift_params = self.get_parameters_by_prefix('swifts')  # keys: 'swift22', 'swift23', ...

        params.swift_idx = OrderedDict()
        for sid in range(22, 26):
            key = f'swift{sid}'
            p = swift_params.get(key)
            if p is None:
                continue

            val = int(p.value)
            params.swift_idx[key] = val

        self.get_logger().info(f'Loaded swift mapping: {params.swift_idx}')

        # For SBG TCP bridge, reuse the existing 'swifts.swiftNN' parameters:
        # - In Gazebo/latent mode they are indices into inc_wave_heights[] (typically 1..4).
        # - In SBG TCP bridge / deployment they are TCP ports (e.g., 3001..).
        # In bridge mode, config should provide ports; in latent mode, indices.
        params.sbg_bridge_port_by_swift = {}
        port_map: dict[int, int] = {}
        for name, val in params.swift_idx.items():
            sid = int(str(name)[-2:])
            port_map[sid] = int(val)

        if params.sbg_bridge_enable:
            params.sbg_bridge_port_by_swift = port_map
            self.get_logger().info(
                f'Loaded SBG bridge port map from swifts.*: {params.sbg_bridge_port_by_swift}'
            )

        self.get_logger().info(f'Resolved node parameters:\n{params}')

    def destroy_node(self):
        try:
            if self.sbg_bridge_service is not None:
                self.sbg_bridge_service.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TheNextWaveNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        # SIGINT may already have shut down the context; try_shutdown is idempotent.
        try:
            rclpy.try_shutdown()
        except AttributeError:
            # Fallback for older rclpy
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
