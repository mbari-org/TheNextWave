#!/usr/bin/env python3

"""ROS 2 node wrapper for TheNextWave near-realtime predictions.

All ROS 2 concerns live here (subscriptions, publishers, message packaging).
The core algorithmic pipeline is implemented in `the_next_wave.the_next_wave.TheNextWave`.
"""

from collections import OrderedDict, deque
import threading
import traceback
import time

import numpy as np

import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import Header

from buoy_api.interface import Interface
from buoy_interfaces.msg import XBRecord
from buoy_interfaces.msg import WavePredictionOutput, WavePredictionPoint
from the_next_wave import TheNextWave, TheNextWaveConfig
from the_next_wave.swift import SBGData, SWIFTArray
from the_next_wave.utilities import generic_coordinate_transform_inverse

# Duration of the data window to maintain (in seconds)
WINDOW_DURATION_SEC = 256.0
# Trigger processing every N seconds (must be > processing time ~1.5s)
PROCESSING_INTERVAL_SEC = 2.0
# Expected sample rate (Hz)
EXPECTED_FS = 5.0
ROTATION = 0.0  # Rotation angle in degrees (0 for unrotated)

N_TE = 10.0  # Match example.py: solve using ~NTe*Te seconds of data


class TheNextWaveNode(Interface):
    def __init__(self, node_name="the_next_wave_node", **kwargs):
        super().__init__(node_name=node_name, **kwargs)
        self.swift_idx = OrderedDict()
        self.swifts = SWIFTArray()
        # Initialize SBG objects with empty lists for 256s windowing.
        self.init_sbg_windows()

        self.last_process_time_us = None
        self.window_ready = False
        self.window_ready_by_swift = {}

        # Optional: downsample incoming latent data to mimic field SBG rate.
        # Set to 0.0 to disable (use all incoming samples).
        self.declare_parameter("downsample_to_hz", 0.0)
        self.downsample_to_hz = float(self.get_parameter("downsample_to_hz").value)
        self._last_accept_t_us_by_swift: dict[int, float] = {}
        self._last_seen_t_us_by_swift: dict[int, float] = {}

        self.predictor = TheNextWave(
            config=TheNextWaveConfig(
                expected_fs=EXPECTED_FS,
                rotation_deg=ROTATION,
                n_te=N_TE,
            ),
            logger=self.get_logger(),
        )

        self.data_lock = threading.Lock()
        self.processing = False
        self._last_wavespec_warn_walltime = 0.0

        # WEC buoy pose for prediction target (set by ahrs_callback)
        self.wec_lat = None
        self.wec_lon = None

        self.set_params()
        self.use_sim_time()

        # Track window readiness only for configured buoys
        configured_swift_nums = [int(name[-2:]) for name in self.swift_idx.keys()]
        if configured_swift_nums:
            self.window_ready_by_swift = {sid: False for sid in configured_swift_nums}
        else:
            # Safe default if no params set (won't trigger unless data arrives)
            self.window_ready_by_swift = {sid: False for sid in range(22, 26)}

        self.pred_publisher = self.create_publisher(WavePredictionOutput, "wave_predictions", 10)

        self.process_timer = self.create_timer(
            PROCESSING_INTERVAL_SEC,
            self.process_timer_callback,
        )

    def reset_swift_window(self, swift_num: int) -> None:
        """Drop accumulated samples for a SWIFT and reset readiness state.

        This is used when we detect a backwards time jump (sim-time reset / clock jump)
        so we don't keep a non-monotonic timestamp history.
        """
        sbg = getattr(self.swifts, f"sbg{swift_num}", None)
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
        self.window_ready = bool(self.window_ready_by_swift) and all(self.window_ready_by_swift.values())

        # Also reset downsampling state so the first post-jump sample is accepted.
        self._last_accept_t_us_by_swift.pop(swift_num, None)

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
            setattr(self.swifts, f"sbg{sid}", sbg)

    def set_params(self):
        UNSET = -1
        for sid in range(22, 26):
            self.declare_parameter(f'swifts.swift{sid}', UNSET)

        swift_params = self.get_parameters_by_prefix('swifts')  # keys: 'swift22', 'swift23', ...

        self.swift_idx = OrderedDict()
        for sid in range(22, 26):
            key = f'swift{sid}'
            p = swift_params.get(key)
            if p is None:
                continue

            val = int(p.value)
            if val == UNSET:
                continue

            self.swift_idx[key] = val

        self.get_logger().info(f'Loaded swift mapping: {self.swift_idx}')

    def ahrs_callback(self, data: XBRecord):
        with self.data_lock:
            self.wec_lat = data.gps.latitude
            self.wec_lon = data.gps.longitude

    def latent_callback(self, data):
        if not data.inc_wave_heights:
            return

        with self.data_lock:
            for swift_name, swift_idx in self.swift_idx.items():
                swift_num = int(swift_name[-2:])
                inc = data.inc_wave_heights[swift_idx]
                sbg = getattr(self.swifts, f"sbg{swift_num}")

                t_us = inc.pose.header.stamp.sec * 1e6 + inc.pose.header.stamp.nanosec / 1e3

                last_seen_t_us = self._last_seen_t_us_by_swift.get(swift_num)
                # Detect timestamp regressions (e.g., sim-time reset / /clock jump).
                # Use a small tolerance to avoid false positives from float rounding.
                if last_seen_t_us is not None and t_us < (last_seen_t_us - 1.0):
                    dt_us = t_us - last_seen_t_us
                    self.get_logger().warn(
                        f"Time jumped backwards for swift{swift_num}: "
                        f"prev={last_seen_t_us:.3f} us new={t_us:.3f} us (dt={dt_us:.3f} us)"
                    )
                    # Drop all pre-jump samples so timestamps remain monotonic.
                    self.reset_swift_window(swift_num)
                    # Refresh reference after reset (lists were cleared).
                    sbg = getattr(self.swifts, f"sbg{swift_num}")
                # Detect large forward jumps (e.g., sim pause/resume or clock discontinuity)
                # that will effectively invalidate the rolling 256s window.
                if last_seen_t_us is not None and (t_us - last_seen_t_us) > (WINDOW_DURATION_SEC * 1e6):
                    dt_us = t_us - last_seen_t_us
                    self.get_logger().warn(
                        f"Time jumped forwards for swift{swift_num}: "
                        f"prev={last_seen_t_us:.3f} us new={t_us:.3f} us (dt={dt_us:.3f} us)"
                    )
                    self.reset_swift_window(swift_num)
                    sbg = getattr(self.swifts, f"sbg{swift_num}")
                self._last_seen_t_us_by_swift[swift_num] = t_us

                if self.downsample_to_hz > 0.0:
                    min_dt_us = 1e6 / self.downsample_to_hz
                    last_t_us = self._last_accept_t_us_by_swift.get(swift_num)
                    if last_t_us is not None and (t_us - last_t_us) < min_dt_us:
                        continue
                    self._last_accept_t_us_by_swift[swift_num] = t_us

                sbg.ShipMotion.time_stamp.append(t_us)
                sbg.ShipMotion.heave.append(inc.pose.pose.position.z)

                sbg.GpsVel.time_stamp.append(t_us)
                sbg.GpsVel.vel_e.append(inc.velocities.x)
                sbg.GpsVel.vel_n.append(inc.velocities.y)

                lat_ref = inc.gps_ref.latitude
                lon_ref = inc.gps_ref.longitude
                x = inc.pose.pose.position.x
                y = inc.pose.pose.position.y

                lat, lon = generic_coordinate_transform_inverse(x, y, lat_ref, lon_ref, ROTATION)
                sbg.GpsPos.time_stamp.append(t_us)
                sbg.GpsPos.lat.append(float(lat))
                sbg.GpsPos.long.append(float(lon))

                self.maintain_sbg_window(sbg, t_us, swift_num)

                if self.last_process_time_us is None:
                    self.last_process_time_us = t_us

    def maintain_sbg_window(self, sbg: SBGData, current_t_us: float, swift_num: int):
        window_us = WINDOW_DURATION_SEC * 1e6
        cutoff_t_us = current_t_us - window_us

        # Deque-based rolling window: pop from the left until we're within cutoff.
        # Assumes timestamps are monotonic; we reset the window on detected time jumps.
        try:
            while sbg.ShipMotion.time_stamp and sbg.ShipMotion.time_stamp[0] < cutoff_t_us:
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
            self.get_logger().warn(f"swift{swift_num} window deques out of sync; resetting")
            self.reset_swift_window(swift_num)
            sbg = getattr(self.swifts, f"sbg{swift_num}")

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
        fs_ready = self.downsample_to_hz if self.downsample_to_hz > 0.0 else EXPECTED_FS
        ready_epsilon_s = (1.0 / fs_ready) if fs_ready and fs_ready > 0.0 else 0.0
        is_ready = time_span_s >= (WINDOW_DURATION_SEC - ready_epsilon_s)
        self.window_ready_by_swift[swift_num] = is_ready
        if was_ready and not is_ready:
            n = len(sbg.ShipMotion.time_stamp)
            first = sbg.ShipMotion.time_stamp[0] if n else float("nan")
            last = sbg.ShipMotion.time_stamp[-1] if n else float("nan")
            self.get_logger().warn(
                f"swift{swift_num} window_ready dropped False: time_span_s={time_span_s:.3f} "
                f"(threshold={WINDOW_DURATION_SEC - ready_epsilon_s:.3f}) n={n} "
                f"first={first:.3f} us last={last:.3f} us"
            )
        self.window_ready = bool(self.window_ready_by_swift) and all(self.window_ready_by_swift.values())

    def process_timer_callback(self):
        self.get_logger().info(f'window ready? {self.window_ready} :: n={len(self.swifts.sbg22.ShipMotion.time_stamp)} samples :: t={(self.swifts.sbg22.ShipMotion.time_stamp[-1] if self.swifts.sbg22.ShipMotion.time_stamp else float("nan"))/1e6 - (self.swifts.sbg22.ShipMotion.time_stamp[0] if self.swifts.sbg22.ShipMotion.time_stamp else float("nan"))/1e6:.3f} s')
        with self.data_lock:
            if self.processing or not self.window_ready:
                return
            # Set this here (before starting the thread) to avoid a race where
            # multiple timer callbacks can launch multiple processing threads.
            self.processing = True
        self.get_logger().info("Triggering wave processing...")

        thread = threading.Thread(target=self.run_wave_processing, daemon=True)
        thread.start()

    def run_wave_processing(self):
        try:
            with self.data_lock:
                swifts_snapshot = SWIFTArray()
                for sid in range(22, 26):
                    sbg_src = getattr(self.swifts, f"sbg{sid}")
                    sbg_dst = SBGData()
                    sbg_dst.ShipMotion.time_stamp = np.array(sbg_src.ShipMotion.time_stamp, dtype=np.float64)
                    sbg_dst.ShipMotion.heave = np.array(sbg_src.ShipMotion.heave, dtype=np.float64)
                    sbg_dst.GpsVel.time_stamp = np.array(sbg_src.GpsVel.time_stamp, dtype=np.float64)
                    sbg_dst.GpsVel.vel_e = np.array(sbg_src.GpsVel.vel_e, dtype=np.float64)
                    sbg_dst.GpsVel.vel_n = np.array(sbg_src.GpsVel.vel_n, dtype=np.float64)
                    sbg_dst.GpsPos.time_stamp = np.array(sbg_src.GpsPos.time_stamp, dtype=np.float64)
                    sbg_dst.GpsPos.lat = np.array(sbg_src.GpsPos.lat, dtype=np.float64)
                    sbg_dst.GpsPos.long = np.array(sbg_src.GpsPos.long, dtype=np.float64)
                    setattr(swifts_snapshot, f"sbg{sid}", sbg_dst)

                wec_lat = self.wec_lat
                wec_lon = self.wec_lon

            results = self.predictor.process(swifts_snapshot, wec_lat=wec_lat, wec_lon=wec_lon)
            self.publish_prediction(results)

        except ValueError as e:
            # Warmup/transient condition: wave spectra may not be usable yet.
            msg = str(e)
            if "No usable wavespec available yet" in msg:
                now = time.monotonic()
                if now - self._last_wavespec_warn_walltime > 5.0:
                    self._last_wavespec_warn_walltime = now
                    self.get_logger().warn(msg)
            else:
                self.get_logger().error(f"Background wave processing failed: {e}")
                self.get_logger().error(traceback.format_exc())
        except Exception as e:
            self.get_logger().error(f"Background wave processing failed: {e}")
            self.get_logger().error(traceback.format_exc())
        finally:
            self.processing = False

    def publish_prediction(self, results: dict) -> None:
        wavespec = results.get("wavespec")
        params = results.get("params")

        msg = WavePredictionOutput()
        msg.header = Header(frame_id="wec_buoy", stamp=self.get_clock().now().to_msg())

        msg.window_start_time = float(results["window_start_time"])
        msg.window_end_time = float(results["window_end_time"])
        msg.n_measurements = int(results["n_samples"])

        if wavespec is not None:
            msg.frequencies = np.asarray(wavespec.f, dtype=float).flatten().tolist()
            msg.directions = np.asarray(wavespec.theta, dtype=float).flatten().tolist()

            Etheta = np.asarray(wavespec.Etheta, dtype=float)
            f = np.asarray(wavespec.f, dtype=float).flatten()
            theta = np.asarray(wavespec.theta, dtype=float).flatten()
            if Etheta.ndim == 2:
                if Etheta.shape == (theta.size, f.size):
                    energy_by_freq = np.sum(Etheta, axis=0)
                elif Etheta.shape == (f.size, theta.size):
                    energy_by_freq = np.sum(Etheta, axis=1)
                else:
                    energy_by_freq = np.sum(Etheta, axis=0)
            else:
                energy_by_freq = np.array([])

            msg.energy_by_freq = np.asarray(energy_by_freq, dtype=float).flatten().tolist()

        msg.centroid_period = float(results.get("Te", 0.0))
        msg.solve_time = float(results.get("solve_time", 0.0))
        msg.num_wavelengths = int(getattr(params, "kx", np.array([])).size) if params is not None else 0

        x_target = float(results.get("x_target", 0.0))
        y_target = float(results.get("y_target", 0.0))
        t_pred = np.asarray(results.get("t_pred", []), dtype=float)
        z_pred = np.asarray(results.get("z_pred", []), dtype=float)
        u_pred = np.asarray(results.get("u_pred", []), dtype=float)
        v_pred = np.asarray(results.get("v_pred", []), dtype=float)

        for i in range(int(t_pred.size)):
            pred_point = WavePredictionPoint()
            pred_point.time = float(t_pred[i])
            pred_point.x = x_target
            pred_point.y = y_target
            pred_point.elevation = float(z_pred[i]) if i < z_pred.size else 0.0
            pred_point.vel_east = float(u_pred[i]) if i < u_pred.size else 0.0
            pred_point.vel_north = float(v_pred[i]) if i < v_pred.size else 0.0
            msg.predictions.append(pred_point)

        t_meas = np.asarray(results.get("t_meas", []), dtype=float)
        z_recon = np.asarray(results.get("z_recon", []), dtype=float)
        u_recon = np.asarray(results.get("u_recon", []), dtype=float)
        v_recon = np.asarray(results.get("v_recon", []), dtype=float)

        if t_meas.ndim == 2 and t_meas.shape[1] > 0:
            msg.reconstruction.time = t_meas[:, 0].flatten().tolist()
        else:
            msg.reconstruction.time = t_meas.flatten().tolist()

        msg.reconstruction.z_recon = z_recon.flatten(order="F").tolist() if z_recon.size else []
        msg.reconstruction.u_recon = u_recon.flatten(order="F").tolist() if u_recon.size else []
        msg.reconstruction.v_recon = v_recon.flatten(order="F").tolist() if v_recon.size else []
        msg.reconstruction.n_samples = int(results.get("n_samples", 0))
        msg.reconstruction.n_buoys = int(results.get("n_buoys", 0))

        self.pred_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TheNextWaveNode()
    try:
        node.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
