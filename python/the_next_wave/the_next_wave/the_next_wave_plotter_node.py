#!/usr/bin/env python3

"""Standalone live plotter for TheNextWave.

This node subscribes to `buoy_interfaces/WavePredictionOutput` and renders the same
Matplotlib view that previously lived inside the solver node, without impacting
solve timing.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from buoy_interfaces.msg import WavePredictionOutput

from the_next_wave.live_plotter import LivePlotData, LivePlotter


def reshape_f(vec: list[float], n_samples: int, n_buoys: int) -> np.ndarray:
    a = np.asarray(vec, dtype=float)
    if n_samples <= 0 or n_buoys <= 0:
        return np.zeros((0, 0), dtype=float)
    if a.size != n_samples * n_buoys:
        # Best-effort: return empty to avoid throwing inside the callback loop.
        return np.zeros((0, 0), dtype=float)
    return a.reshape((n_samples, n_buoys), order="F")


class TheNextWavePlotterNode(Node):
    def __init__(self) -> None:
        super().__init__("the_next_wave_plotter_node")

        self.declare_parameter("prediction_topic", "wave_predictions")
        self.declare_parameter("max_points", 600)
        self.declare_parameter("history_sec", 100.0)

        prediction_topic = str(self.get_parameter("prediction_topic").value)
        max_points = int(self.get_parameter("max_points").value)
        history_sec = float(self.get_parameter("history_sec").value)
        if max_points < 50:
            max_points = 50
        if not np.isfinite(history_sec) or history_sec <= 0.0:
            history_sec = 600.0

        self.plotter = LivePlotter(max_points=max_points)
        self.history_sec = history_sec

        self.lock = threading.Lock()
        self.latest: Optional[WavePredictionOutput] = None
        self.dirty = False

        # Rolling history for WEC actual samples.
        # If a full WEC time series is carried in the message, we plot that directly.
        self.wec_hist = deque()  # (t, z, u, v)

        self.sub = self.create_subscription(
            WavePredictionOutput,
            prediction_topic,
            self.on_msg,
            10,
        )

        self.get_logger().info(f"Plotter subscribing to: {prediction_topic}")

    def on_msg(self, msg: WavePredictionOutput) -> None:
        with self.lock:
            self.latest = msg
            self.dirty = True

    def is_window_open(self) -> bool:
        return self.plotter.is_window_open()

    def plot_once(self) -> None:
        with self.lock:
            if not self.dirty or self.latest is None:
                return
            msg = self.latest
            self.dirty = False

        # --- Decode measurements ---
        m = msg.measurements
        n_samples = int(m.n_samples)
        n_buoys = int(m.n_buoys)

        t_meas = np.asarray(m.time, dtype=float)
        x_meas = reshape_f(m.x_meas, n_samples, n_buoys)
        y_meas = reshape_f(m.y_meas, n_samples, n_buoys)
        z_meas = reshape_f(m.z_meas, n_samples, n_buoys)
        u_meas = reshape_f(m.u_meas, n_samples, n_buoys)
        v_meas = reshape_f(m.v_meas, n_samples, n_buoys)

        # Reconstruction
        r = msg.reconstruction
        rn = int(r.n_samples)
        rb = int(r.n_buoys)
        t_rec = np.asarray(r.time, dtype=float)
        z_recon = reshape_f(r.z_recon, rn, rb)
        u_recon = reshape_f(r.u_recon, rn, rb)
        v_recon = reshape_f(r.v_recon, rn, rb)

        # Make recon times compatible with LivePlotter (it uses t_meas for both).
        # If recon time differs (it shouldn't), fall back to measurement time.
        if t_rec.size and t_rec.size == t_meas.size:
            t_for_plot = t_rec
        else:
            t_for_plot = t_meas

        # Predictions (target)
        t_pred = np.array([float(p.time) for p in msg.predictions], dtype=float)
        z_pred = np.array([float(p.elevation) for p in msg.predictions], dtype=float)
        u_pred = np.array([float(p.vel_east) for p in msg.predictions], dtype=float)
        v_pred = np.array([float(p.vel_north) for p in msg.predictions], dtype=float)

        # Dense target predictions series (optional)
        has_dense_predictions = bool(getattr(msg, "has_dense_predictions", False))
        dense_predictions_time = np.asarray(getattr(msg, "dense_predictions_time", []), dtype=float)
        dense_predictions_z = np.asarray(getattr(msg, "dense_predictions_z", []), dtype=float)
        dense_predictions_u = np.asarray(getattr(msg, "dense_predictions_u", []), dtype=float)
        dense_predictions_v = np.asarray(getattr(msg, "dense_predictions_v", []), dtype=float)
        n_dp = int(
            min(
                dense_predictions_time.size,
                dense_predictions_z.size,
                dense_predictions_u.size,
                dense_predictions_v.size,
            )
        )
        if n_dp <= 0:
            has_dense_predictions = False
            dense_predictions_time = np.array([], dtype=float)
            dense_predictions_z = np.array([], dtype=float)
            dense_predictions_u = np.array([], dtype=float)
            dense_predictions_v = np.array([], dtype=float)
        else:
            dense_predictions_time = dense_predictions_time[:n_dp]
            dense_predictions_z = dense_predictions_z[:n_dp]
            dense_predictions_u = dense_predictions_u[:n_dp]
            dense_predictions_v = dense_predictions_v[:n_dp]

        use_msg_series = bool(getattr(msg, "has_wec_actual_series", False)) and len(getattr(msg, "wec_series_time", [])) > 0
        if not use_msg_series and bool(msg.has_wec_actual):
            # Backward-compatible fallback: single sample per message.
            self.wec_hist.append((float(msg.wec_time), float(msg.wec_z), float(msg.wec_u), float(msg.wec_v)))

        # Trim histories
        t_ref = None
        if t_meas.size:
            t_ref = float(t_meas[-1])
        elif t_pred.size:
            t_ref = float(t_pred[-1])
        if t_ref is not None and np.isfinite(t_ref):
            t_min = t_ref - float(self.history_sec)
            while self.wec_hist and self.wec_hist[0][0] < t_min:
                self.wec_hist.popleft()

        if use_msg_series:
            t_wec_hist = np.asarray(getattr(msg, "wec_series_time", []), dtype=float)
            z_wec_hist = np.asarray(getattr(msg, "wec_series_z", []), dtype=float)
            u_wec_hist = np.asarray(getattr(msg, "wec_series_u", []), dtype=float)
            v_wec_hist = np.asarray(getattr(msg, "wec_series_v", []), dtype=float)
        else:
            t_wec_hist = np.array([p[0] for p in self.wec_hist], dtype=float) if self.wec_hist else np.array([], dtype=float)
            z_wec_hist = np.array([p[1] for p in self.wec_hist], dtype=float) if self.wec_hist else np.array([], dtype=float)
            u_wec_hist = np.array([p[2] for p in self.wec_hist], dtype=float) if self.wec_hist else np.array([], dtype=float)
            v_wec_hist = np.array([p[3] for p in self.wec_hist], dtype=float) if self.wec_hist else np.array([], dtype=float)

        t_wec = float(msg.wec_time) if bool(msg.has_wec_actual) else None
        z_wec = float(msg.wec_z) if bool(msg.has_wec_actual) else None
        u_wec = float(msg.wec_u) if bool(msg.has_wec_actual) else None
        v_wec = float(msg.wec_v) if bool(msg.has_wec_actual) else None

        d = LivePlotData(
            x_meas=x_meas,
            y_meas=y_meas,
            x_target=float(msg.x_target),
            y_target=float(msg.y_target),
            t_meas=t_for_plot,
            z_meas=z_meas,
            u_meas=u_meas,
            v_meas=v_meas,
            z_recon=z_recon,
            u_recon=u_recon,
            v_recon=v_recon,
            t_pred=t_pred,
            z_pred=z_pred,
            u_pred=u_pred,
            v_pred=v_pred,
            has_dense_predictions=has_dense_predictions,
            dense_predictions_time=dense_predictions_time,
            dense_predictions_z=dense_predictions_z,
            dense_predictions_u=dense_predictions_u,
            dense_predictions_v=dense_predictions_v,
            t_wec=t_wec,
            z_wec=z_wec,
            u_wec=u_wec,
            v_wec=v_wec,
            t_wec_hist=t_wec_hist,
            z_wec_hist=z_wec_hist,
            u_wec_hist=u_wec_hist,
            v_wec_hist=v_wec_hist,
        )

        try:
            self.plotter.update(d)
        except Exception as e:
            self.get_logger().warn(f"Plot update failed: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TheNextWavePlotterNode()

    # Create the GUI in the main thread.
    node.plotter.ensure_initialized()

    try:
        import matplotlib.pyplot as plt

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.plot_once()

            # Keep GUI responsive.
            plt.pause(0.05)

            if not node.is_window_open():
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
