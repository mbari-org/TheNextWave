from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LivePlotData:
    # Map / geometry
    x_meas: np.ndarray  # (N, B)
    y_meas: np.ndarray  # (N, B)
    x_target: float
    y_target: float

    # Measurements + recon
    t_meas: np.ndarray  # (N, B) or (N,)
    z_meas: np.ndarray  # (N, B)
    u_meas: np.ndarray  # (N, B)
    v_meas: np.ndarray  # (N, B)

    z_recon: np.ndarray  # (N, B)
    u_recon: np.ndarray  # (N, B)
    v_recon: np.ndarray  # (N, B)

    # Prediction horizon (target)
    t_pred: np.ndarray  # (M,)
    z_pred: np.ndarray  # (M,)
    u_pred: np.ndarray  # (M,)
    v_pred: np.ndarray  # (M,)

    # Dense target prediction over measurement timestamps (optional)
    has_dense_predictions: bool
    dense_predictions_time: np.ndarray  # (N,)
    dense_predictions_z: np.ndarray  # (N,)
    dense_predictions_u: np.ndarray  # (N,)
    dense_predictions_v: np.ndarray  # (N,)

    # Streaming “actual at target”
    t_wec: Optional[float]
    z_wec: Optional[float]
    u_wec: Optional[float]
    v_wec: Optional[float]

    # History of WEC actual samples (inc_wave_heights[0])
    t_wec_hist: np.ndarray  # (K2,)
    z_wec_hist: np.ndarray  # (K2,)
    u_wec_hist: np.ndarray  # (K2,)
    v_wec_hist: np.ndarray  # (K2,)

    # (prediction[0] history removed from plots)


class LivePlotter:
    def __init__(self, max_points: int = 800):
        # Import matplotlib lazily so the node can run headless without failing.
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig = None

        self.ax_map = None
        self.ax_z_in = None
        self.ax_u_in = None
        self.ax_v_in = None
        self.ax_z_rec = None
        self.ax_u_rec = None
        self.ax_v_rec = None
        self.ax_z_pred = None
        self.ax_u_pred = None
        self.ax_v_pred = None

        self.initialized = False
        self.max_points = int(max_points) if max_points and max_points > 0 else 0

    def decimate_1d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.max_points <= 0:
            return x
        if x.size <= self.max_points:
            return x
        step = int(np.ceil(x.size / float(self.max_points)))
        if step <= 1:
            return x
        return x[::step]

    def decimate_2d_rows(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if self.max_points <= 0:
            return a
        if a.ndim != 2:
            return a
        if a.shape[0] <= self.max_points:
            return a
        step = int(np.ceil(a.shape[0] / float(self.max_points)))
        if step <= 1:
            return a
        return a[::step, :]

    def ensure_initialized(self) -> None:
        """Create the figure/axes if not already created.

        Must be called from the main thread when using GUI backends.
        """
        if not self.initialized:
            self.init_figure()

    def is_window_open(self) -> bool:
        """Return True if the plot window still exists."""
        if not self.initialized:
            return True
        return bool(self.plt.fignum_exists(1))

    def init_figure(self):
        plt = self.plt
        plt.ion()
        self.fig = plt.figure(1)
        self.fig.clf()

        # Match the example.py layout as closely as practical.
        self.ax_map = self.fig.add_subplot(2, 2, 2)

        self.ax_z_in = self.fig.add_subplot(6, 2, 1)
        self.ax_u_in = self.fig.add_subplot(6, 2, 3)
        self.ax_v_in = self.fig.add_subplot(6, 2, 5)

        self.ax_z_rec = self.fig.add_subplot(6, 2, 7)
        self.ax_u_rec = self.fig.add_subplot(6, 2, 9)
        self.ax_v_rec = self.fig.add_subplot(6, 2, 11)

        self.ax_z_pred = self.fig.add_subplot(6, 2, 8)
        self.ax_u_pred = self.fig.add_subplot(6, 2, 10)
        self.ax_v_pred = self.fig.add_subplot(6, 2, 12)

        self.initialized = True

    def update(self, d: LivePlotData) -> None:
        if not self.initialized:
            self.init_figure()

        # Clear axes each update (simple + robust).
        axes = (
            self.ax_map,
            self.ax_z_in,
            self.ax_u_in,
            self.ax_v_in,
            self.ax_z_rec,
            self.ax_u_rec,
            self.ax_v_rec,
            self.ax_z_pred,
            self.ax_u_pred,
            self.ax_v_pred,
        )
        for ax in axes:
            ax.cla()

        # --- Map ---
        ax = self.ax_map
        x_meas = self.decimate_2d_rows(np.asarray(d.x_meas, dtype=float))
        y_meas = self.decimate_2d_rows(np.asarray(d.y_meas, dtype=float))
        if x_meas.ndim == 2 and y_meas.ndim == 2 and x_meas.size and y_meas.size:
            ax.plot(x_meas, y_meas, "x", linewidth=2)
        ax.plot([d.x_target], [d.y_target], "ko", linewidth=2, markersize=6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        # --- Measurements + recon ---
        t_meas = np.asarray(d.t_meas, dtype=float)
        if t_meas.ndim == 2 and t_meas.shape[1] > 0:
            t_plot = self.decimate_2d_rows(t_meas)
        else:
            t_plot = self.decimate_1d(t_meas)

        z_meas = self.decimate_2d_rows(np.asarray(d.z_meas, dtype=float))
        u_meas = self.decimate_2d_rows(np.asarray(d.u_meas, dtype=float))
        v_meas = self.decimate_2d_rows(np.asarray(d.v_meas, dtype=float))

        z_recon = self.decimate_2d_rows(np.asarray(d.z_recon, dtype=float))
        u_recon = self.decimate_2d_rows(np.asarray(d.u_recon, dtype=float))
        v_recon = self.decimate_2d_rows(np.asarray(d.v_recon, dtype=float))

        if z_meas.size:
            self.ax_z_in.plot(t_plot, z_meas)
        self.ax_z_in.set_ylabel("z in [m]")

        if u_meas.size:
            self.ax_u_in.plot(t_plot, u_meas)
        self.ax_u_in.set_ylabel("u in [m/s]")

        if v_meas.size:
            self.ax_v_in.plot(t_plot, v_meas)
        self.ax_v_in.set_ylabel("v in [m/s]")

        if z_recon.size:
            self.ax_z_rec.plot(t_plot, z_recon)
        self.ax_z_rec.set_ylabel("z recon [m]")

        if u_recon.size:
            self.ax_u_rec.plot(t_plot, u_recon)
        self.ax_u_rec.set_ylabel("u recon [m/s]")

        if v_recon.size:
            self.ax_v_rec.plot(t_plot, v_recon)
        self.ax_v_rec.set_ylabel("v recon [m/s]")
        self.ax_v_rec.set_xlabel("t [s]")

        # --- Predictions (target) ---
        t_pred = self.decimate_1d(np.asarray(d.t_pred, dtype=float).ravel())

        t_wec_hist = self.decimate_1d(np.asarray(d.t_wec_hist, dtype=float).ravel())

        # Dense target predictions series (optional)
        t_nc = np.asarray(getattr(d, "dense_predictions_time", np.array([])), dtype=float).ravel()
        z_nc = np.asarray(getattr(d, "dense_predictions_z", np.array([])), dtype=float).ravel()
        u_nc = np.asarray(getattr(d, "dense_predictions_u", np.array([])), dtype=float).ravel()
        v_nc = np.asarray(getattr(d, "dense_predictions_v", np.array([])), dtype=float).ravel()
        has_nc = bool(getattr(d, "has_dense_predictions", False)) and t_nc.size and z_nc.size
        if has_nc:
            n_nc = int(min(t_nc.size, z_nc.size, u_nc.size, v_nc.size))
            t_nc = self.decimate_1d(t_nc[:n_nc])
            z_nc = self.decimate_1d(z_nc[:n_nc])
            u_nc = self.decimate_1d(u_nc[:n_nc])
            v_nc = self.decimate_1d(v_nc[:n_nc])

        def plot_pred(axp, y_pred, t_wec, y_wec, y_wec_hist, ylabel: str, y_nc: np.ndarray | None):
            y_pred = self.decimate_1d(np.asarray(y_pred, dtype=float).ravel())
            if t_pred.size and y_pred.size:
                axp.plot(t_pred, y_pred, "k")

            # Dense model-at-target predictions (measurement-rate).
            if has_nc and y_nc is not None:
                y_nc = self.decimate_1d(np.asarray(y_nc, dtype=float).ravel())
                n = min(t_nc.size, y_nc.size)
                if n:
                    axp.plot(t_nc[:n], y_nc[:n], color="C2", linewidth=1.2, alpha=0.9)

            # History of WEC actual samples.
            y_wh = self.decimate_1d(np.asarray(y_wec_hist, dtype=float).ravel())
            if t_wec_hist.size and y_wh.size:
                n = min(t_wec_hist.size, y_wh.size)
                axp.plot(t_wec_hist[:n], y_wh[:n], ".", color="C3", markersize=4, alpha=0.6)

            # Latest actual WEC point (inc_wave_heights[0]) as a “gauge”-like marker.
            if t_wec is not None and y_wec is not None and np.isfinite(t_wec) and np.isfinite(y_wec):
                # Draw a short horizontal bar + crosshair marker at the current value.
                half_width = 0.5
                axp.plot([t_wec - half_width, t_wec + half_width], [y_wec, y_wec], color="C3", linewidth=3)
                axp.plot([t_wec], [y_wec], marker="x", color="C3", markersize=10, mew=2)

            axp.set_ylabel(ylabel)

        plot_pred(self.ax_z_pred, d.z_pred, d.t_wec, d.z_wec, d.z_wec_hist, "z pred [m]", z_nc if has_nc else None)
        plot_pred(self.ax_u_pred, d.u_pred, d.t_wec, d.u_wec, d.u_wec_hist, "u pred [m/s]", u_nc if has_nc else None)
        plot_pred(self.ax_v_pred, d.v_pred, d.t_wec, d.v_wec, d.v_wec_hist, "v pred [m/s]", v_nc if has_nc else None)
        self.ax_v_pred.set_xlabel("t [s]")

        # Keep GUI responsive.
        self.fig.canvas.draw_idle()
        # GUI event pumping is handled by the node's main-thread loop.
