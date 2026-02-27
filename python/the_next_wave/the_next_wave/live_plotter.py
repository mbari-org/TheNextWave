from __future__ import annotations

from collections import deque
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
    window_end_time: float

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

    # Optional: bulk wave params derived from the wavespec used for prediction.
    has_wavespec_bulk: bool
    wavespec_hs: float
    wavespec_tp: float
    wavespec_dp: float  # deg True, FROM
    wavespec_spreadp: float  # deg


class LivePlotter:
    """Render live prediction, reconstruction, and actual-at-target plots."""

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
        self.ax_z_err = None
        self.ax_z_err_meter = None

        self.initialized = False
        self.max_points = int(max_points) if max_points and max_points > 0 else 0
        self.pending_z_forecasts = deque(maxlen=50000)  # (target_t, pred_z, lead_s)
        self.latest_z_error_by_lead: dict[int, float] = {}
        self.z_err_median_abs_history = deque(maxlen=10)
        self.z_err_history_by_lead: dict[int, deque[float]] = {}
        self.latched_y_limits: dict[str, tuple[float, float]] = {}

    def apply_latched_ylim(self, ax, key: str, arrays: list[np.ndarray]) -> None:
        finite_parts = []
        for arr in arrays:
            a = np.asarray(arr, dtype=float).ravel()
            if a.size == 0:
                continue
            a = a[np.isfinite(a)]
            if a.size:
                finite_parts.append(a)

        if finite_parts:
            vals = np.concatenate(finite_parts)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if np.isclose(vmin, vmax):
                pad = max(0.25, abs(vmin) * 0.1)
                cur_lo = vmin - pad
                cur_hi = vmax + pad
            else:
                span = vmax - vmin
                pad = max(0.05 * span, 0.05)
                cur_lo = vmin - pad
                cur_hi = vmax + pad

            prev = self.latched_y_limits.get(key)
            if prev is None:
                latched = (cur_lo, cur_hi)
            else:
                latched = (min(prev[0], cur_lo), max(prev[1], cur_hi))
            self.latched_y_limits[key] = latched
            ax.set_ylim(*latched)
            return

        prev = self.latched_y_limits.get(key)
        if prev is not None:
            ax.set_ylim(*prev)

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
        """
        Create the figure/axes if not already created.

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
        self.ax_map = self.fig.add_axes([0.56, 0.64, 0.40, 0.30])
        self.ax_z_err = self.fig.add_axes([0.56, 0.50, 0.35, 0.10])
        self.ax_z_err_meter = self.fig.add_axes([0.93, 0.50, 0.03, 0.10])

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

        # Local import to avoid matplotlib dependency at module import time.
        from matplotlib.patches import Wedge

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
            self.ax_z_err,
            self.ax_z_err_meter,
        )
        for ax in axes:
            ax.cla()

        # --- Map ---
        ax = self.ax_map
        x_meas = self.decimate_2d_rows(np.asarray(d.x_meas, dtype=float))
        y_meas = self.decimate_2d_rows(np.asarray(d.y_meas, dtype=float))
        if x_meas.ndim == 2 and y_meas.ndim == 2 and x_meas.size and y_meas.size:
            ax.plot(x_meas, y_meas, 'x', linewidth=2)
        ax.plot([d.x_target], [d.y_target], 'ko', linewidth=2, markersize=6)

        # Optional: indicate incident-wave direction and spreading at the target.
        if getattr(d, 'has_wavespec_bulk', False) and np.isfinite(d.wavespec_dp):
            dp = float(d.wavespec_dp)
            spread = float(d.wavespec_spreadp) if np.isfinite(d.wavespec_spreadp) else float('nan')

            # Dp is a nautical/compass direction waves are coming FROM.
            # For display, draw propagation direction (TO = FROM + 180 deg).
            dp_to = (dp + 180.0) % 360.0

            # Compass degrees (0=N, 90=E) to ENU vector in x/y.
            rad = np.deg2rad(dp_to)
            dx_dir = float(np.sin(rad))
            dy_dir = float(np.cos(rad))

            # Pick a reasonable arrow length from the current map extents.
            try:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
                L = max(5.0, 0.08 * float(span))
            except Exception:
                L = 20.0

            ax.arrow(
                d.x_target,
                d.y_target,
                L * dx_dir,
                L * dy_dir,
                head_width=max(1.0, 0.15 * L),
                head_length=max(1.0, 0.20 * L),
                length_includes_head=True,
                color='k',
                alpha=0.8,
            )

            if np.isfinite(spread) and spread > 0.0:
                # Matplotlib Wedge angles are degrees CCW from +x.
                # Convert compass (CW from +y) to math (CCW from +x): a_math = 90 - dp.
                a0 = 90.0 - dp_to
                wedge = Wedge(
                    (d.x_target, d.y_target),
                    r=1.1 * L,
                    theta1=a0 - 0.5 * spread,
                    theta2=a0 + 0.5 * spread,
                    width=0.35 * L,
                    facecolor='k',
                    edgecolor='none',
                    alpha=0.12,
                )
                ax.add_patch(wedge)

            # Small label in the map for quick scanning.
            hs = float(d.wavespec_hs) if np.isfinite(d.wavespec_hs) else float('nan')
            tp = float(d.wavespec_tp) if np.isfinite(d.wavespec_tp) else float('nan')
            label = []
            if np.isfinite(hs):
                label.append(f'Hs {hs:.2f} m')
            if np.isfinite(tp):
                label.append(f'Tp {tp:.2f} s')
            label.append(f'Dp(from) {dp:.0f}°')
            if np.isfinite(spread):
                label.append(f'spread {spread:.0f}°')
            ax.text(
                0.02,
                0.98,
                ', '.join(label),
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                bbox={
                    'boxstyle': 'round,pad=0.2',
                    'facecolor': 'white',
                    'alpha': 0.7,
                    'edgecolor': 'none',
                },
            )
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

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
        self.ax_z_in.set_ylabel('z in [m]')
        self.apply_latched_ylim(self.ax_z_in, 'z_in', [z_meas])

        if u_meas.size:
            self.ax_u_in.plot(t_plot, u_meas)
        self.ax_u_in.set_ylabel('u in [m/s]')
        self.apply_latched_ylim(self.ax_u_in, 'u_in', [u_meas])

        if v_meas.size:
            self.ax_v_in.plot(t_plot, v_meas)
        self.ax_v_in.set_ylabel('v in [m/s]')
        self.apply_latched_ylim(self.ax_v_in, 'v_in', [v_meas])

        if z_recon.size:
            self.ax_z_rec.plot(t_plot, z_recon)
        self.ax_z_rec.set_ylabel('z recon [m]')
        self.apply_latched_ylim(self.ax_z_rec, 'z_recon', [z_recon])

        if u_recon.size:
            self.ax_u_rec.plot(t_plot, u_recon)
        self.ax_u_rec.set_ylabel('u recon [m/s]')
        self.apply_latched_ylim(self.ax_u_rec, 'u_recon', [u_recon])

        if v_recon.size:
            self.ax_v_rec.plot(t_plot, v_recon)
        self.ax_v_rec.set_ylabel('v recon [m/s]')
        self.ax_v_rec.set_xlabel('t [s]')
        self.apply_latched_ylim(self.ax_v_rec, 'v_recon', [v_recon])

        # --- Predictions (target) ---
        t_pred = self.decimate_1d(np.asarray(d.t_pred, dtype=float).ravel())

        t_wec_hist_full = np.asarray(d.t_wec_hist, dtype=float).ravel()
        t_wec_hist = self.decimate_1d(t_wec_hist_full)

        # Dense target predictions series (optional)
        t_nc = np.asarray(getattr(d, 'dense_predictions_time', np.array([])), dtype=float).ravel()
        z_nc = np.asarray(getattr(d, 'dense_predictions_z', np.array([])), dtype=float).ravel()
        u_nc = np.asarray(getattr(d, 'dense_predictions_u', np.array([])), dtype=float).ravel()
        v_nc = np.asarray(getattr(d, 'dense_predictions_v', np.array([])), dtype=float).ravel()
        has_nc = bool(getattr(d, 'has_dense_predictions', False)) and t_nc.size and z_nc.size
        if has_nc:
            n_nc = int(min(t_nc.size, z_nc.size, u_nc.size, v_nc.size))
            t_nc = self.decimate_1d(t_nc[:n_nc])
            z_nc = self.decimate_1d(z_nc[:n_nc])
            u_nc = self.decimate_1d(u_nc[:n_nc])
            v_nc = self.decimate_1d(v_nc[:n_nc])

        def plot_pred(
            axp,
            key: str,
            y_pred,
            t_wec,
            y_wec,
            y_wec_hist,
            ylabel: str,
            y_nc: np.ndarray | None,
        ):
            ylim_arrays: list[np.ndarray] = []
            y_pred = self.decimate_1d(np.asarray(y_pred, dtype=float).ravel())
            if t_pred.size and y_pred.size:
                axp.plot(t_pred, y_pred, 'k')
                ylim_arrays.append(y_pred)

            # Dense model-at-target predictions (measurement-rate).
            if has_nc and y_nc is not None:
                y_nc = self.decimate_1d(np.asarray(y_nc, dtype=float).ravel())
                n = min(t_nc.size, y_nc.size)
                if n:
                    axp.plot(t_nc[:n], y_nc[:n], color='C2', linewidth=1.2, alpha=0.9)
                    ylim_arrays.append(y_nc[:n])

            # History of WEC actual samples.
            y_wh = self.decimate_1d(np.asarray(y_wec_hist, dtype=float).ravel())
            if t_wec_hist.size and y_wh.size:
                n = min(t_wec_hist.size, y_wh.size)
                axp.plot(t_wec_hist[:n], y_wh[:n], '.', color='C3', markersize=4, alpha=0.6)
                ylim_arrays.append(y_wh[:n])

            # Latest actual WEC point (inc_wave_heights[0]) as a “gauge”-like marker.
            if (
                t_wec is not None
                and y_wec is not None
                and np.isfinite(t_wec)
                and np.isfinite(y_wec)
            ):
                # Draw a horizontal gauge bar at the current value.
                half_width = 1.5
                axp.plot(
                    [t_wec - half_width, t_wec + half_width],
                    [y_wec, y_wec],
                    color='C3',
                    linewidth=3,
                )
                ylim_arrays.append(np.array([y_wec], dtype=float))

            axp.set_ylabel(ylabel)
            self.apply_latched_ylim(axp, key, ylim_arrays)

        plot_pred(
            self.ax_z_pred,
            'z_pred',
            d.z_pred,
            d.t_wec,
            d.z_wec,
            d.z_wec_hist,
            'z pred [m]',
            z_nc if has_nc else None,
        )
        plot_pred(
            self.ax_u_pred,
            'u_pred',
            d.u_pred,
            d.t_wec,
            d.u_wec,
            d.u_wec_hist,
            'u pred [m/s]',
            u_nc if has_nc else None,
        )
        plot_pred(
            self.ax_v_pred,
            'v_pred',
            d.v_pred,
            d.t_wec,
            d.v_wec,
            d.v_wec_hist,
            'v pred [m/s]',
            v_nc if has_nc else None,
        )
        self.ax_v_pred.set_xlabel('t [s]')

        # --- Z-only forecast verification panel (actual - forecast) ---
        axe = self.ax_z_err
        axe.set_ylabel('z err [m]')
        axe.set_xlabel('lead [s]')
        axe.grid(True, alpha=0.3)
        axe.axhline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)

        if np.isfinite(getattr(d, 'window_end_time', np.nan)) and t_pred.size:
            issue_t = float(d.window_end_time)
            z_pred_full = np.asarray(d.z_pred, dtype=float).ravel()
            n_pred = min(t_pred.size, z_pred_full.size)
            for i in range(n_pred):
                target_t = float(t_pred[i])
                pred_z = float(z_pred_full[i])
                if np.isfinite(target_t) and np.isfinite(pred_z):
                    lead_s = target_t - issue_t
                    if np.isfinite(lead_s) and lead_s >= 0.0:
                        self.pending_z_forecasts.append((target_t, pred_z, lead_s))

        z_wh_full = np.asarray(d.z_wec_hist, dtype=float).ravel()
        valid_hist = np.isfinite(t_wec_hist_full) & np.isfinite(z_wh_full)
        t_hist = t_wec_hist_full[valid_hist]
        z_hist = z_wh_full[valid_hist]

        if t_hist.size >= 2 and self.pending_z_forecasts:
            order = np.argsort(t_hist)
            t_hist = t_hist[order]
            z_hist = z_hist[order]

            uniq_t, uniq_idx = np.unique(t_hist, return_index=True)
            t_hist = uniq_t
            z_hist = z_hist[uniq_idx]

            if t_hist.size >= 2:
                latest_actual_t = float(t_hist[-1])
                still_pending = deque(maxlen=self.pending_z_forecasts.maxlen)
                resolved_count = 0
                for target_t, pred_z, lead_s in self.pending_z_forecasts:
                    if target_t > latest_actual_t:
                        still_pending.append((target_t, pred_z, lead_s))
                        continue
                    if target_t < t_hist[0]:
                        continue

                    actual_z = float(np.interp(target_t, t_hist, z_hist))
                    if np.isfinite(actual_z):
                        lead_bucket = int(np.round(lead_s))
                        err = actual_z - pred_z
                        self.latest_z_error_by_lead[lead_bucket] = err
                        if lead_bucket not in self.z_err_history_by_lead:
                            self.z_err_history_by_lead[lead_bucket] = deque(maxlen=10)
                        self.z_err_history_by_lead[lead_bucket].append(float(err))
                        resolved_count += 1
                self.pending_z_forecasts = still_pending

                if resolved_count > 0 and self.latest_z_error_by_lead:
                    lead_vals = np.asarray(
                        list(self.latest_z_error_by_lead.values()),
                        dtype=float,
                    )
                    lead_vals = lead_vals[np.isfinite(lead_vals)]
                    if lead_vals.size > 0:
                        self.z_err_median_abs_history.append(float(np.median(np.abs(lead_vals))))

        if self.latest_z_error_by_lead:
            lead_keys = sorted(self.latest_z_error_by_lead.keys())
            axe.set_xlim(min(lead_keys) - 0.7, max(lead_keys) + 0.7)

            half_width_current = 0.18
            half_width_median = 0.33
            for lead in lead_keys:
                current_err = float(self.latest_z_error_by_lead[lead])
                if np.isfinite(current_err):
                    axe.plot(
                        [lead - half_width_current, lead + half_width_current],
                        [current_err, current_err],
                        color='C4',
                        linewidth=3.0,
                        alpha=0.95,
                    )

                hist = self.z_err_history_by_lead.get(lead)
                if not hist:
                    continue

                hist_vals = np.asarray(list(hist), dtype=float)
                hist_vals = hist_vals[np.isfinite(hist_vals)]
                if hist_vals.size == 0:
                    continue

                # Ghost levels from historical signed errors.
                n_hist = hist_vals.size
                for i, hv in enumerate(hist_vals):
                    alpha = 0.08 + 0.20 * ((i + 1) / n_hist)
                    axe.plot(
                        [lead - half_width_median, lead + half_width_median],
                        [hv, hv],
                        color='C1',
                        linewidth=1.1,
                        alpha=alpha,
                    )

                # Current signed median level (distinct color, thicker bar).
                med = float(np.median(hist_vals))
                axe.plot(
                    [lead - half_width_median, lead + half_width_median],
                    [med, med],
                    color='C1',
                    linewidth=2.4,
                    alpha=0.95,
                )

        err_arrays: list[np.ndarray] = []
        if self.latest_z_error_by_lead:
            err_arrays.append(np.asarray(list(self.latest_z_error_by_lead.values()), dtype=float))
        if self.z_err_history_by_lead:
            for hist in self.z_err_history_by_lead.values():
                if hist:
                    err_arrays.append(np.asarray(list(hist), dtype=float))
        self.apply_latched_ylim(axe, 'z_err', err_arrays)

        # --- Narrow magnitude meter for median absolute z-error ---
        axm = self.ax_z_err_meter
        axm.set_ylim(0.0, 3.0)
        axm.set_xlim(0.0, 1.0)
        axm.set_xticks([])
        axm.set_yticks([0.0, 1.0, 2.0, 3.0])
        axm.grid(True, axis='y', alpha=0.2)

        ghost_levels = list(self.z_err_median_abs_history)
        if ghost_levels:
            n_ghost = len(ghost_levels)
            for i, level in enumerate(ghost_levels):
                if not np.isfinite(level):
                    continue
                level = float(np.clip(level, 0.0, 3.0))
                alpha = 0.08 + 0.22 * ((i + 1) / n_ghost)
                axm.plot(
                    [0.12, 0.88],
                    [level, level],
                    color='C4',
                    linewidth=1.2,
                    alpha=alpha,
                )

            current_level = float(np.clip(ghost_levels[-1], 0.0, 3.0))
            axm.plot(
                [0.08, 0.92],
                [current_level, current_level],
                color='C4',
                linewidth=3.0,
                alpha=0.95,
            )
            axm.text(
                0.5,
                3.02,
                f'{current_level:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        # Keep GUI responsive.
        self.fig.canvas.draw_idle()
        # GUI event pumping is handled by the node's main-thread loop.
