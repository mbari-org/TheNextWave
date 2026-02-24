"""Core near-realtime wave prediction pipeline (ROS-free).

This module mirrors the algorithmic flow in `example.py`, but is structured for
streaming / near-realtime use:

1) Raw SBG windows -> `reprocess_swift_array()` -> cleaned SBG + SWIFT structs
2) SWIFT structs -> averaged `WaveSpec` (directional spectrum)
3) Measurement arrays + spectrum -> `leastSquaresWavePropagation()` -> prediction

All ROS 2 concerns (publishers, messages, time stamps) should live in the node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .leastSquaresWavePropagation import leastSquaresWavePropagation
from .reprocess_SBG import reprocess_swift_array
from .swift import SBGData, SWIFTArray, WaveSpec
from .utilities import (
    build_wavespec_from_swifts,
    centroid_period_and_phase_speed,
    generic_coordinate_transform,
    load_raw_arrays_from_sbg,
)


@dataclass
class TheNextWaveConfig:
    expected_fs: float = 5.0
    rotation_deg: float = 0.0
    n_te: float = 10.0
    # If > 0, reuse the most recently computed usable wavespec until this
    # many seconds of (measurement) time have elapsed, then refresh.
    # This can reduce CPU usage and improve warm-start stability (A0).
    wavespec_update_period_sec: float = 0.0
    # Only compute/publish dense_predictions_* for the last N seconds of the
    # measurement window. Set to 0 to use the full window.
    dense_prediction_window: float = 0.0


class TheNextWave:
    """Algorithmic wave prediction core (no ROS dependencies)."""

    def __init__(self, config: TheNextWaveConfig | None = None, logger: Any | None = None):
        self.config = config or TheNextWaveConfig()
        self.logger = logger

        self.A0: np.ndarray | None = None
        self.lat_origin: float | None = None
        self.lon_origin: float | None = None
        self._last_wavespec: WaveSpec | None = None
        self._last_wavespec_time_s: float | None = None

    def _wavespec_is_usable(self, ws: WaveSpec | None) -> bool:
        if ws is None:
            return False
        f = np.asarray(getattr(ws, "f", np.array([])), dtype=float)
        Etheta = np.asarray(getattr(ws, "Etheta", np.array([])), dtype=float)
        if f.size == 0 or Etheta.size == 0:
            return False
        if not np.all(np.isfinite(f)):
            return False
        # Require some positive finite energy to avoid 0/0 centroid period.
        finite_energy = Etheta[np.isfinite(Etheta)]
        return finite_energy.size > 0 and float(np.nansum(finite_energy)) > 0.0

    def process(self, swifts: SWIFTArray, wec_lat: float | None = None, wec_lon: float | None = None) -> dict:
        """Run wave spectral analysis and prediction.

        Args:
            swifts: SWIFTArray containing SBG windows (sbg22-25)
            wec_lat/wec_lon: WEC buoy target position. If missing, target defaults
                to the origin (0,0) of the local projection.

        Returns:
            Results dictionary containing the intermediate products (cleaned SBG,
            spectrum, recon) and the final predictions.
        """
        results: dict[str, Any] = {"wave_stats": {}}

        cleaned_sbg, swift_structs = reprocess_swift_array(swifts, fs=self.config.expected_fs)

        # Wave statistics per buoy (from reprocess_swift_array outputs)
        for sid in range(22, 26):
            swift_name = f"swift{sid}"
            swift_data = getattr(swift_structs, swift_name, None)
            if swift_data is None or getattr(swift_data, "sigwaveheight", np.array([])).size == 0:
                continue

            Hs = float(swift_data.sigwaveheight[0])
            Tp = float(swift_data.peakwaveperiod[0])
            Dp = float(swift_data.peakwavedirT[0])
            results["wave_stats"][swift_name] = {"Hs": Hs, "Tp": Tp, "Dp": Dp}

        zin, uin, vin, tin, xin, yin, fs = self._stack_measurement_data(cleaned_sbg)
        fs = float(fs)
        if not np.isfinite(fs) or fs <= 0.0:
            if self.logger is not None:
                self.logger.warn(f"Invalid fs={fs}; falling back to expected_fs={self.config.expected_fs}")
            fs = float(self.config.expected_fs)

        # Decide whether to refresh the wavespec.
        # Use the measurement time base (tin) to determine age.
        current_t_s = float(np.nanmax(tin)) if np.size(tin) else 0.0
        use_cached = False
        if self.config.wavespec_update_period_sec > 0.0 and self._wavespec_is_usable(self._last_wavespec):
            if self._last_wavespec_time_s is not None and np.isfinite(current_t_s):
                age_s = current_t_s - float(self._last_wavespec_time_s)
                use_cached = age_s >= 0.0 and age_s < float(self.config.wavespec_update_period_sec)

        wavespec_new = None
        if not use_cached:
            wavespec_new = self._build_averaged_wavespec(swift_structs)

        if use_cached:
            wavespec = self._last_wavespec
        elif self._wavespec_is_usable(wavespec_new):
            wavespec = wavespec_new
            self._last_wavespec = wavespec_new
            self._last_wavespec_time_s = current_t_s
        elif self._wavespec_is_usable(self._last_wavespec):
            wavespec = self._last_wavespec
            if self.logger is not None:
                self.logger.warn("No usable wavespec from current window; reusing last valid wavespec")
        else:
            raise ValueError("No usable wavespec available yet (SBGwaves may still be failing / low energy)")

        Te, ce = centroid_period_and_phase_speed(wavespec)
        if not np.isfinite(Te) or Te <= 0.0:
            raise ValueError("Computed centroid period Te is invalid (spectrum has zero/NaN energy)")

        # Match example.py: solve using ~NTe*Te seconds of data (here we take the most recent)
        n_total = int(zin.shape[0])
        win_len = int(round(self.config.n_te * Te * fs))
        if win_len < 1:
            win_len = n_total
        win_len = min(win_len, n_total)
        input_slice = slice(n_total - win_len, n_total)

        # Target position in local projection
        if wec_lat is None or wec_lon is None:
            x_target = 0.0
            y_target = 0.0
        else:
            if self.lat_origin is None or self.lon_origin is None:
                # This should be set by _stack_measurement_data; keep safe fallback.
                self.lat_origin = float(np.asarray(getattr(cleaned_sbg, "sbg22").GpsPos.lat)[0])
                self.lon_origin = float(np.asarray(getattr(cleaned_sbg, "sbg22").GpsPos.long)[0])

            x_target, y_target = generic_coordinate_transform(
                wec_lat,
                wec_lon,
                self.lat_origin,
                self.lon_origin,
                self.config.rotation_deg,
            )
            x_target = float(np.asarray(x_target).reshape(-1)[0])
            y_target = float(np.asarray(y_target).reshape(-1)[0])

        # Lead time from maximum sensor-to-target distance and phase speed
        dist = np.sqrt((xin[input_slice, :] - x_target) ** 2 + (yin[input_slice, :] - y_target) ** 2)
        max_target_distance = float(np.nanmax(dist))
        leadtime = float(max_target_distance / ce) if ce and np.isfinite(ce) else 0.0

        # Match example.py behavior: 1 Hz predictions with horizon determined
        # by target distance / phase speed.
        n_lead = int(np.floor(leadtime)) if np.isfinite(leadtime) and leadtime > 0.0 else 1
        if n_lead < 1:
            n_lead = 1

        # Avoid pathological message sizes.
        max_pred_points = 5000
        if n_lead > max_pred_points:
            n_lead = max_pred_points

        t_start = float(np.nanmin(tin[input_slice, :]))
        t_end = float(np.nanmax(tin[input_slice, :]))

        # example.py uses 1 Hz predictions: tpred = t_end + arange(1, n_lead+1)
        tpred = t_end + np.arange(1, n_lead + 1, dtype=float)
        xpred = np.full_like(tpred, x_target, dtype=float)
        ypred = np.full_like(tpred, y_target, dtype=float)

        # Solver mutates wavespec internals; pass copies each call.
        ws = WaveSpec()
        ws.theta = wavespec.theta.copy()
        ws.f = wavespec.f.copy()
        ws.Etheta = wavespec.Etheta.copy()

        pred_vec, recon_vec, params, solve_time = leastSquaresWavePropagation(
            zin[input_slice, :],
            uin[input_slice, :],
            vin[input_slice, :],
            tin[input_slice, :],
            xin[input_slice, :],
            yin[input_slice, :],
            tpred.reshape((-1, 1)),
            xpred.reshape((-1, 1)),
            ypred.reshape((-1, 1)),
            ws,
            A0=self.A0,
        )
        self.A0 = params.A

        prediction = np.asarray(pred_vec).reshape((tpred.size, -1), order="F")
        zout = prediction[:, 0]
        uout = prediction[:, 1] if prediction.shape[1] > 1 else np.zeros_like(zout)
        vout = prediction[:, 2] if prediction.shape[1] > 2 else np.zeros_like(zout)

        # Dense model evaluation at the target (WEC) for measurement timestamps.
        # This provides high-rate model-vs-actual comparison without changing the
        # future prediction time grid.
        t_now = np.asarray(tin[input_slice, :], dtype=float)
        if t_now.ndim == 2 and t_now.shape[1] > 0:
            t_now = t_now[:, 0]
        t_now = t_now.reshape((-1,))

        # Optionally clip dense predictions to the last N seconds of the window.
        dpw = float(self.config.dense_prediction_window) if np.isfinite(self.config.dense_prediction_window) else 0.0
        if dpw > 0.0 and t_now.size and np.all(np.isfinite(t_now)):
            t_end_now = float(np.nanmax(t_now))
            t_min_now = t_end_now - dpw
            keep = t_now >= t_min_now
            if np.any(keep):
                t_now = t_now[keep]

        dense_predictions_time = np.array([], dtype=float)
        dense_predictions_z = np.array([], dtype=float)
        dense_predictions_u = np.array([], dtype=float)
        dense_predictions_v = np.array([], dtype=float)

        try:
            kx = np.asarray(getattr(params, "kx", np.array([])), dtype=float).reshape((-1,))
            ky = np.asarray(getattr(params, "ky", np.array([])), dtype=float).reshape((-1,))
            omega = np.asarray(getattr(params, "omega", np.array([])), dtype=float).reshape((-1,))
            A = np.asarray(getattr(params, "A", np.array([])), dtype=float).reshape((-1,))
            ncomp = int(kx.size)
            if ncomp > 0 and A.size == 2 * ncomp and t_now.size > 0:
                x = np.full((t_now.size, 1), float(x_target), dtype=float)
                y = np.full((t_now.size, 1), float(y_target), dtype=float)
                t = t_now.reshape((-1, 1))

                kx_row = kx.reshape((1, -1))
                ky_row = ky.reshape((1, -1))
                om_row = omega.reshape((1, -1))

                phi = x * kx_row + y * ky_row - t * om_row
                c = np.cos(phi)
                s = np.sin(phi)

                Ac = A[:ncomp]
                As = A[ncomp:]

                z_nc = (c @ Ac) + (s @ As)
                u_nc = np.zeros_like(z_nc)
                v_nc = np.zeros_like(z_nc)

                if bool(getattr(params, "use_vel", False)):
                    kn = np.sqrt(kx * kx + ky * ky)
                    kn[kn == 0.0] = np.nan
                    cu = (kx / kn) * omega
                    cv = (ky / kn) * omega
                    cu = np.nan_to_num(cu, nan=0.0, posinf=0.0, neginf=0.0)
                    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
                    cu_row = cu.reshape((1, -1))
                    cv_row = cv.reshape((1, -1))
                    u_nc = (c * cu_row) @ Ac + (s * cu_row) @ As
                    v_nc = (c * cv_row) @ Ac + (s * cv_row) @ As

                dense_predictions_time = t_now
                dense_predictions_z = z_nc.reshape((-1,))
                dense_predictions_u = u_nc.reshape((-1,))
                dense_predictions_v = v_nc.reshape((-1,))
        except Exception:
            # Best-effort: dense predictions are optional.
            pass

        nbuoys = int(zin.shape[1])
        reconstruction = np.asarray(recon_vec).reshape((win_len, -1), order="F")
        zr = reconstruction[:, 0:nbuoys]
        ur = reconstruction[:, nbuoys : 2 * nbuoys]
        vr = reconstruction[:, 2 * nbuoys : 3 * nbuoys]

        results.update(
            {
                "cleaned_sbg": cleaned_sbg,
                "swift_structs": swift_structs,
                "wavespec": wavespec,
                "Te": float(Te),
                "ce": float(ce),
                "fs": float(fs),
                "window_start_time": t_start,
                "window_end_time": t_end,
                "n_samples": int(win_len),
                "n_buoys": int(nbuoys),
                "lat_origin": self.lat_origin,
                "lon_origin": self.lon_origin,
                "rotation_deg": float(self.config.rotation_deg),
                "x_target": float(x_target),
                "y_target": float(y_target),
                "max_target_distance": max_target_distance,
                "leadtime": float(leadtime),
                "t_pred": tpred,
                "z_pred": zout,
                "u_pred": uout,
                "v_pred": vout,
                "dense_predictions_time": dense_predictions_time,
                "dense_predictions_z": dense_predictions_z,
                "dense_predictions_u": dense_predictions_u,
                "dense_predictions_v": dense_predictions_v,
                "t_meas": tin[input_slice, :],
                "x_meas": xin[input_slice, :],
                "y_meas": yin[input_slice, :],
                "z_meas": zin[input_slice, :],
                "u_meas": uin[input_slice, :],
                "v_meas": vin[input_slice, :],
                "z_recon": zr,
                "u_recon": ur,
                "v_recon": vr,
                "params": params,
                "solve_time": float(solve_time),
            }
        )

        return results

    def _stack_measurement_data(self, cleaned_sbg: SWIFTArray) -> tuple[np.ndarray, ...]:
        sbgs: list[SBGData] = []
        for swift_num in range(22, 26):
            sbg = getattr(cleaned_sbg, f"sbg{swift_num}", None)
            if sbg is not None and len(sbg.ShipMotion.heave) > 0:
                sbgs.append(sbg)

        if not sbgs:
            raise ValueError("No valid SBG data found")

        if self.lat_origin is None or self.lon_origin is None:
            self.lat_origin = float(np.asarray(sbgs[0].GpsPos.lat, dtype=float)[0])
            self.lon_origin = float(np.asarray(sbgs[0].GpsPos.long, dtype=float)[0])

        return load_raw_arrays_from_sbg(sbgs, self.lat_origin, self.lon_origin, self.config.rotation_deg)

    def _build_averaged_wavespec(self, swift_structs: SWIFTArray) -> WaveSpec:
        swifts = []
        for swift_name in ("swift22", "swift23", "swift24", "swift25"):
            swift_data = getattr(swift_structs, swift_name, None)
            if swift_data is not None and swift_data.wavespectra.energy.size > 0:
                swifts.append(swift_data)

        if len(swifts) == 0:
            ws = WaveSpec()
            ws.theta = np.array([])
            ws.f = np.array([])
            ws.Etheta = np.array([[]])
            return ws

        return build_wavespec_from_swifts(swifts, recip=True)
