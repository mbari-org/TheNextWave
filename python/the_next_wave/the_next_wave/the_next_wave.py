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


class TheNextWave:
    """Algorithmic wave prediction core (no ROS dependencies)."""

    def __init__(self, config: TheNextWaveConfig | None = None, logger: Any | None = None):
        self.config = config or TheNextWaveConfig()
        self.logger = logger

        self.A0: np.ndarray | None = None
        self.lat_origin: float | None = None
        self.lon_origin: float | None = None
        self._last_wavespec: WaveSpec | None = None

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

        wavespec_new = self._build_averaged_wavespec(swift_structs)
        if self._wavespec_is_usable(wavespec_new):
            wavespec = wavespec_new
            self._last_wavespec = wavespec_new
        elif self._wavespec_is_usable(self._last_wavespec):
            wavespec = self._last_wavespec
            if self.logger is not None:
                self.logger.warn("No usable wavespec from current window; reusing last valid wavespec")
        else:
            raise ValueError("No usable wavespec available yet (SBGwaves may still be failing / low energy)")

        zin, uin, vin, tin, xin, yin, fs = self._stack_measurement_data(cleaned_sbg)
        fs = float(fs)
        if not np.isfinite(fs) or fs <= 0.0:
            if self.logger is not None:
                self.logger.warn(f"Invalid fs={fs}; falling back to expected_fs={self.config.expected_fs}")
            fs = float(self.config.expected_fs)

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

        n_lead = int(np.floor(leadtime))
        if n_lead < 1:
            n_lead = 1

        t_start = float(np.nanmin(tin[input_slice, :]))
        t_end = float(np.nanmax(tin[input_slice, :]))

        # example.py: tpred = t_end + arange(1, n_lead+1)
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
