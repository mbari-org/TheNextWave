"""Utility functions for wave prediction and coordinate transformations.

Reusable functions extracted from example.py for use in the_next_wave_node.py and other modules.
"""

import numpy as np
import utm

from .swift import SWIFTArray, WaveSpec
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra


def as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape((-1,))


def select_sbg_burst_struct(sbg_data, prefer_longest: bool = True):
    """Select a single raw SBG burst struct from a MATLAB-loaded `sbgData`."""
    try:
        size = int(getattr(sbg_data, "size", 1))
    except Exception:
        size = 1

    if size <= 1:
        return sbg_data

    if not prefer_longest:
        try:
            return sbg_data[0]
        except Exception:
            return sbg_data

    best = None
    best_n = -1
    for i in range(size):
        try:
            cand = sbg_data[i]
            t = as_1d(getattr(getattr(cand, "ShipMotion"), "time_stamp"))
            n = int(t.size)
        except Exception:
            continue

        if n > best_n:
            best_n = n
            best = cand

    return best if best is not None else sbg_data


def load_raw_sbg_arrays(
    sbg,
    *,
    start_index: int = 0,
    end_index: int | None = None,
):
    """Extract aligned raw arrays from one SBG burst struct.

    Returns arrays in the raw SWIFT / SBG coordinate conventions.

    Outputs
    -------
    t_us : ndarray (N,)
        Raw SBG time_stamp values (microseconds)
    heave : ndarray (N,)
    vel_e : ndarray (N,)
    vel_n : ndarray (N,)
    lat : ndarray (N,)
    lon : ndarray (N,)
    """
    t_us = as_1d(getattr(getattr(sbg, "ShipMotion"), "time_stamp"))
    heave = as_1d(getattr(getattr(sbg, "ShipMotion"), "heave"))
    vel_e = as_1d(getattr(getattr(sbg, "GpsVel"), "vel_e"))
    vel_n = as_1d(getattr(getattr(sbg, "GpsVel"), "vel_n"))
    lat = as_1d(getattr(getattr(sbg, "GpsPos"), "lat"))
    lon = as_1d(getattr(getattr(sbg, "GpsPos"), "long"))

    n = int(min(t_us.size, heave.size, vel_e.size, vel_n.size, lat.size, lon.size))
    if n <= 0:
        raise ValueError("No usable samples in SBG burst")

    start = max(0, int(start_index))
    stop = n if end_index is None else int(end_index)
    if stop < 0 or stop > n:
        stop = n
    if start >= stop:
        raise ValueError(f"Invalid start/end: {start}..{stop} (n={n})")

    sl = slice(start, stop)
    return (
        np.asarray(t_us[sl]),
        np.asarray(heave[sl]),
        np.asarray(vel_e[sl]),
        np.asarray(vel_n[sl]),
        np.asarray(lat[sl]),
        np.asarray(lon[sl]),
    )


def generic_coordinate_transform(lat, lon, lat0, lon0, rotation_deg):
    """
    Convert lat/lon coordinates to local x/y using UTM with rotation.
    
    Parameters
    ----------
    lat, lon : array-like
        Latitude and longitude in degrees
    lat0, lon0 : float
        Reference latitude and longitude in degrees
    rotation_deg : float
        Rotation angle in degrees (clockwise positive)
    
    Returns
    -------
    x, y : ndarray
        Local Cartesian coordinates (east, north) in meters, rotated
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    e0, n0, zone_num, zone_let = utm.from_latlon(float(lat0), float(lon0))

    e = np.empty_like(lat, dtype=float)
    n = np.empty_like(lat, dtype=float)
    for i in range(lat.size):
        ei, ni, zn, zl = utm.from_latlon(float(lat.flat[i]), float(lon.flat[i]))
        e.flat[i] = ei
        n.flat[i] = ni

    dx = e - e0
    dy = n - n0

    ang = np.deg2rad(rotation_deg)
    c = np.cos(ang)
    s = np.sin(ang)

    x = dx * c + dy * s
    y = -dx * s + dy * c
    return x, y


def generic_coordinate_transform_inverse(x, y, lat0, lon0, rotation_deg):
    """
    Convert local x/y coordinates back to lat/lon (inverse of generic_coordinate_transform).
    
    Parameters
    ----------
    x, y : array-like
        Local Cartesian coordinates (east, north) in meters
    lat0, lon0 : float
        Reference latitude and longitude in degrees
    rotation_deg : float
        Rotation angle in degrees (clockwise positive)
    
    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude in degrees
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Undo rotation
    ang = np.deg2rad(rotation_deg)
    c = np.cos(ang)
    s = np.sin(ang)
    
    dx = x * c - y * s
    dy = x * s + y * c
    
    # Get reference UTM coordinates
    e0, n0, zone_num, zone_let = utm.from_latlon(float(lat0), float(lon0))
    
    e = dx + e0
    n = dy + n0
    
    # Convert back to lat/lon
    lat = np.empty_like(x, dtype=float)
    lon = np.empty_like(x, dtype=float)
    for i in range(x.size):
        # print(f'Converting back to lat/lon for point {i}: e={e.flat[i]}, n={n.flat[i]}, zone={zone_num}{zone_let}, ref={lat0},{lon0} rotation={rotation_deg} deg')
        lat.flat[i], lon.flat[i] = utm.to_latlon(float(e.flat[i]), float(n.flat[i]), zone_num, zone_let)
    
    return lat, lon


def build_wavespec_from_swifts(swifts, recip=True):
    """
    Build averaged directional spectrum from multiple SWIFT structures.
    
    Parameters
    ----------
    swifts : list-like of SWIFTData
        SWIFT structures with computed wave spectra
    recip : bool, optional
        Whether to use reciprocal direction convention (default: True)
    
    Returns
    -------
    ws : WaveSpec
        Averaged directional spectrum across all SWIFT buoys
    """
    Ethetas = []
    theta0 = None
    f0 = None

    for sw in swifts:
        Etheta, theta, E, f, _, spread, spread2, _ = SWIFTdirectionalspectra(sw, plotflag=False, recip=recip)

        Etheta = np.asarray(Etheta, dtype=float)
        f = np.asarray(f, dtype=float).ravel()
        theta = np.asarray(theta, dtype=float).ravel()

        # Accept only finite, non-negative, non-zero-energy spectra.
        if Etheta.size == 0 or f.size == 0 or theta.size == 0:
            continue
        if not np.isfinite(Etheta).any():
            continue
        energy_sum = float(np.nansum(Etheta[np.isfinite(Etheta)]))
        if not np.isfinite(energy_sum) or energy_sum <= 0.0:
            continue

        Ethetas.append(Etheta)
        if theta0 is None:
            theta0 = theta.copy()
            f0 = f.copy()

    ws = WaveSpec()
    ws.theta = theta0
    ws.f = f0
    if len(Ethetas) == 0:
        ws.Etheta = np.array([[]], dtype=float)
        ws.theta = np.array([], dtype=float)
        ws.f = np.array([], dtype=float)
        return ws

    ws.Etheta = np.nanmean(np.stack(Ethetas, axis=2), axis=2)
    return ws


def centroid_period_and_phase_speed(ws):
    """
    Compute centroid period and phase speed from wave spectrum.
    
    Parameters
    ----------
    ws : WaveSpec
        Wave spectrum with Etheta (directional spectrum), f (frequencies)
    
    Returns
    -------
    Te : float
        Centroid (energy-weighted) period [s]
    ce : float
        Phase speed [m/s]
    """
    Etheta = np.asarray(ws.Etheta)
    f = np.asarray(ws.f)

    if f.size == 0 or Etheta.size == 0:
        return float("nan"), float("nan")

    if Etheta.shape[0] != f.size and Etheta.shape[1] == f.size:
        Etheta = Etheta.T

    Ef = np.sum(Etheta, axis=1)
    denom = float(np.sum(Ef * f))
    numer = float(np.sum(Etheta))
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(numer):
        return float("nan"), float("nan")

    Te = numer / denom
    ce = 9.8 * Te / (2.0 * 3.14)
    return float(Te), float(ce)


def load_raw_arrays_from_sbg(sbgs, *args):
    """
    Extract and stack raw SBG data from multiple buoys.
    
    Converts lat/lon to local x/y coordinates, stacks data from all buoys,
    negates heave (for upside-down SBG mount), and computes sampling rate.
    
    Supports both call signatures (for compatibility with `example.py`):

    1) load_raw_arrays_from_sbg(sbgs, latorigin, lonorigin, rotation)
    2) load_raw_arrays_from_sbg(sbgs, skipwarmup, burstend, latorigin, lonorigin, rotation)

    Parameters
    ----------
    sbgs : list of SBGData
        Raw SBG data structures from buoys
    skipwarmup : int, optional
        Number of initial samples to skip (MATLAB "skipwarmup")
    burstend : int, optional
        End sample index (exclusive) for truncation (MATLAB "burstend")
    latorigin, lonorigin : float
        Reference latitude and longitude in degrees
    rotation : float
        Rotation angle in degrees for coordinate transformation
    
    Returns
    -------
    zin, uin, vin : ndarray (N, nbuoys)
        Stacked heave and velocity arrays (z negated for upside-down mount)
    tin, xin, yin : ndarray (N, nbuoys)
        Stacked time and position arrays
    fs : float
        Sampling rate [Hz]
    """
    # Parse args for backward/forward compatibility.
    if len(args) == 3:
        skipwarmup = None
        burstend = None
        latorigin, lonorigin, rotation = args
    elif len(args) == 5:
        skipwarmup, burstend, latorigin, lonorigin, rotation = args
    else:
        raise TypeError(
            "load_raw_arrays_from_sbg expects (sbgs, lat0, lon0, rotation) or "
            "(sbgs, skipwarmup, burstend, lat0, lon0, rotation)"
        )

    zin = []
    uin = []
    vin = []
    tin = []
    xin = []
    yin = []

    # Raw SBG fields can occasionally be off-by-one across buoys (or even within
    # a buoy after cleaning / truncation). Additionally, when ingesting streams
    # in realtime, each buoy can be a sample ahead/behind at the moment we
    # snapshot. We align by timestamp using nearest-neighbor matching (no
    # interpolation) to produce a rectangular matrix for the solver.
    per_buoy = []

    for sbg in sbgs:
        z = np.asarray(sbg.ShipMotion.heave, dtype=float)
        ztime = np.asarray(sbg.ShipMotion.time_stamp, dtype=float) / 1e6

        u = np.asarray(sbg.GpsVel.vel_e, dtype=float)
        v = np.asarray(sbg.GpsVel.vel_n, dtype=float)

        lat = np.asarray(sbg.GpsPos.lat, dtype=float)
        lon = np.asarray(sbg.GpsPos.long, dtype=float)

        if skipwarmup is not None or burstend is not None:
            start = int(skipwarmup) if skipwarmup is not None else 0
            stop = int(burstend) if burstend is not None else None
            z = z[start:stop]
            ztime = ztime[start:stop]
            u = u[start:stop]
            v = v[start:stop]
            lat = lat[start:stop]
            lon = lon[start:stop]

        n_local = int(min(z.size, ztime.size, u.size, v.size, lat.size, lon.size))
        if n_local <= 0:
            continue
        if n_local != z.size:
            z = z[:n_local]
        if n_local != ztime.size:
            ztime = ztime[:n_local]
        if n_local != u.size:
            u = u[:n_local]
        if n_local != v.size:
            v = v[:n_local]
        if n_local != lat.size:
            lat = lat[:n_local]
        if n_local != lon.size:
            lon = lon[:n_local]

        x, y = generic_coordinate_transform(lat, lon, latorigin, lonorigin, rotation)

        x = np.asarray(x, dtype=float).reshape((-1,))
        y = np.asarray(y, dtype=float).reshape((-1,))
        ztime = np.asarray(ztime, dtype=float).reshape((-1,))
        z = np.asarray(z, dtype=float).reshape((-1,))
        u = np.asarray(u, dtype=float).reshape((-1,))
        v = np.asarray(v, dtype=float).reshape((-1,))

        n_local2 = int(min(ztime.size, z.size, u.size, v.size, x.size, y.size))
        if n_local2 <= 0:
            continue
        if n_local2 != ztime.size:
            ztime = ztime[:n_local2]
        if n_local2 != z.size:
            z = z[:n_local2]
        if n_local2 != u.size:
            u = u[:n_local2]
        if n_local2 != v.size:
            v = v[:n_local2]
        if n_local2 != x.size:
            x = x[:n_local2]
        if n_local2 != y.size:
            y = y[:n_local2]

        per_buoy.append({"t": ztime, "z": z, "u": u, "v": v, "x": x, "y": y})

    if not per_buoy:
        raise ValueError("No usable samples in SBG inputs")

    # Choose a reference time grid. In the main pipeline sbgs are ordered (swift22..)
    # so using the first buoy provides stable behavior across runs.
    t_ref = np.asarray(per_buoy[0]["t"], dtype=float).reshape((-1,))
    if t_ref.size < 2:
        raise ValueError("Not enough reference samples for alignment")

    # Estimate a reasonable matching tolerance from the reference sampling.
    dt_ref = np.diff(t_ref)
    dt_ref = dt_ref[np.isfinite(dt_ref) & (dt_ref > 0.0)]
    if dt_ref.size == 0:
        raise ValueError("Invalid reference timestamps for alignment")
    period_ref = float(np.nanmedian(dt_ref))
    if not np.isfinite(period_ref) or period_ref <= 0.0:
        raise ValueError("Invalid reference sampling period")
    # Nearest-neighbor without interpolation: accept matches within ~half a sample.
    tol = 0.55 * period_ref

    # For each buoy, find the nearest index in its time vector for each ref time.
    # Keep only those ref times that match all buoys within tolerance.
    indices_by_buoy = []
    valid = np.ones((t_ref.size,), dtype=bool)
    for b in per_buoy:
        t = np.asarray(b["t"], dtype=float).reshape((-1,))
        if t.size == 0:
            valid[:] = False
            indices_by_buoy.append(np.zeros_like(t_ref, dtype=int))
            continue

        j = np.searchsorted(t, t_ref, side="left")
        j0 = np.clip(j - 1, 0, t.size - 1)
        j1 = np.clip(j, 0, t.size - 1)

        d0 = np.abs(t[j0] - t_ref)
        d1 = np.abs(t[j1] - t_ref)
        use1 = d1 < d0
        idx = np.where(use1, j1, j0).astype(int)

        # Enforce non-decreasing indices to avoid occasional backwards picks.
        idx = np.maximum.accumulate(idx)

        d = np.abs(t[idx] - t_ref)
        valid &= np.isfinite(d) & (d <= tol)
        indices_by_buoy.append(idx)

    t_common = t_ref[valid]
    if t_common.size < 2:
        raise ValueError("Not enough aligned samples across buoys")

    # Build aligned per-buoy vectors at the matched indices; assign common time grid.
    for b, idx in zip(per_buoy, indices_by_buoy, strict=True):
        ii = np.asarray(idx, dtype=int)[valid]
        zin.append(np.asarray(b["z"], dtype=float)[ii])
        uin.append(np.asarray(b["u"], dtype=float)[ii])
        vin.append(np.asarray(b["v"], dtype=float)[ii])
        xin.append(np.asarray(b["x"], dtype=float)[ii])
        yin.append(np.asarray(b["y"], dtype=float)[ii])
        tin.append(np.asarray(t_common, dtype=float))

    zin = np.column_stack(zin)
    uin = np.column_stack(uin)
    vin = np.column_stack(vin)
    tin = np.column_stack(tin)
    xin = np.column_stack(xin)
    yin = np.column_stack(yin)

    zin = -zin  # SBG mounted upside-down on SWIFT

    fs = 1.0 / float(np.nanmean(np.diff(tin, axis=0)))
    return zin, uin, vin, tin, xin, yin, fs
