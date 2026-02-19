"""Utility functions for wave prediction and coordinate transformations.

Reusable functions extracted from example.py for use in the_next_wave_node.py and other modules.
"""

import numpy as np
import utm

from .swift import SWIFTArray, WaveSpec
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra


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

        x, y = generic_coordinate_transform(lat, lon, latorigin, lonorigin, rotation)

        zin.append(z)
        uin.append(u)
        vin.append(v)
        tin.append(ztime)
        xin.append(x)
        yin.append(y)

    zin = np.column_stack(zin)
    uin = np.column_stack(uin)
    vin = np.column_stack(vin)
    tin = np.column_stack(tin)
    xin = np.column_stack(xin)
    yin = np.column_stack(yin)

    zin = -zin  # SBG mounted upside-down on SWIFT

    fs = 1.0 / float(np.nanmean(np.diff(tin, axis=0)))
    return zin, uin, vin, tin, xin, yin, fs
