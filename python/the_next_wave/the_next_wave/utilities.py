"""
Utilities for the_next_wave.

Utility functions for wave prediction and coordinate transformations.

Frame conventions used throughout the Python port
-----------------------------------------------

Unless explicitly documented otherwise, this code assumes a local Cartesian frame where:

- $x$ is East (meters)
- $y$ is North (meters)
- $z$ (heave / eta) is up-positive (meters)
- $u$ is East (m/s)
- $v$ is North (m/s)

Important: `rotation_deg` rotates the *projected x/y coordinates* (clockwise-positive).
The current pipeline does not automatically rotate (u,v) when `rotation_deg != 0`, so
keep `rotation_deg = 0` unless you also rotate velocities consistently.

Reusable functions extracted from example.py for use in the_next_wave_node.py and other modules.
"""

import numpy as np
import utm

from .swift import WaveSpec
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra

try:
    from pyproj import CRS, Transformer

    HAS_PYPROJ = True
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None
    HAS_PYPROJ = False

UTM_TRANSFORMER_CACHE: dict[tuple[int, bool], tuple[object, object]] = {}


def get_pyproj_utm_transformers(zone_num: int, northern: bool):
    """
    Return (fwd, inv) pyproj Transformers for WGS84<->UTM zone.

    Uses EPSG:
      - 326xx for northern hemisphere
      - 327xx for southern hemisphere
    """
    key = (int(zone_num), bool(northern))
    cached = UTM_TRANSFORMER_CACHE.get(key)
    if cached is not None:
        return cached

    if not HAS_PYPROJ:
        raise RuntimeError('pyproj not available')

    crs_ll = CRS.from_epsg(4326)
    epsg = (32600 + int(zone_num)) if northern else (32700 + int(zone_num))
    crs_utm = CRS.from_epsg(epsg)
    fwd = Transformer.from_crs(crs_ll, crs_utm, always_xy=True)
    inv = Transformer.from_crs(crs_utm, crs_ll, always_xy=True)
    UTM_TRANSFORMER_CACHE[key] = (fwd, inv)
    return fwd, inv


def as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape((-1,))


def select_sbg_burst_struct(sbg_data, prefer_longest: bool = True):
    """Select a single raw SBG burst struct from a MATLAB-loaded `sbgData`."""
    try:
        size = int(getattr(sbg_data, 'size', 1))
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
            t = as_1d(getattr(getattr(cand, 'ShipMotion'), 'time_stamp'))
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
    """
    Extract aligned raw arrays from one SBG burst struct.

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
    t_us = as_1d(getattr(getattr(sbg, 'ShipMotion'), 'time_stamp'))
    heave = as_1d(getattr(getattr(sbg, 'ShipMotion'), 'heave'))
    vel_e = as_1d(getattr(getattr(sbg, 'GpsVel'), 'vel_e'))
    vel_n = as_1d(getattr(getattr(sbg, 'GpsVel'), 'vel_n'))
    lat = as_1d(getattr(getattr(sbg, 'GpsPos'), 'lat'))
    lon = as_1d(getattr(getattr(sbg, 'GpsPos'), 'long'))

    n = int(min(t_us.size, heave.size, vel_e.size, vel_n.size, lat.size, lon.size))
    if n <= 0:
        raise ValueError('No usable samples in SBG burst')

    start = max(0, int(start_index))
    stop = n if end_index is None else int(end_index)
    if stop < 0 or stop > n:
        stop = n
    if start >= stop:
        raise ValueError(f'Invalid start/end: {start}..{stop} (n={n})')

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
        Rotation angle in degrees (clockwise positive). When `rotation_deg == 0`, the
        returned x/y are East/North offsets in meters. When nonzero, the returned x/y are
        in a rotated local frame (and are no longer East/North).

    Returns
    -------
    x, y : ndarray
        Local Cartesian coordinates (meters). With `rotation_deg == 0`, x=East and y=North.
        With `rotation_deg != 0`, x/y are rotated axes.

    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    e0, n0, zone_num, zone_let = utm.from_latlon(float(lat0), float(lon0))

    # Fast path: vectorized pyproj transform (dramatically faster than per-sample
    # Python loop over utm.from_latlon). Fall back to utm if pyproj unavailable.
    e = np.empty_like(lat, dtype=float)
    n = np.empty_like(lat, dtype=float)
    if HAS_PYPROJ:
        try:
            northern = bool(float(lat0) >= 0.0)
            fwd, inv_unused = get_pyproj_utm_transformers(int(zone_num), northern)
            lon_flat = np.asarray(lon, dtype=float).reshape((-1,))
            lat_flat = np.asarray(lat, dtype=float).reshape((-1,))
            e_flat, n_flat = fwd.transform(lon_flat, lat_flat)
            e[:] = np.asarray(e_flat, dtype=float).reshape(lat.shape)
            n[:] = np.asarray(n_flat, dtype=float).reshape(lat.shape)
        except Exception:
            for i in range(lat.size):
                ei, ni, zn_unused, zl_unused = utm.from_latlon(
                    float(lat.flat[i]), float(lon.flat[i])
                )
                e.flat[i] = ei
                n.flat[i] = ni
    else:
        for i in range(lat.size):
            ei, ni, zn_unused, zl_unused = utm.from_latlon(
                float(lat.flat[i]), float(lon.flat[i])
            )
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

    e = e0 + dx
    n = n0 + dy

    lat = np.empty_like(x, dtype=float)
    lon = np.empty_like(x, dtype=float)
    if HAS_PYPROJ:
        try:
            northern = bool(float(lat0) >= 0.0)
            fwd_unused, inv = get_pyproj_utm_transformers(int(zone_num), northern)
            e_flat = np.asarray(e, dtype=float).reshape((-1,))
            n_flat = np.asarray(n, dtype=float).reshape((-1,))
            lon_flat, lat_flat = inv.transform(e_flat, n_flat)
            lat[:] = np.asarray(lat_flat, dtype=float).reshape(x.shape)
            lon[:] = np.asarray(lon_flat, dtype=float).reshape(x.shape)
        except Exception:
            for i in range(x.size):
                lat.flat[i], lon.flat[i] = utm.to_latlon(
                    float(e.flat[i]),
                    float(n.flat[i]),
                    zone_num,
                    zone_let,
                )
    else:
        for i in range(x.size):
            lat.flat[i], lon.flat[i] = utm.to_latlon(
                float(e.flat[i]),
                float(n.flat[i]),
                zone_num,
                zone_let,
            )

    return lat, lon


def build_wavespec_from_swifts(
    swifts,
    recip: bool = False,
    *,
    mem_moment_cap: float | None = None,
):
    """
    Build averaged directional spectrum from multiple SWIFT structures.

    Parameters
    ----------
    swifts : list-like of SWIFTData
        SWIFT structures with computed wave spectra
    recip : bool, optional
        Pass-through of MATLAB `SWIFTdirectionalspectra(..., recip=...)` behavior.

    mem_moment_cap : float or None, optional
        Optional cap applied to MEM directional moments to reduce numerical
        instability in low-energy bands.

        Important: in the MATLAB implementation this flag is asymmetric:
        - It flips the *Etheta/theta* axis when `recip=True`.
        - It flips the moment-derived `dir` output when `recip=False`.

        In this repo, `WaveSpec.theta` is treated as compass degrees True,
        direction waves are coming FROM (and any propagation arrow uses
        TO = FROM + 180, wrapped to 0–360). Callers should choose `recip` so
        that `WaveSpec.theta` matches that convention.

    Returns
    -------
    ws : WaveSpec
        Averaged directional spectrum across all SWIFT buoys

    """
    Ethetas = []
    theta0 = None
    f0 = None

    for sw in swifts:
        Etheta, theta, E, f, unused_1, spread, spread2, unused_2 = SWIFTdirectionalspectra(
            sw,
            plotflag=False,
            recip=recip,
            mem_moment_cap=mem_moment_cap,
        )

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
        return float('nan'), float('nan')

    if Etheta.shape[0] != f.size and Etheta.shape[1] == f.size:
        Etheta = Etheta.T

    Ef = np.sum(Etheta, axis=1)
    denom = float(np.sum(Ef * f))
    numer = float(np.sum(Etheta))
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(numer):
        return float('nan'), float('nan')

    Te = numer / denom
    ce = 9.8 * Te / (2.0 * 3.14)
    return float(Te), float(ce)


def wrap_360(deg: float) -> float:
    if not np.isfinite(deg):
        return float('nan')
    out = float(deg) % 360.0
    if out < 0.0:
        out += 360.0
    return out


def deg_to_compass_16(deg: float) -> str:
    if not np.isfinite(deg):
        return 'nan'
    labels = [
        'N',
        'NNE',
        'NE',
        'ENE',
        'E',
        'ESE',
        'SE',
        'SSE',
        'S',
        'SSW',
        'SW',
        'WSW',
        'W',
        'WNW',
        'NW',
        'NNW',
    ]
    idx = int(np.floor((wrap_360(deg) + 11.25) / 22.5)) % 16
    return labels[idx]


def bulk_wave_params_from_1d_spectrum(
    f_hz: np.ndarray,
    E_m2_per_hz: np.ndarray,
) -> dict:
    """
    Compute a few standard bulk parameters from a 1D spectrum.

    Assumes E(f) is in units of m^2/Hz and f is in Hz.
    """
    f = np.asarray(f_hz, dtype=float).ravel()
    E = np.asarray(E_m2_per_hz, dtype=float).ravel()
    ok = np.isfinite(f) & np.isfinite(E) & (f > 0.0) & (E >= 0.0)
    if np.count_nonzero(ok) < 2:
        return {
            'Hs_m': float('nan'),
            'Tp_s': float('nan'),
            'fp_hz': float('nan'),
            'Tm01_s': float('nan'),
            'Tm02_s': float('nan'),
            'm0': float('nan'),
        }

    f = f[ok]
    E = E[ok]
    order = np.argsort(f)
    f = f[order]
    E = E[order]

    # NumPy 2.x: np.trapz was removed; use np.trapezoid instead.
    m0 = float(np.trapezoid(E, f))
    m1 = float(np.trapezoid(f * E, f))
    m2 = float(np.trapezoid((f**2) * E, f))
    Hs = 4.0 * np.sqrt(m0) if np.isfinite(m0) and m0 > 0.0 else float('nan')

    i_peak = int(np.nanargmax(E)) if np.isfinite(E).any() else 0
    fp = float(f[i_peak]) if f.size else float('nan')
    Tp = 1.0 / fp if np.isfinite(fp) and fp > 0.0 else float('nan')

    Tm01 = (m0 / m1) if np.isfinite(m0) and np.isfinite(m1) and m1 > 0.0 else float('nan')
    Tm02 = (
        np.sqrt(m0 / m2)
        if np.isfinite(m0) and np.isfinite(m2) and m0 > 0.0 and m2 > 0.0
        else float('nan')
    )

    return {
        'Hs_m': float(Hs),
        'Tp_s': float(Tp),
        'fp_hz': float(fp),
        'Tm01_s': float(Tm01),
        'Tm02_s': float(Tm02),
        'm0': float(m0),
    }


def bulk_dir_params_from_Etheta(
    f_hz: np.ndarray,
    theta_deg: np.ndarray,
    Etheta: np.ndarray,
) -> dict:
    """
    Compute peak/mean direction and a simple spread estimate from Etheta.

    Uses circular moments of the energy-weighted directional distribution.
    Theta is assumed in *compass degrees True* (0°=North, 90°=East), and represents
    the direction waves are coming FROM (as produced by SWIFTdirectionalspectra).
    """
    f = np.asarray(f_hz, dtype=float).ravel()
    theta = np.asarray(theta_deg, dtype=float).ravel()
    S = np.asarray(Etheta, dtype=float)

    if f.size == 0 or theta.size == 0 or S.size == 0:
        return {'Dp_deg': float('nan'), 'Dm_deg': float('nan'), 'spreadp_deg': float('nan')}

    # Expected SWIFTdirectionalspectra shape: (nfreq, ntheta)
    if S.shape == (theta.size, f.size):
        S = S.T
    if S.shape[0] != f.size or S.shape[1] != theta.size:
        return {'Dp_deg': float('nan'), 'Dm_deg': float('nan'), 'spreadp_deg': float('nan')}

    dtheta_deg = float(np.nanmedian(np.diff(np.sort(theta)))) if theta.size > 1 else 1.0
    # MEM_directionalestimator normalizes using radians (see MATLAB: tot=sum(S)*dtheta*dr).
    # Therefore Etheta is per-radian, and integrations over theta must use dtheta_rad.
    dtheta = float(dtheta_deg * (np.pi / 180.0))
    theta_rad = np.deg2rad(theta)

    # Nautical/compass directions are measured clockwise from North.
    # Represent each direction as an EN unit vector:
    #   east  = sin(theta)
    #   north = cos(theta)
    east = np.sin(theta_rad)
    north = np.cos(theta_rad)

    # 1D energy spectrum (m^2/Hz): integrate directional density over theta
    E = np.sum(S * dtheta, axis=1)
    if not np.isfinite(E).any() or float(np.nansum(E)) <= 0.0:
        return {'Dp_deg': float('nan'), 'Dm_deg': float('nan'), 'spreadp_deg': float('nan')}

    i_peak = int(np.nanargmax(E))

    # Peak-direction moments
    w_peak = S[i_peak, :] * dtheta
    denom_peak = float(np.nansum(w_peak))
    if not np.isfinite(denom_peak) or denom_peak <= 0.0:
        Dp = float('nan')
        spreadp = float('nan')
    else:
        mean_e = float(np.nansum(w_peak * east) / denom_peak)
        mean_n = float(np.nansum(w_peak * north) / denom_peak)
        # Heading in compass degrees: atan2(east, north)
        Dp = wrap_360(np.rad2deg(np.arctan2(mean_e, mean_n)))
        R = float(np.hypot(mean_e, mean_n))
        spreadp = float(np.rad2deg(np.sqrt(max(0.0, 2.0 * (1.0 - R)))))

    # Mean direction across all frequencies, energy-weighted
    df = np.diff(f)
    df = np.concatenate([df, df[-1:]]) if df.size else np.array([1.0])
    df = df.reshape((-1, 1))
    w_all = S * dtheta * df
    denom_all = float(np.nansum(w_all))
    if not np.isfinite(denom_all) or denom_all <= 0.0:
        Dm = float('nan')
    else:
        mean_e = float(np.nansum(w_all * east.reshape((1, -1))) / denom_all)
        mean_n = float(np.nansum(w_all * north.reshape((1, -1))) / denom_all)
        Dm = wrap_360(np.rad2deg(np.arctan2(mean_e, mean_n)))

    return {'Dp_deg': float(Dp), 'Dm_deg': float(Dm), 'spreadp_deg': float(spreadp)}


def bulk_wave_params_from_wavespec(ws) -> dict:
    """Compute bulk parameters from a WaveSpec (directional spectrum)."""
    f = np.asarray(getattr(ws, 'f', np.array([])), dtype=float).ravel()
    theta = np.asarray(getattr(ws, 'theta', np.array([])), dtype=float).ravel()
    S = np.asarray(getattr(ws, 'Etheta', np.array([[]])), dtype=float)
    if f.size == 0 or theta.size == 0 or S.size == 0:
        return {
            **bulk_wave_params_from_1d_spectrum(np.array([]), np.array([])),
            'Dp_deg': float('nan'),
            'Dm_deg': float('nan'),
            'spreadp_deg': float('nan'),
        }

    if S.shape == (theta.size, f.size):
        S = S.T
    dtheta_deg = float(np.nanmedian(np.diff(np.sort(theta)))) if theta.size > 1 else 1.0
    dtheta = float(dtheta_deg * (np.pi / 180.0))
    E = np.sum(S * dtheta, axis=1)

    out = dict(bulk_wave_params_from_1d_spectrum(f, E))
    out.update(bulk_dir_params_from_Etheta(f, theta, S))
    return out


def format_bulk_wave_params(params: dict, label: str = '') -> str:
    prefix = f'{label}: ' if label else ''

    Hs = params.get('Hs_m', float('nan'))
    Tp = params.get('Tp_s', float('nan'))
    Tm01 = params.get('Tm01_s', float('nan'))
    Tm02 = params.get('Tm02_s', float('nan'))
    Dp = params.get('Dp_deg', float('nan'))
    Dm = params.get('Dm_deg', float('nan'))
    spreadp = params.get('spreadp_deg', float('nan'))

    parts = [
        f'Hs={Hs:.2f} m' if np.isfinite(Hs) else 'Hs=nan',
        f'Tp={Tp:.2f} s' if np.isfinite(Tp) else 'Tp=nan',
        f'Tm01={Tm01:.2f} s' if np.isfinite(Tm01) else 'Tm01=nan',
        f'Tm02={Tm02:.2f} s' if np.isfinite(Tm02) else 'Tm02=nan',
    ]

    if np.isfinite(Dp):
        parts.append(f'Dp={Dp:.1f}° ({deg_to_compass_16(Dp)})')
    else:
        parts.append('Dp=nan')

    if np.isfinite(Dm):
        parts.append(f'Dm={Dm:.1f}° ({deg_to_compass_16(Dm)})')
    else:
        parts.append('Dm=nan')

    if np.isfinite(spreadp):
        parts.append(f'spread@peak={spreadp:.1f}°')

    return prefix + ', '.join(parts)


def load_raw_arrays_from_sbg(sbgs, *args, flip_z_sign: bool = True):
    """
    Extract and stack raw SBG data from multiple buoys.

    Converts lat/lon to local x/y coordinates, stacks data from all buoys,
    optionally negates heave (for upside-down SBG mount), and computes sampling rate.

    Notes on frames
    ---------------
    - Input velocities are taken as East/North from SBG (`vel_e`, `vel_n`).
    - Positions are projected to local x/y using `generic_coordinate_transform`.
    - If `rotation_deg != 0`, x/y are rotated but (u,v) are still East/North. To avoid
      mixing frames, prefer `rotation_deg = 0` unless you also rotate (u,v) consistently.

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
    flip_z_sign : bool, optional
        If True (default), negate the stacked heave signal to correct for the
        real SWIFT SBG being mounted upside-down. Set False when the incoming
        z/heave is already in the desired sign convention (e.g., gz sim).

    Returns
    -------
    zin, uin, vin : ndarray (N, nbuoys)
        Stacked heave and velocity arrays. `uin/vin` are East/North (m/s).
        `zin` is up-positive (meters) when `flip_z_sign` is configured correctly.
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
            'load_raw_arrays_from_sbg expects (sbgs, lat0, lon0, rotation) or '
            '(sbgs, skipwarmup, burstend, lat0, lon0, rotation)'
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

        # Optional cached absolute UTM coordinates (computed at ingest time).
        # When present and aligned, these avoid re-projecting the full window.
        east = None
        north = None
        try:
            east = np.asarray(getattr(sbg.GpsPos, 'easting'), dtype=float)
            north = np.asarray(getattr(sbg.GpsPos, 'northing'), dtype=float)
        except Exception:
            east = None
            north = None

        if skipwarmup is not None or burstend is not None:
            start = int(skipwarmup) if skipwarmup is not None else 0
            stop = int(burstend) if burstend is not None else None
            z = z[start:stop]
            ztime = ztime[start:stop]
            u = u[start:stop]
            v = v[start:stop]
            lat = lat[start:stop]
            lon = lon[start:stop]
            if east is not None and north is not None:
                east = east[start:stop]
                north = north[start:stop]

        n_local = int(min(z.size, ztime.size, u.size, v.size, lat.size, lon.size))
        if east is not None and north is not None:
            n_local = int(min(n_local, east.size, north.size))
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

        if east is not None and north is not None:
            if n_local != east.size:
                east = east[:n_local]
            if n_local != north.size:
                north = north[:n_local]

        # If cached UTM is present but mostly non-finite, ignore it.
        if east is not None and north is not None:
            try:
                if int(np.count_nonzero(np.isfinite(east) & np.isfinite(north))) < max(
                    1, int(0.9 * n_local)
                ):
                    east = None
                    north = None
            except Exception:
                east = None
                north = None

        if east is not None and north is not None:
            # Convert absolute UTM -> local x/y in meters relative to origin.
            e0, n0, zone_num_unused, zone_let_unused = utm.from_latlon(
                float(latorigin), float(lonorigin)
            )
            dx = np.asarray(east, dtype=float) - float(e0)
            dy = np.asarray(north, dtype=float) - float(n0)
            ang = np.deg2rad(rotation)
            c = np.cos(ang)
            s = np.sin(ang)
            x = dx * c + dy * s
            y = -dx * s + dy * c
        else:
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

        per_buoy.append({'t': ztime, 'z': z, 'u': u, 'v': v, 'x': x, 'y': y})

    if not per_buoy:
        raise ValueError('No usable samples in SBG inputs')

    # Choose a reference time grid. In the main pipeline sbgs are ordered (swift22..)
    # so using the first buoy provides stable behavior across runs.
    t_ref = np.asarray(per_buoy[0]['t'], dtype=float).reshape((-1,))
    if t_ref.size < 2:
        raise ValueError('Not enough reference samples for alignment')

    # Estimate a reasonable matching tolerance from the reference sampling.
    dt_ref = np.diff(t_ref)
    dt_ref = dt_ref[np.isfinite(dt_ref) & (dt_ref > 0.0)]
    if dt_ref.size == 0:
        raise ValueError('Invalid reference timestamps for alignment')
    period_ref = float(np.nanmedian(dt_ref))
    if not np.isfinite(period_ref) or period_ref <= 0.0:
        raise ValueError('Invalid reference sampling period')
    # Nearest-neighbor without interpolation: accept matches within ~half a sample.
    tol = 0.55 * period_ref

    # For each buoy, find the nearest index in its time vector for each ref time.
    # Keep only those ref times that match all buoys within tolerance.
    indices_by_buoy = []
    valid = np.ones((t_ref.size,), dtype=bool)
    for b in per_buoy:
        t = np.asarray(b['t'], dtype=float).reshape((-1,))
        if t.size == 0:
            valid[:] = False
            indices_by_buoy.append(np.zeros_like(t_ref, dtype=int))
            continue

        j = np.searchsorted(t, t_ref, side='left')
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
        raise ValueError('Not enough aligned samples across buoys')

    # Build aligned per-buoy vectors at the matched indices; assign common time grid.
    for b, idx in zip(per_buoy, indices_by_buoy, strict=True):
        ii = np.asarray(idx, dtype=int)[valid]
        zin.append(np.asarray(b['z'], dtype=float)[ii])
        uin.append(np.asarray(b['u'], dtype=float)[ii])
        vin.append(np.asarray(b['v'], dtype=float)[ii])
        xin.append(np.asarray(b['x'], dtype=float)[ii])
        yin.append(np.asarray(b['y'], dtype=float)[ii])
        tin.append(np.asarray(t_common, dtype=float))

    zin = np.column_stack(zin)
    uin = np.column_stack(uin)
    vin = np.column_stack(vin)
    tin = np.column_stack(tin)
    xin = np.column_stack(xin)
    yin = np.column_stack(yin)

    if bool(flip_z_sign):
        zin = -zin  # Real SBG mounted upside-down on SWIFT

    fs = 1.0 / float(np.nanmean(np.diff(tin, axis=0)))
    return zin, uin, vin, tin, xin, yin, fs
