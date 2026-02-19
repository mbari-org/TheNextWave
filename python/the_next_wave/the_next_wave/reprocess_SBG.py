import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Optional, Tuple

from .swift import SBGData, SWIFTData, SWIFTArray
from .sbg_waves import sbg_waves

"""
Process SWIFT v4 SBG wave data from real-time streaming or stored arrays.

Follows the MATLAB reprocess_SBG.m workflow:
1. Clean outliers in raw SBG data (heave, velocities, positions)
2. Remove NaNs
3. Call SBGwaves to compute wave spectra
4. Convert peak direction from "to" to "from" convention
5. Populate SWIFTData structures for use with SWIFTdirectionalspectra

Pipeline for the_next_wave_node.py:
  Raw SBG → reprocess_swift_array() → cleaned SBG + SWIFT structures
  → SWIFTdirectionalspectra() → averaged WaveSpec
  → leastSquaresWavePropagation(raw SBG, WaveSpec) → predictions

Original MATLAB code:
M. Schwendeman, 01/2017
J. Thomson, 10/2017 - add reprocessing to batch read of raw data
K. Zeiden, July 2024 - reformatting for use in master postprocessing script

Python port: 2026
"""


def filloutliers(data, method='linear', threshold=3.0):
    """
    Replace outliers in data using interpolation (MATLAB filloutliers equivalent).
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic')
    threshold : float, optional
        Number of standard deviations for outlier detection
        
    Returns
    -------
    np.ndarray
        Data with outliers replaced
    """
    data = np.array(data, dtype=float)
    median = np.median(data)
    std = np.std(data)
    outliers = np.abs(data - median) > threshold * std
    
    if np.any(outliers) and np.sum(~outliers) > 1:
        good_idx = np.where(~outliers)[0]
        outlier_idx = np.where(outliers)[0]
        interp_func = interp1d(
            good_idx, data[good_idx],
            kind=method, 
            bounds_error=False,
            fill_value='extrapolate'
        )
        data[outliers] = interp_func(outlier_idx)
    
    return data


def convert_direction_to_from(dirto: float) -> float:
    """
    Convert wave direction from 'to' to 'from' convention (MATLAB code).
    
    Parameters
    ----------
    dirto : float
        Direction waves are going TO [deg]
        
    Returns
    -------
    float
        Direction waves are coming FROM [deg]
    """
    if dirto >= 180:
        return dirto - 180
    elif dirto < 180:
        return dirto + 180
    else:
        return dirto


def reprocess_swift_array(
    swifts: SWIFTArray,
    fs: float = 5.0,
    min_samples: Optional[int] = None,
    tstart: int = 0
) -> Tuple[SWIFTArray, SWIFTArray]:
    """
    Process all SBG data following MATLAB reprocess_SBG.m workflow.
    
    This is the main function for the_next_wave_node.py. It:
    1. Cleans outliers in raw SBG data
    2. Removes NaNs
    3. Calls SBGwaves to compute wave spectra
    4. Converts peak direction from "to" to "from"
    5. Populates SWIFTData structures
    
    Parameters
    ----------
    swifts : SWIFTArray
        SWIFTArray containing sbg22-25 with raw accumulated data
    fs : float, optional
        Sampling rate in Hz (default: 5.0)
    min_samples : int, optional
        Minimum samples required (default: 256s window = 1280 samples)
    tstart : int, optional
        Number of seconds to skip from beginning (default: 0)
        
    Returns
    -------
    swifts_cleaned : SWIFTArray
        Cleaned raw SBG data (for leastSquaresWavePropagation)
    swifts_processed : SWIFTArray
        Populated SWIFTData structures (for SWIFTdirectionalspectra)
        
    Example
    -------
    >>> # In the_next_wave_node.py after collecting 256s window:
    >>> cleaned_sbg, swift_structs = reprocess_swift_array(
    ...     swifts_snapshot
    ... )
    >>> # Get averaged directional spectrum:
    >>> wavespec = build_wavespec_from_swifts(
    ...     [swift_structs.swift22, swift_structs.swift23, 
    ...      swift_structs.swift24, swift_structs.swift25]
    ... )
    >>> # Run prediction with cleaned raw data:
    >>> pred = leastSquaresWavePropagation(
    ...     cleaned_sbg.sbg22.ShipMotion.heave, ..., wavespec
    ... )
    """
    window_duration_s = 256.0
    if min_samples is None:
        # Keep as a fallback when timestamps are missing; when timestamps exist,
        # we validate by time span and estimate fs per buoy.
        min_samples = int(window_duration_s * fs)
    
    # Create output structures
    swifts_cleaned = SWIFTArray()
    swifts_processed = SWIFTArray()
    
    for swift_num in range(22, 26):
        sbg = getattr(swifts, f'sbg{swift_num}')
        swift_name = f'swift{swift_num}'
        
        # Extract data arrays (handle both lists and numpy arrays)
        if isinstance(sbg.ShipMotion.heave, list):
            t_us = np.array(sbg.ShipMotion.time_stamp, dtype=float)
            z = np.array(sbg.ShipMotion.heave, dtype=float)
            u = np.array(sbg.GpsVel.vel_e, dtype=float)
            v = np.array(sbg.GpsVel.vel_n, dtype=float)
            lat = np.array(sbg.GpsPos.lat, dtype=float)
            lon = np.array(sbg.GpsPos.long, dtype=float)
        else:
            t_us = np.asarray(sbg.ShipMotion.time_stamp, dtype=float).copy()
            z = sbg.ShipMotion.heave.copy()
            u = sbg.GpsVel.vel_e.copy()
            v = sbg.GpsVel.vel_n.copy()
            lat = sbg.GpsPos.lat.copy()
            lon = sbg.GpsPos.long.copy()
        
        # Crop tstart if requested (MATLAB: z(tstart*5:end))
        if tstart > 0:
            crop_idx = int(tstart * fs)
            t_us = t_us[crop_idx:]
            z = z[crop_idx:]
            u = u[crop_idx:]
            v = v[crop_idx:]
            lat = lat[crop_idx:]
            lon = lon[crop_idx:]

        # Estimate actual sample rate from timestamps (microseconds).
        fs_used = float(fs)
        if t_us.size >= 2:
            dt_s = np.diff(t_us) / 1e6
            dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
            if dt_s.size > 0:
                fs_est = float(1.0 / np.nanmean(dt_s))
                if np.isfinite(fs_est) and fs_est > 0.0:
                    fs_used = fs_est
        
        # Check we have a full window of data. Prefer time-span check if timestamps exist.
        if t_us.size >= 2:
            span_s = float((t_us[-1] - t_us[0]) / 1e6)
            if not np.isfinite(span_s) or span_s < window_duration_s:
                print(f'{swift_name}: insufficient time span ({span_s:.1f}s < {window_duration_s:.1f}s)')
                continue
        else:
            if len(z) < min_samples:
                print(f'{swift_name}: insufficient samples ({len(z)} < {min_samples})')
                continue
        
        # Clean outliers (MATLAB filloutliers)
        z = filloutliers(z, method='linear')
        u = filloutliers(u, method='linear')
        v = filloutliers(v, method='linear')
        lat = filloutliers(lat, method='linear')
        lon = filloutliers(lon, method='linear')
        
        # Remove NaNs (MATLAB: ibad = isnan(z + x + y + u + v + lat + lon))
        ibad = np.isnan(z + u + v + lat + lon)
        if np.any(ibad):
            t_us = t_us[~ibad]
            z = z[~ibad]
            u = u[~ibad]
            v = v[~ibad]
            lat = lat[~ibad]
            lon = lon[~ibad]
        
        # Store cleaned raw data
        sbg_clean = SBGData()
        sbg_clean.ShipMotion.time_stamp = t_us
        sbg_clean.ShipMotion.heave = z
        sbg_clean.GpsVel.time_stamp = t_us
        sbg_clean.GpsVel.vel_e = u
        sbg_clean.GpsVel.vel_n = v
        sbg_clean.GpsPos.time_stamp = t_us
        sbg_clean.GpsPos.lat = lat
        sbg_clean.GpsPos.long = lon
        setattr(swifts_cleaned, f'sbg{swift_num}', sbg_clean)
        
        # Compute wave spectra (MATLAB: SBGwaves)
        try:
            Hs, Tp, Dp, E, f, a1, b1, a2, b2, check = sbg_waves(u, v, z, fs_used)
            
            # Check for error codes (MATLAB checks for 9999)
            if Hs == 9999:
                print(f'{swift_name}: wave processing returned 9999 for Hs')
                Hs = np.nan
                Tp = np.nan
                Dp = np.nan
            
            if Dp > 9000:  # Sometimes only directions fail
                Dp = np.nan
            
            if np.sum(E) < 1:
                print(f'{swift_name}: low energy sum ({np.sum(E)})')
            
            # Convert direction from "to" to "from" (MATLAB code)
            if not np.isnan(Dp):
                Dp = convert_direction_to_from(Dp)
            
            # Populate SWIFTData structure
            swift_data = SWIFTData()
            swift_data.sigwaveheight = np.array([Hs])
            swift_data.peakwaveperiod = np.array([Tp])
            swift_data.peakwavedirT = np.array([Dp])
            swift_data.wavespectra.freq = f
            swift_data.wavespectra.energy = E.reshape(1, -1)
            swift_data.wavespectra.a1 = a1.reshape(1, -1)
            swift_data.wavespectra.b1 = b1.reshape(1, -1)
            swift_data.wavespectra.a2 = a2.reshape(1, -1)
            swift_data.wavespectra.b2 = b2.reshape(1, -1)
            swift_data.wavespectra.check = check.reshape(1, -1)
            
            # Store cleaned lat/lon for position reference
            swift_data.sbg_lat = lat
            swift_data.sbg_lon = lon
            
            setattr(swifts_processed, f'swift{swift_num}', swift_data)
            
        except Exception as e:
            print(f'{swift_name}: wave processing failed - {e}')
            continue
    
    return swifts_cleaned, swifts_processed