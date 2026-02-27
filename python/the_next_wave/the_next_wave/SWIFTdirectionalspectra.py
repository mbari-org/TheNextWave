
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .mem_directionalestimator import MEM_directionalestimator
from .swift import SWIFTData


def SWIFTdirectionalspectra(
    SWIFT: SWIFTData,
    plotflag: bool = False,
    recip: bool = False,
    *,
    mem_moment_cap: float | None = None,
) -> Tuple[
    npt.NDArray[np.float64],  # Etheta
    npt.NDArray[np.float64],  # theta
    npt.NDArray[np.float64],  # E
    npt.NDArray[np.float64],  # f
    npt.NDArray[np.float64],  # dir
    npt.NDArray[np.float64],  # spread
    npt.NDArray[np.float64],  # spread2
    npt.NDArray[np.float64],  # spread2alt
]:
    """
    Make directional spectra from a SWIFT data structure.

    Can have multiple spectral results (and the results will average it) using
    MEM estimator from direction moments (call to function
    MEM_directionalestimator.m) and rotating from cartesion directions to
    nautical convention compass direction FROM which waves are coming also
    reports the dominant direction at each frequency and the spread.

        Notes on direction conventions
        ------------------------------
        - Angles are compass degrees True (0°=North, 90°=East).
        - "FROM" means the direction waves are coming from (the common oceanographic
            reporting convention). A propagation direction is TO = FROM + 180.
        - The legacy MATLAB implementation's `recip` flag is asymmetric: it flips
            the Etheta/theta axis when `recip=True`, while the moment-derived `dir`
            output is flipped when `recip=False`. This Python port preserves that
            behavior for compatibility.

    This is intended to be used after post-processing wave data with
    'reprocess_IMU.m' which uses 'XYZwaves.m' to get coefficients

    [Etheta theta E f dir spread spread2 spread2alt ] =
    SWIFTdirectionalspectra(SWIFT, plotflag, recip);

    J. Thomson, 10/2015, based on NCEX codes (J. Thomson, 2002)
                 4/2016 editted by Fabrice Ardhuin to energy weight coefs in
                     determining average
                 5/2016 corrected typo in spread1
                 8/2016 enable reciprocal flag, for post-procesing vs onboard processing
                 1/2017 output multiple estimates of spread and add binary plotting flag
                12/2018   add plots of directional distributions distinct freqs
                 3/2019 add second binary varargin to control reciprocal direction
    """
    wd = Path.cwd().name  # get current folder name without path

    # frequency vector and df
    f = np.asarray(SWIFT.wavespectra.freq).ravel()
    df = np.median(np.diff(f))

    # get spectral arrays (try to coerce to 1D frequency arrays)
    def to_1d(arr):
        a = np.asarray(arr)
        if a.size == 0:
            return a.ravel()
        a = np.squeeze(a)
        if a.ndim == 0:
            return a.reshape((-1,))  # scalar -> single-element array
        if a.ndim > 1:
            # If shape is higher-rank after squeeze, collapse over axis 0.
            # (this mirrors treating a single SWIFT; adapt if you need different behavior)
            return np.nanmean(a, axis=0)
        return a

    energy = to_1d(SWIFT.wavespectra.energy)
    a1_i = to_1d(SWIFT.wavespectra.a1)
    a2_i = to_1d(SWIFT.wavespectra.a2)
    b1_i = to_1d(SWIFT.wavespectra.b1)
    b2_i = to_1d(SWIFT.wavespectra.b2)

    # replace 9999 and NaN with zero as in MATLAB
    energy = np.where((energy == 9999) | np.isnan(energy), 0.0, energy)
    a1_i = np.where((a1_i == 9999) | np.isnan(a1_i), 0.0, a1_i)
    a2_i = np.where((a2_i == 9999) | np.isnan(a2_i), 0.0, a2_i)
    b1_i = np.where((b1_i == 9999) | np.isnan(b1_i), 0.0, b1_i)
    b2_i = np.where((b2_i == 9999) | np.isnan(b2_i), 0.0, b2_i)

    # following original logic: include this SWIFT if sigwaveheight in (0,20) and a1 exists
    cond = (
        (getattr(SWIFT, 'sigwaveheight', None) is not None)
        and (SWIFT.sigwaveheight > 0)
        and (SWIFT.sigwaveheight < 20)
        and np.all(~np.isnan(a1_i))
    )
    if cond:
        # for single SWIFT the weighted-sum / divide-by-E logic reduces:
        E = energy.copy()
        a1 = a1_i.copy()
        a2 = a2_i.copy()
        b1 = b1_i.copy()
        b2 = b2_i.copy()
        counter = 1
    else:
        # mirror MATLAB: if not included, produce zeros and counter 0
        E = np.zeros_like(f, dtype=float)
        a1 = np.zeros_like(f, dtype=float)
        a2 = np.zeros_like(f, dtype=float)
        b1 = np.zeros_like(f, dtype=float)
        b2 = np.zeros_like(f, dtype=float)
        counter = 0

    # indices where E > 0 and not NaN
    valid_idx = np.where((E > 0) & (~np.isnan(E)))[0]
    if counter == 0 or valid_idx.size == 0:
        Hs = 0.0
    else:
        # for single SWIFT we already have a1,a2,b1,b2 as the moment arrays
        Hs = 4.0 * np.sqrt(np.sum(E[valid_idx] * df))

    # calc MEM estimate of full dir distribution spectrum
    # (MATLAB calls MEM_directionalestimator with convert=0 here)

    # Optional stabilization for perfectly coherent plane-wave inputs.
    # When enabled, clamp moment magnitudes so |a1+ib1| and |a2+ib2| are < 1.
    # This is NOT part of the original MATLAB SWIFTdirectionalspectra.m logic,
    # so it is disabled by default and must be explicitly requested by callers.
    if mem_moment_cap is not None:
        cap = float(mem_moment_cap)
        if np.isfinite(cap) and cap > 0.0:
            c1 = a1.astype(float) + 1j * b1.astype(float)
            c2 = a2.astype(float) + 1j * b2.astype(float)

            m1 = np.abs(c1)
            m2 = np.abs(c2)

            with np.errstate(divide='ignore', invalid='ignore'):
                s1 = np.where(m1 > cap, cap / m1, 1.0)
                s2 = np.where(m2 > cap, cap / m2, 1.0)

            c1 = c1 * s1
            c2 = c2 * s2
            a1 = np.real(c1)
            b1 = np.imag(c1)
            a2 = np.real(c2)
            b2 = np.imag(c2)

    Ethetanorm, Etheta = MEM_directionalestimator(a1, a2, b1, b2, E, 0)

    # build theta vector identical to MATLAB's -[-180:dtheta:179]
    dtheta = 2.0
    theta = -np.arange(-180.0, 180.0, dtheta)

    # rotate, flip and sort (match MATLAB)
    theta = theta + 90.0
    theta[theta < 0] = theta[theta < 0] + 360.0

    if recip:
        westdirs = theta > 180.0
        eastdirs = theta < 180.0
        theta[westdirs] = theta[westdirs] - 180.0
        theta[eastdirs] = theta[eastdirs] + 180.0

    # sort theta and reorder Etheta columns
    sort_idx = np.argsort(theta)
    theta = theta[sort_idx]
    # Etheta expected shape: (nfreq, ntheta)
    Etheta = Etheta[:, sort_idx]

    # spectral dirs and spread
    dir1 = np.arctan2(b1, a1)
    dir2 = np.arctan2(b2, a2) / 2.0
    with np.errstate(invalid='ignore'):
        spread1 = np.sqrt(2.0 * (1.0 - np.sqrt(a1**2 + b1**2)))
        spread2 = np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2.0 * dir2) + b2 * np.sin(2.0 * dir2))))
        spread2alt = np.sqrt(np.abs(0.5 - 0.5 * (a2**2 + b2**2)))

    # rotate and convert to degrees
    dir_deg = -180.0 / np.pi * dir1
    dir_deg = dir_deg + 90.0
    dir_deg[dir_deg < 0] = dir_deg[dir_deg < 0] + 360.0

    # rotate and flip (match MATLAB)
    if not recip:
        westdirs = dir_deg > 180.0
        eastdirs = dir_deg < 180.0
        dir_deg[westdirs] = dir_deg[westdirs] - 180.0
        dir_deg[eastdirs] = dir_deg[eastdirs] + 180.0

    # spread in degrees (match MATLAB's 3.14 constant)
    if np.isrealobj(spread1):
        spread = 180.0 / 3.14 * spread1
    else:
        spread = np.full_like(spread1, np.nan)

    spread2 = 180.0 / 3.14 * spread2
    spread2alt = 180.0 / 3.14 * spread2alt

    # make theta complete circle (match MATLAB)
    theta = np.concatenate(([0.0], theta[:180], [360.0]))
    Etheta = np.concatenate((Etheta[:, :1], Etheta[:, :180], Etheta[:, :1]), axis=1)

    # plotting (kept faithful but minimal)
    if plotflag:
        # directional spectra (approximate polar pcolor)
        plt.figure(3)
        plt.clf()
        theta_plot = np.concatenate([theta, [360.0]])
        Etheta_plot = np.concatenate([Etheta, Etheta[:, :1]], axis=1)
        plt.pcolormesh(f, theta_plot, np.log10(Etheta_plot.T), shading='auto')
        plt.title(f'{wd}, Hs = {Hs:.2f} m')
        plt.ylabel('Direction [deg]')
        plt.xlabel('freq [Hz]')
        plt.colorbar(label='log10(E)')
        plt.savefig(f'{wd}_directionalspectra.png', dpi=150)

        # debugging plots as in original
        plt.figure(4)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(f, E, 'k', f, np.sum(Etheta * dtheta, axis=1), 'k--', linewidth=2)
        plt.ylabel('Energy [m^2/Hz]')
        plt.xlim([0.05, np.max(f)])
        plt.title(f'{wd}, Hs = {Hs:.2f} m')

        plt.subplot(3, 1, 2)
        plt.errorbar(f, dir_deg, spread, fmt='k', markersize=4, linewidth=1)
        plt.ylabel('Dir [deg T]')
        plt.xlim([0.05, np.max(f)])
        plt.ylim([0, 360])
        plt.yticks([0, 90, 180, 270, 360])

        plt.subplot(3, 1, 3)
        plt.plot(f, a1, label='a1')
        plt.plot(f, a2, label='a2')
        plt.plot(f, b1, label='b1')
        plt.plot(f, b2, label='b2')
        plt.plot([np.min(f), np.max(f)], [0, 0], 'k:')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.savefig(f'{wd}_directionalmoments.png', dpi=150)

        # distributions at freqs
        plt.figure(5)
        plt.clf()
        r, c = 3, 3
        n = r * c
        for fi in range(1, n + 1):
            thisf = fi * (int(np.ceil(len(f) / n)) - 1)
            if thisf < 0:
                thisf = 0
            if thisf >= Etheta.shape[0]:
                thisf = Etheta.shape[0] - 1
            plt.subplot(r, c, fi)
            plt.plot(theta, Etheta[thisf, :])
            plt.plot([dir_deg[thisf], dir_deg[thisf]], [0, np.max(Etheta[thisf, :])], 'r-')
            plt.plot(
                [dir_deg[thisf] + spread[thisf], dir_deg[thisf] - spread[thisf]],
                [np.max(Etheta[thisf, :]) / 2, np.max(Etheta[thisf, :]) / 2],
                'r:',
            )
            plt.xlim([0, 360])
            plt.title(f'f = {f[thisf]:.3f} Hz')
        plt.savefig(f'{wd}_directiondistributions.png', dpi=150)

    return Etheta, theta, E, f, dir_deg, spread, spread2, spread2alt
