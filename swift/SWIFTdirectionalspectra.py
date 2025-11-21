
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .mem_directionalestimator import MEM_directionalestimator
from .swift import SWIFTData


def SWIFTdirectionalspectra(
        SWIFT:  SWIFTData,
        plotflag: bool = False,
        recip: bool = False
    ) -> Tuple[
        npt.NDArray[np.float64],  # Etheta
        npt.NDArray[np.float64],  # theta
        npt.NDArray[np.float64],  # E
        npt.NDArray[np.float64],  # f
        npt.NDArray[np.float64],  # dir
        npt.NDArray[np.float64],  # spread
        npt.NDArray[np.float64],  # spread2
        npt.NDArray[np.float64]  # spread2alt
    ]:
    """
    Make directional spectra from a SWIFT data structure.

    Can have multiple spectral results (and the results will average it) using MEM estimator from
    direction moments (call to function MEM_directionalestimator.m) and rotating from cartesion
    directions to nautical convention compass direction FROM which waves are coming also reports the
    dominant direction at each frequency and the spread.

    This is intended to be used after post-processing wave data with 'reprocess_IMU.m' which uses
    'XYZwaves.m' to get coefficients

    [Etheta theta E f dir spread spread2 spread2alt ] = SWIFTdirectionalspectra(SWIFT, plotflag, recip);

    J. Thomson, 10/2015, based on NCEX codes (J. Thomson, 2002)
                 4/2016 editted by Fabrice Ardhuin to energy weight coefs in determining average
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
    def _to_1d(arr):
        a = np.asarray(arr)
        if a.size == 0:
            return a.ravel()
        a = np.squeeze(a)
        if a.ndim == 0:
            return a.reshape((-1,))  # scalar -> single-element array
        if a.ndim > 1:
            # if shape is (1, n) or (n,1) squeeze above handled it; otherwise take mean across axis 0
            # (this mirrors treating a single SWIFT; adapt if you need different behavior)
            return np.nanmean(a, axis=0)
        return a

    energy = _to_1d(SWIFT.wavespectra.energy)
    a1_i = _to_1d(SWIFT.wavespectra.a1)
    a2_i = _to_1d(SWIFT.wavespectra.a2)
    b1_i = _to_1d(SWIFT.wavespectra.b1)
    b2_i = _to_1d(SWIFT.wavespectra.b2)

    # replace 9999 and NaN with zero as in MATLAB
    energy = np.where((energy == 9999) | np.isnan(energy), 0.0, energy)
    a1_i = np.where((a1_i == 9999) | np.isnan(a1_i), 0.0, a1_i)
    a2_i = np.where((a2_i == 9999) | np.isnan(a2_i), 0.0, a2_i)
    b1_i = np.where((b1_i == 9999) | np.isnan(b1_i), 0.0, b1_i)
    b2_i = np.where((b2_i == 9999) | np.isnan(b2_i), 0.0, b2_i)

    # following original logic: include this SWIFT if sigwaveheight in (0,20) and a1 exists
    cond = (getattr(SWIFT, "sigwaveheight", None) is not None) and (SWIFT.sigwaveheight > 0) and (SWIFT.sigwaveheight < 20) and np.all(~np.isnan(a1_i))
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
    I = np.where((E > 0) & (~np.isnan(E)))[0]
    if counter == 0 or I.size == 0:
        Hs = 0.0
    else:
        # for single SWIFT we already have a1,a2,b1,b2 as the moment arrays
        Hs = 4.0 * np.sqrt(np.sum(E[I] * df))

    # call MEM directional estimator (assumed provided elsewhere)
    # MEM_directionalestimator should return (Ethetanorm, Etheta) as in original code
    Ethetanorm, Etheta = MEM_directionalestimator(a1, a2, b1, b2, E, 0)

    # build theta vector identical to MATLAB's -[-180:dtheta:179]
    dtheta = 2.0
    theta = -np.arange(-180.0, 180.0, dtheta)
    theta = theta + 90.0
    theta[theta < 0] = theta[theta < 0] + 360.0

    if recip:
        print('taking reciprical directions (sanity check results)')
        westdirs = theta > 180.0
        eastdirs = theta < 180.0
        theta[westdirs] = theta[westdirs] - 180.0
        theta[eastdirs] = theta[eastdirs] + 180.0  # keep the same reciprocal transform

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

    if not recip:
        westdirs = dir_deg > 180.0
        eastdirs = dir_deg < 180.0
        dir_deg[westdirs] = dir_deg[westdirs] - 180.0
        dir_deg[eastdirs] = dir_deg[eastirs] + 180.0

    # spread in degrees
    if np.isrealobj(spread1):
        spread = 180.0 / np.pi * spread1
    else:
        spread = np.full_like(spread1, np.nan)

    spread2 = 180.0 / np.pi * spread2
    spread2alt = 180.0 / np.pi * spread2alt

    # plotting (kept faithful but minimal)
    if plotflag:
        # directional spectra (approximate polar pcolor)
        plt.figure(3); plt.clf()
        theta_plot = np.concatenate([theta, [360.0]])
        Etheta_plot = np.concatenate([Etheta, Etheta[:, :1]], axis=1)
        plt.pcolormesh(f, theta_plot, np.log10(Etheta_plot.T), shading='auto')
        plt.title(f"{wd}, Hs = {Hs:.2f} m")
        plt.ylabel('Direction [deg]')
        plt.xlabel('freq [Hz]')
        plt.colorbar(label='log10(E)')
        plt.savefig(f"{wd}_directionalspectra.png", dpi=150)

        # debugging plots as in original
        plt.figure(4); plt.clf()
        plt.subplot(3,1,1)
        plt.plot(f, E, 'k', f, np.sum(Etheta * dtheta, axis=1), 'k--', linewidth=2)
        plt.ylabel('Energy [m^2/Hz]')
        plt.xlim([0.05, np.max(f)])
        plt.title(f"{wd}, Hs = {Hs:.2f} m")

        plt.subplot(3,1,2)
        plt.errorbar(f, dir_deg, spread, fmt='k', markersize=4, linewidth=1)
        plt.ylabel('Dir [deg T]')
        plt.xlim([0.05, np.max(f)])
        plt.ylim([0, 360]); plt.yticks([0,90,180,270,360])

        plt.subplot(3,1,3)
        plt.plot(f, a1, label='a1')
        plt.plot(f, a2, label='a2')
        plt.plot(f, b1, label='b1')
        plt.plot(f, b2, label='b2')
        plt.plot([np.min(f), np.max(f)], [0,0], 'k:')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.savefig(f"{wd}_directionalmoments.png", dpi=150)

        # distributions at freqs
        plt.figure(5); plt.clf()
        r,c = 3,3; n = r*c
        for fi in range(1, n+1):
            thisf = fi * (int(np.ceil(len(f)/n)) - 1)
            if thisf < 0: thisf = 0
            if thisf >= Etheta.shape[0]: thisf = Etheta.shape[0] - 1
            plt.subplot(r,c,fi)
            plt.plot(theta, Etheta[thisf,:])
            plt.plot([dir_deg[thisf], dir_deg[thisf]],[0, np.max(Etheta[thisf,:])], 'r-')
            plt.plot([dir_deg[thisf]+spread[thisf], dir_deg[thisf]-spread[thisf]],
                     [np.max(Etheta[thisf,:])/2, np.max(Etheta[thisf,:])/2], 'r:')
            plt.xlim([0,360])
            plt.title(f"f = {f[thisf]:.3f} Hz")
        plt.savefig(f"{wd}_directiondistributions.png", dpi=150)

    return Etheta, theta, E, f, dir_deg, spread, spread2, spread2alt

"""
    # Average the spectra in the input SWIFT structure (if more than one).
    f = SWIFT.wavespectra.freq.reshape((-1, 1), order='F')
    df = np.median(np.diff(f, axis=0))
    energy = SWIFT.wavespectra.energy.copy()
    a1 = SWIFT.wavespectra.a1.copy()
    a2 = SWIFT.wavespectra.a2.copy()
    b1 = SWIFT.wavespectra.b1.copy()
    b2 = SWIFT.wavespectra.b2.copy()
    Hs = SWIFT.sigwaveheight.copy()
    [print(f'{v.shape=}') for v in [f, energy, a1, a2, b1, b2, Hs]]

    # clean data
    for arr in [energy, a1, a2, b1, b2]:
        arr[(arr == 9999) | np.isnan(arr)] = 0.0

    # mask on time samples
    good_mask = (
        (Hs > 0.0) &
        (Hs < 20.0) &
        np.all(~np.isnan(a1), axis=1)
    )

    # keep good data
    energy = energy[good_mask, :]
    a1     = a1[good_mask, :]
    a2     = a2[good_mask, :]
    b1     = b1[good_mask, :]
    b2     = b2[good_mask, :]
    counter = energy.shape[0]

    # Weighted sums across time axis
    E = np.sum(energy, axis=0)
    a1 = np.sum(a1 * energy, axis=0)
    a2 = np.sum(a2 * energy, axis=0)
    b1 = np.sum(b1 * energy, axis=0)
    b2 = np.sum(b2 * energy, axis=0)

    # Normalize
    E /= counter
    mask = (E > 0.0) & (~np.isnan(E))
    a1[mask] /= E[mask] * counter
    a2[mask] /= E[mask] * counter
    b1[mask] /= E[mask] * counter
    b2[mask] /= E[mask] * counter

    # recompute sig wave height
    Hs = 4.0 * np.sqrt(np.sum(E[mask] * df))

    # calc MEM estimate of full dir distribution spectrum, then convert to nautical convention
    # (compass dir FROM)
    Ethetanorm, Etheta = MEM_directionalestimator(a1, a2, b1, b2, E, False)
    dtheta = 2.
    # start with cartesion (a1 is positive east velocities, b1 is positive north)
    theta = -np.arange(-180., 180., dtheta)

    # rotate, flip and sort
    theta += 90.
    theta[theta < 0.] += 360.

    if recip:
        print('taking reciprical directions (sanity check results)')
        westdirs = theta > 180.
        eastdirs = theta < 180.
        theta[westdirs] = theta[westdirs] - 180.  # take reciprocal such wave direction is FROM, not TOWARDS
        theta[eastdirs] = theta[eastdirs] + 180.  # take reciprocal such wave direction is FROM, not TOWARDS

    dsort = np.argsort(theta)
    theta = theta[dsort].reshape((-1, 1), order='F')
    Etheta = Etheta[:, dsort]

    # spectral directions and spread, converted to nautical convention

    dir1 = np.arctan2(b1, a1)  # [rad], 4 quadrant
    dir2 = np.arctan2(b2, a2) / 2.  # [rad], only 2 quadrant
    spread1 = np.sqrt(2. * (1. - np.sqrt(a1**2. + b1**2.)))  # radians?
    # this is the usual definitionn e.g. OReilly et al. 1996
    spread2 = np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2. * dir2) + b2 * np.sin(2. * dir2))))  # radians?
    # Alternatively one can use (this is what is coded in WW3), and can be compared to tiltmeter data (Ardhuin et al. GRL 2016)
    spread2alt = np.sqrt(np.abs(0.5 - 0.5 * (a2**2. + b2**2.)))  # radians?

    # rotate and flip
    dir = -180. / np.pi * dir1  # switch from rad to deg, and CCW to CW (negate)
    dir += 90.  # rotate from eastward = 0 to northward  = 0
    dir[dir < 0.] += 360.  # take NW quadrant from negative to 270-360 range
    if not recip:
        westdirs = (dir > 180.)
        eastdirs = (dir < 180.)
        dir[westdirs] -= 180.  # take reciprocal such wave direction is FROM, not TOWARDS
        dir[eastdirs] += 180.  # take reciprocal such wave direction is FROM, not TOWARDS

    if np.allclose(np.imag(spread1), 0.):
        spread = (180. / np.pi) * np.real(spread1)
    else:
        spread = np.full(np.shape(spread1), np.nan)

    spread2 = 180. / np.pi * spread2
    spread2alt = 180. / np.pi * spread2alt

    return Etheta, theta, E, f, dir, spread, spread2, spread2alt


    if plotflag:
        figure(3), clf
        % 
        % subplot(2,1,1)
        % plot(f,E,'k',f,sum(Etheta*dtheta,2),'k--','linewidth',2), hold on
        % ylabel('Energy [m^2/Hz')
        % set(gca,'xlim',[0.05 0.5])
        % %title(['SWIFT   ' datestr(mean([SWIFT.time]),1) ', Hs = ' num2str(Hs,2) ' m'])

        %subplot(2,1,2)
        %pcolor(f,theta,log10(Etheta')), shading flat;
        if iscolumn(f),
            polarPcolor(f',theta(1:180),log10(Etheta(:,1:180)'));
        elseif isrow(f), 
            polarPcolor(f,theta(1:180),log10(Etheta(:,1:180)'));
        else
            disp('Problem with the size of frequency vector')
        end
        %ylabel('Direction [deg T]')
        %xlabel('freq [Hz]')
        %colorbar('peer',gca,'west')
        %legend('log_{10} (E)')
        title([ wd ', Hs = ' num2str(Hs,2) ' m'],'interpreter','none')


        print('-dpng',[ wd '_directionalspectra.png'])
        %print('-dpng',['SWIFT_dirwavespectra_' datestr(mean([SWIFT.time]),1) '.png'])

        %% debugging plots

        figure(4),clf

        subplot(3,1,1)
        plot(f,E,'k',f,sum(Etheta*dtheta,2),'k--','linewidth',2), hold on
        set(gca,'Fontsize',14,'fontweight','demi')
        ylabel('Energy [m^2/Hz]')
        set(gca,'xlim',[0.05 max(f)])
        title([ wd ', Hs = ' num2str(Hs,2) ' m'],'interpreter','none')


        subplot(3,1,2)
        errorbar(f,dir,spread,'k','markersize',16,'linewidth',2), hold on
        set(gca,'Fontsize',14,'fontweight','demi')
        ylabel('Dir [deg T]')
        set(gca,'xlim',[0.05 max(f)])
        set(gca,'ylim',[0 360],'YTick',[0 90 180 270 360])


        subplot(3,1,3)
        plot(f,a1,f,a2,f,b1,f,b2,'linewidth',2);
        set(gca,'Fontsize',14,'fontweight','demi')
        hold on
        plot([min(f) max(f)],[0 0],'k:')
        legend('a1','a2','b1','b2')
        set(gca,'xlim',[0.05 max(f)])
        set(gca,'ylim',[-1 1])
        xlabel('Frequency [Hz]')
        ylabel('Moments []')

        print('-dpng',[ wd '_directionalmoments.png'])


        %% distributions at freqs

        figure(5), clf

        r = 3; c = 3;
        n = r * c;

        for fi=1:n,
            subplot(r,c,fi)
            thisf = fi * ( ceil( length(f) / n ) - 1 );
            plot( theta , Etheta(thisf, :) ), hold on
            plot( [dir(thisf) dir(thisf)],[0 max(Etheta(thisf, :))], 'r-' )
            plot( [dir(thisf) + spread(thisf); dir(thisf) - spread(thisf); ],[max(Etheta(thisf, :))/2 max(Etheta(thisf, :))/2], 'r:' )
            title(['f = ' num2str( f(thisf) ) 'Hz' ] )
            set(gca,'XLim',[0 360])
            xlabel('\theta [deg]')
            ylabel('E')
        end

        print('-dpng',[ wd '_directiondistributions.png'])


        else 
        end

"""
