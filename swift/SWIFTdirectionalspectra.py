
from pathlib import Path
from typing import Tuple

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

    # Average the spectra in the input SWIFT structure (if more than one).
    f = SWIFT.wavespectra.freq.reshape((-1, 1), order='F')
    df = np.median(np.diff(f, axis=0))
    energy = SWIFT.wavespectra.energy.copy()
    a1 = SWIFT.wavespectra.a1.copy()
    a2 = SWIFT.wavespectra.a2.copy()
    b1 = SWIFT.wavespectra.b1.copy()
    b2 = SWIFT.wavespectra.b2.copy()
    Hs = SWIFT.significant_wave_height.copy()

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


    """ TODO
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


