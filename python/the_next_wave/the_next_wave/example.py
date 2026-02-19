#!/usr/bin/python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from .swift import Prediction, SWIFTArray, WaveSpec
from .leastSquaresWavePropagation import leastSquaresWavePropagation
from .utilities import (
    build_wavespec_from_swifts,
    centroid_period_and_phase_speed,
    load_raw_arrays_from_sbg,
)

from .download_example_data import get_example_data_dir


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--movie',
        type=str,
        default=None,
        help='Write an animation to this path (.mp4 or .gif).',
    )
    p.add_argument('--fps', type=float, default=10.0, help='Movie frames per second.')
    p.add_argument('--dpi', type=int, default=150, help='Output DPI for the movie frames.')
    p.add_argument(
        '--show',
        action='store_true',
        help='Show the interactive window even if --movie is set.',
    )
    return p.parse_args()


A0 = None
all_preds = Prediction()
def main():
    global A0
    global all_preds
    args = _parse_args()

    # match MATLAB example
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation = 180

    xtarget = 200.0
    ytarget = 200.0

    skipwarmup = 200
    burstend = 2740

    example_data_dir = get_example_data_dir()
    swiftdat = (
        example_data_dir / 'SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        example_data_dir / 'SWIFT23_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        example_data_dir / 'SWIFT24_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        example_data_dir / 'SWIFT25_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
    )

    sbgdat = (
        example_data_dir / 'SWIFT22_SBG_12Sep2022_07_01.mat',
        example_data_dir / 'SWIFT23_SBG_12Sep2022_07_01.mat',
        example_data_dir / 'SWIFT24_SBG_12Sep2022_07_01.mat',
        example_data_dir / 'SWIFT25_SBG_12Sep2022_07_01.mat',
    )

    select_idx = 91  # MATLAB burst index 92
    swifts = SWIFTArray.from_mdat(swiftdat, sbgdat, select_idx)

    # 1) wavespec via SWIFTdirectionalspectra() on processed SWIFT burst structs
    wavespec_base = build_wavespec_from_swifts(swifts.bursts(raw_sbg=False), recip=True)
    Te, ce = centroid_period_and_phase_speed(wavespec_base)

    # 2) raw SBGData burst structs
    zin, uin, vin, tin, xin, yin, fs = load_raw_arrays_from_sbg(
        swifts.bursts(raw_sbg=True), skipwarmup, burstend, latorigin, lonorigin, rotation
    )

    nbuoys = zin.shape[1]
    n = zin.shape[0]

    NTe = 10
    win_len = int(round(NTe * Te * fs))
    step = int(round(fs))  # ~1 second increments

    if args.movie and not args.show:
        plt.ioff()
    else:
        plt.ion()

    fig = plt.figure(1)

    writer = None
    movie_path = None
    if args.movie:
        movie_path = Path(args.movie)
        movie_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = movie_path.suffix.lower()
        if suffix == '.gif':
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps)

    def run_loop(grab_frame=False):
        global A0
        global all_preds

        for ti in range(0, n, step):
            inputwindow = ti + np.arange(win_len)
            if inputwindow[-1] >= n:
                break

            dist = np.sqrt((xin[inputwindow, :] - xtarget) ** 2 + (yin[inputwindow, :] - ytarget) ** 2)
            maxtargetdistance = float(np.nanmax(dist))
            leadtime = maxtargetdistance / ce

            n_lead = int(np.floor(leadtime))
            if n_lead < 1:
                n_lead = 1

            t_start = float(np.nanmin(tin[inputwindow, :]))
            t_end = float(np.nanmax(tin[inputwindow, :]))
            print(f'solving prediction window: [{t_start}, {t_end}] s')
            tpred = t_end + np.arange(1, n_lead + 1, dtype=float)

            xpred = np.full_like(tpred, xtarget, dtype=float)
            ypred = np.full_like(tpred, ytarget, dtype=float)

            # solver mutates wavespec internals; pass copies each call
            ws = WaveSpec()
            ws.theta = wavespec_base.theta.copy()
            ws.f = wavespec_base.f.copy()
            ws.Etheta = wavespec_base.Etheta.copy()

            pred_vec, recon_vec, params, comp_time = leastSquaresWavePropagation(
                zin[inputwindow, :],
                uin[inputwindow, :],
                vin[inputwindow, :],
                tin[inputwindow, :],
                xin[inputwindow, :],
                yin[inputwindow, :],
                tpred.reshape((-1, 1)),
                xpred.reshape((-1, 1)),
                ypred.reshape((-1, 1)),
                ws,
                A0=A0,
            )
            A0 = params.A

            prediction = np.asarray(pred_vec).reshape((tpred.size, -1), order='F')
            zout = prediction[:, 0]
            uout = prediction[:, 1]
            vout = prediction[:, 2]

            reconstruction = np.asarray(recon_vec).reshape((inputwindow.size, -1), order='F')
            zr = reconstruction[:, 0:nbuoys]
            ur = reconstruction[:, nbuoys:2 * nbuoys]
            vr = reconstruction[:, 2 * nbuoys:3 * nbuoys]

            all_preds.append_window(
                window_start_time=t_start,
                tm=tin[inputwindow, :],
                zm=zin[inputwindow, :],
                um=uin[inputwindow, :],
                vm=vin[inputwindow, :],
                xm=xin[inputwindow, :],
                ym=yin[inputwindow, :],
                zc=zr,
                uc=ur,
                vc=vr,
                tp=tpred,
                zp=zout,
                up=uout,
                vp=vout,
                params=params,
                comp_time=float(comp_time),
            )

            fig.clf()

            ax = fig.add_subplot(2, 2, 2)
            ax.plot(xin[inputwindow, :], yin[inputwindow, :], 'x', linewidth=2)
            ax.plot(xpred, ypred, 'ko', linewidth=2, markersize=6)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 500)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')

            fig.add_subplot(6, 2, 1)
            plt.plot(tin[inputwindow, :], zin[inputwindow, :])
            plt.ylabel('z in [m]')

            fig.add_subplot(6, 2, 3)
            plt.plot(tin[inputwindow, :], uin[inputwindow, :])
            plt.ylabel('u in [m/s]')

            fig.add_subplot(6, 2, 5)
            plt.plot(tin[inputwindow, :], vin[inputwindow, :])
            plt.ylabel('v in [m/s]')

            fig.add_subplot(6, 2, 7)
            plt.plot(tin[inputwindow, :], zr)
            plt.ylabel('z recon [m]')

            fig.add_subplot(6, 2, 9)
            plt.plot(tin[inputwindow, :], ur)
            plt.ylabel('u recon [m/s]')

            fig.add_subplot(6, 2, 11)
            plt.plot(tin[inputwindow, :], vr)
            plt.ylabel('v recon [m/s]')
            plt.xlabel('t [s]')

            fig.add_subplot(6, 2, 8)
            plt.plot(tpred, zout, 'k')
            plt.ylabel('z pred [m]')

            fig.add_subplot(6, 2, 10)
            plt.plot(tpred, uout, 'k')
            plt.ylabel('u pred [m/s]')

            fig.add_subplot(6, 2, 12)
            plt.plot(tpred, vout, 'k')
            plt.ylabel('v pred [m/s]')
            plt.xlabel('t [s]')

            if grab_frame:
                writer.grab_frame()
            else:
                plt.pause(0.001)

    if writer is not None:
        with writer.saving(fig, str(movie_path), dpi=args.dpi):
            run_loop(grab_frame=True)
    else:
        run_loop(grab_frame=False)

    all_preds.to_netcdf('predictions.nc')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
