#!/usr/bin/python3

from pathlib import Path
import argparse

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import utm

from .swift import SWIFTArray, WaveSpec
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra
from .leastSquaresWavePropagation import leastSquaresWavePropagation


def _loadmat_struct(path: str):
    return spio.loadmat(path, struct_as_record=False, squeeze_me=True)


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


def generic_coordinate_transform(lat, lon, lat0, lon0, rotation_deg):
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


def build_wavespec_from_swifts(swifts, recip=True):
    Ethetas = []
    theta0 = None
    f0 = None

    for sw in swifts:
        Etheta, theta, E, f, _, spread, spread2, _ = SWIFTdirectionalspectra(sw, plotflag=False, recip=recip)
        Ethetas.append(Etheta)
        if theta0 is None:
            theta0 = theta.copy()
            f0 = f.copy()

    ws = WaveSpec()
    ws.theta = theta0
    ws.f = f0
    ws.Etheta = np.nanmean(np.stack(Ethetas, axis=2), axis=2)
    return ws


def centroid_period_and_phase_speed(ws):
    Etheta = ws.Etheta
    f = ws.f

    if Etheta.shape[0] != f.size and Etheta.shape[1] == f.size:
        Etheta = Etheta.T

    Ef = np.sum(Etheta, axis=1)
    Te = np.sum(Etheta) / np.sum(Ef * f)
    ce = 9.8 * Te / (2.0 * 3.14)
    return Te, ce


def load_raw_arrays_from_sbg(sbgs, skipwarmup, burstend, latorigin, lonorigin, rotation):
    sl = slice(skipwarmup - 1, burstend)

    zin = []
    uin = []
    vin = []
    tin = []
    xin = []
    yin = []

    for sbg in sbgs:
        z = np.asarray(sbg.ShipMotion.heave)[sl].astype(float)
        ztime = np.asarray(sbg.ShipMotion.time_stamp)[sl].astype(float) / 1e6

        u = np.asarray(sbg.GpsVel.vel_e)[sl].astype(float)
        v = np.asarray(sbg.GpsVel.vel_n)[sl].astype(float)

        lat = np.asarray(sbg.GpsPos.lat)[sl].astype(float)
        lon = np.asarray(sbg.GpsPos.long)[sl].astype(float)

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


if __name__ == '__main__':
    args = _parse_args()

    # local geometry (match MATLAB example)
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation = 180

    xtarget = 200.0
    ytarget = 200.0

    skipwarmup = 200
    burstend = 2740

    swiftdat = (
        'test_data/mat/SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        'test_data/mat/SWIFT23_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        'test_data/mat/SWIFT24_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
        'test_data/mat/SWIFT25_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat',
    )

    sbgdat = (
        'test_data/mat/SWIFT22_SBG_12Sep2022_07_01.mat',
        'test_data/mat/SWIFT23_SBG_12Sep2022_07_01.mat',
        'test_data/mat/SWIFT24_SBG_12Sep2022_07_01.mat',
        'test_data/mat/SWIFT25_SBG_12Sep2022_07_01.mat',
    )

    select_idx = 91  # MATLAB “burst index 92”
    swifts = SWIFTArray.from_mdat(swiftdat, sbgdat, select_idx)

    # 1) wavespec via SWIFTdirectionalspectra() on processed SWIFT burst structs
    swift_bursts = [swifts.swift22, swifts.swift23, swifts.swift24, swifts.swift25]
    wavespec_base = build_wavespec_from_swifts(swift_bursts, recip=True)
    Te, ce = centroid_period_and_phase_speed(wavespec_base)

    # 2) raw 5 Hz arrays via sbgData burst structs (like MATLAB example)
    sbg_bursts = [swifts.sbg22, swifts.sbg23, swifts.sbg24, swifts.sbg25]
    zin, uin, vin, tin, xin, yin, fs = load_raw_arrays_from_sbg(
        sbg_bursts, skipwarmup, burstend, latorigin, lonorigin, rotation
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

    A0 = None

    def run_loop(grab_frame=False):
        global A0

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

            t_end = float(np.nanmax(tin[inputwindow, :]))
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

    plt.ioff()
    plt.show()
