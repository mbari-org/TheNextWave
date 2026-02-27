#!/usr/bin/python3

import argparse
from pathlib import Path

from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from .download_example_data import get_example_data_dir
from .leastSquaresWavePropagation import leastSquaresWavePropagation
from .swift import Prediction, SWIFTArray, WaveSpec
from .utilities import (
    build_wavespec_from_swifts,
    centroid_period_and_phase_speed,
    bulk_wave_params_from_wavespec,
    load_raw_arrays_from_sbg,
    format_bulk_wave_params,
)


def parse_args():
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
    args = parse_args()

    # match MATLAB example
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation = 0.0  # 180.0

    xtarget = 200.0
    ytarget = 200.0

    skipwarmup = 200
    burstend = 2740

    example_data_dir = get_example_data_dir()
    swiftdat = (
        (
            example_data_dir
            / 'SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat'
        ),
        (
            example_data_dir
            / 'SWIFT23_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat'
        ),
        (
            example_data_dir
            / 'SWIFT24_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat'
        ),
        (
            example_data_dir
            / 'SWIFT25_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat'
        ),
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
    # Direction convention: use compass degrees True, direction waves come FROM.
    # (The plotter then draws propagation TO = FROM + 180.)
    wavespec_base = build_wavespec_from_swifts(swifts.bursts(raw_sbg=False), recip=False)
    Te, ce = centroid_period_and_phase_speed(wavespec_base)
    wavespec_bulk = bulk_wave_params_from_wavespec(wavespec_base)
    print(format_bulk_wave_params(wavespec_bulk, 'wavespec'))

    # 2) raw SBGData burst structs
    # The example dataset uses an SBG heave sign convention that needs inversion
    # (mirrors the MATLAB example's explicit `zin = -zin`). For gz sim, do NOT
    # apply this flip.
    zin, uin, vin, tin, xin, yin, fs = load_raw_arrays_from_sbg(
        swifts.bursts(raw_sbg=True),
        skipwarmup,
        burstend,
        latorigin,
        lonorigin,
        rotation,
        flip_z_sign=True,
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
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass

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

            dist = np.sqrt(
                (xin[inputwindow, :] - xtarget) ** 2 + (yin[inputwindow, :] - ytarget) ** 2
            )
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

            # Single consistent layout (avoid overlapping subplot grids).
            gs = fig.add_gridspec(nrows=6, ncols=2)

            ax_z_in = fig.add_subplot(gs[0, 0])
            ax_u_in = fig.add_subplot(gs[1, 0], sharex=ax_z_in)
            ax_v_in = fig.add_subplot(gs[2, 0], sharex=ax_z_in)
            ax_z_rc = fig.add_subplot(gs[3, 0], sharex=ax_z_in)
            ax_u_rc = fig.add_subplot(gs[4, 0], sharex=ax_z_in)
            ax_v_rc = fig.add_subplot(gs[5, 0], sharex=ax_z_in)

            ax_map = fig.add_subplot(gs[0:3, 1])
            ax_z_pr = fig.add_subplot(gs[3, 1])
            ax_u_pr = fig.add_subplot(gs[4, 1], sharex=ax_z_pr)
            ax_v_pr = fig.add_subplot(gs[5, 1], sharex=ax_z_pr)

            # Map: plot buoy tracks + a representative per-buoy position marker.
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
            xw = np.asarray(xin[inputwindow, :], dtype=float)
            yw = np.asarray(yin[inputwindow, :], dtype=float)

            buoy_x = []
            buoy_y = []
            for j in range(nbuoys):
                xj = xw[:, j]
                yj = yw[:, j]
                ok = np.isfinite(xj) & np.isfinite(yj)
                if not np.any(ok):
                    continue
                ax_map.plot(xj[ok], yj[ok], '.', color=colors[j % len(colors)], alpha=0.25, markersize=2)
                x_med = float(np.nanmedian(xj[ok]))
                y_med = float(np.nanmedian(yj[ok]))
                buoy_x.append(x_med)
                buoy_y.append(y_med)
                ax_map.plot(
                    [x_med],
                    [y_med],
                    marker='x',
                    color=colors[j % len(colors)],
                    markersize=8,
                    mew=2,
                    linestyle='None',
                    label=f'swift{22 + j}',
                )

            ax_map.plot([float(xtarget)], [float(ytarget)], 'ko', markersize=6, label='target')

            # Auto-scale around all finite buoy points and the target.
            x_all = np.concatenate([np.asarray(buoy_x, dtype=float), np.asarray([xtarget], dtype=float)])
            y_all = np.concatenate([np.asarray(buoy_y, dtype=float), np.asarray([ytarget], dtype=float)])
            ok_all = np.isfinite(x_all) & np.isfinite(y_all)
            if np.any(ok_all):
                xmin = float(np.min(x_all[ok_all]))
                xmax = float(np.max(x_all[ok_all]))
                ymin = float(np.min(y_all[ok_all]))
                ymax = float(np.max(y_all[ok_all]))
                dx = max(xmax - xmin, 1.0)
                dy = max(ymax - ymin, 1.0)
                pad = 0.15 * max(dx, dy)
                ax_map.set_xlim(xmin - pad, xmax + pad)
                ax_map.set_ylim(ymin - pad, ymax + pad)
            ax_map.set_xlabel('x [m]')
            ax_map.set_ylabel('y [m]')
            ax_map.grid(True)
            ax_map.set_aspect('equal', adjustable='box')
            ax_map.legend(loc='best', fontsize='small')

            # Indicate dominant incident-wave direction/spread from the averaged wavespec.
            # Convention: nautical compass degrees True, direction waves are coming FROM.
            # Plot arrow shows propagation direction (TO = FROM + 180 deg).
            try:
                from matplotlib.patches import Wedge

                dp = float(wavespec_bulk.get('Dp_deg', float('nan')))
                spread = float(wavespec_bulk.get('spreadp_deg', float('nan')))
                hs = float(wavespec_bulk.get('Hs_m', float('nan')))
                tp = float(wavespec_bulk.get('Tp_s', float('nan')))

                if np.isfinite(dp):
                    dp_to = (dp + 180.0) % 360.0

                    # Pick a reasonable arrow length from the current map extents.
                    xlim = ax_map.get_xlim()
                    ylim = ax_map.get_ylim()
                    span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
                    L = max(5.0, 0.08 * float(span))

                    # Compass degrees (0=N, 90=E) to ENU vector in x/y.
                    rad = np.deg2rad(dp_to)
                    dx_dir = float(np.sin(rad))
                    dy_dir = float(np.cos(rad))

                    ax_map.arrow(
                        float(xtarget),
                        float(ytarget),
                        L * dx_dir,
                        L * dy_dir,
                        head_width=max(1.0, 0.15 * L),
                        head_length=max(1.0, 0.20 * L),
                        length_includes_head=True,
                        color='k',
                        alpha=0.8,
                    )

                    if np.isfinite(spread) and spread > 0.0:
                        # Matplotlib Wedge angles are degrees CCW from +x.
                        # Convert compass (CW from +y) to math (CCW from +x): a_math = 90 - dp.
                        a0 = 90.0 - dp_to
                        wedge = Wedge(
                            (float(xtarget), float(ytarget)),
                            r=1.1 * L,
                            theta1=a0 - 0.5 * spread,
                            theta2=a0 + 0.5 * spread,
                            width=0.35 * L,
                            facecolor='k',
                            edgecolor='none',
                            alpha=0.12,
                        )
                        ax_map.add_patch(wedge)

                    label = []
                    if np.isfinite(hs):
                        label.append(f'Hs {hs:.2f} m')
                    if np.isfinite(tp):
                        label.append(f'Tp {tp:.2f} s')
                    label.append(f'Dp(from) {dp:.0f}°')
                    if np.isfinite(spread):
                        label.append(f'spread {spread:.0f}°')
                    ax_map.text(
                        0.02,
                        0.98,
                        ', '.join(label),
                        transform=ax_map.transAxes,
                        ha='left',
                        va='top',
                        fontsize=9,
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                        ),
                    )
            except Exception:
                # If plotting backends or patches aren't available, keep the example running.
                pass

            ax_z_in.plot(tin[inputwindow, :], zin[inputwindow, :])
            ax_z_in.set_ylabel('z in [m]')

            ax_u_in.plot(tin[inputwindow, :], uin[inputwindow, :])
            ax_u_in.set_ylabel('u in [m/s]')

            ax_v_in.plot(tin[inputwindow, :], vin[inputwindow, :])
            ax_v_in.set_ylabel('v in [m/s]')

            ax_z_rc.plot(tin[inputwindow, :], zr)
            ax_z_rc.set_ylabel('z recon [m]')

            ax_u_rc.plot(tin[inputwindow, :], ur)
            ax_u_rc.set_ylabel('u recon [m/s]')

            ax_v_rc.plot(tin[inputwindow, :], vr)
            ax_v_rc.set_ylabel('v recon [m/s]')
            ax_v_rc.set_xlabel('t [s]')

            ax_z_pr.plot(tpred, zout, 'k')
            ax_z_pr.set_ylabel('z pred [m]')

            ax_u_pr.plot(tpred, uout, 'k')
            ax_u_pr.set_ylabel('u pred [m/s]')

            ax_v_pr.plot(tpred, vout, 'k')
            ax_v_pr.set_ylabel('v pred [m/s]')
            ax_v_pr.set_xlabel('t [s]')

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
