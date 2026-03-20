#!/usr/bin/python3
"""
Leave-one-out (LOO) validation example.

One buoy is withheld from the solver constellation and used as ground truth.
Its measured z/u/v are plotted as a continuous timeseries; the solver's
prediction at that buoy's location is overlaid wherever it falls in time.

Default: SWIFT25 (index 3) is the held-out target.
SWIFT22, 23, 24 (indices 0-2) drive the solver.
"""

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
    bulk_wave_params_from_wavespec,
    centroid_period_and_phase_speed,
    format_bulk_wave_params,
    load_raw_arrays_from_sbg,
)

# Buoy labels in order matching swiftdat / sbgdat tuples (0-indexed)
BUOY_LABELS = ['swift22', 'swift23', 'swift24', 'swift25']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--loo-idx',
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help='Column index of the buoy to leave out (0=swift22 .. 3=swift25). Default 3.',
    )
    p.add_argument('--movie', type=str, default=None,
                   help='Write animation to this path (.mp4 or .gif).')
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--fig-width',  type=float, default=12.0)
    p.add_argument('--fig-height', type=float, default=12.0)
    p.add_argument('--solver-max-iter', type=int, default=5,
                   help='Maximum solver iterations.')
    p.add_argument('--solver-backend', type=str, default='auto',
                   choices=('auto', 'scipy', 'jax'))
    p.add_argument(
        '--wavespec-swift',
        type=int,
        default=None,
        choices=(1, 2, 3, 4),
        help='Use a single SWIFT buoy (1=SWIFT22, 2=SWIFT23, 3=SWIFT24, 4=SWIFT25) '
             'for the wavespec used in predictions. '
             'If omitted, uses the mean of all 4 buoys (default).',
    )
    return p.parse_args()


A0 = None
A0_indices = None
all_preds = Prediction()


def main():
    global A0, A0_indices, all_preds

    args = parse_args()
    loo_idx        = int(args.loo_idx)
    in_idxs        = [i for i in range(len(BUOY_LABELS)) if i != loo_idx]
    max_iter       = int(args.solver_max_iter)
    solver_backend = str(args.solver_backend)

    loo_label = BUOY_LABELS[loo_idx]
    in_labels = [BUOY_LABELS[i] for i in in_idxs]
    print(f'LOO: held-out = {loo_label}  |  input = {in_labels}')
    print(f'solver backend: {solver_backend}  max_iter: {max_iter}')

    # --- data -----------------------------------------------------------
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation  = 180.0  # MATLAB convention: rotation=180 → x=+East, y=+North
    skipwarmup = 200
    burstend   = 2740

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

    select_idx = 91
    swifts = SWIFTArray.from_mdat(swiftdat, sbgdat, select_idx)

    all_bursts = swifts.bursts(raw_sbg=False)
    swift_names = ['SWIFT22', 'SWIFT23', 'SWIFT24', 'SWIFT25']
    wavespec_swift_idx = args.wavespec_swift

    if wavespec_swift_idx is not None:
        sel_burst = all_bursts[wavespec_swift_idx - 1]
        sel_label = swift_names[wavespec_swift_idx - 1]
        wavespec_base = build_wavespec_from_swifts([sel_burst], recip=True)
        print(f'wavespec for predictions: {sel_label} (--wavespec-swift {wavespec_swift_idx})')
    else:
        wavespec_base = build_wavespec_from_swifts(all_bursts, recip=True)
        print('wavespec for predictions: mean of all 4 SWIFTs (default)')

    Te, ce = centroid_period_and_phase_speed(wavespec_base)
    wavespec_bulk = bulk_wave_params_from_wavespec(wavespec_base)
    print(format_bulk_wave_params(wavespec_bulk, 'wavespec'))

    zin_all, uin_all, vin_all, tin_all, xin_all, yin_all, fs = load_raw_arrays_from_sbg(
        swifts.bursts(raw_sbg=True),
        skipwarmup,
        burstend,
        latorigin,
        lonorigin,
        rotation,
        flip_z_sign=True,
    )

    # Input constellation
    zin = zin_all[:, in_idxs]
    uin = uin_all[:, in_idxs]
    vin = vin_all[:, in_idxs]
    tin = tin_all[:, in_idxs]
    xin = xin_all[:, in_idxs]
    yin = yin_all[:, in_idxs]

    # Held-out buoy full timeseries (ground truth)
    z_loo = zin_all[:, loo_idx]
    u_loo = uin_all[:, loo_idx]
    v_loo = vin_all[:, loo_idx]
    t_loo = tin_all[:, loo_idx]
    x_loo = xin_all[:, loo_idx]
    y_loo = yin_all[:, loo_idx]

    n_in = len(in_idxs)
    n    = zin.shape[0]

    NTe     = 10
    win_len = int(round(NTe * Te * fs))
    step    = int(round(fs))  # ~1 s increments

    # --- Figure 1: animation (map + input + recon + prediction) ---------
    if args.movie:
        plt.ioff()
    else:
        plt.ion()

    fw = max(float(args.fig_width),  8.0)
    fh = max(float(args.fig_height), 8.0)

    fig1 = plt.figure(1, figsize=(fw, fh))
    fig1.set_size_inches(fw, fh, forward=True)
    fig1.clf()
    try:
        fig1.set_constrained_layout(False)
    except Exception:
        pass

    gs1 = fig1.add_gridspec(
        nrows=6, ncols=2,
        left=0.10, right=0.985, bottom=0.07, top=0.965,
        wspace=0.30, hspace=0.33,
    )

    ax_z_in = fig1.add_subplot(gs1[0, 0])
    ax_u_in = fig1.add_subplot(gs1[1, 0], sharex=ax_z_in)
    ax_v_in = fig1.add_subplot(gs1[2, 0], sharex=ax_z_in)
    ax_z_rc = fig1.add_subplot(gs1[3, 0], sharex=ax_z_in)
    ax_u_rc = fig1.add_subplot(gs1[4, 0], sharex=ax_z_in)
    ax_v_rc = fig1.add_subplot(gs1[5, 0], sharex=ax_z_in)
    ax_map  = fig1.add_subplot(gs1[0:3, 1])
    ax_z_pr = fig1.add_subplot(gs1[3, 1])
    ax_u_pr = fig1.add_subplot(gs1[4, 1], sharex=ax_z_pr)
    ax_v_pr = fig1.add_subplot(gs1[5, 1], sharex=ax_z_pr)

    axes_fig1 = (
        ax_map,
        ax_z_in, ax_u_in, ax_v_in,
        ax_z_rc, ax_u_rc, ax_v_rc,
        ax_z_pr, ax_u_pr, ax_v_pr,
    )

    # --- Figure 2: study-style error summary ----------------------------
    fig2, axes2 = plt.subplots(3, 3, figsize=(fw * 1.1, fh * 1.1), num=2)
    fig2.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.93,
                         wspace=0.35, hspace=0.45)

    ax_z_ts   = axes2[0, 0]
    ax_u_ts   = axes2[0, 1]
    ax_v_ts   = axes2[0, 2]
    ax_z_err  = axes2[1, 0]
    ax_uv_err = axes2[1, 1]
    ax_ct     = axes2[1, 2]
    ax_xcorr  = axes2[2, 0]
    ax_peak   = axes2[2, 1]
    ax_npeak  = axes2[2, 2]

    # Storage for study metrics
    all_horizons_z  = []
    all_z_errors    = []
    all_horizons_uv = []
    all_uv_errors   = []
    all_comp_times  = []
    all_xcorr_lags  = []
    all_peak_errors = []
    all_npeak_diffs = []

    pred_latched_y_limits: dict[str, tuple[float, float]] = {}

    def apply_latched_ylim(ax, key: str, values) -> None:
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            prev = pred_latched_y_limits.get(key)
            if prev is not None:
                ax.set_ylim(*prev)
            return
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if np.isclose(vmin, vmax):
            pad = max(0.25, abs(vmin) * 0.1)
            cur_lo, cur_hi = vmin - pad, vmax + pad
        else:
            span = vmax - vmin
            pad = max(0.05 * span, 0.05)
            cur_lo, cur_hi = vmin - pad, vmax + pad
        prev = pred_latched_y_limits.get(key)
        if prev is None:
            latched = (cur_lo, cur_hi)
        else:
            latched = (min(prev[0], cur_lo), max(prev[1], cur_hi))
        pred_latched_y_limits[key] = latched
        ax.set_ylim(*latched)

    writer = None
    movie_path = None
    if args.movie:
        movie_path = Path(args.movie)
        movie_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = movie_path.suffix.lower()
        writer = PillowWriter(fps=args.fps) if suffix == '.gif' else FFMpegWriter(fps=args.fps)

    # Row 0 of Fig 2: draw the full LOO measured timeseries once (sticky baseline)
    ax_z_ts.plot(t_loo, z_loo, color='limegreen', lw=1.0, label=f'{loo_label}')
    ax_z_ts.set_ylabel('z [m]')
    ax_z_ts.set_xlabel('t [s]')
    ax_z_ts.set_title('z timeseries', fontsize=9)
    ax_z_ts.legend(loc='best', fontsize='small')

    ax_u_ts.plot(t_loo, u_loo, color='limegreen', lw=1.0)
    ax_u_ts.set_ylabel('u [m/s]')
    ax_u_ts.set_xlabel('t [s]')
    ax_u_ts.set_title('u timeseries', fontsize=9)

    ax_v_ts.plot(t_loo, v_loo, color='limegreen', lw=1.0)
    ax_v_ts.set_ylabel('v [m/s]')
    ax_v_ts.set_xlabel('t [s]')
    ax_v_ts.set_title('v timeseries', fontsize=9)

    def run_loop(grab_frame=False):
        global A0, A0_indices, all_preds

        for ti in range(0, n, step):
            inputwindow = ti + np.arange(win_len)
            if inputwindow[-1] >= n:
                break

            # Target = LOO buoy median position over the input window
            x_loo_win = x_loo[inputwindow]
            y_loo_win = y_loo[inputwindow]
            ok_pos = np.isfinite(x_loo_win) & np.isfinite(y_loo_win)
            if not np.any(ok_pos):
                continue
            xtarget = float(np.nanmedian(x_loo_win[ok_pos]))
            ytarget = float(np.nanmedian(y_loo_win[ok_pos]))

            # Lead time from LOO buoy to itself is 0, so use constellation spread
            dist = np.sqrt(
                (xin[inputwindow, :] - xtarget) ** 2 + (yin[inputwindow, :] - ytarget) ** 2
            )
            maxtargetdistance = float(np.nanmax(dist))
            leadtime = maxtargetdistance / ce
            n_lead = max(1, int(np.floor(leadtime)))

            t_start = float(np.nanmin(tin[inputwindow, :]))
            t_end   = float(np.nanmax(tin[inputwindow, :]))
            print(f'solving [{t_start:.1f}, {t_end:.1f}] s  target=({xtarget:.1f},{ytarget:.1f})')

            # tpred is derived entirely from the input constellation timing
            tpred = t_end + np.arange(1, n_lead + 1, dtype=float)
            xpred = np.full_like(tpred, xtarget)
            ypred = np.full_like(tpred, ytarget)

            # --- solve --------------------------------------------------
            ws = WaveSpec()
            ws.theta  = wavespec_base.theta.copy()
            ws.f      = wavespec_base.f.copy()
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
                max_iter=max_iter,
                A0_active_indices=None,
                solver_backend=solver_backend,
                ridge=0.0,
                use_spectrum_weighted_ridge=False,
                spectrum_ridge_floor=1e-6,
                diagnostics=True,
                gtol=0.1,
                lambda_time=0.0,
                lambda_freq_smooth=0.0,
                lambda_theta_smooth=0.0,
                freq_energy_frac=0.0,
                dir_energy_frac=0.0,
                active_grid_pad=1,
                print_losses=True,
                use_rank_reduction=False,
                use_row_scale=False,
                use_col_scale=False,
            )
            A0 = params.A
            A0_indices = params.active_good_indices
            print(f'  solve_time={comp_time:.3f}s')

            prediction     = np.asarray(pred_vec).reshape((tpred.size, -1), order='F')
            reconstruction = np.asarray(recon_vec).reshape((inputwindow.size, -1), order='F')
            zout = prediction[:, 0]
            uout = prediction[:, 1]
            vout = prediction[:, 2]
            zr   = reconstruction[:, 0:n_in]
            ur   = reconstruction[:, n_in:2 * n_in]
            vr   = reconstruction[:, 2 * n_in:3 * n_in]

            all_preds.append_window(
                window_start_time=t_start,
                tm=tin[inputwindow, :], zm=zin[inputwindow, :],
                um=uin[inputwindow, :], vm=vin[inputwindow, :],
                xm=xin[inputwindow, :], ym=yin[inputwindow, :],
                zc=zr, uc=ur, vc=vr,
                tp=tpred, zp=zout, up=uout, vp=vout,
                params=params, comp_time=float(comp_time),
            )

            all_comp_times.append(float(comp_time))

            # --- error metrics vs LOO ground truth ----------------------
            # Interpolate LOO onto tpred wherever tpred falls within t_loo
            ok_loo = np.isfinite(t_loo) & np.isfinite(z_loo)
            if np.sum(ok_loo) > 1:
                t_loo_ok = t_loo[ok_loo]
                within   = (tpred >= t_loo_ok[0]) & (tpred <= t_loo_ok[-1])
                if np.any(within):
                    tp_in    = tpred[within]
                    horizons = tp_in - t_end

                    z_gt_i = np.interp(tp_in, t_loo_ok, z_loo[ok_loo])
                    u_gt_i = np.interp(tp_in, t_loo_ok, u_loo[ok_loo])
                    v_gt_i = np.interp(tp_in, t_loo_ok, v_loo[ok_loo])

                    z_err  = np.abs(zout[within] - z_gt_i)
                    uv_err = np.sqrt(
                        0.5 * ((uout[within] - u_gt_i) ** 2 +
                               (vout[within] - v_gt_i) ** 2)
                    )

                    all_horizons_z.extend(horizons.tolist())
                    all_z_errors.extend(z_err.tolist())
                    all_horizons_uv.extend(horizons.tolist())
                    all_uv_errors.extend(uv_err.tolist())

                    # xcorr lag on z
                    try:
                        from scipy.signal import correlate
                        n_sig = len(zout[within])
                        xc    = correlate(
                            zout[within] - np.mean(zout[within]),
                            z_gt_i - np.mean(z_gt_i),
                            mode='full',
                        )
                        lags  = np.arange(-(n_sig - 1), n_sig)
                        lag_s = float(lags[np.argmax(xc)]) / fs
                        all_xcorr_lags.append(lag_s)
                    except Exception:
                        pass

                    # peak timing error
                    try:
                        from scipy.signal import find_peaks
                        pk_pred, _ = find_peaks(zout[within])
                        pk_gt,   _ = find_peaks(z_gt_i)
                        if pk_pred.size > 0 and pk_gt.size > 0:
                            for pp in pk_pred:
                                nearest = pk_gt[np.argmin(np.abs(pk_gt - pp))]
                                all_peak_errors.append(
                                    float(tp_in[pp] - tp_in[nearest])
                                )
                        all_npeak_diffs.append(int(pk_pred.size) - int(pk_gt.size))
                    except Exception:
                        pass

            # --- Fig 1: animation plots ---------------------------------
            for ax in axes_fig1:
                ax.cla()

            colors = (
                plt.rcParams['axes.prop_cycle']
                .by_key()
                .get('color', ['C0', 'C1', 'C2', 'C3'])
            )
            xw = np.asarray(xin[inputwindow, :], dtype=float)
            yw = np.asarray(yin[inputwindow, :], dtype=float)

            buoy_x, buoy_y = [], []
            for j, lbl in enumerate(in_labels):
                xj, yj = xw[:, j], yw[:, j]
                ok = np.isfinite(xj) & np.isfinite(yj)
                if not np.any(ok):
                    continue
                ax_map.plot(xj[ok], yj[ok], '.', color=colors[j % len(colors)],
                            alpha=0.25, markersize=2)
                x_med = float(np.nanmedian(xj[ok]))
                y_med = float(np.nanmedian(yj[ok]))
                buoy_x.append(x_med)
                buoy_y.append(y_med)
                ax_map.plot([x_med], [y_med], marker='x', color=colors[j % len(colors)],
                            markersize=8, mew=2, linestyle='None', label=lbl)

            # LOO buoy track (grey) + current position
            loo_xw = x_loo[inputwindow]
            loo_yw = y_loo[inputwindow]
            ok_loo_w = np.isfinite(loo_xw) & np.isfinite(loo_yw)
            if np.any(ok_loo_w):
                ax_map.plot(loo_xw[ok_loo_w], loo_yw[ok_loo_w], '.', color='gray',
                            alpha=0.25, markersize=2)
            ax_map.plot([xtarget], [ytarget], 'k^', markersize=8,
                        label=f'{loo_label} (target/truth)')

            x_all = np.concatenate([np.asarray(buoy_x), [xtarget]])
            y_all = np.concatenate([np.asarray(buoy_y), [ytarget]])
            ok_all = np.isfinite(x_all) & np.isfinite(y_all)
            if np.any(ok_all):
                dx  = max(np.max(x_all[ok_all]) - np.min(x_all[ok_all]), 1.0)
                dy  = max(np.max(y_all[ok_all]) - np.min(y_all[ok_all]), 1.0)
                pad = 0.15 * max(dx, dy)
                ax_map.set_xlim(np.min(x_all[ok_all]) - pad, np.max(x_all[ok_all]) + pad)
                ax_map.set_ylim(np.min(y_all[ok_all]) - pad, np.max(y_all[ok_all]) + pad)
            ax_map.set_xlabel('x [m]')
            ax_map.set_ylabel('y [m]')
            ax_map.grid(True)
            ax_map.set_aspect('equal', adjustable='box')
            ax_map.legend(loc='best', fontsize='small')

            try:
                from matplotlib.patches import Wedge
                dp     = float(wavespec_bulk.get('Dp_deg',     float('nan')))
                spread = float(wavespec_bulk.get('spreadp_deg', float('nan')))
                hs     = float(wavespec_bulk.get('Hs_m',       float('nan')))
                tp_s   = float(wavespec_bulk.get('Tp_s',       float('nan')))
                if np.isfinite(dp):
                    dp_to = (dp + 180.0) % 360.0
                    xlim = ax_map.get_xlim(); ylim = ax_map.get_ylim()
                    L = max(5.0, 0.08 * max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])))
                    rad = np.deg2rad(dp_to)
                    ax_map.arrow(xtarget, ytarget,
                                 L * float(np.sin(rad)), L * float(np.cos(rad)),
                                 head_width=max(1.0, 0.15 * L), head_length=max(1.0, 0.20 * L),
                                 length_includes_head=True, color='k', alpha=0.8)
                    if np.isfinite(spread) and spread > 0.0:
                        a0 = 90.0 - dp_to
                        ax_map.add_patch(Wedge(
                            (xtarget, ytarget), r=1.1 * L,
                            theta1=a0 - 0.5 * spread, theta2=a0 + 0.5 * spread,
                            width=0.35 * L, facecolor='k', edgecolor='none', alpha=0.12,
                        ))
                    parts = []
                    if np.isfinite(hs):    parts.append(f'Hs {hs:.2f} m')
                    if np.isfinite(tp_s):  parts.append(f'Tp {tp_s:.2f} s')
                    parts.append(f'Dp(from) {dp:.0f}°')
                    if np.isfinite(spread): parts.append(f'spread {spread:.0f}°')
                    ax_map.text(0.02, 0.98, ', '.join(parts), transform=ax_map.transAxes,
                                ha='left', va='top', fontsize=9,
                                bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white',
                                      'alpha': 0.7, 'edgecolor': 'none'})
            except Exception:
                pass

            # Input panels (constellation buoys only, n_in lines per panel)
            ax_z_in.plot(tin[inputwindow, :], zin[inputwindow, :])
            ax_z_in.set_ylabel('z in [m]')
            apply_latched_ylim(ax_z_in, 'z_in', zin)

            ax_u_in.plot(tin[inputwindow, :], uin[inputwindow, :])
            ax_u_in.set_ylabel('u in [m/s]')
            apply_latched_ylim(ax_u_in, 'u_in', uin)

            ax_v_in.plot(tin[inputwindow, :], vin[inputwindow, :])
            ax_v_in.set_ylabel('v in [m/s]')
            apply_latched_ylim(ax_v_in, 'v_in', vin)

            ax_z_rc.plot(tin[inputwindow, :], zr)
            ax_z_rc.set_ylabel('z recon [m]')
            apply_latched_ylim(ax_z_rc, 'z_recon', zr)

            ax_u_rc.plot(tin[inputwindow, :], ur)
            ax_u_rc.set_ylabel('u recon [m/s]')
            apply_latched_ylim(ax_u_rc, 'u_recon', ur)

            ax_v_rc.plot(tin[inputwindow, :], vr)
            ax_v_rc.set_ylabel('v recon [m/s]')
            ax_v_rc.set_xlabel('t [s]')
            apply_latched_ylim(ax_v_rc, 'v_recon', vr)

            # Prediction panels: LOO segment spanning input+prediction window (green) + prediction (black)
            loo_win = (t_loo >= t_start) & (t_loo <= tpred[-1])
            t_loo_w = t_loo[loo_win]
            z_loo_w = z_loo[loo_win]
            u_loo_w = u_loo[loo_win]
            v_loo_w = v_loo[loo_win]

            ax_z_pr.plot(t_loo_w, z_loo_w, color='limegreen', lw=1.0,
                         label=f'{loo_label} measured')
            ax_z_pr.plot(tpred, zout, 'k', lw=1.5, label='predicted')
            ax_z_pr.set_ylabel('z pred [m]')
            ax_z_pr.legend(loc='best', fontsize='small')
            apply_latched_ylim(ax_z_pr, 'z_pred',
                               np.concatenate([z_loo_w[np.isfinite(z_loo_w)], zout]))

            ax_u_pr.plot(t_loo_w, u_loo_w, color='limegreen', lw=1.0)
            ax_u_pr.plot(tpred, uout, 'k', lw=1.5)
            ax_u_pr.set_ylabel('u pred [m/s]')
            apply_latched_ylim(ax_u_pr, 'u_pred',
                               np.concatenate([u_loo_w[np.isfinite(u_loo_w)], uout]))

            ax_v_pr.plot(t_loo_w, v_loo_w, color='limegreen', lw=1.0)
            ax_v_pr.plot(tpred, vout, 'k', lw=1.5)
            ax_v_pr.set_ylabel('v pred [m/s]')
            ax_v_pr.set_xlabel('t [s]')
            apply_latched_ylim(ax_v_pr, 'v_pred',
                               np.concatenate([v_loo_w[np.isfinite(v_loo_w)], vout]))

            fig1.suptitle(
                f'LOO={loo_label}  t=[{t_start:.0f},{t_end:.0f}]s  '
                f'horizon={n_lead}s  ct={comp_time:.3f}s',
                fontsize=9,
            )

            # --- Fig 2: study-style summary (accumulated) ---------------
            # Only clear error/metric panels (row 0 is sticky)
            for ax in [ax_z_err, ax_uv_err, ax_ct, ax_xcorr, ax_peak, ax_npeak]:
                ax.cla()

            # Row 0: append black prediction window (green LOO line drawn once outside loop)
            ax_z_ts.plot(tpred, zout, 'k', lw=0.8, alpha=0.4)
            ax_u_ts.plot(tpred, uout, 'k', lw=0.8, alpha=0.4)
            ax_v_ts.plot(tpred, vout, 'k', lw=0.8, alpha=0.4)

            # Row 1: error vs horizon
            if all_horizons_z:
                hz = np.asarray(all_horizons_z)
                ez = np.asarray(all_z_errors)
                ax_z_err.scatter(hz, ez, s=4, color='steelblue', alpha=0.3)
                sort_idx = np.argsort(hz)
                k = min(50, len(ez))
                ax_z_err.plot(hz[sort_idx],
                              np.convolve(ez[sort_idx], np.ones(k) / k, mode='same'),
                              color='steelblue', lw=1.5, label=f'iter={max_iter}')
                ax_z_err.legend(loc='best', fontsize='small')
            ax_z_err.set_xlabel('prediction horizon [s]')
            ax_z_err.set_ylabel('|z error| [m]')
            ax_z_err.set_title('z prediction error vs horizon', fontsize=9)

            if all_horizons_uv:
                huv = np.asarray(all_horizons_uv)
                euv = np.asarray(all_uv_errors)
                ax_uv_err.scatter(huv, euv, s=4, color='steelblue', alpha=0.3)
                sort_idx = np.argsort(huv)
                k = min(50, len(euv))
                ax_uv_err.plot(huv[sort_idx],
                               np.convolve(euv[sort_idx], np.ones(k) / k, mode='same'),
                               color='steelblue', lw=1.5)
            ax_uv_err.set_xlabel('prediction horizon [s]')
            ax_uv_err.set_ylabel('RMS(u,v) error [m/s]')
            ax_uv_err.set_title('u+v prediction error vs horizon', fontsize=9)

            if all_comp_times:
                ax_ct.boxplot([all_comp_times], positions=[1],
                              widths=0.4, patch_artist=True,
                              boxprops=dict(facecolor='steelblue', alpha=0.5),
                              medianprops=dict(color='k', lw=1.5))
                ax_ct.set_xticks([1])
                ax_ct.set_xticklabels([str(max_iter)])
            ax_ct.set_ylabel('solve time [s]')
            ax_ct.set_xlabel('max_iter')
            ax_ct.set_title('solve time per window', fontsize=9)

            # Row 2: distribution plots
            for ax, data, title, ylabel in [
                (ax_xcorr, all_xcorr_lags,  'xcorr lag (+ = prediction early)', 'lag [s]'),
                (ax_peak,  all_peak_errors, 'peak/trough timing error',          '\u0394t peak [s]'),
                (ax_npeak, all_npeak_diffs, 'peak count diff vs reference',      '\u0394n peaks'),
            ]:
                if len(data) > 1:
                    ax.violinplot([data], positions=[1],
                                  showmedians=True, showextrema=True)
                elif len(data) == 1:
                    ax.plot([1], data, 'ko', markersize=5)
                ax.axhline(0, color='k', lw=0.8, ls='--')
                ax.set_title(title, fontsize=9)
                ax.set_ylabel(ylabel)
                ax.set_xlabel('max_iter')
                ax.set_xticks([1])
                ax.set_xticklabels([str(max_iter)])

            fig2.suptitle(
                f'LOO={loo_label}  error summary  '
                f'(windows processed: {len(all_comp_times)})',
                fontsize=9,
            )

            if grab_frame:
                writer.grab_frame()
            else:
                plt.pause(0.001)

    if writer is not None:
        with writer.saving(fig1, str(movie_path), dpi=args.dpi):
            run_loop(grab_frame=True)
    else:
        run_loop(grab_frame=False)

    all_preds.to_netcdf(f'loo_{loo_label}_predictions.nc')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
