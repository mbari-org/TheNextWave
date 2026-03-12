#!/usr/bin/python3
"""
Leave-one-out (LOO) validation example.

One buoy is withheld from the solver constellation and used as ground truth.
Its measured z/u/v at prediction times are compared directly against the
solver's prediction at that buoy's location.

Default: SWIFT25 (index 3) is the held-out target.
SWIFT22, 23, 24 (indices 0-2) drive the solver.
"""

import argparse
from pathlib import Path

from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
    p.add_argument(
        '--movie',
        type=str,
        default=None,
        help='Write animation to this path (.mp4 or .gif).',
    )
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument(
        '--fig-width', type=float, default=12.0,
        help='Figure width in inches.',
    )
    p.add_argument(
        '--fig-height', type=float, default=12.0,
        help='Figure height in inches.',
    )
    p.add_argument(
        '--matlab-pred-nc',
        type=str,
        default=None,
        help='Optional MATLAB NetCDF predictions file for secondary overlay.',
    )
    p.add_argument(
        '--matlab-window-warn-sec',
        type=float,
        default=0.5,
        help='Warn if MATLAB vs Python input window start/end differ by more than this [s].',
    )
    p.add_argument(
        '--solver-max-iter',
        type=int,
        default=5,
        help='Maximum solver iterations.',
    )
    p.add_argument(
        '--solver-backend',
        type=str,
        default='auto',
        choices=('auto', 'scipy', 'jax'),
        help='Solver backend.',
    )
    return p.parse_args()


A0 = None
all_preds = Prediction()
prev_t_start_diff = None
matlab_ti_offset = 2
matlab_ti_offset_increment = 1


def main():
    global A0, all_preds, prev_t_start_diff
    global matlab_ti_offset, matlab_ti_offset_increment

    args = parse_args()
    loo_idx = int(args.loo_idx)
    in_idxs = [i for i in range(len(BUOY_LABELS)) if i != loo_idx]
    max_iter = int(args.solver_max_iter)
    solver_backend = str(args.solver_backend)

    loo_label = BUOY_LABELS[loo_idx]
    in_labels  = [BUOY_LABELS[i] for i in in_idxs]
    print(f'LOO: held-out = {loo_label}  |  input = {in_labels}')
    print(f'solver backend: {solver_backend}  max_iter: {max_iter}')

    # --- data -----------------------------------------------------------
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation  = 0.0
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

    # Build wavespec from all-buoy processed burst structs (same as full example)
    wavespec_base = build_wavespec_from_swifts(swifts.bursts(raw_sbg=False), recip=True)
    Te, ce = centroid_period_and_phase_speed(wavespec_base)
    wavespec_bulk = bulk_wave_params_from_wavespec(wavespec_base)
    print(format_bulk_wave_params(wavespec_bulk, 'wavespec'))

    # Load full 4-buoy raw arrays, then split into input vs LOO columns
    zin_all, uin_all, vin_all, tin_all, xin_all, yin_all, fs = load_raw_arrays_from_sbg(
        swifts.bursts(raw_sbg=True),
        skipwarmup,
        burstend,
        latorigin,
        lonorigin,
        rotation,
        flip_z_sign=True,
    )

    # Input constellation (all columns except loo_idx)
    zin = zin_all[:, in_idxs]
    uin = uin_all[:, in_idxs]
    vin = vin_all[:, in_idxs]
    tin = tin_all[:, in_idxs]
    xin = xin_all[:, in_idxs]
    yin = yin_all[:, in_idxs]

    # Held-out buoy (LOO ground truth)
    z_loo = zin_all[:, loo_idx]   # (N,)
    u_loo = uin_all[:, loo_idx]
    v_loo = vin_all[:, loo_idx]
    t_loo = tin_all[:, loo_idx]
    x_loo = xin_all[:, loo_idx]
    y_loo = yin_all[:, loo_idx]

    n_in  = len(in_idxs)
    n     = zin.shape[0]

    matlab_ds = None
    matlab_ti = None
    if args.matlab_pred_nc:
        matlab_ds = xr.open_dataset(args.matlab_pred_nc)
        if 'ti' in matlab_ds:
            matlab_ti = np.asarray(matlab_ds['ti'].values, dtype=float).ravel()
        else:
            print('WARNING: matlab-pred-nc provided but missing variable `ti`; overlay disabled')
            matlab_ds = None

    NTe    = 10
    win_len = int(round(NTe * Te * fs))
    step    = int(round(fs))  # ~1 s increments

    if args.movie:
        plt.ioff()
    else:
        plt.ion()

    fw = max(float(args.fig_width),  8.0)
    fh = max(float(args.fig_height), 8.0)

    fig = plt.figure(1, figsize=(fw, fh))
    fig.set_size_inches(fw, fh, forward=True)
    fig.clf()
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass

    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        left=0.10, right=0.985, bottom=0.07, top=0.965,
        wspace=0.30, hspace=0.33,
    )

    ax_z_in = fig.add_subplot(gs[0, 0])
    ax_u_in = fig.add_subplot(gs[1, 0], sharex=ax_z_in)
    ax_v_in = fig.add_subplot(gs[2, 0], sharex=ax_z_in)
    ax_z_rc = fig.add_subplot(gs[3, 0], sharex=ax_z_in)
    ax_u_rc = fig.add_subplot(gs[4, 0], sharex=ax_z_in)
    ax_v_rc = fig.add_subplot(gs[5, 0], sharex=ax_z_in)

    ax_map  = fig.add_subplot(gs[0:3, 1])
    ax_z_pr = fig.add_subplot(gs[3, 1])
    ax_u_pr = fig.add_subplot(gs[4, 1], sharex=ax_z_pr)
    ax_v_pr = fig.add_subplot(gs[5, 1], sharex=ax_z_pr)

    axes_all = (
        ax_map,
        ax_z_in, ax_u_in, ax_v_in,
        ax_z_rc, ax_u_rc, ax_v_rc,
        ax_z_pr, ax_u_pr, ax_v_pr,
    )

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

    warned_mismatch = False

    def run_loop(grab_frame=False):
        global A0, all_preds, prev_t_start_diff
        global matlab_ti_offset, matlab_ti_offset_increment

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

            tpred = t_end + np.arange(1, n_lead + 1, dtype=float)
            xpred = np.full_like(tpred, xtarget)
            ypred = np.full_like(tpred, ytarget)

            # Ground truth: LOO buoy measured at prediction times
            t_loo_win  = t_loo[inputwindow]
            t_loo_full = t_loo   # full time series for interpolation

            # Interpolate LOO measurements at tpred times.
            # flip_z_sign=True negates z but not u/v.  The solver absorbs the z
            # sign flip into the fitted amplitudes and predicts zout in the
            # *original* (unflipped) sign convention.  To compare fairly, undo
            # the flip on z_loo so both sides are in the same space.
            ok_loo = np.isfinite(t_loo_full) & np.isfinite(z_loo)
            if np.sum(ok_loo) > 1:
                t_ok   = t_loo_full[ok_loo]
                z_gt   = np.interp(tpred, t_ok, -z_loo[ok_loo])  # undo flip_z_sign
                u_gt   = np.interp(tpred, t_ok,  u_loo[ok_loo])
                v_gt   = np.interp(tpred, t_ok,  v_loo[ok_loo])
            else:
                z_gt = u_gt = v_gt = np.full_like(tpred, np.nan)

            # Optional MATLAB overlay
            matlab_tpred = matlab_z = matlab_u = matlab_v = None

            if matlab_ds is not None and matlab_ti is not None and matlab_ti.size:
                ti_target = float(ti + 1 + matlab_ti_offset)
                idx = int(np.argmin(np.abs(matlab_ti - ti_target)))
                try:
                    start_1based = int(
                        np.asarray(matlab_ds['window_start_idx'].values).ravel()[idx]
                    )
                    count = int(np.asarray(matlab_ds['window_count'].values).ravel()[idx])
                    if count > 0:
                        start = start_1based - 1
                        stop  = start + count
                        matlab_tpred = np.asarray(matlab_ds['tpred_flat'].values[start:stop], dtype=float)
                        matlab_z     = np.asarray(matlab_ds['zout_flat'].values[start:stop], dtype=float)
                        matlab_u     = np.asarray(matlab_ds['uout_flat'].values[start:stop], dtype=float)
                        matlab_v     = np.asarray(matlab_ds['vout_flat'].values[start:stop], dtype=float)

                        warn_sec = float(args.matlab_window_warn_sec)
                        t_start_diff = abs(matlab_tpred[0] - tpred[0])
                        if t_start_diff > warn_sec:
                            if (
                                not warned_mismatch
                                and prev_t_start_diff is not None
                                and t_start_diff > prev_t_start_diff
                            ):
                                matlab_ti_offset_increment *= -1
                            prev_t_start_diff = t_start_diff
                            matlab_ti_offset += matlab_ti_offset_increment
                            matlab_tpred = matlab_z = matlab_u = matlab_v = None
                except Exception as exc:
                    if not warned_mismatch:
                        print(f'WARNING: failed to read MATLAB window (ti={ti_target}): {exc}')

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
                solver_backend=solver_backend,
            )
            A0 = params.A
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

            # --- plot ---------------------------------------------------
            for ax in axes_all:
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

            # Prediction panels: python prediction vs LOO ground truth (+ optional MATLAB)
            ax_z_pr.plot(tpred, zout, 'k',   label='predicted')
            ax_z_pr.plot(tpred, z_gt, color='limegreen', lw=1.5, label=f'{loo_label} measured')
            if matlab_tpred is not None:
                ax_z_pr.plot(matlab_tpred, matlab_z, color='steelblue', lw=1.0, ls='--', label='matlab')
            ax_z_pr.set_ylabel('z pred [m]')
            ax_z_pr.legend(loc='best', fontsize='small')
            z_lim = np.concatenate([zout, z_gt] + ([matlab_z] if matlab_z is not None else []))
            apply_latched_ylim(ax_z_pr, 'z_pred', z_lim)

            ax_u_pr.plot(tpred, uout, 'k')
            ax_u_pr.plot(tpred, u_gt, color='limegreen', lw=1.5)
            if matlab_tpred is not None:
                ax_u_pr.plot(matlab_tpred, matlab_u, color='steelblue', lw=1.0, ls='--')
            ax_u_pr.set_ylabel('u pred [m/s]')
            u_lim = np.concatenate([uout, u_gt] + ([matlab_u] if matlab_u is not None else []))
            apply_latched_ylim(ax_u_pr, 'u_pred', u_lim)

            ax_v_pr.plot(tpred, vout, 'k')
            ax_v_pr.plot(tpred, v_gt, color='limegreen', lw=1.5)
            if matlab_tpred is not None:
                ax_v_pr.plot(matlab_tpred, matlab_v, color='steelblue', lw=1.0, ls='--')
            ax_v_pr.set_ylabel('v pred [m/s]')
            ax_v_pr.set_xlabel('t [s]')
            v_lim = np.concatenate([vout, v_gt] + ([matlab_v] if matlab_v is not None else []))
            apply_latched_ylim(ax_v_pr, 'v_pred', v_lim)

            fig.suptitle(
                f'LOO={loo_label}  t=[{t_start:.0f},{t_end:.0f}]s  '
                f'horizon={n_lead}s  ct={comp_time:.3f}s',
                fontsize=9,
            )

            if grab_frame:
                writer.grab_frame()
            else:
                plt.pause(0.001)

    if writer is not None:
        with writer.saving(fig, str(movie_path), dpi=args.dpi):
            run_loop(grab_frame=True)
    else:
        run_loop(grab_frame=False)

    all_preds.to_netcdf(f'loo_{loo_label}_predictions.nc')
    if matlab_ds is not None:
        matlab_ds.close()


if __name__ == '__main__':
    main()
