#!/usr/bin/python3
"""
max_iter sweep study.

Runs each prediction window through all max_iter values in the sweep list.

Figure 1 (reproj): same panel layout as example.py, updated per window, using
                   the middle max_iter case only.
Figure 2 (study):  one row of z/u/v prediction panels per max_iter value, all
                   windows accumulated on the same axes.  Bottom summary row:
                   z-error vs prediction horizon, u+v-error vs prediction horizon,
                   and solve-time box plot.  Error and timing data are collected
                   only after the Python window aligns with the MATLAB reference.
"""

import argparse
from pathlib import Path

from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from scipy.signal import find_peaks
import xarray as xr

from .download_example_data import (
    get_default_example_name,
    get_example_select_idx,
    get_example_sbg_paths,
    get_example_swift_paths,
    load_example_sbg_bursts,
    load_example_wavespec,
)
from .leastSquaresWavePropagation import leastSquaresWavePropagation
from .swift import SWIFTArray, WaveSpec
from .utilities import (
    build_wavespec_from_swifts,
    bulk_wave_params_from_wavespec,
    centroid_period_and_phase_speed,
    format_bulk_wave_params,
    load_raw_arrays_from_sbg,
)

DEFAULT_MAX_ITERS = [1, 5, 15, 60]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--example-name',
        type=str,
        default=None,
        help='Folder name under example_data/ to use. Defaults to ExampleData1 when present, else the first available dataset.',
    )
    p.add_argument(
        '--solver-max-iters',
        type=str,
        default=','.join(str(m) for m in DEFAULT_MAX_ITERS),
        help='Comma-separated max_iter values to sweep, e.g. "1,5,15,60".',
    )
    p.add_argument(
        '--solver-backend',
        type=str,
        default='auto',
        choices=('auto', 'scipy', 'jax'),
        help='Solver backend.',
    )
    p.add_argument(
        '--matlab-pred-nc',
        type=str,
        default=None,
        help='Optional MATLAB NetCDF predictions file for overlay and error comparison.',
    )
    p.add_argument(
        '--matlab-window-warn-sec',
        type=float,
        default=0.5,
        help='Alignment threshold for MATLAB vs Python window start [s].',
    )
    p.add_argument(
        '--movie',
        type=str,
        default=None,
        help='Write reproj figure animation to this path (.mp4 or .gif).',
    )
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--fig-width', type=float, default=14.0)
    p.add_argument('--fig-height', type=float, default=12.0)
    p.add_argument(
        '--swift-gt-idx',
        type=int,
        default=None,
        choices=(0, 1, 2, 3),
        help=(
            'Hold out this SWIFT buoy (0=swift22 .. 3=swift25) as ground truth '
            'instead of MATLAB data.  The remaining buoys drive the solver; the '
            'held-out buoy is used for overlay and error statistics.'
        ),
    )
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


def _iter_colors(n):
    """Return n visually distinct colors from tab10."""
    cmap = mcm.get_cmap('tab10')
    return [cmap(i / max(n - 1, 1) * 0.9) for i in range(n)]


def main():
    args = parse_args()

    max_iters = sorted(int(x.strip()) for x in args.solver_max_iters.split(',') if x.strip())
    if not max_iters:
        max_iters = list(DEFAULT_MAX_ITERS)
    n_iters = len(max_iters)
    mid_idx = n_iters // 2  # index into max_iters used for the reproj figure

    solver_backend = str(args.solver_backend)
    print(f'solver backend: {solver_backend}')
    print(f'max_iter sweep: {max_iters}  (reproj uses max_iter={max_iters[mid_idx]})')

    iter_colors = _iter_colors(n_iters)

    # --- data -----------------------------------------------------------
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation = 180.0
    xtarget = 200.0
    ytarget = 200.0
    skipwarmup = 200
    burstend = 2740

    example_name = args.example_name or get_default_example_name()
    sbgdat = get_example_sbg_paths(example_name=example_name)
    sbg_bursts = load_example_sbg_bursts(example_name=example_name)

    print(f'example dataset: {example_name}')

    wavespec_base = load_example_wavespec(example_name=example_name)
    print(f'wavespec for predictions: wavespec.mat from {example_name} (default)')

    swiftdat = get_example_swift_paths(example_name=example_name)
    if args.wavespec_swift is not None:
        if swiftdat is None:
            print(
                'WARNING: --wavespec-swift requested, but this example bundle does not '
                'include per-buoy processed SWIFT files; using wavespec.mat instead'
            )
        else:
            swifts = SWIFTArray.from_mdat(
                swiftdat,
                sbgdat,
                get_example_select_idx(example_name=example_name),
            )
            all_bursts = swifts.bursts(raw_sbg=False)
            swift_names = ['SWIFT22', 'SWIFT23', 'SWIFT24', 'SWIFT25']
            wavespec_swift_idx = args.wavespec_swift
            sel_burst = all_bursts[wavespec_swift_idx - 1]
            sel_label = swift_names[wavespec_swift_idx - 1]
            wavespec_base = build_wavespec_from_swifts([sel_burst], recip=True)
            print(
                f'wavespec for predictions overridden with {sel_label} '
                f'(--wavespec-swift {wavespec_swift_idx})'
            )

    Te, ce = centroid_period_and_phase_speed(wavespec_base)
    wavespec_bulk = bulk_wave_params_from_wavespec(wavespec_base)
    print(format_bulk_wave_params(wavespec_bulk, 'wavespec'))

    swift_gt_idx = args.swift_gt_idx
    BUOY_LABELS = ['swift22', 'swift23', 'swift24', 'swift25']

    zin_all, uin_all, vin_all, tin_all, xin_all, yin_all, fs = load_raw_arrays_from_sbg(
        sbg_bursts,
        skipwarmup,
        burstend,
        latorigin,
        lonorigin,
        rotation,
        flip_z_sign=True,
    )

    if swift_gt_idx is not None:
        in_idxs = [i for i in range(len(BUOY_LABELS)) if i != swift_gt_idx]
        zin = zin_all[:, in_idxs]
        uin = uin_all[:, in_idxs]
        vin = vin_all[:, in_idxs]
        tin = tin_all[:, in_idxs]
        xin = xin_all[:, in_idxs]
        yin = yin_all[:, in_idxs]
        z_gt_full = zin_all[:, swift_gt_idx]
        u_gt_full = uin_all[:, swift_gt_idx]
        v_gt_full = vin_all[:, swift_gt_idx]
        t_gt_full = tin_all[:, swift_gt_idx]
        x_gt_full = xin_all[:, swift_gt_idx]
        y_gt_full = yin_all[:, swift_gt_idx]
        gt_label  = BUOY_LABELS[swift_gt_idx]
        in_labels = [BUOY_LABELS[i] for i in in_idxs]
        print(f'GT buoy: {gt_label}  |  input: {in_labels}')
    else:
        zin = zin_all
        uin = uin_all
        vin = vin_all
        tin = tin_all
        xin = xin_all
        yin = yin_all
        z_gt_full = u_gt_full = v_gt_full = t_gt_full = None
        x_gt_full = y_gt_full = None
        gt_label  = None
        in_labels = BUOY_LABELS

    ref_label = gt_label if swift_gt_idx is not None else 'matlab'

    nbuoys = zin.shape[1]
    n = zin.shape[0]

    matlab_ds = None
    matlab_ti = None
    if args.matlab_pred_nc:
        matlab_ds = xr.open_dataset(args.matlab_pred_nc)
        if 'ti' in matlab_ds:
            matlab_ti = np.asarray(matlab_ds['ti'].values, dtype=float).ravel()
        else:
            print('WARNING: matlab-pred-nc provided but missing variable `ti`; overlay disabled')
            matlab_ds = None

    NTe = 10
    win_len = int(round(NTe * Te * fs))
    step = int(round(fs))

    # --- warm-start A0 per iter slot ------------------------------------
    A0_list = [None] * n_iters

    # --- accumulated study data ----------------------------------------
    # errors_z[i], errors_uv[i]: list of 1-D arrays (one per aligned window)
    errors_z_by_iter = [[] for _ in range(n_iters)]
    errors_uv_by_iter = [[] for _ in range(n_iters)]
    times_by_iter = [[] for _ in range(n_iters)]
    xcorr_lags_by_iter = [[] for _ in range(n_iters)]      # seconds, one per aligned window
    peak_dt_by_iter = [[] for _ in range(n_iters)]          # seconds, all matched peaks
    peak_count_diff_by_iter = [[] for _ in range(n_iters)]  # int, one per aligned window

    # --- figures --------------------------------------------------------
    if args.movie:
        plt.ioff()
    else:
        plt.ion()

    fw = max(float(args.fig_width), 8.0)
    fh = max(float(args.fig_height), 8.0)

    # Figure 1: reproj (middle iter, updated each window)
    fig_r = plt.figure(1, figsize=(fw, fh))
    fig_r.clf()
    try:
        fig_r.set_constrained_layout(False)
    except Exception:
        pass

    gs_r = fig_r.add_gridspec(
        nrows=6, ncols=2,
        left=0.10, right=0.985, bottom=0.07, top=0.965,
        wspace=0.30, hspace=0.33,
    )
    ax_z_in = fig_r.add_subplot(gs_r[0, 0])
    ax_u_in = fig_r.add_subplot(gs_r[1, 0], sharex=ax_z_in)
    ax_v_in = fig_r.add_subplot(gs_r[2, 0], sharex=ax_z_in)
    ax_z_rc = fig_r.add_subplot(gs_r[3, 0], sharex=ax_z_in)
    ax_u_rc = fig_r.add_subplot(gs_r[4, 0], sharex=ax_z_in)
    ax_v_rc = fig_r.add_subplot(gs_r[5, 0], sharex=ax_z_in)
    ax_map  = fig_r.add_subplot(gs_r[0:3, 1])
    ax_z_pr = fig_r.add_subplot(gs_r[3, 1])
    ax_u_pr = fig_r.add_subplot(gs_r[4, 1], sharex=ax_z_pr)
    ax_v_pr = fig_r.add_subplot(gs_r[5, 1], sharex=ax_z_pr)

    axes_reproj = (
        ax_map, ax_z_in, ax_u_in, ax_v_in,
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

    # Figure 2: study — n_iters rows of prediction panels + 2 summary rows
    n_study_rows = n_iters + 2
    study_fig_w = max(fw * 1.1, 16.0)
    study_fig_h = max(fh * 1.1, 3.0 * n_study_rows)
    fig_s = plt.figure(2, figsize=(study_fig_w, study_fig_h))
    fig_s.clf()
    try:
        fig_s.set_constrained_layout(False)
    except Exception:
        pass

    gs_s = fig_s.add_gridspec(
        nrows=n_study_rows, ncols=3,
        left=0.08, right=0.97, bottom=0.06, top=0.96,
        wspace=0.30, hspace=0.40,
    )

    # Prediction panels  axes_study_pred[i] = (ax_z, ax_u, ax_v)
    axes_study_pred = []
    for i in range(n_iters):
        az = fig_s.add_subplot(gs_s[i, 0])
        au = fig_s.add_subplot(gs_s[i, 1], sharex=az)
        av = fig_s.add_subplot(gs_s[i, 2], sharex=az)
        axes_study_pred.append((az, au, av))

    # Summary row 0: error vs horizon + solve time
    ax_err_z  = fig_s.add_subplot(gs_s[n_iters, 0])
    ax_err_uv = fig_s.add_subplot(gs_s[n_iters, 1])
    ax_time   = fig_s.add_subplot(gs_s[n_iters, 2])

    # Summary row 1: phase / peak alignment
    ax_xcorr  = fig_s.add_subplot(gs_s[n_iters + 1, 0])
    ax_pkdt   = fig_s.add_subplot(gs_s[n_iters + 1, 1])
    ax_pkdt2  = fig_s.add_subplot(gs_s[n_iters + 1, 2])

    # Label study prediction rows
    for i, mi in enumerate(max_iters):
        az, au, av = axes_study_pred[i]
        az.set_ylabel(f'z [m]\niter={mi}', fontsize=8)
        au.set_ylabel(f'u [m/s]\niter={mi}', fontsize=8)
        av.set_ylabel(f'v [m/s]\niter={mi}', fontsize=8)
    ax_err_z.set_xlabel('prediction horizon [s]')
    ax_err_z.set_ylabel('|z error| [m]')
    ax_err_z.set_title('z prediction error vs horizon')
    ax_err_uv.set_xlabel('prediction horizon [s]')
    ax_err_uv.set_ylabel('RMS(u,v) error [m/s]')
    ax_err_uv.set_title('u+v prediction error vs horizon')
    ax_time.set_xlabel('max_iter')
    ax_time.set_ylabel('solve time [s]')
    ax_time.set_title('solve time per window')

    # --- movie writer ---------------------------------------------------
    writer = None
    movie_path = None
    if args.movie:
        movie_path = Path(args.movie)
        movie_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = movie_path.suffix.lower()
        writer = PillowWriter(fps=args.fps) if suffix == '.gif' else FFMpegWriter(fps=args.fps)

    # --- MATLAB alignment state ----------------------------------------
    prev_t_start_diff = None
    matlab_ti_offset = 2
    matlab_ti_offset_increment = 1
    warned_mismatch = False

    def run_loop(grab_frame=False):
        nonlocal A0_list
        nonlocal prev_t_start_diff, matlab_ti_offset, matlab_ti_offset_increment
        nonlocal warned_mismatch

        for ti in range(0, n, step):
            inputwindow = ti + np.arange(win_len)
            if inputwindow[-1] >= n:
                break

            # Resolve per-window target position
            if swift_gt_idx is not None:
                ok_pos = np.isfinite(x_gt_full[inputwindow]) & np.isfinite(y_gt_full[inputwindow])
                if not np.any(ok_pos):
                    continue
                xt = float(np.nanmedian(x_gt_full[inputwindow][ok_pos]))
                yt = float(np.nanmedian(y_gt_full[inputwindow][ok_pos]))
            else:
                xt, yt = float(xtarget), float(ytarget)

            dist = np.sqrt(
                (xin[inputwindow, :] - xt) ** 2 + (yin[inputwindow, :] - yt) ** 2
            )
            maxtargetdistance = float(np.nanmax(dist))
            leadtime = maxtargetdistance / ce
            n_lead = max(1, int(np.floor(leadtime)))

            t_start = float(np.nanmin(tin[inputwindow, :]))
            t_end   = float(np.nanmax(tin[inputwindow, :]))
            print(f'solving prediction window: [{t_start:.1f}, {t_end:.1f}] s')
            tpred = t_end + np.arange(1, n_lead + 1, dtype=float)

            # -- MATLAB reference overlay --------------------------------
            matlab_tpred = matlab_z = matlab_u = matlab_v = None
            aligned = False

            if matlab_ds is not None and matlab_ti is not None and matlab_ti.size:
                ti_target = float(ti + 1 + matlab_ti_offset)
                idx = int(np.argmin(np.abs(matlab_ti - ti_target)))
                print(f'  ti target={ti_target:.1f}, matlab ti={matlab_ti[idx]:.1f} (idx={idx})')
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
                        print(f'  window start diff = {t_start_diff:.3f}s (threshold {warn_sec:.3f}s)')

                        if t_start_diff > warn_sec:
                            if (
                                not warned_mismatch
                                and prev_t_start_diff is not None
                                and t_start_diff > prev_t_start_diff
                            ):
                                matlab_ti_offset_increment *= -1
                            prev_t_start_diff = t_start_diff
                            matlab_ti_offset += matlab_ti_offset_increment
                        else:
                            aligned = True
                except Exception as exc:
                    if not warned_mismatch:
                        print(f'WARNING: failed to read MATLAB window (ti={ti_target}): {exc}')
                        warned_mismatch = True

            xpred = np.full_like(tpred, xt)
            ypred = np.full_like(tpred, yt)

            # If a SWIFT GT buoy is selected, populate matlab_* from it (drop-in for MATLAB)
            if swift_gt_idx is not None and t_gt_full is not None:
                matlab_tpred = t_gt_full
                matlab_z     = z_gt_full
                matlab_u     = u_gt_full
                matlab_v     = v_gt_full
                aligned = True

            # -- solve for each max_iter value ---------------------------
            results = []
            for iter_idx, mi in enumerate(max_iters):
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
                    A0=A0_list[iter_idx],
                    max_iter=mi,
                    solver_backend=solver_backend,
                )
                A0_list[iter_idx] = params.A
                print(f'  max_iter={mi:4d}  solve_time={comp_time:.3f}s')

                prediction    = np.asarray(pred_vec).reshape((tpred.size, -1), order='F')
                reconstruction = np.asarray(recon_vec).reshape((inputwindow.size, -1), order='F')
                results.append({
                    'zout': prediction[:, 0],
                    'uout': prediction[:, 1],
                    'vout': prediction[:, 2],
                    'zr': reconstruction[:, 0:nbuoys],
                    'ur': reconstruction[:, nbuoys:2 * nbuoys],
                    'vr': reconstruction[:, 2 * nbuoys:3 * nbuoys],
                    'comp_time': float(comp_time),
                })

                # accumulate timing (always)
                times_by_iter[iter_idx].append(float(comp_time))

                # accumulate errors (aligned windows only)
                if aligned and matlab_tpred is not None and matlab_tpred.size > 0:
                    mz_i = np.interp(tpred, matlab_tpred, matlab_z)
                    mu_i = np.interp(tpred, matlab_tpred, matlab_u)
                    mv_i = np.interp(tpred, matlab_tpred, matlab_v)
                    pz   = prediction[:, 0]
                    ez   = np.abs(pz - mz_i)
                    euv  = np.sqrt(
                        (prediction[:, 1] - mu_i) ** 2 + (prediction[:, 2] - mv_i) ** 2
                    )
                    errors_z_by_iter[iter_idx].append(ez)
                    errors_uv_by_iter[iter_idx].append(euv)

                    # cross-correlation lag (positive = prediction is early / leads reference)
                    n_sig = len(pz)
                    if n_sig > 4:
                        pz_c  = pz - pz.mean()
                        mz_c  = mz_i - mz_i.mean()
                        xc    = np.correlate(pz_c, mz_c, mode='full')
                        lags  = np.arange(-(n_sig - 1), n_sig)
                        dt_s  = float(tpred[1] - tpred[0]) if n_sig > 1 else 1.0
                        lag_s = float(lags[int(np.argmax(xc))]) * dt_s
                        xcorr_lags_by_iter[iter_idx].append(lag_s)

                    # peak/trough timing error — nearest-neighbour match
                    prominence = float(np.std(mz_i)) * 0.5 if np.std(mz_i) > 0 else 0.1
                    m_pk, _  = find_peaks( mz_i, prominence=prominence)
                    m_tr, _  = find_peaks(-mz_i, prominence=prominence)
                    m_all    = np.sort(np.concatenate([m_pk, m_tr]))
                    p_pk, _  = find_peaks( pz,   prominence=prominence)
                    p_tr, _  = find_peaks(-pz,   prominence=prominence)
                    p_all    = np.sort(np.concatenate([p_pk, p_tr]))

                    peak_count_diff_by_iter[iter_idx].append(len(p_all) - len(m_all))

                    if m_all.size > 0 and p_all.size > 0:
                        dt_s = float(tpred[1] - tpred[0]) if len(tpred) > 1 else 1.0
                        for m_idx in m_all:
                            nearest_p = p_all[int(np.argmin(np.abs(p_all - m_idx)))]
                            peak_dt_by_iter[iter_idx].append(
                                float(nearest_p - m_idx) * dt_s
                            )

            # mid-iter result used for reproj figure
            mid = results[mid_idx]

            # -- update reproj figure (middle iter) ----------------------
            for ax in axes_reproj:
                ax.cla()

            colors = (
                plt.rcParams['axes.prop_cycle']
                .by_key()
                .get('color', ['C0', 'C1', 'C2', 'C3'])
            )
            xw = np.asarray(xin[inputwindow, :], dtype=float)
            yw = np.asarray(yin[inputwindow, :], dtype=float)
            buoy_x, buoy_y = [], []
            for j in range(nbuoys):
                xj, yj = xw[:, j], yw[:, j]
                ok = np.isfinite(xj) & np.isfinite(yj)
                if not np.any(ok):
                    continue
                ax_map.plot(xj[ok], yj[ok], '.', color=colors[j % len(colors)], alpha=0.25, markersize=2)
                x_med, y_med = float(np.nanmedian(xj[ok])), float(np.nanmedian(yj[ok]))
                buoy_x.append(x_med)
                buoy_y.append(y_med)
                ax_map.plot([x_med], [y_med], marker='x', color=colors[j % len(colors)],
                            markersize=8, mew=2, linestyle='None', label=f'swift{22 + j}')

            if swift_gt_idx is not None:
                loo_xw = x_gt_full[inputwindow]
                loo_yw = y_gt_full[inputwindow]
                ok_loo_w = np.isfinite(loo_xw) & np.isfinite(loo_yw)
                if np.any(ok_loo_w):
                    ax_map.plot(loo_xw[ok_loo_w], loo_yw[ok_loo_w], '.', color='gray',
                                alpha=0.25, markersize=2)
                ax_map.plot([xt], [yt], 'k^', markersize=8, label=gt_label)
            else:
                ax_map.plot([xt], [yt], 'ko', markersize=6, label='target')
            x_all = np.concatenate([np.asarray(buoy_x), [xt]])
            y_all = np.concatenate([np.asarray(buoy_y), [yt]])
            ok_all = np.isfinite(x_all) & np.isfinite(y_all)
            if np.any(ok_all):
                dx = max(np.max(x_all[ok_all]) - np.min(x_all[ok_all]), 1.0)
                dy = max(np.max(y_all[ok_all]) - np.min(y_all[ok_all]), 1.0)
                pad = 0.15 * max(dx, dy)
                ax_map.set_xlim(np.min(x_all[ok_all]) - pad, np.max(x_all[ok_all]) + pad)
                ax_map.set_ylim(np.min(y_all[ok_all]) - pad, np.max(y_all[ok_all]) + pad)
            ax_map.set_xlabel('x [m]'); ax_map.set_ylabel('y [m]')
            ax_map.grid(True); ax_map.set_aspect('equal', adjustable='box')
            ax_map.legend(loc='best', fontsize='small')
            ax_map.set_title(f'max_iter={max_iters[mid_idx]} (mid)  ct={mid["comp_time"]:.3f}s')

            try:
                from matplotlib.patches import Wedge
                dp = float(wavespec_bulk.get('Dp_deg', float('nan')))
                spread = float(wavespec_bulk.get('spreadp_deg', float('nan')))
                hs = float(wavespec_bulk.get('Hs_m', float('nan')))
                tp_s = float(wavespec_bulk.get('Tp_s', float('nan')))
                if np.isfinite(dp):
                    dp_to = (dp + 180.0) % 360.0
                    xlim = ax_map.get_xlim(); ylim = ax_map.get_ylim()
                    L = max(5.0, 0.08 * max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])))
                    rad = np.deg2rad(dp_to)
                    ax_map.arrow(xt, yt,
                                 L * float(np.sin(rad)), L * float(np.cos(rad)),
                                 head_width=max(1.0, 0.15 * L), head_length=max(1.0, 0.20 * L),
                                 length_includes_head=True, color='k', alpha=0.8)
                    if np.isfinite(spread) and spread > 0.0:
                        a0 = 90.0 - dp_to
                        ax_map.add_patch(Wedge(
                            (xt, yt), r=1.1 * L,
                            theta1=a0 - 0.5 * spread, theta2=a0 + 0.5 * spread,
                            width=0.35 * L, facecolor='k', edgecolor='none', alpha=0.12,
                        ))
                    label_parts = []
                    if np.isfinite(hs): label_parts.append(f'Hs {hs:.2f} m')
                    if np.isfinite(tp_s): label_parts.append(f'Tp {tp_s:.2f} s')
                    label_parts.append(f'Dp(from) {dp:.0f}°')
                    if np.isfinite(spread): label_parts.append(f'spread {spread:.0f}°')
                    ax_map.text(0.02, 0.98, ', '.join(label_parts), transform=ax_map.transAxes,
                                ha='left', va='top', fontsize=9,
                                bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white',
                                      'alpha': 0.7, 'edgecolor': 'none'})
            except Exception:
                pass

            ax_z_in.plot(tin[inputwindow, :], zin[inputwindow, :])
            ax_z_in.set_ylabel('z in [m]')
            apply_latched_ylim(ax_z_in, 'z_in', zin)

            ax_u_in.plot(tin[inputwindow, :], uin[inputwindow, :])
            ax_u_in.set_ylabel('u in [m/s]')
            apply_latched_ylim(ax_u_in, 'u_in', uin)

            ax_v_in.plot(tin[inputwindow, :], vin[inputwindow, :])
            ax_v_in.set_ylabel('v in [m/s]')
            apply_latched_ylim(ax_v_in, 'v_in', vin)

            ax_z_rc.plot(tin[inputwindow, :], mid['zr'])
            ax_z_rc.set_ylabel('z recon [m]')
            apply_latched_ylim(ax_z_rc, 'z_recon', mid['zr'])

            ax_u_rc.plot(tin[inputwindow, :], mid['ur'])
            ax_u_rc.set_ylabel('u recon [m/s]')
            apply_latched_ylim(ax_u_rc, 'u_recon', mid['ur'])

            ax_v_rc.plot(tin[inputwindow, :], mid['vr'])
            ax_v_rc.set_ylabel('v recon [m/s]')
            ax_v_rc.set_xlabel('t [s]')
            apply_latched_ylim(ax_v_rc, 'v_recon', mid['vr'])

            ax_z_pr.plot(tpred, mid['zout'], 'k', label='python')
            if matlab_tpred is not None:
                ax_z_pr.plot(matlab_tpred, matlab_z, color='g', lw=1.5, label=ref_label)
                ax_z_pr.legend(loc='best', fontsize='small')
            ax_z_pr.set_ylabel('z pred [m]')
            z_lim_data = mid['zout'] if matlab_z is None else np.concatenate((mid['zout'], matlab_z))
            apply_latched_ylim(ax_z_pr, 'z_pred', z_lim_data)

            ax_u_pr.plot(tpred, mid['uout'], 'k')
            if matlab_tpred is not None:
                ax_u_pr.plot(matlab_tpred, matlab_u, color='g', lw=1.5)
            ax_u_pr.set_ylabel('u pred [m/s]')
            u_lim_data = mid['uout'] if matlab_u is None else np.concatenate((mid['uout'], matlab_u))
            apply_latched_ylim(ax_u_pr, 'u_pred', u_lim_data)

            ax_v_pr.plot(tpred, mid['vout'], 'k')
            if matlab_tpred is not None:
                ax_v_pr.plot(matlab_tpred, matlab_v, color='g', lw=1.5)
            ax_v_pr.set_ylabel('v pred [m/s]')
            ax_v_pr.set_xlabel('t [s]')
            v_lim_data = mid['vout'] if matlab_v is None else np.concatenate((mid['vout'], matlab_v))
            apply_latched_ylim(ax_v_pr, 'v_pred', v_lim_data)

            fig_r.suptitle(
                f't=[{t_start:.0f},{t_end:.0f}]s  max_iter={max_iters[mid_idx]}  '
                f'ct={mid["comp_time"]:.3f}s',
                fontsize=9,
            )

            if grab_frame:
                writer.grab_frame()

            # -- update study figure -------------------------------------
            horizon = tpred - t_end  # [1, 2, ..., n_lead] seconds ahead

            for iter_idx, mi in enumerate(max_iters):
                r = results[iter_idx]
                c = iter_colors[iter_idx]
                az, au, av = axes_study_pred[iter_idx]

                az.plot(tpred, r['zout'], color=c, lw=0.6, alpha=0.4)
                au.plot(tpred, r['uout'], color=c, lw=0.6, alpha=0.4)
                av.plot(tpred, r['vout'], color=c, lw=0.6, alpha=0.4)
                if matlab_tpred is not None:
                    az.plot(matlab_tpred, matlab_z, color='g', lw=0.6, alpha=0.4)
                    au.plot(matlab_tpred, matlab_u, color='g', lw=0.6, alpha=0.4)
                    av.plot(matlab_tpred, matlab_v, color='g', lw=0.6, alpha=0.4)

            # error and timing summary panels
            ax_err_z.cla()
            ax_err_uv.cla()
            ax_time.cla()

            ax_err_z.set_xlabel('prediction horizon [s]')
            ax_err_z.set_ylabel('|z error| [m]')
            ax_err_z.set_title('z prediction error vs horizon')
            ax_err_uv.set_xlabel('prediction horizon [s]')
            ax_err_uv.set_ylabel('RMS(u,v) error [m/s]')
            ax_err_uv.set_title('u+v prediction error vs horizon')
            ax_time.set_xlabel('max_iter')
            ax_time.set_ylabel('solve time [s]')
            ax_time.set_title('solve time per window')

            for iter_idx, mi in enumerate(max_iters):
                c = iter_colors[iter_idx]
                label = f'iter={mi}'

                # error plots — only once we have aligned windows
                errs_z = errors_z_by_iter[iter_idx]
                errs_uv = errors_uv_by_iter[iter_idx]
                if errs_z:
                    max_len = max(len(e) for e in errs_z)
                    mat_z  = np.full((len(errs_z),  max_len), np.nan)
                    mat_uv = np.full((len(errs_uv), max_len), np.nan)
                    for w, (ez, euv) in enumerate(zip(errs_z, errs_uv)):
                        mat_z[w,  :len(ez)]  = ez
                        mat_uv[w, :len(euv)] = euv
                    h_ax = np.arange(1, max_len + 1, dtype=float)
                    mean_z  = np.nanmean(mat_z,  axis=0)
                    std_z   = np.nanstd(mat_z,   axis=0)
                    mean_uv = np.nanmean(mat_uv, axis=0)
                    std_uv  = np.nanstd(mat_uv,  axis=0)

                    # thin per-window lines
                    for w in range(mat_z.shape[0]):
                        valid = np.isfinite(mat_z[w, :])
                        if np.any(valid):
                            ax_err_z.plot(h_ax[valid], mat_z[w, valid],
                                          color=c, lw=0.5, alpha=0.25)
                    for w in range(mat_uv.shape[0]):
                        valid = np.isfinite(mat_uv[w, :])
                        if np.any(valid):
                            ax_err_uv.plot(h_ax[valid], mat_uv[w, valid],
                                           color=c, lw=0.5, alpha=0.25)

                    # bold mean ± 1σ shading
                    valid_h = np.isfinite(mean_z)
                    ax_err_z.plot(h_ax[valid_h], mean_z[valid_h],
                                  color=c, lw=2.0, label=label)
                    ax_err_z.fill_between(
                        h_ax[valid_h],
                        (mean_z - std_z)[valid_h],
                        (mean_z + std_z)[valid_h],
                        color=c, alpha=0.15,
                    )

                    valid_h = np.isfinite(mean_uv)
                    ax_err_uv.plot(h_ax[valid_h], mean_uv[valid_h],
                                   color=c, lw=2.0, label=label)
                    ax_err_uv.fill_between(
                        h_ax[valid_h],
                        (mean_uv - std_uv)[valid_h],
                        (mean_uv + std_uv)[valid_h],
                        color=c, alpha=0.15,
                    )

            if any(errors_z_by_iter[i] for i in range(n_iters)):
                ax_err_z.legend(loc='best', fontsize='small')
                ax_err_uv.legend(loc='best', fontsize='small')

            # solve-time box plot
            times_data = [times_by_iter[i] for i in range(n_iters)]
            x_ticks = list(range(1, n_iters + 1))
            bp = ax_time.boxplot(
                [t if t else [float('nan')] for t in times_data],
                positions=x_ticks,
                widths=0.5,
                patch_artist=True,
                medianprops={'color': 'k', 'lw': 1.5},
            )
            for patch, c in zip(bp['boxes'], iter_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax_time.set_xticks(x_ticks)
            ax_time.set_xticklabels([str(m) for m in max_iters])

            # -- summary row 1: xcorr lag + peak timing + peak count ----
            ax_xcorr.cla()
            ax_pkdt.cla()
            ax_pkdt2.cla()

            ax_xcorr.axhline(0.0, color='k', lw=0.8, ls='--')
            ax_xcorr.set_xlabel('max_iter')
            ax_xcorr.set_ylabel('lag [s]')
            ax_xcorr.set_title('xcorr lag  (+  =  prediction early)')

            ax_pkdt.axhline(0.0, color='k', lw=0.8, ls='--')
            ax_pkdt.set_xlabel('max_iter')
            ax_pkdt.set_ylabel('\u0394t peak [s]')
            ax_pkdt.set_title('peak/trough timing error')

            ax_pkdt2.axhline(0.0, color='k', lw=0.8, ls='--')
            ax_pkdt2.set_xlabel('max_iter')
            ax_pkdt2.set_ylabel('\u0394n peaks')
            ax_pkdt2.set_title('peak count diff vs reference')

            def _draw_dist(ax, data_by_iter):
                """Violin when >=4 points per group, else box, else scatter dots."""
                positions, vdata = [], []
                for i, d in enumerate(data_by_iter):
                    arr = [v for v in d if np.isfinite(v)]
                    if arr:
                        positions.append(i + 1)
                        vdata.append(arr)
                if not vdata:
                    return
                if all(len(d) >= 4 for d in vdata):
                    parts = ax.violinplot(
                        vdata, positions=positions,
                        showmedians=True, showextrema=True,
                    )
                    for pc, pos in zip(parts['bodies'], positions):
                        pc.set_facecolor(iter_colors[pos - 1])
                        pc.set_alpha(0.55)
                    parts['cmedians'].set_color('k')
                    parts['cmins'].set_color('k')
                    parts['cmaxes'].set_color('k')
                    parts['cbars'].set_color('k')
                else:
                    bp = ax.boxplot(
                        vdata, positions=positions, widths=0.5,
                        patch_artist=True,
                        medianprops={'color': 'k', 'lw': 1.5},
                    )
                    for patch, pos in zip(bp['boxes'], positions):
                        patch.set_facecolor(iter_colors[pos - 1])
                        patch.set_alpha(0.6)
                ax.set_xticks(list(range(1, n_iters + 1)))
                ax.set_xticklabels([str(m) for m in max_iters])

            _draw_dist(ax_xcorr, xcorr_lags_by_iter)
            _draw_dist(ax_pkdt,  peak_dt_by_iter)
            _draw_dist(ax_pkdt2, peak_count_diff_by_iter)

            fig_s.canvas.draw_idle()

            if not grab_frame:
                plt.pause(0.001)

    if writer is not None:
        with writer.saving(fig_r, str(movie_path), dpi=args.dpi):
            run_loop(grab_frame=True)
    else:
        run_loop(grab_frame=False)

    if matlab_ds is not None:
        matlab_ds.close()

    # Save study figure
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    iters_tag = '-'.join(str(m) for m in max_iters)
    study_path = Path(f'study_iters{iters_tag}_{ts}.png')
    fig_s.savefig(str(study_path), dpi=150, bbox_inches='tight')
    print(f'study figure saved: {study_path}')


if __name__ == '__main__':
    main()
