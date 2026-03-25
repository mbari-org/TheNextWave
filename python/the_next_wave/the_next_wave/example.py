#!/usr/bin/env python3

import argparse
from pathlib import Path

from matplotlib.animation import FFMpegWriter, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import xarray as xr

from .download_example_data import get_example_data_dir
from .leastSquaresWavePropagation import leastSquaresWavePropagation
from .swift import Prediction, SWIFTArray, WaveSpec
from .utilities import (
    build_wavespec_from_swifts,
    bulk_dir_params_from_Etheta,
    bulk_wave_params_from_1d_spectrum,
    bulk_wave_params_from_wavespec,
    centroid_period_and_phase_speed,
    format_bulk_wave_params,
    load_raw_arrays_from_sbg,
)


def plot_wavespec_comparison(
    wavespec_base,
    wavespec_bulk,
    example_data_dir,
    wavespec_swift22_base=None,
    wavespec_swift22_bulk=None,
) -> None:
    """Plot a side-by-side comparison of the MATLAB and Python directional wave spectra.

    Three 2D+1D column panels:
      col 0 — MATLAB wavespec.mat  (single SWIFT 92)
      col 1 — Python mean of SWIFTs 22-25
      col 2 — Python SWIFT 22 only  (apples-to-apples with MATLAB)
    2D panels use compass-convention polar plots (0=N at top, clockwise, radial = frequency).
    """
    # Locate wavespec.mat
    mat_path = Path(example_data_dir) / 'wavespec.mat'
    if not mat_path.exists():
        mat_path = Path(__file__).parents[3] / 'ExampleData' / 'wavespec.mat'
    if not mat_path.exists():
        print(f'[comparison] wavespec.mat not found; skipping comparison figure')
        return

    m = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    ws_m = m['wavespec']
    Etheta_mat = np.asarray(ws_m.Etheta, dtype=float)
    theta_mat  = np.asarray(ws_m.theta,  dtype=float).ravel()
    f_mat      = np.asarray(ws_m.f,      dtype=float).ravel()

    # Ensure (nfreq, ntheta) — MATLAB saves (42, 180) = (f.size, theta.size)
    if Etheta_mat.shape == (theta_mat.size, f_mat.size):
        Etheta_mat = Etheta_mat.T

    # ------------------------------------------------------------------ MATLAB bulk params
    # wavespec.mat stores Etheta/theta/f only.  The MATLAB E(f) that was used to
    # compute Hs at the time the file was saved came directly from
    # SWIFT(ai).wavespectra.energy — it was never derived by integrating Etheta.
    # That raw E(f) is NOT in wavespec.mat, so the only reconstruction available
    # is numerical integration:
    #
    #   E(f) = ∫ Etheta(f, θ) dθ   (in radians)
    #
    # MEM_directionalestimator normalises Sn so that ∫ Sn dθ_rad = 1, making
    # Etheta units m²/Hz/rad.  Therefore dtheta must be in RADIANS here.
    # Using dtheta_deg (≈57× larger) would give physically impossible Hs (~18 m).
    dtheta_mat = float(np.nanmedian(np.diff(np.sort(theta_mat)))) * (np.pi / 180.0)  # rad
    E_mat      = np.sum(Etheta_mat * dtheta_mat, axis=1)   # (nfreq,)  ← ∫ Etheta dθ_rad
    mat_bulk   = bulk_wave_params_from_1d_spectrum(f_mat, E_mat)

    Hs_mat = mat_bulk['Hs_m']
    Tp_mat = mat_bulk['Tp_s']

    # Derive Dp/Dm/spread from Etheta_mat by computing first circular moments.
    # SWIFTdirectionalspectra outputs per-frequency direction as atan2(a1, b1)
    # in compass convention (a1=east=sin(θ), b1=north=cos(θ)).  Reconstruct
    # dir_freqband here so bulk_dir_params_from_Etheta gives MATLAB-faithful values.
    _theta_rad_m = np.deg2rad(theta_mat)
    with np.errstate(invalid='ignore', divide='ignore'):
        _a1_m = np.where(E_mat > 0,
                         np.sum(Etheta_mat * np.sin(_theta_rad_m) * dtheta_mat, axis=1) / E_mat,
                         np.nan)
        _b1_m = np.where(E_mat > 0,
                         np.sum(Etheta_mat * np.cos(_theta_rad_m) * dtheta_mat, axis=1) / E_mat,
                         np.nan)
    dir_freqband_mat = np.mod(np.rad2deg(np.arctan2(_a1_m, _b1_m)), 360.0)
    mat_dir_bulk = bulk_dir_params_from_Etheta(
        f_mat, theta_mat, Etheta_mat, dir_freqband=dir_freqband_mat
    )
    Dp_mat     = mat_dir_bulk['Dp_deg']
    Dm_mat     = mat_dir_bulk['Dm_deg']
    spread_mat = mat_dir_bulk['spreadp_deg']

    # ------------------------------------------------------------------ Python side (4-buoy mean)
    Etheta_py = np.asarray(wavespec_base.Etheta, dtype=float)
    theta_py  = np.asarray(wavespec_base.theta,  dtype=float).ravel()
    f_py      = np.asarray(wavespec_base.f,       dtype=float).ravel()
    E_py      = np.asarray(wavespec_base.E,        dtype=float).ravel()

    if Etheta_py.shape == (theta_py.size, f_py.size):
        Etheta_py = Etheta_py.T

    Hs_py      = float(wavespec_bulk.get('Hs_m',       float('nan')))
    Tp_py      = float(wavespec_bulk.get('Tp_s',        float('nan')))
    Dp_py      = float(wavespec_bulk.get('Dp_deg',      float('nan')))
    Dm_py      = float(wavespec_bulk.get('Dm_deg',      float('nan')))
    spread_py  = float(wavespec_bulk.get('spreadp_deg', float('nan')))

    # ------------------------------------------------------------------ Python side (SWIFT22 only)
    have_s22 = wavespec_swift22_base is not None
    if have_s22:
        Etheta_s22 = np.asarray(wavespec_swift22_base.Etheta, dtype=float)
        theta_s22  = np.asarray(wavespec_swift22_base.theta,  dtype=float).ravel()
        f_s22      = np.asarray(wavespec_swift22_base.f,       dtype=float).ravel()
        E_s22      = np.asarray(wavespec_swift22_base.E,        dtype=float).ravel()
        if Etheta_s22.shape == (theta_s22.size, f_s22.size):
            Etheta_s22 = Etheta_s22.T
        _bk22      = wavespec_swift22_bulk or {}
        Hs_s22     = float(_bk22.get('Hs_m',       float('nan')))
        Tp_s22     = float(_bk22.get('Tp_s',        float('nan')))
        Dp_s22     = float(_bk22.get('Dp_deg',      float('nan')))
        Dm_s22     = float(_bk22.get('Dm_deg',      float('nan')))
        spread_s22 = float(_bk22.get('spreadp_deg', float('nan')))
    else:
        Etheta_s22 = np.zeros_like(Etheta_py)
        theta_s22  = theta_py.copy()
        f_s22      = f_py.copy()
        E_s22      = np.zeros_like(E_py)
        Hs_s22 = Tp_s22 = Dp_s22 = Dm_s22 = spread_s22 = float('nan')

    # ------------------------------------------------------------------ Figure
    fig_cmp = plt.figure(num='wavespec_comparison', figsize=(19, 12))
    fig_cmp.clf()
    fig_cmp.suptitle('MATLAB vs Python wave spectrum comparison', fontsize=12)

    gs = fig_cmp.add_gridspec(3, 3, hspace=0.48, wspace=0.38,
                               height_ratios=[1.4, 1.0, 0.9])
    ax_m2d    = fig_cmp.add_subplot(gs[0, 0], projection='polar')
    ax_p2d    = fig_cmp.add_subplot(gs[0, 1], projection='polar')
    ax_s22_2d = fig_cmp.add_subplot(gs[0, 2], projection='polar')
    ax_m1d    = fig_cmp.add_subplot(gs[1, 0])
    ax_p1d    = fig_cmp.add_subplot(gs[1, 1])
    ax_s22_1d = fig_cmp.add_subplot(gs[1, 2])
    ax_diff   = fig_cmp.add_subplot(gs[2, :])

    vmax = max(
        float(np.nanmax(Etheta_mat)) if Etheta_mat.size else 1.0,
        float(np.nanmax(Etheta_py))  if Etheta_py.size  else 1.0,
        float(np.nanmax(Etheta_s22)) if (have_s22 and Etheta_s22.size) else 1.0,
    )

    def draw_polar_panel(ax, Etheta, theta_deg, f, title):
        # Compass convention: 0=N at top, angles increase clockwise.
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Close the angular gap at 0°/360° (North) so there is no white wedge.
        # SWIFTdirectionalspectra outputs theta = 2°, 4°, …, 358° (no 0°/360°).
        # pcolormesh leaves the 358°→2° arc (crossing North) empty without wrapping.
        theta_s = np.asarray(theta_deg, dtype=float)
        Etheta_s = np.asarray(Etheta, dtype=float)
        if 0.0 not in theta_s and 360.0 not in theta_s:
            # Average first (≈2°) and last (≈358°) columns for the wrap cell at 0°/360°
            wrap_col = 0.5 * (Etheta_s[:, [0]] + Etheta_s[:, [-1]])
            theta_s   = np.concatenate([[0.0],  theta_s,  [360.0]])
            Etheta_s  = np.hstack([wrap_col, Etheta_s, wrap_col])

        theta_rad = np.deg2rad(theta_s)
        pcm = ax.pcolormesh(
            theta_rad, f, Etheta_s,
            shading='auto', cmap='viridis',
            vmin=0.0, vmax=vmax,
        )
        ax.set_title(title, pad=12)
        # Cardinal labels on the angular axis
        ax.set_thetagrids([0, 90, 180, 270], labels=['N', 'E', 'S', 'W'], fontsize=8)
        ax.set_ylabel('f [Hz]', labelpad=28, fontsize=8)
        fig_cmp.colorbar(pcm, ax=ax, label='E(f,\u03b8) [m\u00b2/Hz/rad]', pad=0.10, shrink=0.8)

    draw_polar_panel(ax_m2d, Etheta_mat, theta_mat, f_mat,
                     'MATLAB wavespec.mat  (SWIFT 22)')
    draw_polar_panel(ax_p2d, Etheta_py, theta_py, f_py,
                     'Python — mean of SWIFTs 22–25')
    draw_polar_panel(ax_s22_2d, Etheta_s22, theta_s22, f_s22,
                     'Python — SWIFT 22 only')

    # 1-D spectra
    ax_m1d.plot(f_mat, E_mat, color='tab:blue')
    ax_m1d.set_xlabel('Frequency [Hz]')
    ax_m1d.set_ylabel('E(f) [m\u00b2/Hz]')
    ax_m1d.set_title('MATLAB 1-D spectrum (∫Etheta dθ)  — SWIFT 22')
    ax_m1d.grid(True, alpha=0.3)

    def nf(v, spec, unit=''):
        """Format a scalar; return 'nan' if not finite."""
        return (format(v, spec) + (f' {unit}' if unit else '')) if np.isfinite(v) else 'nan'

    deg = "\u00b0"
    theta = "\u03b8"
    int_symbol = "\u222b"
    nl = chr(10)

    mat_lines = [
        f"Hs*      = {nf(Hs_mat, '.3f', 'm')}",
        f"Tp*      = {nf(Tp_mat, '.2f', 's')}",
        f"Dp*      = {nf(Dp_mat, '.1f', deg)}",
        f"Dm*      = {nf(Dm_mat, '.1f', deg)}",
        f"spread@p*= {nf(spread_mat, '.1f', deg)}",
        f"* reconstructed via {int_symbol}Etheta d{theta}_rad",
        "  (MATLAB used raw E(f), not saved here)",
    ]
    mat_txt = nl.join(mat_lines)

    ax_m1d.text(
        0.97, 0.95, mat_txt, transform=ax_m1d.transAxes,
        ha='right', va='top', fontsize=9, family='monospace',
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.85},
    )

    ax_p1d.plot(f_py, E_py, color='tab:orange')
    ax_p1d.set_xlabel('Frequency [Hz]')
    ax_p1d.set_ylabel('E(f) [m\u00b2/Hz]')
    ax_p1d.set_title('Python  \u2014 mean of SWIFTs 22\u201325')
    ax_p1d.grid(True, alpha=0.3)

    py_lines = [
        f'Hs       = {nf(Hs_py,    ".3f", "m")}',
        f'Tp       = {nf(Tp_py,    ".2f", "s")}',
        f'Dp       = {nf(Dp_py,    ".1f", deg)}',
        f'Dm       = {nf(Dm_py,    ".1f", deg)}',
        f'spread@p = {nf(spread_py, ".1f", deg)}'
    ]
    py_txt = nl.join(py_lines)

    ax_p1d.text(
        0.97, 0.95, py_txt, transform=ax_p1d.transAxes,
        ha='right', va='top', fontsize=9, family='monospace',
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.85},
    )

    ax_s22_1d.plot(f_s22, E_s22, color='tab:green')
    ax_s22_1d.set_xlabel('Frequency [Hz]')
    ax_s22_1d.set_ylabel('E(f) [m\u00b2/Hz]')
    ax_s22_1d.set_title('Python  \u2014 SWIFT 22 only')
    ax_s22_1d.grid(True, alpha=0.3)

    s22_lines = [
        f'Hs       = {nf(Hs_s22,    ".3f", "m")}',
        f'Tp       = {nf(Tp_s22,    ".2f", "s")}',
        f'Dp       = {nf(Dp_s22,    ".1f", deg)}',
        f'Dm       = {nf(Dm_s22,    ".1f", deg)}',
        f'spread@p = {nf(spread_s22, ".1f", deg)}'
    ]
    s22_txt = nl.join(s22_lines)
    ax_s22_1d.text(
        0.97, 0.95, s22_txt, transform=ax_s22_1d.transAxes,
        ha='right', va='top', fontsize=9, family='monospace',
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.85},
    )

    # ------------------------------------------------------------------ Difference row
    # Interpolate all three spectra onto a common frequency grid for direct residuals.
    f_common = f_py if f_py.size >= f_mat.size else f_mat
    E_mat_interp = np.interp(f_common, f_mat, E_mat,
                             left=float('nan'), right=float('nan'))
    E_py_interp  = np.interp(f_common, f_py,  E_py,
                             left=float('nan'), right=float('nan'))
    E_s22_interp = np.interp(f_common, f_s22, E_s22,
                             left=float('nan'), right=float('nan'))
    E_diff    = E_py_interp  - E_mat_interp   # 4-buoy mean minus MATLAB
    E_diff_22 = E_s22_interp - E_mat_interp   # SWIFT22-only minus MATLAB

    ax_diff.fill_between(f_common, E_diff, 0,
                         where=(E_diff >= 0), alpha=0.25, color='tab:orange',
                         label='Python mean-4 > MATLAB')
    ax_diff.fill_between(f_common, E_diff, 0,
                         where=(E_diff < 0),  alpha=0.25, color='tab:blue',
                         label='MATLAB > Python mean-4')
    ax_diff.plot(f_common, E_diff,    color='tab:orange', linewidth=1.0,
                 label=f'\u0394E  mean-4  (\u0394Hs={nf(Hs_py - Hs_mat, "+.3f", "m")})')
    ax_diff.plot(f_common, E_diff_22, color='tab:green',  linewidth=1.0, linestyle='--',
                 label=f'\u0394E  SWIFT22  (\u0394Hs={nf(Hs_s22 - Hs_mat, "+.3f", "m")})')
    ax_diff.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_diff.set_xlabel('Frequency [Hz]')
    ax_diff.set_ylabel('\u0394E(f)  [m\u00b2/Hz]\n(Python \u2212 MATLAB)', fontsize=9)
    ax_diff.set_title(
        f'1-D spectrum residuals  vs MATLAB   '
        f'(MATLAB Hs from \u222bEtheta\u202fd\u03b8, not original E)',
        fontsize=9,
    )
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(fontsize=8, loc='upper right')

    fig_cmp.tight_layout()
    fig_cmp.savefig('wavespec_comparison.png', dpi=120, bbox_inches='tight')
    print('[comparison] saved wavespec_comparison.png')
    plt.show(block=False)
    fig_cmp.canvas.flush_events()
    return fig_cmp


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
        '--fig-width',
        type=float,
        default=12.0,
        help='Figure width in inches (used for both live window and movie frame size).',
    )
    p.add_argument(
        '--fig-height',
        type=float,
        default=12.0,
        help='Figure height in inches (used for both live window and movie frame size).',
    )
    p.add_argument(
        '--matlab-pred-nc',
        type=str,
        default=None,
        help='Optional MATLAB NetCDF predictions file for overlay comparison.',
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
        help='Maximum number of iterations for the solver.',
    )
    p.add_argument(
        '--solver-backend',
        type=str,
        default='auto',
        choices=('auto', 'scipy', 'jax', 'trust-constr'),
        help='Solver backend: auto (default, uses jax/GPU when available, else scipy), '
             'jax (always use jax — warns if no GPU found), '
             'scipy (always use scipy L-BFGS-B, fastest on CPU), '
             'trust-constr (second-order trust-region with exact hessp; '
             'converges reliably where L-BFGS-B stalls).',
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
    p.add_argument(
        '--reproj-swift',
        type=int,
        default=None,
        choices=(1, 2, 3, 4),
        help='Add three extra axes with measured vs reprojected z/u/v for a single '
             'buoy (1=SWIFT22, 2=SWIFT23, 3=SWIFT24, 4=SWIFT25). '
             'If omitted, these extra axes are not shown.',
    )
    return p.parse_args()


A0 = None
A0_indices = None
all_preds = Prediction()
max_iter = 5
solver_backend = 'auto'
prev_t_start_diff = None
matlab_ti_offset = 1  # index to offset MATLAB array to align with Python prediction window
matlab_ti_offset_increment = 1  # iterative adjustment to offset

def main():
    global A0
    global A0_indices
    global all_preds
    global max_iter
    global solver_backend

    args = parse_args()
    max_iter = args.solver_max_iter
    solver_backend = str(args.solver_backend)
    print(f'solver backend: {solver_backend}')

    # match MATLAB example
    latorigin = 41.6878
    lonorigin = -9.0545
    rotation = 180.0

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
    _all_bursts = swifts.bursts(raw_sbg=False)   # [swift22, swift23, swift24, swift25]
    _swift_names = ['SWIFT22', 'SWIFT23', 'SWIFT24', 'SWIFT25']
    _wavespec_swift_idx = args.wavespec_swift  # 1-based index or None

    # The 4-buoy mean is always computed — it is always shown in the comparison plot (col 1).
    wavespec_all4_base = build_wavespec_from_swifts(_all_bursts, recip=True)
    wavespec_all4_bulk = bulk_wave_params_from_wavespec(wavespec_all4_base)

    if _wavespec_swift_idx is not None:
        _sel = _all_bursts[_wavespec_swift_idx - 1]
        _label = _swift_names[_wavespec_swift_idx - 1]
        wavespec_base = build_wavespec_from_swifts([_sel], recip=True)
        print(f'wavespec for predictions: {_label} (--wavespec-swift {_wavespec_swift_idx})')
    else:
        wavespec_base = wavespec_all4_base
        print('wavespec for predictions: mean of all 4 SWIFTs (default)')
    Te, ce = centroid_period_and_phase_speed(wavespec_base)
    wavespec_bulk = bulk_wave_params_from_wavespec(wavespec_base)
    print(format_bulk_wave_params(wavespec_bulk, 'wavespec'))

    # SWIFT22-only wavespec always used for the comparison plot (col 2).
    wavespec_swift22_base = build_wavespec_from_swifts([swifts.swift22], recip=True)
    wavespec_swift22_bulk = bulk_wave_params_from_wavespec(wavespec_swift22_base)
    print(format_bulk_wave_params(wavespec_swift22_bulk, 'wavespec_swift22'))
    # plot_wavespec_comparison moved to after plt.ion() is set (below)

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

    reproj_swift_idx = args.reproj_swift
    reproj_buoy_col = None
    reproj_buoy_label = None
    if reproj_swift_idx is not None:
        reproj_buoy_col = int(reproj_swift_idx) - 1
        if reproj_buoy_col < 0 or reproj_buoy_col >= nbuoys:
            print(
                f'WARNING: --reproj-swift {reproj_swift_idx} out of range for '
                f'{nbuoys} buoys; disabling extra reproj axes'
            )
            reproj_buoy_col = None
        else:
            reproj_buoy_label = f'swift{22 + reproj_buoy_col}'
            print(
                f'extra reproj axes enabled for {reproj_buoy_label} '
                f'(--reproj-swift {reproj_swift_idx})'
            )

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
    step = int(round(fs))  # ~1 second increments

    # Set interactive mode before creating any figures so the event loop keeps
    # both the comparison window and the prediction window alive simultaneously.
    if args.movie:
        plt.ioff()
    else:
        plt.ion()

    # Keep the returned figure reference alive so the window is not closed/GC'd
    # while the prediction loop is running.  plt.pause() only pumps events for
    # the *active* figure, so holding this reference is what keeps fig_cmp alive.
    fig_cmp = plot_wavespec_comparison(
        wavespec_all4_base, wavespec_all4_bulk, example_data_dir,
        wavespec_swift22_base, wavespec_swift22_bulk,
    )

    fig_width = float(args.fig_width)
    fig_height = float(args.fig_height)
    if not np.isfinite(fig_width) or fig_width <= 0.0:
        fig_width = 16.0
    if not np.isfinite(fig_height) or fig_height <= 0.0:
        fig_height = 10.0

    fig = plt.figure(num='prediction', figsize=(fig_width, fig_height))
    fig.set_size_inches(fig_width, fig_height, forward=True)
    fig.clf()
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass

    # Build fixed axes once so panel positions/sizes stay stable across frames.
    gs = fig.add_gridspec(
        nrows=6,
        ncols=3 if reproj_buoy_col is not None else 2,
        left=0.10,
        right=0.985,
        bottom=0.07,
        top=0.965,
        wspace=0.30,
        hspace=0.33,
    )

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

    ax_z_cmp = None
    ax_u_cmp = None
    ax_v_cmp = None
    if reproj_buoy_col is not None:
        ax_z_cmp = fig.add_subplot(gs[3, 2])
        ax_u_cmp = fig.add_subplot(gs[4, 2], sharex=ax_z_cmp)
        ax_v_cmp = fig.add_subplot(gs[5, 2], sharex=ax_z_cmp)

    axes_all = [
        ax_map,
        ax_z_in,
        ax_u_in,
        ax_v_in,
        ax_z_rc,
        ax_u_rc,
        ax_v_rc,
        ax_z_pr,
        ax_u_pr,
        ax_v_pr,
    ]
    if ax_z_cmp is not None:
        axes_all.extend([ax_z_cmp, ax_u_cmp, ax_v_cmp])
    axes_all = tuple(axes_all)

    pred_latched_y_limits: dict[str, tuple[float, float]] = {}

    def apply_latched_ylim(ax, key: str, values) -> None:
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            prev = pred_latched_y_limits.get(key)
            if prev is not None:
                ax.set_ylim(*prev)
            return

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if np.isclose(vmin, vmax):
            pad = max(0.25, abs(vmin) * 0.1)
            cur_lo = vmin - pad
            cur_hi = vmax + pad
        else:
            span = vmax - vmin
            pad = max(0.05 * span, 0.05)
            cur_lo = vmin - pad
            cur_hi = vmax + pad

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
        if suffix == '.gif':
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps)

    def run_loop(grab_frame=False):
        global A0
        global A0_indices
        global all_preds
        global max_iter
        global solver_backend
        global prev_t_start_diff
        global matlab_ti_offset
        global matlab_ti_offset_increment

        warned_mismatch = False

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

            matlab_tpred = None
            matlab_z = None
            matlab_u = None
            matlab_v = None
            if matlab_ds is not None and matlab_ti is not None and matlab_ti.size:
                # MATLAB loop is 1-based: ti_matlab = ti_python + 1
                ti_target = float(ti + 1 + matlab_ti_offset)
                idx = int(np.argmin(np.abs(matlab_ti - ti_target)))
                print(f'ti target={ti_target:.1f}, matlab ti={matlab_ti[idx]:.1f} (idx={idx})')
                if True:  # np.isfinite(matlab_ti[idx]) and abs(matlab_ti[idx] - ti_target) <= 0.5:
                    try:
                        matlab_t_start = float(
                            np.asarray(matlab_ds['input_t_start'].values).ravel()[idx]
                        )
                        matlab_t_end = float(
                            np.asarray(matlab_ds['input_t_end'].values).ravel()[idx]
                        )
                        start_1based = int(
                            np.asarray(matlab_ds['window_start_idx'].values).ravel()[idx]
                        )
                        count = int(np.asarray(matlab_ds['window_count'].values).ravel()[idx])
                        if count > 0:
                            start = start_1based - 1
                            stop = start + count
                            matlab_tpred = np.asarray(
                                matlab_ds['tpred_flat'].values[start:stop],
                                dtype=float,
                            )
                            matlab_z = np.asarray(
                                matlab_ds['zout_flat'].values[start:stop],
                                dtype=float,
                            )
                            matlab_u = np.asarray(
                                matlab_ds['uout_flat'].values[start:stop],
                                dtype=float,
                            )
                            matlab_v = np.asarray(
                                matlab_ds['vout_flat'].values[start:stop],
                                dtype=float,
                            )

                            warn_sec = float(args.matlab_window_warn_sec)
                            t_start_diff = abs(matlab_tpred[0] - tpred[0])
                            if (
                                not warned_mismatch
                                and (
                                     t_start_diff > warn_sec
                                    # or abs(matlab_t_end - t_end) > warn_sec
                                )
                            ):
                                print(
                                    'WARNING: MATLAB/Python window mismatch exceeds threshold: '
                                    f'|start|={t_start_diff:.3f}s, '
                                    f'|end|={abs(matlab_tpred[-1] - tpred[-1]):.3f}s, '
                                    f'threshold={warn_sec:.3f}s (ti_py={ti}, '
                                    f'ti_mat={matlab_ti[idx]:.1f})'
                                )
                                if (
                                    prev_t_start_diff is not None
                                    and t_start_diff > prev_t_start_diff
                                ):
                                    matlab_ti_offset_increment *= -1
                                prev_t_start_diff = t_start_diff
                                matlab_ti_offset += matlab_ti_offset_increment
                                # warned_mismatch = True
                    except Exception as exc:
                        if not warned_mismatch:
                            print(
                                f'WARNING: failed to read MATLAB overlay window '
                                f'(ti={ti_target}): {exc}'
                            )
                            warned_mismatch = True

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
                max_iter=max_iter,
                A0_active_indices=None,  # A0_indices,
                solver_backend=solver_backend,
                ridge=0.0,  # set 0.0 to disable base ridge
                use_spectrum_weighted_ridge=False,  # set True to enable spectrum-weighted ridge
                spectrum_ridge_floor=1e-6,
                diagnostics=True,
                gtol=0.1,
                lambda_time=0.0,  # 0.1,            # set 0.0 to disable temporal continuity loss
                lambda_freq_smooth=0.0,  # 1e-3,    # set 0.0 to disable frequency smoothing loss
                lambda_theta_smooth=0.0,   # 1e-3,   # set 0.0 to disable directional smoothing loss
                freq_energy_frac=0.0,      # 0.05,  # set 0.0 to disable frequency active-grid pruning
                dir_energy_frac=0.0,       # 0.10,  # set 0.0 to disable directional active-grid pruning
                active_grid_pad=1,
                print_losses=True,
                use_rank_reduction=False,  # set True to try SVD compression (bounds approximate)
                use_row_scale=False,        # set False to disable row conditioning
                use_col_scale=False,        # set False to disable column conditioning
            )
            A0 = params.A
            A0_indices = params.active_good_indices
            print(f'solve time = {comp_time:.2f}s  nit={params.solver_nit}/{max_iter}')
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

            for ax in axes_all:
                ax.cla()

            # Map: plot buoy tracks + a representative per-buoy position marker.
            colors = (
                plt.rcParams['axes.prop_cycle']
                .by_key()
                .get('color', ['C0', 'C1', 'C2', 'C3'])
            )
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
                ax_map.plot(
                    xj[ok],
                    yj[ok],
                    '.',
                    color=colors[j % len(colors)],
                    alpha=0.25,
                    markersize=2,
                )
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
            x_all = np.concatenate(
                [
                    np.asarray(buoy_x, dtype=float),
                    np.asarray([xtarget], dtype=float),
                ]
            )
            y_all = np.concatenate(
                [
                    np.asarray(buoy_y, dtype=float),
                    np.asarray([ytarget], dtype=float),
                ]
            )
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
                        bbox={
                            'boxstyle': 'round,pad=0.2',
                            'facecolor': 'white',
                            'alpha': 0.7,
                            'edgecolor': 'none',
                        },
                    )
            except Exception:
                # If plotting backends or patches aren't available, keep the example running.
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

            if reproj_buoy_col is not None and ax_z_cmp is not None:
                t_cmp = tin[inputwindow, reproj_buoy_col]
                z_meas = zin[inputwindow, reproj_buoy_col]
                u_meas = uin[inputwindow, reproj_buoy_col]
                v_meas = vin[inputwindow, reproj_buoy_col]
                z_recon = zr[:, reproj_buoy_col]
                u_recon = ur[:, reproj_buoy_col]
                v_recon = vr[:, reproj_buoy_col]

                ax_z_cmp.plot(t_cmp, z_meas, color='k', label='meas')
                ax_z_cmp.plot(t_cmp, z_recon, color='C1', label='reproj')
                ax_z_cmp.set_ylabel('z [m]')
                ax_z_cmp.set_title(f'{reproj_buoy_label}: measured vs reprojected')
                apply_latched_ylim(
                    ax_z_cmp,
                    'z_cmp',
                    np.concatenate((np.asarray(z_meas, dtype=float), np.asarray(z_recon, dtype=float))),
                )
                ax_z_cmp.legend(loc='best', fontsize='small')

                ax_u_cmp.plot(t_cmp, u_meas, color='k')
                ax_u_cmp.plot(t_cmp, u_recon, color='C1')
                ax_u_cmp.set_ylabel('u [m/s]')
                apply_latched_ylim(
                    ax_u_cmp,
                    'u_cmp',
                    np.concatenate((np.asarray(u_meas, dtype=float), np.asarray(u_recon, dtype=float))),
                )

                ax_v_cmp.plot(t_cmp, v_meas, color='k')
                ax_v_cmp.plot(t_cmp, v_recon, color='C1')
                ax_v_cmp.set_ylabel('v [m/s]')
                ax_v_cmp.set_xlabel('t [s]')
                apply_latched_ylim(
                    ax_v_cmp,
                    'v_cmp',
                    np.concatenate((np.asarray(v_meas, dtype=float), np.asarray(v_recon, dtype=float))),
                )

            ax_z_pr.plot(tpred, zout, 'k', label='python')
            if matlab_tpred is not None:
                ax_z_pr.plot(matlab_tpred, matlab_z, color='g', linewidth=1.5, label='matlab')
            ax_z_pr.set_ylabel('z pred [m]')
            z_pred_for_ylim = zout if matlab_z is None else np.concatenate((zout, matlab_z))
            apply_latched_ylim(ax_z_pr, 'z_pred', z_pred_for_ylim)
            if matlab_tpred is not None:
                ax_z_pr.legend(loc='best', fontsize='small')

            ax_u_pr.plot(tpred, uout, 'k')
            if matlab_tpred is not None:
                ax_u_pr.plot(matlab_tpred, matlab_u, color='g', linewidth=1.5)
            ax_u_pr.set_ylabel('u pred [m/s]')
            u_pred_for_ylim = uout if matlab_u is None else np.concatenate((uout, matlab_u))
            apply_latched_ylim(ax_u_pr, 'u_pred', u_pred_for_ylim)

            ax_v_pr.plot(tpred, vout, 'k')
            if matlab_tpred is not None:
                ax_v_pr.plot(matlab_tpred, matlab_v, color='g', linewidth=1.5)
            ax_v_pr.set_ylabel('v pred [m/s]')
            ax_v_pr.set_xlabel('t [s]')
            v_pred_for_ylim = vout if matlab_v is None else np.concatenate((vout, matlab_v))
            apply_latched_ylim(ax_v_pr, 'v_pred', v_pred_for_ylim)

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
    if matlab_ds is not None:
        matlab_ds.close()
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    main()
