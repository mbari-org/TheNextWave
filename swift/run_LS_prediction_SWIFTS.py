#!/usr/bin/python3

import numpy as np
import scientimate as sm
import scipy.signal as sps

from .swift import SWIFTArray, WaveSpec, Prediction, WFA, LSQWavePropParams
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra
from .WFA_sim_grid import WFA_sim_grid
from .leastSquaresWavePropagation import leastSquaresWavePropagation


def filtdat(x, order, cutoff, ftype='low', fs=None):
    """
    Zero-phase Butterworth filter (filtfilt wrapper).

    Parameters
    ----------
    x : array_like
        Input signal (1-D or ND; filtering applied along axis=0 by default).
    order : int
        Butterworth filter order.
    cutoff : float
        If fs is None: cutoff is *normalized* (0 < cutoff < 1) where 1 == Nyquist (MATLAB style).
        If fs is not None: cutoff is in Hz and will be normalized by fs/2 inside the function.
    ftype : {'low','high','bandpass','bandstop'}
        Filter type (defaults to 'low').
    fs : float or None
        Sampling frequency in Hz. If provided, cutoff is treated as Hz.

    Returns
    -------
    y : ndarray
        Filtered signal (same shape as x).
    """
    x = np.asarray(x)
    # compute normalized critical frequency Wn in (0,1)
    if fs is None:
        Wn = float(cutoff)
    else:
        nyq = 0.5 * float(fs)
        if nyq <= 0:
            raise ValueError("fs must be positive when provided")
        Wn = float(cutoff) / nyq

    # SciPy requires 0 < Wn < 1 (strict). Clamp only if Wn == 1 or slightly >1 due to numerics.
    if Wn <= 0.0:
        raise ValueError(f"Normalized cutoff must be > 0; got Wn={Wn} (cutoff={cutoff}, fs={fs})")
    if not (Wn < 1.0):
        # clamp to largest representable float < 1.0 to avoid scipy.error
        Wn = np.nextafter(1.0, 0.0)

    # design and apply filter
    b, a = sps.butter(order, Wn, btype=ftype, analog=False)
    # filtfilt applies along axis=0 by default — keep that unless you need a different axis
    y = sps.filtfilt(b, a, x, axis=0)

    return y


def to_seconds(t):
    """Convert a datetime64[ns] to float seconds since Unix epoch."""
    return (t - np.datetime64(0, 's')) / np.timedelta64(1, 's')


def run_LS_prediction_SWIFTS(
        array: SWIFTArray,
        data_deny: bool = False
    ) -> None:
    """Predict next incoming wave from SWIFT array data."""
    N = 9
    Nf = 1

    Etheta22, theta, E22, f, _, spread22, spread2_22, _ = SWIFTdirectionalspectra(array.swift22, False, True)
    Etheta23, _, E23 , _, _, spread23, spread2_23, _ = SWIFTdirectionalspectra(array.swift23, False, True)
    Etheta24, _, E24, _, _, spread24, spread2_24, _ = SWIFTdirectionalspectra(array.swift24, False, True)
    Etheta25, _, E25, _, _, spread25, spread2_25, _ = SWIFTdirectionalspectra(array.swift25, False, True)

    # print(f'{E22.shape=}')
    # print(f'{E23.shape=}')
    # print(f'{E24.shape=}')
    # print(f'{E25.shape=}')

    Etheta = np.stack([Etheta22, Etheta23, Etheta24, Etheta25], axis=2)
    E = np.column_stack([
        np.asarray(E22).ravel(order='F'),
        np.asarray(E23).ravel(order='F'),
        np.asarray(E24).ravel(order='F'),
        np.asarray(E25).ravel(order='F')
    ])
    spread = np.column_stack([
        np.asarray(spread22).ravel(order='F'),
        np.asarray(spread23).ravel(order='F'),
        np.asarray(spread24).ravel(order='F'),
        np.asarray(spread25).ravel(order='F')
    ])
    spread2 = np.column_stack([
        np.asarray(spread2_22).ravel(order='F'),
        np.asarray(spread2_23).ravel(order='F'),
        np.asarray(spread2_24).ravel(order='F'),
        np.asarray(spread2_25).ravel(order='F')
    ])

    #print('Etheta', Etheta.shape)
    #print('E', E.shape)
    #print('spread', spread.shape)
    #print('spread2', spread2.shape)
    #print('f', f.shape)
    #print('theta', theta.shape)

    wavespec = WaveSpec()
    wavespec.theta = theta.copy()
    wavespec.f = f.copy()
    wavespec.Etheta = np.nanmean(Etheta, axis=2)
    wavespec.spread = np.nanmean(spread, axis=1).reshape((-1, 1), order='F')
    wavespec.spread2 = np.nanmean(spread2, axis=1).reshape((-1, 1), order='F')
    E = np.nanmean(E, axis=1).reshape((-1, 1), order='F')
    # print(f'{E.shape=}\n{E=}')
    # print(f'{wavespec.f.shape=}\n{wavespec.f=}')

    #print('ws Etheta', wavespec.Etheta.shape)
    #print('E mean', E.shape)
    #print('ws spread', wavespec.spread.shape)
    #print('ws spread2', wavespec.spread2.shape)
    #print('ws f', wavespec.f.shape)
    #print('ws theta', wavespec.theta.shape)

    #print('trapz num', np.trapz(E, x=wavespec.f).shape)
    #print('trapz arg', (E * wavespec.f).shape)
    #print('trapz den', np.trapz(E * wavespec.f, x=wavespec.f).shape)

    # print(f'{np.trapz(E.T, x=wavespec.f.T)=}')
    # print(f'{np.trapz(E.T * wavespec.f.T, x=wavespec.f.T)=}')


    TM0 = np.trapz(E.T, x=wavespec.f.T) / np.trapz(E.T * wavespec.f.T, x=wavespec.f.T)
    # print(f'{TM0.shape=}\n{TM0=}')
    k, L, C, Cg = sm.wavedispersion(95., TM0, kCalcMethod='exact')
    s = 2. * np.pi / TM0
    Cp = s / k

    t0 = np.min([
             to_seconds(array.swift22.rawtime[0]),
             to_seconds(array.swift23.rawtime[0]),
             to_seconds(array.swift24.rawtime[0]),
             to_seconds(array.swift25.rawtime[0])
         ])
    x0 = np.array([
             array.swift22.x[0],
             array.swift23.x[0],
             array.swift24.x[0],
             array.swift25.x[0]
         ]).mean()
    y0 = np.array([
             array.swift22.y[0],
             array.swift23.y[0],
             array.swift24.y[0],
             array.swift25.y[0]
         ]).mean()
    # print(f'{array.swift22.wavespectra.a1.shape=}')
    # print(f'{array.swift23.wavespectra.a1.shape=}')
    # print(f'{array.swift24.wavespectra.a1.shape=}')
    # print(f'{array.swift25.wavespectra.a1.shape=}')
    a1 = np.nanmean(np.vstack([
             array.swift22.wavespectra.a1,
             array.swift23.wavespectra.a1,
             array.swift24.wavespectra.a1,
             array.swift25.wavespectra.a1
         ]), axis=0).reshape((-1, 1), order='F')
    b1 = np.nanmean(np.vstack([
             array.swift22.wavespectra.b1,
             array.swift23.wavespectra.b1,
             array.swift24.wavespectra.b1,
             array.swift25.wavespectra.b1
         ]), axis=0).reshape((-1, 1), order='F')
    a1 = np.trapz(E.T * a1.T, x=wavespec.f.T) / np.trapz(E.T, x=wavespec.f.T)
    # print(f'{a1.shape=}\n{a1=}')
    b1 = np.trapz(E.T * b1.T, x=wavespec.f.T) / np.trapz(E.T, x=wavespec.f.T)
    # print(f'{b1.shape=}\n{b1=}')
    dmo = np.degrees(np.arctan2(a1, b1))
    dmo[dmo < 0.] += 360.

    prediction = Prediction()
    prediction.tp = to_seconds(array.swift25.rawtime.reshape((-1, 1), order='F')) - t0
    prediction.params = [LSQWavePropParams() for _ in range(prediction.tp.shape[0])]
    prediction.comp_time = np.zeros(prediction.tp.shape[0])
    if data_deny:
        prediction.tm = np.hstack([
                            to_seconds(array.swift22.rawtime.reshape((-1, 1), order='F')),
                            to_seconds(array.swift23.rawtime.reshape((-1, 1), order='F')),
                            to_seconds(array.swift24.rawtime.reshape((-1, 1), order='F'))
                        ]) - t0
        # print(f'{prediction.tm.shape=}')
        prediction.zm = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.zc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.um = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.vm = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.uc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.vc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.zp = np.full_like(prediction.tp, np.nan, dtype=np.float64)
        prediction.up = np.full_like(prediction.tp, np.nan, dtype=np.float64)
        prediction.vp = np.full_like(prediction.tp, np.nan, dtype=np.float64)
        prediction.zt = np.full_like(prediction.tp, np.nan, dtype=np.float64)
        prediction.ut = np.full_like(prediction.tp, np.nan, dtype=np.float64)
        prediction.vt = np.full_like(prediction.tp, np.nan, dtype=np.float64)

        X = np.sqrt(
            (
                np.nanmean(array.swift25.x) - np.nanmean(array.swift24.x)
            )**2. + (
                np.nanmean(array.swift25.y) - np.nanmean(array.swift24.y)
            )**2.
        )
        Theta = 289  # approximate bearing between SWIFT 24 & SWIFT 25
        #Cp = s / k
        # Nlead=round(0.5.*X.*cosd(Theta-dmo)./Cp);
        Nlead = 5
    else:
        # TODO fix this else clause to match constructions above
        WFA1 = WFA()
        WFA1.x, WFA1.y, WFA1.lon, WFA1.lat, WFA1.x0, WFA1.y0, WFA1.lon, WFA1.lat = WFA_sim_grid()

        prediction.tm = np.vstack([
                            to_seconds(array.swift22.time.reshape((-1, 1), order='F')),
                            to_seconds(array.swift23.time.reshape((-1, 1), order='F')),
                            to_seconds(array.swift24.time.reshape((-1, 1), order='F')),
                            to_seconds(array.swift25.time.reshape((-1, 1), order='F'))
                        ]) - t0
        prediction.zm = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.zc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.um = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.vm = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.uc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.vc = np.full_like(prediction.tm, np.nan, dtype=np.float64)
        prediction.zp = np.full((prediction.tp.shape[0], WFA1.x.shape[0], WFA1.x.shape[1]), np.nan)
        Theta = 335.
        X = np.sqrt(
            (
                np.nanmean(WFA1.x.flatten(order='F')) - np.nanmean(array.swift25.x)
            )**2. + (
                np.nanmean(WFA1.y.flatten(order='F')) - np.nanmean(array.swift25.y)
            )**2.
        )
        Nlead = np.round(0.5 * X * np.cos(np.radians(Theta - dmo)) / Cp)

    #for n = np.round(N * TM0 * 5.) : Nf * 5 : len(prediction.tp) - Nf * 5 - Nlead * 5

    n_start_mat = int(np.round(N * TM0 * 5))
    n_start = n_start_mat - 1
    n_end_mat = int(len(prediction.tp) - Nf*5 - Nlead*5)
    n_end = n_end_mat
    n_step = int(Nf * 5)

    for n in range(n_start, n_end, n_step):
        ss_start = n - int(np.round(N*TM0*5 - 1))  # n is already zero-based so keep the same
        ss_end = n + 1  # inclusive of n to match matlab
        subsample = np.arange(ss_start, ss_end).astype(int)
        ts_start = n + 1  # n is already zero-based so keep the same
        ts_end = n + int(Nf*5) + 1  # inclusive of end to match matlab
        target_samp = (np.arange(ts_start, ts_end) + Nlead * 5).astype(int)
        if data_deny:
            zt = -array.swift25.z[target_samp]
            ut = array.swift25.u[target_samp]
            vt = array.swift25.v[target_samp]
            xt = array.swift25.x[target_samp] - x0
            yt = array.swift25.y[target_samp] - y0
            tp = to_seconds(array.swift25.rawtime[target_samp]) - t0

            zk = -np.vstack([array.swift22.z[subsample],
                   array.swift23.z[subsample],
                   array.swift24.z[subsample]]).T
            xk = np.vstack([array.swift22.x[subsample],
                  array.swift23.x[subsample],
                  array.swift24.x[subsample]]).T - x0
            yk = np.vstack([array.swift22.y[subsample],
                  array.swift23.y[subsample],
                  array.swift24.y[subsample]]).T - y0
            uk = np.vstack([sps.detrend(filtdat(array.swift22.u[subsample], 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.u[subsample], 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.u[subsample], 5, 1./2., 'low'))]).T
            vk = np.vstack([sps.detrend(filtdat(array.swift22.v[subsample], 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.v[subsample], 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.v[subsample], 5, 1./2., 'low'))]).T
            tk = np.vstack([to_seconds(array.swift22.rawtime[subsample]),
                  to_seconds(array.swift23.rawtime[subsample]),
                  to_seconds(array.swift24.rawtime[subsample])]).T - t0
            # print(f'{zt.shape=} {ut.shape=} {vt.shape=} {xt.shape=} {yt.shape=} {tp.shape=}')
            # print(f'{zk.shape=} {xk.shape=} {yk.shape=} {uk.shape=} {vk.shape=} {tk.shape=}')

        else:
            # TODO fix this else clause to match constructions above
            xt = np.tile(WFA1.x.ravel(order='F')[None, :], (len(target_samp), 1)) - x0
            yt = np.tile(WFA1.y.ravel(order='F')[None, :], (len(target_samp), 1)) - y0
            tp = (array.swift25.time[target_samp].T - t0) * 24. * 3600.
            tp = np.tile(tp[:, None], (1, WFA1.x.size))
            tp = tp.ravel(order='F')
            xt = xt.ravel(order='F')
            yt = yt.ravel(order='F')

            zk = -[array.swift22.z[subsample].T,
                   array.swift23.z[subsample].T,
                   array.swift24.z[subsample].T,
                   array.swift25.z[subsample].T]
            xk = [array.swift22.x[subsample].T,
                  array.swift23.x[subsample].T,
                  array.swift24.x[subsample].T,
                  array.swift25.x[subsample].T] - x0
            yk = [array.swift22.y[subsample].T,
                  array.swift23.y[subsample].T,
                  array.swift24.y[subsample].T,
                  array.swift25.y[subsample].T] - x0
            uk = [sps.detrend(filtdat(array.swift22.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift25.u[subsample].T, 5, 1./2., 'low'))]
            vk = [sps.detrend(filtdat(array.swift22.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift25.v[subsample].T, 5, 1./2., 'low'))]

            tk = [to_seconds(array.swift22.rawtime[subsample].T),
                  to_seconds(array.swift23.rawtime[subsample].T),
                  to_seconds(array.swift24.rawtime[subsample].T),
                  to_seconds(array.swift25.rawtime[subsample].T)] - t0

        zp, zc, params, t = leastSquaresWavePropagation(zk, uk, vk, tk, xk, yk, tp, xt, yt, wavespec)

        # print(f'{zp.shape=} {tp.shape=}')
        # print(f'{zp[:len(tp), :].T.shape=}')
        # print(f'{prediction.zp[target_samp, :].shape=}')

        if data_deny:
            prediction.zp[target_samp, :] = zp[:len(tp), :]
            prediction.zt[target_samp] = zt.reshape((-1, 1))
        else:
            # TODO fix this else clause to match constructions above
            reshaped = zp[:len(tp)].reshape(len(target_samp), *WFA1.x.shape)
            prediction.zp[target_samp, :, :] = reshaped

        for tsidx in target_samp:
            prediction.params[tsidx] = params
            prediction.comp_time[tsidx] = t

        n_rows = len(subsample)
        n_cols = prediction.tm.shape[1]
        prediction.zc[subsample, :] = zc[:zk.size].reshape(n_rows, n_cols)
        prediction.zm[subsample, :] = zk.reshape(n_rows, n_cols)

        # [print(f'{v=}') for v in [zp, zc, zk, params, t, prediction.tm]]

    valid = np.array([p.A.size > 0 for p in prediction.params])
    prediction.tp = prediction.tp[valid]
    prediction.zp = prediction.zp[valid]
    prediction.zt = prediction.zt[valid]
    prediction.ut = prediction.ut[valid] if prediction.ut.size else prediction.ut
    prediction.vt = prediction.vt[valid] if prediction.vt.size else prediction.vt

    prediction.params = [prediction.params[i] for i in np.where(valid)[0]]
    prediction.comp_time = prediction.comp_time[valid]

    prediction.tm = prediction.tm[valid]
    prediction.zm = prediction.zm[valid]
    prediction.zc = prediction.zc[valid]
    prediction.um = prediction.um[valid]
    prediction.vm = prediction.vm[valid]
    prediction.uc = prediction.uc[valid]
    prediction.vc = prediction.vc[valid]

    return array, prediction
