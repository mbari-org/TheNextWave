#!/usr/bin/python3
import scipy.signal as sps

from .swift import SWIFTArray, WaveSpec, Prediction
from .SWIFTdirectionalspectra import SWIFTdirectionalspectra
from .WFA_sim_grid import WFA_sim_grid


def filtdat(x, order, cutoff, ftype):
    """Assume this is what filtdat func does in MATLAB code"""
    nyquist = 0.5
    b, a = sps.butter(order, cutoff / nyquist, btype=ftype)
    return sps.filtfilt(b, a, x)


def run_LS_prediction_SWIFTS(
        array: SWIFTArray,
        data_deny: bool = False
    ) -> None:
    """Predict next incoming wave from SWIFT array data."""
    N = 9
    Nf = 1

    Etheta22, theta, E22, f, _, spread22, spread2_22 = SWIFTdirectionalspectra(array.swift22, false, true)
    Etheta23, _, E23 , _, _, spread23, spread2_23 = SWIFTdirectionalspectra(array.swift23, false, true)
    Etheta24, _, E24, _, _, spread24, spread2_24 = SWIFTdirectionalspectra(array.swift24, false, true)
    Etheta25, _, E25, _, _, spread25, spread2_25 = SWIFTdirectionalspectra(array.swift25, false, true)

    Etheta = np.stack([Etheta22, Etheta23, Etheta24, Etheta25], axis=2)
    E = np.column_stack([
        np.asarray(E22).ravel(),
        np.asarray(E23).ravel(),
        np.asarray(E24).ravel(),
        np.asarray(E25).ravel()
    ])
    spread = np.column_stack([
        np.asarray(spread22).ravel(),
        np.asarray(spread23).ravel(),
        np.asarray(spread24).ravel(),
        np.asarray(spread25).ravel()
    ])
    spread2 = np.column_stack([
        np.asarray(spread2_22).ravel(),
        np.asarray(spread2_23).ravel(),
        np.asarray(spread2_24).ravel(),
        np.asarray(spread2_25).ravel()
    ])

    wavespec = WaveSpec()
    wavespec.theta = theta
    wavespec.f = f
    wavespec.Etheta = Etheta.mean(axis=2)
    wavespec.spread = spread.mean(axis=1)
    wavespec.spread2 = spread2.mean(axis=1)
    E = E.mean(axis=1)
    TM0 = np.trapz(E, wavespec.f) / np.trapz(E * wavespec.f, wavespec.f)
    k, L, C, Cg = sm.wavedispersion(95., TM0, kCalcMethod='exact')
    s = 2. * np.pi / TM0

    t0 = np.column_stack([
             array.swift22.rawtime[0],
             array.swift23.rawtime[0],
             array.swift24.rawtime[0],
             array.swift25.rawtime[0]
         ])
    x0 = np.column_stack([
             array.swift22.sbg_x[0],
             array.swift23.sbg_x[0],
             array.swift24.sbg_x[0],
             array.swift25.sbg_x[0]
         ])
    y0 = np.column_stack([
             array.swift22.sbg_y[0],
             array.swift23.sbg_y[0],
             array.swift24.sbg_y[0],
             array.swift25.sbg_y[0]
         ])
    a1 = np.column_stack([
             array.swift22.wavespectra.a1,
             array.swift23.wavespectra.a1,
             array.swift24.wavespectra.a1,
             array.swift25.wavespectra.a1
         ]).mean(axis=1)
    b1 = np.column_stack([
             array.swift22.wavespectra.b1,
             array.swift23.wavespectra.b1,
             array.swift24.wavespectra.b1,
             array.swift25.wavespectra.b1
         ]).mean(axis=1)
    a1 = np.trapz(E * a1, wavespec.f) / np.trapz(E, wavespec.f)
    b1 = np.trapz(E * b1, wavespec.f) / np.trapz(E, wavespec.f)
    dmo = np.degrees(np.arctan2(a1, b1))
    dmo[dmo < 0.] += 360.

    prediction = Prediction()
    prediction.tp = (array.swift25.rawtime.T - t0) * 24. * 3600.
    if data_deny:
        prediction.tm = ([array.swift22.rawtime.T,
                          array.swift23.rawtime.T,
                          array.swift24.rawtime.T] - t0) * 24. * 3600.
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
                np.nanmean(array.swift25.sbg_x) - np.nanmean(array.swift24.sbg_x)
            )**2. + (
                np.nanmean(array.swift25.sbg_y) - np.nanmean(array.swift24.sbg_y)
            )**2.
        )
        Theta = 289  # approximate bearing between SWIFT 24 & SWIFT 25
        Cp = s / k
        # Nlead=round(0.5.*X.*cosd(Theta-dmo)./Cp);
        Nlead = 5
    else:
        WFA1.x, WFA1.y, WFA1.lon, WFA1.lat, WFA1.x0, WFA1.y0 = WFA_sim_grid

        prediction.tm = ([array.swift22.rawtime.T,
                          array.swift23.rawtime.T,
                          array.swift24.rawtime.T,
                          array.swift25.rawtime.T] - t0) * 24. * 3600.
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
                np.nanmean(WFA1.x[:]) - np.nanmean(array.swift25.sbg_x)
            )**2. + (
                np.nanmean(WFA1.y[:]) - np.nanmean(array.swift25.sbg_y)
            )**2.
        )
        Nlead = np.round(0.5 * X * np.cos(np.radians(Theta - dmo)) / Cp)

    #for n = np.round(N * TM0 * 5.) : Nf * 5 : len(prediction.tp) - Nf * 5 - Nlead * 5
    for n in range(
        int(np.round(N * TM0 * 5)),
        len(prediction.tp) - Nf * 5 - Nlead * 5 + 1,
        Nf * 5
    ):
        subsample = np.arange(int(n - np.round(N * TM0 * 5 - 1)), n + 1)
        target_samp = np.arange(n + 1, n + Nf * 5 + 1) + Nlead * 5
        if data_deny:
            zt = -array.swift25.z[target_samp]
            ut = array.swift25.u[target_samp]
            vt = array.swift25.v[target_samp]
            xt = array.swift25.sbg_x[target_samp] - x0
            yt = array.swift25.sbg_y[target_samp] - y0
            tp = (array.swift25.rawtime[target_samp].T - t0) * 24. * 3600.

            zk = -[array.swift22.z[subsample].T,
                   array.swift23.z[subsample].T,
                   array.swift24.z[subsample].T]
            xk = [array.swift22.sbg_x[subsample].T,
                  array.swift23.sbg_x[subsample].T,
                  array.swift24.sbg_x[subsample].T] - x0
            yk = [array.swift22.sbg_y[subsample].T,
                  array.swift23.sbg_y[subsample].T,
                  array.swift24.sbg_y[subsample].T] - y0
            uk = [sps.detrend(filtdat(array.swift22.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.u[subsample].T, 5, 1./2., 'low'))]
            vk = [sps.detrend(filtdat(array.swift22.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.v[subsample].T, 5, 1./2., 'low'))]
            tk = ([array.swift22.rawtime[subsample].T,
                   array.swift23.rawtime[subsample].T,
                   array.swift24.rawtime[subsample].T] - t0) * 24. * 3600.

        else
            xt = np.tile(WFA1.x.ravel()[None, :], (len(target_samp), 1)) - x0
            yt = np.tile(WFA1.y.ravel()[None, :], (len(target_samp), 1)) - y0
            tp = (array.swift25.rawtime[target_samp].T - t0) * 24. * 3600.
            tp = np.tile(tp[:, None], (1, WFA1.x.size))
            tp = tp.ravel()
            xt = xt.ravel()
            yt = yt.ravel()

            zk = -[array.swift22.z[subsample].T,
                   array.swift23.z[subsample].T,
                   array.swift24.z[subsample].T,
                   array.swift25.z[subsample].T]
            xk = [array.swift22.sbg_x[subsample].T,
                  array.swift23.sbg_x[subsample].T,
                  array.swift24.sbg_x[subsample].T,
                  array.swift25.sbg_x[subsample].T] - x0
            yk = [array.swift22.sbg_y[subsample].T,
                  array.swift23.sbg_y[subsample].T,
                  array.swift24.sbg_y[subsample].T,
                  array.swift25.sbg_y[subsample].T] - x0
            uk = [sps.detrend(filtdat(array.swift22.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.u[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift25.u[subsample].T, 5, 1./2., 'low'))]
            vk = [sps.detrend(filtdat(array.swift22.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift23.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift24.v[subsample].T, 5, 1./2., 'low')),
                  sps.detrend(filtdat(array.swift25.v[subsample].T, 5, 1./2., 'low'))]

            tk = ([array.swift22.rawtime[subsample].T,
                   array.swift23.rawtime[subsample].T,
                   array.swift24.rawtime[subsample].T,
                   array.swift25.rawtime[subsample].T] - t0) * 24. * 3600.

        zp, zc, params, t = leastSquaresWavePropagation(zk, uk, vk, tk, xk, yk, tp, xt, yt, wavespec)

        if data_deny:
            prediction.zp[target_samp, :] = zp[:len(tp), :].T
            prediction.zt[target_samp] = zt.T
        else
            reshaped = zp[:len(tp)].reshape(len(target_samp), *WFA1.x.shape)
            prediction.zp[target_samp, :, :] = reshaped

        prediction.params[target_samp] = params
        prediction.comp_time[target_samp] = t

        n_rows = len(subsample)
        n_cols = prediction.tm.shape[1]
        prediction.zc[subsample, :] = zc[:zk.size].reshape(n_rows, n_cols)
        prediction.zm[subsample, :] = zk.reshape(n_rows, n_cols)

    return array, prediction
