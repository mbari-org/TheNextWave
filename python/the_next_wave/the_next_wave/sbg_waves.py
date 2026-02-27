from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import signal


def sbg_waves(
    u: Iterable[float],
    v: Iterable[float],
    heave: Iterable[float],
    fs: float,
) -> Tuple[
    float,  # Hs
    float,  # Tp
    float,  # Dp
    np.ndarray,  # E
    np.ndarray,  # f
    np.ndarray,  # a1
    np.ndarray,  # b1
    np.ndarray,  # a2
    np.ndarray,  # b2
    np.ndarray,  # check
]:
    """
    Python port of MATLAB SBGwaves.m.

    Estimate wave height, period, direction, and spectral moments from:
      - u: east velocity [m/s]
      - v: north velocity [m/s]
      - heave: vertical heave [m, positive down]
      - fs: sampling rate [Hz]

    Returns (Hs, Tp, Dp, E, f, a1, b1, a2, b2, check).
    Invalid results are returned as 9999 (including arrays filled with 9999).
    """
    # --- fixed parameters (from MATLAB) ---
    wsecs = 256  # window length in seconds
    merge = 3  # number of neighboring freq bands to merge
    recip = True  # flip wave directions (FROM vs TOWARDS)
    RC = 3.5  # RC filter constant
    fmin = 0.05
    fmax = 1.0

    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    heave = np.asarray(heave, dtype=float).ravel()

    if not (u.size == v.size == heave.size):
        raise ValueError('u, v, and heave must have the same length')

    pts = int(u.size)
    w = int(np.round(fs * wsecs))
    if w % 2 != 0:
        w -= 1

    # Number of windows with 75% overlap
    # MATLAB: nwin = floor( 4*(pts/w - 1)+1 );
    nwin = int(np.floor(4 * (pts / w - 1) + 1))

    # Frequency range and bandwidth (MATLAB logic)
    # MATLAB: n = (w/2) / merge;
    n = int((w // 2) // merge)
    nyquist = 0.5 * fs
    bandwidth = nyquist / n
    # MATLAB: f = 1/(wsecs) + bandwidth/2 + bandwidth.*(0:(n-1))
    f = (1.0 / wsecs) + (bandwidth / 2.0) + bandwidth * np.arange(n, dtype=float)

    def invalid_outputs() -> Tuple[
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        f_out = f[f <= fmax].copy()
        nf = int(f_out.size)
        fill = np.full((nf,), 9999.0, dtype=float)
        return (
            9999.0,  # Hs
            9999.0,  # Tp
            9999.0,  # Dp
            fill.copy(),  # E
            f_out,  # f
            fill.copy(),  # a1
            fill.copy(),  # b1
            fill.copy(),  # a2
            fill.copy(),  # b2
            fill.copy(),  # check
        )

    # MATLAB: if pts >= w && fs > 1
    if not (pts >= w and fs > 1.0 and nwin >= 1):
        return invalid_outputs()

    # --- High-pass filter (RC) ---
    alpha = RC / (RC + 1.0 / fs)

    u_f = u.copy()
    v_f = v.copy()
    h_f = heave.copy()
    for i in range(1, pts):
        u_f[i] = alpha * u_f[i - 1] + alpha * (u[i] - u[i - 1])
        v_f[i] = alpha * v_f[i - 1] + alpha * (v[i] - v[i - 1])
        h_f[i] = alpha * h_f[i - 1] + alpha * (heave[i] - heave[i - 1])

    u = u_f
    v = v_f
    heave = h_f

    # --- Break into windows (75% overlap) ---
    step = w // 4  # 25% of window length
    if step <= 0:
        return invalid_outputs()

    uw = np.zeros((w, nwin), dtype=float)
    vw = np.zeros((w, nwin), dtype=float)
    zw = np.zeros((w, nwin), dtype=float)

    for q in range(nwin):
        start = q * step
        end = start + w
        if end > pts:
            # Should not occur given nwin formula, but be safe.
            return invalid_outputs()
        uw[:, q] = u[start:end]
        vw[:, q] = v[start:end]
        zw[:, q] = heave[start:end]

    # --- Detrend each window ---
    uw = signal.detrend(uw, axis=0, type='linear')
    vw = signal.detrend(vw, axis=0, type='linear')
    zw = signal.detrend(zw, axis=0, type='linear')

    # --- Taper and rescale (preserve variance) ---
    taper = np.sin(np.arange(1, w + 1, dtype=float) * np.pi / w).reshape(-1, 1)
    taper = np.repeat(taper, nwin, axis=1)

    uw_t = uw * taper
    vw_t = vw * taper
    zw_t = zw * taper

    # MATLAB var uses N-1 by default; ratio mostly cancels, but match via ddof=1.
    with np.errstate(invalid='ignore', divide='ignore'):
        fact_u = np.sqrt(np.var(uw, axis=0, ddof=1) / np.var(uw_t, axis=0, ddof=1))
        fact_v = np.sqrt(np.var(vw, axis=0, ddof=1) / np.var(vw_t, axis=0, ddof=1))
        fact_z = np.sqrt(np.var(zw, axis=0, ddof=1) / np.var(zw_t, axis=0, ddof=1))

    # Replace non-finite scale factors with 1.0 (avoids propagating NaNs)
    fact_u = np.where(np.isfinite(fact_u), fact_u, 1.0)
    fact_v = np.where(np.isfinite(fact_v), fact_v, 1.0)
    fact_z = np.where(np.isfinite(fact_z), fact_z, 1.0)

    uw_r = uw_t * fact_u.reshape(1, -1)
    vw_r = vw_t * fact_v.reshape(1, -1)
    zw_r = zw_t * fact_z.reshape(1, -1)

    # --- FFT ---
    U = np.fft.fft(uw_r, axis=0)
    V = np.fft.fft(vw_r, axis=0)
    Z = np.fft.fft(zw_r, axis=0)

    # Keep first half (MATLAB removed (w/2+1):w, i.e., kept 1..w/2)
    half = w // 2
    U = U[:half, :]
    V = V[:half, :]
    Z = Z[:half, :]

    # Remove mean (DC) by shifting bins down and zeroing last bin
    # MATLAB:
    # U(1:(w/2-1),:) = U(2:(w/2),:); U(w/2,:)=0;
    U[:-1, :] = U[1:, :]
    V[:-1, :] = V[1:, :]
    Z[:-1, :] = Z[1:, :]
    U[-1, :] = 0.0
    V[-1, :] = 0.0
    Z[-1, :] = 0.0

    # --- Spectra ---
    UUw = np.real(U * np.conj(U))
    VVw = np.real(V * np.conj(V))
    ZZw = np.real(Z * np.conj(Z))

    UVw = U * np.conj(V)
    UZw = U * np.conj(Z)
    VZw = V * np.conj(Z)

    # --- Merge neighboring frequency bands (truncate to full merge blocks) ---
    m = (half // merge)
    take = m * merge
    UUw = UUw[:take, :]
    VVw = VVw[:take, :]
    ZZw = ZZw[:take, :]
    UVw = UVw[:take, :]
    UZw = UZw[:take, :]
    VZw = VZw[:take, :]

    # Reshape to (m, merge, nwin) then average across merge
    UUwm = UUw.reshape(m, merge, nwin).mean(axis=1)
    VVwm = VVw.reshape(m, merge, nwin).mean(axis=1)
    ZZwm = ZZw.reshape(m, merge, nwin).mean(axis=1)
    UVwm = UVw.reshape(m, merge, nwin).mean(axis=1)
    UZwm = UZw.reshape(m, merge, nwin).mean(axis=1)
    VZwm = VZw.reshape(m, merge, nwin).mean(axis=1)

    # --- Ensemble average windows together and normalize to PSD ---
    denom = (half * fs)
    UU = np.nanmean(UUwm, axis=1) / denom
    VV = np.nanmean(VVwm, axis=1) / denom
    ZZ = np.nanmean(ZZwm, axis=1) / denom
    UV = np.nanmean(UVwm, axis=1) / denom
    UZ = np.nanmean(UZwm, axis=1) / denom
    VZ = np.nanmean(VZwm, axis=1) / denom

    # Ensure f length matches merged spectral vectors (n == m by construction)
    f = f[:m].copy()

    # --- Convert to displacement spectra from GPS velocities ---
    w_rad = 2.0 * np.pi * f
    with np.errstate(divide='ignore', invalid='ignore'):
        Exx = UU / (w_rad**2)
        Eyy = VV / (w_rad**2)
        Ezz = ZZ

        Qxz = np.imag(UZ) / (w_rad**1)  # noqa: F841
        Cxz = np.real(UZ) / (w_rad**1)

        Qyz = np.imag(VZ) / (w_rad**1)  # noqa: F841
        Cyz = np.real(VZ) / (w_rad**1)

        Cxy = np.real(UV) / (w_rad**2)

    # --- Wave spectral moments ---
    with np.errstate(divide='ignore', invalid='ignore'):
        denom1 = np.sqrt((Exx + Eyy) * Ezz)
        a1 = Cxz / denom1
        b1 = Cyz / denom1
        a2 = (Exx - Eyy) / (Exx + Eyy)
        b2 = (2.0 * Cxy) / (Exx + Eyy)

    # --- Directions ---
    dir1 = np.arctan2(b1, a1)
    # dir2 = np.arctan2(b2, a2) / 2.0  # computed but unused (kept for parity)

    # Orbit shape check
    with np.errstate(divide='ignore', invalid='ignore'):
        check = Ezz / (Eyy + Exx)

    E = Ezz.copy()

    # Wave stats over limited band
    fwaves = (f > fmin) & (f < fmax)

    if not np.any(fwaves) or not np.isfinite(E[fwaves]).any():
        return invalid_outputs()

    # Significant wave height
    Hs = 4.0 * np.sqrt(np.nansum(E[fwaves]) * bandwidth)

    # Peak period (from E)
    fpindex = int(np.nanargmax(E))
    Tp = 1.0 / f[fpindex] if f[fpindex] > 0 else 9999.0

    # Spectral directions
    # MATLAB: dir = - 180 ./ 3.14 * dir1; dir = dir + 90; wrap; recip adjustments
    dir_deg = -(180.0 / np.pi) * dir1
    dir_deg = dir_deg + 90.0
    dir_deg = np.where(dir_deg < 0.0, dir_deg + 360.0, dir_deg)

    if recip:
        west = dir_deg > 180.0
        east = dir_deg < 180.0
        dir_deg = dir_deg.copy()
        dir_deg[west] = dir_deg[west] - 180.0
        dir_deg[east] = dir_deg[east] + 180.0

    Dp = float(dir_deg[fpindex])

    # Screen for noisy direction estimate
    inds = np.array([fpindex - 1, fpindex, fpindex + 1], dtype=int)
    if np.all((inds >= 0) & (inds < dir_deg.size)):
        dirnoise = float(np.nanstd(dir_deg[inds], ddof=0))
        if dirnoise > 45.0:
            Dp = 9999.0
    else:
        Dp = 9999.0

    # Prune high frequency results
    keep = f <= fmax
    f = f[keep]
    E = E[keep]
    a1 = a1[keep]
    b1 = b1[keep]
    a2 = a2[keep]
    b2 = b2[keep]
    check = check[keep]

    # Quality control (MATLAB: if Tp > 20 => bulk metrics invalid)
    if np.isfinite(Tp) and Tp > 20.0:
        Hs = 9999.0
        Tp = 9999.0
        Dp = 9999.0

    # Replace any remaining non-finite spectral values with 9999? MATLAB does not,
    # but returning NaNs can be awkward for callers; keep as-is unless desired.
    return float(Hs), float(Tp), float(Dp), E, f, a1, b1, a2, b2, check
