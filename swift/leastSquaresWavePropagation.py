import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import scipy.interpolate as spint
import scipy.linalg as splin

from .swift import LSQWavePropParams


def leastSquaresWavePropagation(z1, u1, v1, t1, x1, y1, t2, x2, y2, wavespec, A0=None):
    """
    Phase-resolved prediction of sea surface elevation at a specified time & location
    using an inverse linear model, following the MATLAB LSQ wave propagation method.

    Parameters
    ----------
    z1 : array-like
        Vertical displacement time series (M samples × P instruments).
    u1, v1 : array-like
        East and north velocities at the sea surface (same shape as z1) [m/s].
        If empty, the inversion uses displacement only.
    t1 : array-like
        Measurement times [s].
    x1, y1 : array-like
        Easting and northing of measurement locations [m].
    t2 : array-like
        Target times for prediction [s].
    x2, y2 : array-like
        Target easting and northing for prediction [m].
    wavespec : object
        Must have attributes:
          - Etheta : 2D array, directional wave spectrum (freq × direction)
          - f      : 1D array, frequencies [Hz]
          - theta  : 1D array, directions [deg, nautical convention]
    """

    if len(u1) > 0 and len(v1) > 0:
        use_vel = True
    else:
        use_vel = False

    # convert wave spectrum to Cartesian coordinates (direction waves move TOWARDS)
    if wavespec.Etheta.shape[0] == len(wavespec.theta):
        wavespec.Etheta = wavespec.Etheta.T

    theta_deg = wavespec.theta
    Etheta = wavespec.Etheta

    # unique(theta,'last') behaviour
    unique_vals = np.unique(theta_deg)  # sorted unique values
    idx_last = []
    for v in unique_vals:
        idxs = np.where(theta_deg == v)[0]
        idx_last.append(idxs[-1])       # last occurrence
    idx_last = np.array(idx_last, dtype=int)

    theta_u = unique_vals
    Etheta_u = Etheta[:, idx_last]

    # shift by 180°, sort by that, but only permute Etheta
    t = theta_u + 180.0
    t[t > 360.0] -= 360.0
    I_sort = np.argsort(t)

    wavespec.theta = theta_u.copy()
    wavespec.Etheta = Etheta_u[:, I_sort]

    E_for_peak = wavespec.Etheta
    n_freq, n_dir = E_for_peak.shape
    flat_F = E_for_peak.flatten(order="F")
    idx_flat = int(np.argmax(flat_F))
    col_idx = idx_flat // n_freq
    DTp = np.deg2rad(wavespec.theta[col_idx])  # radians

    f = np.asarray(wavespec.f, dtype=float)
    df = np.gradient(f)

    # wavespec.E = trapz(wavespec.theta, wavespec.Etheta')
    wavespec.E = np.trapz(wavespec.Etheta.T, x=wavespec.theta, axis=0)

    mask = (df * wavespec.E) / np.max(df * wavespec.E) >= 0.05
    frange_idx = np.nonzero(mask)[0]
    if frange_idx.size == 0:
        raise RuntimeError("No frequencies satisfy 5% cutoff")

    omega = np.logspace(
        np.log10(f[frange_idx[0]]),
        np.log10(f[frange_idx[-1]]),
        40
    ) * 2.0 * np.pi
    k = omega**2 / 9.81  # (40,)

    # 25 directions around DTp
    theta = np.linspace(DTp - np.pi/2.0, DTp + np.pi/2.0, 25)
    theta[theta > 2.0*np.pi] -= 2.0*np.pi
    theta[theta < 0.0]      += 2.0*np.pi
    theta = np.sort(theta)  # (25,)

    # Reshape, build kx, ky, omega
    print(f'{DTp=}')
    k = k.flatten(order='F')
    print(f'{k.mean()=}')
    theta = theta.flatten(order='F')
    print(f'{theta=}')

    kx = np.outer(k, np.sin(theta))
    ky = np.outer(k, np.cos(theta))
    omega = np.outer(np.sqrt(9.81 * k), np.ones_like(theta))

    kx = kx.flatten(order='F')
    ky = ky.flatten(order='F')
    omega = omega.flatten(order='F')
    x1 = x1.flatten(order='F')
    y1 = y1.flatten(order='F')
    t1 = t1.flatten(order='F')
    z1 = z1.flatten(order='F')
    u1 = u1.flatten(order='F')
    v1 = v1.flatten(order='F')
    x2 = x2.flatten(order='F')
    y2 = y2.flatten(order='F')
    t2 = t2.flatten(order='F')

    N_input_pts = len(z1)
    if len(x1) != N_input_pts or len(y1) != N_input_pts or len(t1) != N_input_pts:
        print('Error: All input vectors must be equal length')

    N_output_pts = len(t2)
    if len(x2) != N_output_pts or len(y2) != N_output_pts:
        print('Error: All output vectors must be equal length')

    # Interpolate measured spectrum to solution space
    F, T = np.meshgrid(wavespec.f, wavespec.theta)       # (M, N)
    f2, thet2 = np.meshgrid(np.sqrt(k * 9.8), theta)     # target grid
    points = np.column_stack((F.ravel(), T.ravel()))
    values = np.log10(wavespec.Etheta.T).ravel()
    xi = (f2 / (2. * np.pi), np.degrees(thet2))
    zi_log = spint.griddata(points, values, xi, method='linear')
    Ei = 10.0 ** zi_log
    Ei[np.isnan(Ei)] = 0.0

    Ei *= np.trapz(wavespec.E, x=wavespec.f, axis=0) / np.trapz(
        np.trapz(Ei, x=np.degrees(thet2[:, 0]), axis=0),
        x=f2[0, :] / (2. * np.pi),
        axis=0
    )

    amps = np.sqrt(
        Ei * np.diff(
            f2[0, :] / (2. * np.pi),
            prepend=0.0
        ) * sps.mode(
            np.diff(np.degrees(thet2[:, 0])),
            axis=None,
            keepdims=False
        ).mode.item()
    )
    amps = amps.T
    amps = np.concatenate((amps.flatten(order='F'),
                           amps.flatten(order='F')), axis=0)
    amps[np.isnan(amps)] = 0.0

    # reshape coordinates for phi matrices
    x1 = x1.reshape((-1, 1), order='F')
    x2 = x2.reshape((-1, 1), order='F')
    y1 = y1.reshape((-1, 1), order='F')
    y2 = y2.reshape((-1, 1), order='F')
    kx = kx.reshape((-1, 1), order='F')
    print(f'{kx.mean()=}')
    ky = ky.reshape((-1, 1), order='F')
    print(f'{ky.mean()=}')
    t1 = t1.reshape((-1, 1), order='F')
    t2 = t2.reshape((-1, 1), order='F')
    omega = omega.reshape((-1, 1), order='F')

    # Propagator matrices
    phi1 = x1 @ kx.T + y1 @ ky.T - t1 @ omega.T
    phi2 = x2 @ kx.T + y2 @ ky.T - t2 @ omega.T

    if use_vel:
        P1_11 = np.cos(phi1)
        P1_21 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1)
        P1_31 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1)
        P1_12 = np.sin(phi1)
        P1_22 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)
        P1_32 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)

        row0 = np.hstack((P1_11, P1_12))
        row1 = np.hstack((P1_21, P1_22))
        row2 = np.hstack((P1_31, P1_32))

        P1 = np.vstack((row0, row1, row2))

        P2_11 = np.cos(phi2)
        P2_21 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2)
        P2_31 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2)
        P2_12 = np.sin(phi2)
        P2_22 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)
        P2_32 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)

        row0 = np.hstack((P2_11, P2_12))
        row1 = np.hstack((P2_21, P2_22))
        row2 = np.hstack((P2_31, P2_32))

        P2 = np.vstack((row0, row1, row2))
    else:
        P1 = np.hstack([np.cos(phi1), np.sin(phi1)])
        P2 = np.hstack([np.cos(phi2), np.sin(phi2)])

    good = np.nonzero(amps[:np.size(Ei)] != 0)[0]
    zero_mask = (amps == 0)
    if zero_mask.any():
        keep_mask = ~zero_mask
        P1 = P1[:, keep_mask]
        P2 = P2[:, keep_mask]
        amps = amps[keep_mask]

    # RHS vector
    if use_vel:
        b = np.concatenate((np.asarray(z1).ravel(order='F'),
                            np.asarray(u1).ravel(order='F'),
                            np.asarray(v1).ravel(order='F')), axis=0)
    else:
        b = np.asarray(z1).ravel(order='F')

    print(f'{P1.shape=}')
    print(f'{P2.mean()=}')
    print(f'{b.mean()=}')

    """
    PtP = P1.T @ P1     # shape (2000×2000)
    # Condition number estimate using 2-norm (SVD)
    u, s, vt = np.linalg.svd(PtP)
    cond_number = s[0] / s[-1]
    print("cond(P1ᵀP1) =", cond_number)
    print("smallest singular val =", s[-1])
    print("largest singular val  =", s[0])

    rank_est = np.linalg.matrix_rank(P1, tol=1e-10)
    print("matrix_rank(P1) =", rank_est, "out of", P1.shape[1])

    # singular values of P1 (economy SVD)
    u, s, vt = np.linalg.svd(P1, full_matrices=False)
    import matplotlib.pyplot as plt
    plt.semilogy(s)
    plt.title("Singular values of P1")
    plt.show()

    nz = np.count_nonzero(P1)
    tot = P1.size
    print("sparsity =", 1 - nz / tot)

    # Normalize columns to unit length
    Pnorm = P1 / np.linalg.norm(P1, axis=0, keepdims=True)
    # Compute correlation of first 20 columns
    C = Pnorm[:, :20].T @ Pnorm[:, :20]
    print(C)

    C_full = Pnorm.T @ Pnorm
    max_corr = np.max(C_full - np.eye(C_full.shape[0]))
    print("max off-diagonal correlation =", max_corr)

    Q, R = np.linalg.qr(P1)
    rank_qr = np.sum(np.abs(np.diag(R)) > 1e-10)
    print("QR rank =", rank_qr)
    """

    def _solve_box_ridge_lbfgsb(P, b, lb, ub, x0=None, ridge=1e-6, max_iter=80):
        '''
        Solve: min 0.5||P x - b||^2 + 0.5*ridge*||x||^2  s.t. lb <= x <= ub

        Uses:
          - row scaling (improves conditioning)
          - column scaling (improves conditioning)
          - L-BFGS-B (fast, supports warm start via x0)
        '''
        # Row scaling: scale each row by its RMS to avoid huge disparities
        row_rms = np.sqrt(np.mean(P * P, axis=1))
        row_rms[row_rms == 0.0] = 1.0
        w = 1.0 / row_rms
        Pw = P * w[:, None]
        bw = b * w

        # Column scaling: normalize columns
        col_rms = np.sqrt(np.mean(Pw * Pw, axis=0))
        col_rms[col_rms == 0.0] = 1.0
        Ps = Pw / col_rms[None, :]

        # Variable scaling: x = xs / col_rms  <=>  Ps xs ~ bw
        lb_s = lb * col_rms
        ub_s = ub * col_rms

        if x0 is None:
            xs0 = np.zeros(P.shape[1])
        else:
            xs0 = x0 * col_rms

        xs0 = np.minimum(np.maximum(xs0, lb_s), ub_s)

        def fun(xs):
            r = Ps @ xs - bw
            return 0.5 * (r @ r) + 0.5 * ridge * (xs @ xs)

        def jac(xs):
            r = Ps @ xs - bw
            return Ps.T @ r + ridge * xs

        bounds = list(zip(lb_s, ub_s))

        res = spo.minimize(
            fun,
            xs0,
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6}
        )

        xs = res.x
        x = xs / col_rms
        return x, res

    import time
    t_0 = time.time()

    """
    Q, R, piv = splin.qr(P1, mode='economic', pivoting=True)
    tol = np.max(P1.shape) * R[0,0] * 1e-12
    print(f'{tol=}')
    r = np.sum(np.abs(np.diag(R)) > tol)
    P1r = P1[:, piv[:r]]
    amps_r = amps[piv[:r]]

    lower = -amps_r / np.sqrt(2)
    upper =  amps_r / np.sqrt(2)

    result = spo.lsq_linear(
        P1r,
        b,
        bounds=(lower, upper),
        method="trf",        # <-- MATLAB equivalent
        lsq_solver="lsmr",   # <-- Efficient iterative solver
        max_iter=100,        # <-- You can adjust
        verbose=1
    )

    A_r = result.x  # reduced solution (r,)
    A = np.zeros(P1.shape[1])
    A[piv[:r]] = A_r
    """

    lb = -amps / np.sqrt(2.0)
    ub =  amps / np.sqrt(2.0)

    A, info = _solve_box_ridge_lbfgsb(
        P1,
        b,
        lb,
        ub,
        x0=A0,          # pass previous A here for warm-start (see below)
        ridge=1e-6,       # start here; bump to 1e-5 or 1e-4 if still unstable
        max_iter=60
    )

    t = time.time() - t_0
    print(f"bounded_lsqr solve time: {t:.4f} s")
    print(f"{A.sum()=}")

    # reconstructed fields
    zc = P1 @ A
    z2 = P2 @ A
    print(f'{zc=} {z2=}')

    # bookkeeping into params
    params = LSQWavePropParams()
    params.A = A
    params.Etheta = np.zeros_like(Ei.flatten(order='F')).T
    params.Etheta[good] = (A[: (len(A) // 2)]**2. + A[(len(A) // 2):]**2.) / 2.0
    params.Etheta = params.Etheta.reshape((len(k), len(theta)), order='F').T
    params.Etheta /= (
        np.diff(f2[0, :] / (2. * np.pi), prepend=0.0)
        * sps.mode(
            np.diff(np.degrees(thet2[:, 0])),
            axis=None,
            keepdims=False
        ).mode.item()
    )

    params.f = (f2[0, :] / (2. * np.pi)).flatten()
    params.theta = np.degrees(thet2[:, 0])
    params.theta += 180.0
    params.theta[params.theta > 360.0] -= 360.0
    I = np.argsort(params.theta)
    params.theta = params.theta[I].flatten()
    params.Etheta = params.Etheta[I, :].T

    params.kx = kx[good].flatten()
    params.ky = ky[good].flatten()
    params.omega = omega[good].flatten()
    params.use_vel = use_vel

    #import sys; sys.exit(0)

    # return shapes consistent with MATLAB-style (column vectors)
    return (
        z2.reshape((-1, 1), order='F'),
        zc.reshape((-1, 1), order='F'),
        params,
        t,
    )
