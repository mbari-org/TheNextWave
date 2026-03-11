import sys
import time

import numpy as np
import scipy.interpolate as spint
import scipy.optimize as spo
import scipy.stats as sps

from .swift import LSQWavePropParams

try:
    from .solve_box_ridge_lbfgsb_gpu import solve_box_ridge_lbfgsb_gpu
except Exception:
    solve_box_ridge_lbfgsb_gpu = None

try:
    from .solve_box_ridge_lbfgsb_jax import solve_box_ridge_lbfgsb_jax as solve_box_ridge_lbfgsb_jax
except Exception:
    solve_box_ridge_lbfgsb_jax = None

trapz = getattr(np, 'trapz', np.trapezoid)


def solve_box_ridge_lbfgsb(
    P,
    b,
    lb,
    ub,
    x0=None,
    ridge=1e-6,
    max_iter=80,
    ridge_sigma_x=None,
    backend='auto',
):
    """
    Solve: min 0.5||P x - b||^2 + 0.5*ridge*||x||^2  s.t. lb <= x <= ub.

    Uses:
      - row scaling (improves conditioning)
      - column scaling (improves conditioning)
      - L-BFGS-B (fast, supports warm start via x0)
    """
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {'auto', 'gpu', 'scipy', 'jax'}:
        raise ValueError(
            f"Invalid backend='{backend}'. Expected one of: auto, gpu, scipy, jax"
        )

    # --- explicit backend requests ---
    if backend_norm == 'gpu':
        if solve_box_ridge_lbfgsb_gpu is None:
            raise RuntimeError(
                "backend='gpu' requested but solve_box_ridge_lbfgsb_gpu module is not available"
            )
        return solve_box_ridge_lbfgsb_gpu(
            np.asarray(P, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
            x0=None if x0 is None else np.asarray(x0, dtype=np.float64),
            ridge=float(ridge),
            max_iter=int(max_iter),
            ridge_sigma_x=None
            if ridge_sigma_x is None
            else np.asarray(ridge_sigma_x, dtype=np.float64),
        )

    if backend_norm == 'jax':
        if solve_box_ridge_lbfgsb_jax is None:
            raise RuntimeError(
                "backend='jax' requested but jax/jaxopt are not available.\n"
                "  CPU-only:  pip install jax jaxopt\n"
                "  CUDA 13:   pip install 'jax[cuda13]' jaxopt"
            )
        return solve_box_ridge_lbfgsb_jax(
            np.asarray(P, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
            x0=None if x0 is None else np.asarray(x0, dtype=np.float64),
            ridge=float(ridge),
            max_iter=int(max_iter),
            ridge_sigma_x=None
            if ridge_sigma_x is None
            else np.asarray(ridge_sigma_x, dtype=np.float64),
        )

    # --- auto priority: scipy -> jax -> gpu ---
    # 'scipy' always falls through to the scipy path below.
    # 'auto' selects scipy (always available); jax and gpu are only used when
    # explicitly requested so that the known-correct scipy result is the default.
    if backend_norm == 'auto' and solve_box_ridge_lbfgsb_jax is not None:
        # uncomment the next two lines to promote jax above scipy in auto order:
        # return solve_box_ridge_lbfgsb_jax(...)
        pass  # currently auto == scipy

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

    # Warm-start safety: the number of columns can change when the spectrum-driven
    # pruning changes, so ignore an incompatible initial guess instead of failing.
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size == 0:
            print(
                'WARNING: warm-start A0 provided but empty; starting from zeros',
                file=sys.stderr,
                flush=True,
            )
            x0 = None
        elif not np.all(np.isfinite(x0)):
            n_bad = int(np.size(x0) - np.count_nonzero(np.isfinite(x0)))
            print(
                'WARNING: warm-start A0 contains non-finite values '
                f'({n_bad} bad / {x0.size}); starting from zeros',
                file=sys.stderr,
                flush=True,
            )
            x0 = None
        elif x0.shape[0] != P.shape[1]:
            print(
                'WARNING: warm-start A0 length mismatch '
                f'(len(A0)={x0.shape[0]} vs n_cols={P.shape[1]}); '
                'starting from zeros',
                file=sys.stderr,
                flush=True,
            )
            x0 = None

    # Variable scaling: x = xs / col_rms  <=>  Ps xs ~ bw
    lb_s = lb * col_rms
    ub_s = ub * col_rms

    if x0 is None:
        xs0 = np.zeros(P.shape[1])
    else:
        xs0 = x0 * col_rms

    xs0 = np.minimum(np.maximum(xs0, lb_s), ub_s)

    # Optional spectrum-weighted ridge (MAP prior): penalize physical x scaled by sigma.
    # penalty = 0.5 * ridge * sum((x / sigma_x)^2), where x = xs / col_rms
    # => in scaled variables: sum((xs / (col_rms * sigma_x))^2)
    ridge_xs_scale2 = None
    if ridge_sigma_x is not None:
        ridge_sigma_x = np.asarray(ridge_sigma_x, dtype=float).reshape(-1)
        if ridge_sigma_x.shape[0] != P.shape[1]:
            raise ValueError(
                'ridge_sigma_x length mismatch '
                f'(len={ridge_sigma_x.shape[0]} vs n_cols={P.shape[1]})'
            )
        denom = col_rms * ridge_sigma_x
        denom[denom == 0.0] = 1.0
        ridge_xs_scale2 = 1.0 / (denom * denom)

    def fun(xs):
        r = Ps @ xs - bw
        if ridge_xs_scale2 is None:
            return 0.5 * (r @ r) + 0.5 * ridge * (xs @ xs)
        return 0.5 * (r @ r) + 0.5 * ridge * np.sum((xs * xs) * ridge_xs_scale2)

    def jac(xs):
        r = Ps @ xs - bw
        if ridge_xs_scale2 is None:
            return Ps.T @ r + ridge * xs
        return Ps.T @ r + ridge * (xs * ridge_xs_scale2)

    bounds = list(zip(lb_s, ub_s))

    res = spo.minimize(
        fun,
        xs0,
        jac=jac,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6},
    )

    xs = res.x
    x = xs / col_rms
    return x, res


def leastSquaresWavePropagation(
    z1,
    u1,
    v1,
    t1,
    x1,
    y1,
    t2,
    x2,
    y2,
    wavespec,
    A0=None,
    ridge=1e-6,
    max_iter=60,
    solver_backend='auto',
    use_spectrum_weighted_ridge=True,
    spectrum_ridge_floor=1e-6,
    diagnostics=False,
    near_bound_ratio=0.95,
    gram_diagnostics=False,
    gram_diag_out_dir=None,
    gram_diag_prefix='lsq',
    gram_diag_subset_size=256,
    profiling=None,
):
    """
    Phase-resolved prediction of sea surface elevation.

    At a specified time and location using an inverse linear model, following
    the MATLAB LSQ wave propagation method.

    Parameters
    ----------
    z1 : array-like
        Vertical displacement time series (M samples x P instruments).
    u1 : array-like
        Eastward velocity time series at the sea surface (same shape as z1) [m/s].
    v1 : array-like
        Northward velocity time series at the sea surface (same shape as z1) [m/s].
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
          - Etheta : 2D array, directional wave spectrum (freq x direction)
          - f      : 1D array, frequencies [Hz]
          - theta  : 1D array, directions [deg True, FROM]
    A0 : array-like, optional
        Warm-start amplitudes from the previous solve.
    ridge : float, optional
        Ridge (L2) regularization strength.
    max_iter : int, optional
        Maximum optimizer iterations.
    solver_backend : str, optional
        Solver backend selector: 'auto', 'gpu', or 'scipy'.
    use_spectrum_weighted_ridge : bool, optional
        If True, scale ridge penalty per component using spectrum-derived bounds.
    spectrum_ridge_floor : float, optional
        Minimum per-component ridge floor when weighted ridge is enabled.
    diagnostics : bool, optional
        If True, print solver diagnostics (iteration count, status, bound ratios).
    near_bound_ratio : float, optional
        Threshold (0-1) for counting coefficients as near the box bounds.
    gram_diagnostics : bool, optional
        If True, compute and optionally plot normalized Gram-matrix diagnostics.
    gram_diag_out_dir : str or None, optional
        If provided and matplotlib is available, write PNG plots in this directory.
    gram_diag_prefix : str, optional
        Prefix for Gram diagnostic output filenames.
    gram_diag_subset_size : int, optional
        Number of columns to visualize exactly in the correlation heatmap.
    profiling : dict or None, optional
        Mutable dictionary used to collect stage timing breakdowns in seconds.

    """
    prof = profiling if isinstance(profiling, dict) else None
    if prof is not None:
        prof.clear()
        prof['t0'] = time.perf_counter()

    def prof_mark(key: str, t_start: float) -> float:
        """Record elapsed time (seconds) under `key` and return a new start time."""
        if prof is not None:
            prof[key] = float(time.perf_counter() - t_start)
        return time.perf_counter()

    t_stage = time.perf_counter()

    if len(u1) > 0 and len(v1) > 0:
        use_vel = True
    else:
        use_vel = False

    # Convert wave spectrum to Cartesian coordinates.
    # Input convention: `wavespec.theta` is compass degrees True, direction waves come FROM.
    # Internally, this solver shifts by 180° to work in the propagation (TO) direction.
    # IMPORTANT: do not mutate `wavespec` in-place; callers may want to reuse it and
    # we cache derived interpolation products on the object.
    theta_deg = np.asarray(wavespec.theta, dtype=float).reshape((-1,))
    Etheta = np.asarray(wavespec.Etheta, dtype=float)
    if Etheta.ndim == 2 and Etheta.shape[0] == theta_deg.size:
        Etheta = Etheta.T

    f_in = np.asarray(wavespec.f, dtype=float).reshape((-1,))

    # Cache expensive spectrum->solution-space interpolation when wavespec is unchanged.
    # The cache lives on the `wavespec` instance.
    cache = getattr(wavespec, 'lsq_interp_cache', None)

    # NOTE: `wavespec` may be reused and its arrays may be updated either by
    # reallocation (new ndarray) or in-place (same underlying buffer). Pointer-
    # only signatures are fast but can be stale under in-place updates.
    # Add a couple cheap numeric fingerprints to make cache invalidation robust
    # while keeping overhead negligible relative to `griddata`.
    def fingerprint_1d(a: np.ndarray) -> tuple[float, float, float]:
        a = np.asarray(a, dtype=float).reshape((-1,))
        if a.size == 0:
            return (0.0, 0.0, 0.0)
        i_mid = int(a.size // 2)
        return (float(a[0]), float(a[i_mid]), float(a[-1]))

    def fingerprint_2d(a: np.ndarray) -> tuple[float, float, float]:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return (0.0, 0.0, 0.0)
        flat = a.reshape((-1,))
        i_mid = int(flat.size // 2)
        return (float(flat[0]), float(flat[i_mid]), float(flat[-1]))

    theta_fp = fingerprint_1d(theta_deg)
    f_fp = fingerprint_1d(f_in)
    Etheta_fp = fingerprint_2d(Etheta)
    # Sum-of-squares helps distinguish spectra with identical sum.
    try:
        Etheta_ss = float(np.nansum(np.asarray(Etheta, dtype=float) ** 2.0))
    except Exception:
        Etheta_ss = float('nan')
    sig = (
        int(theta_deg.size),
        tuple(theta_deg.shape),
        int(f_in.size),
        tuple(f_in.shape),
        tuple(Etheta.shape),
        theta_fp,
        f_fp,
        Etheta_fp,
        Etheta_ss,
        int(Etheta.__array_interface__['data'][0])
        if hasattr(Etheta, '__array_interface__')
        else 0,
        int(f_in.__array_interface__['data'][0]) if hasattr(f_in, '__array_interface__') else 0,
        int(theta_deg.__array_interface__['data'][0])
        if hasattr(theta_deg, '__array_interface__')
        else 0,
    )

    # unique(theta,'last') behaviour
    unique_vals = np.unique(theta_deg)  # sorted unique values
    idx_last = []
    for v in unique_vals:
        idxs = np.where(theta_deg == v)[0]
        idx_last.append(idxs[-1])  # last occurrence
    idx_last = np.array(idx_last, dtype=int)

    theta_u = unique_vals
    Etheta_u = Etheta[:, idx_last]

    # shift by 180°, sort by that, but only permute Etheta
    t = theta_u + 180.0
    t[t > 360.0] -= 360.0
    I_sort = np.argsort(t)

    theta_u_sorted = theta_u.copy()
    Etheta_u_sorted = Etheta_u[:, I_sort]

    t_stage = prof_mark('spectrum_prep_s', t_stage)

    E_for_peak = Etheta_u_sorted
    n_freq, n_dir = E_for_peak.shape
    flat_F = E_for_peak.flatten(order='F')
    idx_flat = int(np.argmax(flat_F))
    col_idx = idx_flat // n_freq
    DTp = np.deg2rad(theta_u_sorted[col_idx])  # radians

    f = f_in
    df = np.gradient(f)

    E_1d = trapz(Etheta_u_sorted.T, x=theta_u_sorted, axis=0)
    mask = (df * E_1d) / np.max(df * E_1d) >= 0.05
    frange_idx = np.nonzero(mask)[0]
    if frange_idx.size == 0:
        raise RuntimeError('No frequencies satisfy 5% cutoff')

    omega = np.logspace(
        np.log10(f[frange_idx[0]]),
        np.log10(f[frange_idx[-1]]),
        40,
    ) * 2.0 * np.pi
    k = omega**2 / 9.81  # (40,)

    # 25 directions around DTp
    theta = np.linspace(DTp - np.pi / 2.0, DTp + np.pi / 2.0, 25)
    theta[theta > 2.0 * np.pi] -= 2.0 * np.pi
    theta[theta < 0.0] += 2.0 * np.pi
    theta = np.sort(theta)  # (25,)

    t_stage = prof_mark('build_k_theta_s', t_stage)

    # Reshape, build kx, ky, omega
    # print(f'{DTp=}')
    k = k.flatten(order='F')
    # print(f'{k.mean()=}')
    theta = theta.flatten(order='F')
    # print(f'{theta=}')

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
        raise ValueError('All input vectors must be equal length')

    N_output_pts = len(t2)
    if len(x2) != N_output_pts or len(y2) != N_output_pts:
        raise ValueError('All output vectors must be equal length')

    # Interpolate measured spectrum to solution space.
    # This is expensive (`griddata`), so reuse cached results when possible.
    if isinstance(cache, dict) and cache.get('sig') == sig:
        k = cache['k']
        theta = cache['theta']
        kx = cache['kx']
        ky = cache['ky']
        omega = cache['omega']
        Ei = cache['Ei']
        good = cache['good']
        amps_base = cache['amps_base']
        f2 = cache['f2']
        thet2 = cache['thet2']
        if prof is not None and 'interp_griddata_s' not in prof:
            # For legacy logs that still print `griddata=...`, make cache hits
            # show up as 0.0ms instead of NaN.
            prof['interp_griddata_s'] = 0.0
        t_stage = prof_mark('interp_cache_hit_s', t_stage)
    else:
        F, T = np.meshgrid(f, theta_u_sorted)  # (n_dir, n_freq)
        f2, thet2 = np.meshgrid(np.sqrt(k * 9.8), theta)  # target grid
        points = np.column_stack((F.ravel(), T.ravel()))
        values = np.log10(Etheta_u_sorted.T).ravel()
        xi = (f2 / (2.0 * np.pi), np.degrees(thet2))
        zi_log = spint.griddata(points, values, xi, method='linear')
        Ei = 10.0 ** zi_log
        Ei[np.isnan(Ei)] = 0.0
        t_stage = prof_mark('interp_griddata_s', t_stage)

        Ei *= trapz(E_1d, x=f, axis=0) / trapz(
            trapz(Ei, x=np.degrees(thet2[:, 0]), axis=0),
            x=f2[0, :] / (2.0 * np.pi),
            axis=0,
        )
        t_stage = prof_mark('interp_renorm_s', t_stage)

        dtheta_mode_deg = (
            sps.mode(
                np.diff(np.degrees(thet2[:, 0])),
                axis=None,
                keepdims=False,
            )
            .mode
            .item()
        )
        df2 = np.diff(f2[0, :] / (2.0 * np.pi), prepend=0.0)
        amps = np.sqrt(Ei * df2 * dtheta_mode_deg)
        amps_base = amps.T.flatten(order='F')
        amps_base[np.isnan(amps_base)] = 0.0
        good = np.nonzero(amps_base != 0.0)[0]
        t_stage = prof_mark('build_amps_s', t_stage)

        try:
            wavespec.lsq_interp_cache = {
                'sig': sig,
                'k': k,
                'theta': theta,
                'kx': kx,
                'ky': ky,
                'omega': omega,
                'Ei': Ei,
                'good': good,
                'amps_base': amps_base,
                'f2': f2,
                'thet2': thet2,
            }
        except Exception:
            pass

    # Drop zero-energy base components before building phi/cos/sin.
    good_base = np.asarray(good, dtype=int).reshape((-1,))
    if good_base.size > 0 and good_base.size < int(np.asarray(kx).size):
        kx = np.asarray(kx, dtype=float).reshape((-1,))[good_base]
        ky = np.asarray(ky, dtype=float).reshape((-1,))[good_base]
        omega = np.asarray(omega, dtype=float).reshape((-1,))[good_base]
        amps_base = np.asarray(amps_base, dtype=float).reshape((-1,))[good_base]

    amps = np.concatenate((amps_base, amps_base), axis=0).astype(np.float32, copy=False)

    # reshape coordinates for phi matrices (float32 for faster trig + lower memory bandwidth)
    work_dtype = np.float32
    x1 = np.asarray(x1, dtype=work_dtype).reshape((-1, 1), order='F')
    x2 = np.asarray(x2, dtype=work_dtype).reshape((-1, 1), order='F')
    y1 = np.asarray(y1, dtype=work_dtype).reshape((-1, 1), order='F')
    y2 = np.asarray(y2, dtype=work_dtype).reshape((-1, 1), order='F')
    kx = np.asarray(kx, dtype=work_dtype).reshape((-1, 1), order='F')
    ky = np.asarray(ky, dtype=work_dtype).reshape((-1, 1), order='F')
    t1 = np.asarray(t1, dtype=work_dtype).reshape((-1, 1), order='F')
    t2 = np.asarray(t2, dtype=work_dtype).reshape((-1, 1), order='F')
    omega = np.asarray(omega, dtype=work_dtype).reshape((-1, 1), order='F')

    # Reused velocity factors (avoid recomputing kx/sqrt(kx^2+ky^2) and ky/sqrt(...) many times).
    k_norm = np.sqrt(kx * kx + ky * ky)
    k_norm[k_norm == 0.0] = 1.0
    vel_x = (kx / k_norm) * omega
    vel_y = (ky / k_norm) * omega

    # Propagator matrices
    phi1 = x1 @ kx.T + y1 @ ky.T - t1 @ omega.T
    t_stage = prof_mark('build_phi_input_s', t_stage)
    phi2 = x2 @ kx.T + y2 @ ky.T - t2 @ omega.T

    t_stage = prof_mark('build_phi_output_s', t_stage)

    if use_vel:
        cos1 = np.cos(phi1)
        sin1 = np.sin(phi1)
        t_stage = prof_mark('trig_input_s', t_stage)

        n_in = cos1.shape[0]
        n_comp = cos1.shape[1]
        P1 = np.empty((3 * n_in, 2 * n_comp), dtype=work_dtype)

        P1[:n_in, :n_comp] = cos1
        P1[:n_in, n_comp:] = sin1

        vel_x_t = vel_x.T
        vel_y_t = vel_y.T
        P1[n_in:2 * n_in, :n_comp] = vel_x_t * cos1
        P1[n_in:2 * n_in, n_comp:] = vel_x_t * sin1
        P1[2 * n_in:, :n_comp] = vel_y_t * cos1
        P1[2 * n_in:, n_comp:] = vel_y_t * sin1
        t_stage = prof_mark('assemble_P_input_s', t_stage)

        cos2 = np.cos(phi2)
        sin2 = np.sin(phi2)
        t_stage = prof_mark('trig_output_s', t_stage)

        n_out = cos2.shape[0]
        P2 = np.empty((3 * n_out, 2 * n_comp), dtype=work_dtype)

        P2[:n_out, :n_comp] = cos2
        P2[:n_out, n_comp:] = sin2
        P2[n_out:2 * n_out, :n_comp] = vel_x_t * cos2
        P2[n_out:2 * n_out, n_comp:] = vel_x_t * sin2
        P2[2 * n_out:, :n_comp] = vel_y_t * cos2
        P2[2 * n_out:, n_comp:] = vel_y_t * sin2
        t_stage = prof_mark('assemble_P_output_s', t_stage)
    else:
        cos1 = np.cos(phi1)
        sin1 = np.sin(phi1)
        t_stage = prof_mark('trig_input_s', t_stage)

        n_in = cos1.shape[0]
        n_comp = cos1.shape[1]
        P1 = np.empty((n_in, 2 * n_comp), dtype=work_dtype)
        P1[:, :n_comp] = cos1
        P1[:, n_comp:] = sin1
        t_stage = prof_mark('assemble_P_input_s', t_stage)

        cos2 = np.cos(phi2)
        sin2 = np.sin(phi2)
        t_stage = prof_mark('trig_output_s', t_stage)

        n_out = cos2.shape[0]
        P2 = np.empty((n_out, 2 * n_comp), dtype=work_dtype)
        P2[:, :n_comp] = cos2
        P2[:, n_comp:] = sin2
        t_stage = prof_mark('assemble_P_output_s', t_stage)

    if prof is not None:
        prof['build_P_mats_s'] = float(
            prof.get('trig_input_s', 0.0)
            + prof.get('assemble_P_input_s', 0.0)
            + prof.get('trig_output_s', 0.0)
            + prof.get('assemble_P_output_s', 0.0)
        )

    # Safety: handle any remaining exact zeros.
    zero_mask = (amps == 0)
    if zero_mask.any():
        keep_mask = ~zero_mask
        P1 = P1[:, keep_mask]
        P2 = P2[:, keep_mask]
        amps = amps[keep_mask]

    t_stage = prof_mark('prune_zero_cols_s', t_stage)

    # RHS vector
    if use_vel:
        b = np.concatenate((np.asarray(z1).ravel(order='F'),
                            np.asarray(u1).ravel(order='F'),
                            np.asarray(v1).ravel(order='F')), axis=0)
    else:
        b = np.asarray(z1).ravel(order='F')
    b = np.asarray(b, dtype=work_dtype)

    t_stage = prof_mark('build_rhs_s', t_stage)

    # print(f'{P1.shape=}')
    # print(f'{P2.mean()=}')
    # print(f'{b.mean()=}')

    t_0 = time.time()
    t_stage_solve = time.perf_counter()

    lb = -amps / np.sqrt(2.0)
    ub = amps / np.sqrt(2.0)

    ridge_sigma_x = None
    if use_spectrum_weighted_ridge:
        # Use the spectrum-derived coefficient scale as a prior std-dev for each component.
        # This is physically motivated: low-energy components should have small coefficients.
        # Use the same scale used by the box bounds (ub) so the prior matches the constraint.
        ridge_sigma_x = np.maximum(np.asarray(ub, dtype=float), float(spectrum_ridge_floor))

    # Keep matrix construction fast in float32, but solve in float64 for optimizer stability.
    P1_solve = np.asarray(P1, dtype=np.float64)
    P2_solve = np.asarray(P2, dtype=np.float64)
    b_solve = np.asarray(b, dtype=np.float64)
    lb_solve = np.asarray(lb, dtype=np.float64)
    ub_solve = np.asarray(ub, dtype=np.float64)

    A, info = solve_box_ridge_lbfgsb(
        P1_solve,
        b_solve,
        lb_solve,
        ub_solve,
        x0=A0,  # pass previous A here for warm-start (see below)
        ridge=float(ridge),
        max_iter=int(max_iter),
        ridge_sigma_x=ridge_sigma_x,
        backend=solver_backend,
    )

    t = time.time() - t_0
    if prof is not None:
        prof['solve_total_s'] = float(time.perf_counter() - t_stage_solve)
    if diagnostics:
        print(f'solve time: {t:.4f} s', flush=True)
    # print(f'{A.sum()=}')

    gram_diag = None
    if gram_diagnostics:
        try:
            from .gram_diagnostics import gram_diagnostics as gram_diagnostics_fn

            gram_diag = gram_diagnostics_fn(
                P1_solve,
                subset_size=int(gram_diag_subset_size),
                out_dir=gram_diag_out_dir,
                prefix=str(gram_diag_prefix),
            )
        except Exception as exc:
            if diagnostics:
                print(f'Gram diag: failed: {exc}', flush=True)

    t_stage = prof_mark('gram_diag_s', t_stage)

    # Diagnostics: are coefficients sitting on bounds?
    # This helps distinguish a well-posed solve from an extrapolation/under-regularized solve.
    bound_ratio = np.abs(A) / ub
    bound_ratio = bound_ratio[np.isfinite(bound_ratio)]
    if bound_ratio.size:
        frac_near = float(np.mean(bound_ratio >= float(near_bound_ratio)))
        n_near = int(np.sum(bound_ratio >= float(near_bound_ratio)))
        max_ratio = float(np.max(bound_ratio))
        p95_ratio = float(np.percentile(bound_ratio, 95.0))
    else:
        frac_near = 0.0
        n_near = 0
        max_ratio = float('nan')
        p95_ratio = float('nan')

    if diagnostics:
        try:
            nit = int(getattr(info, 'nit', -1))
            success = bool(getattr(info, 'success', False))
            status = int(getattr(info, 'status', -1))
            message = str(getattr(info, 'message', '') or '')
        except Exception:
            nit = -1
            success = False
            status = -1
            message = ''

        message = message.replace('\n', ' ').strip()
        if len(message) > 120:
            message = message[:117] + '...'

        hit_maxiter = (nit >= 0) and (int(max_iter) > 0) and (nit >= int(max_iter))

        print(
            'LSQ diag: '
            f'n_cols={P1_solve.shape[1]} nit={nit}/{int(max_iter)} '
            f'hit_maxiter={hit_maxiter} status={status} success={success} '
            f'msg="{message}" '
            f'near_bound(thr={near_bound_ratio:.2f})={n_near}/{P1_solve.shape[1]} '
            f'frac={frac_near:.3f} max={max_ratio:.3f} p95={p95_ratio:.3f}',
            flush=True,
        )

        if gram_diag is not None:
            print(
                'Gram diag: '
                f'm={gram_diag.n_rows} n={gram_diag.n_cols} '
                f'offdiag_F_est={gram_diag.offdiag_fro_norm_est:.3e} '
                f'offdiag_rms_est={gram_diag.offdiag_rms_est:.3e} '
                f'max_abs_offdiag_sample={gram_diag.max_abs_offdiag_sample:.3e} '
                f'n_pairs={gram_diag.n_pairs_sampled}',
                flush=True,
            )

    t_stage = prof_mark('diagnostics_s', t_stage)

    # reconstructed fields
    zc = P1_solve @ A
    z2 = P2_solve @ A
    # print(f'{zc=} {z2=}')

    t_stage = prof_mark('recon_s', t_stage)

    # bookkeeping into params
    params = LSQWavePropParams()
    params.A = A
    params.bound_near_ratio_threshold = float(near_bound_ratio)
    params.bound_frac_ge_threshold = float(frac_near)
    params.bound_ratio_max = float(max_ratio)
    params.bound_ratio_p95 = float(p95_ratio)
    params.solver_success = bool(getattr(info, 'success', False))
    params.solver_nit = int(getattr(info, 'nit', 0) or 0)
    params.solver_status = int(getattr(info, 'status', 0) or 0)
    params.Etheta = np.zeros_like(Ei.flatten(order='F')).T
    params.Etheta[good_base] = (A[: (len(A) // 2)] ** 2.0 + A[(len(A) // 2):] ** 2.0) / 2.0
    params.Etheta = params.Etheta.reshape((len(k), len(theta)), order='F').T
    params.Etheta /= (
        np.diff(f2[0, :] / (2.0 * np.pi), prepend=0.0)
        * sps.mode(
            np.diff(np.degrees(thet2[:, 0])),
            axis=None,
            keepdims=False
        ).mode.item()
    )

    params.f = (f2[0, :] / (2.0 * np.pi)).flatten()
    params.theta = np.degrees(thet2[:, 0])
    params.theta += 180.0
    params.theta[params.theta > 360.0] -= 360.0
    sort_idx = np.argsort(params.theta)
    params.theta = params.theta[sort_idx].flatten()
    params.Etheta = params.Etheta[sort_idx, :].T

    # After pruning, `kx/ky/omega` already correspond to the kept components.
    # Do not index them with `good` (which refers to base-grid indices).
    params.kx = np.asarray(kx, dtype=float).reshape((-1,))
    params.ky = np.asarray(ky, dtype=float).reshape((-1,))
    params.omega = np.asarray(omega, dtype=float).reshape((-1,))
    params.use_vel = use_vel

    t_stage = prof_mark('params_pack_s', t_stage)

    if prof is not None:
        prof['total_lsq_func_s'] = float(time.perf_counter() - float(prof.get('t0', t_stage)))
        print(
            'LSQ profile [ms]: '
            f"spectrum={1e3*float(prof.get('spectrum_prep_s', 0.0)):.1f} "
            f"k_theta={1e3*float(prof.get('build_k_theta_s', 0.0)):.1f} "
            f"interp={1e3*float(prof.get('interp_griddata_s', 0.0)):.1f} "
            f"phi_in={1e3*float(prof.get('build_phi_input_s', 0.0)):.1f} "
            f"phi_out={1e3*float(prof.get('build_phi_output_s', 0.0)):.1f} "
            f"P={1e3*float(prof.get('build_P_mats_s', 0.0)):.1f} "
            f"rhs={1e3*float(prof.get('build_rhs_s', 0.0)):.1f} "
            f"solve={1e3*float(prof.get('solve_total_s', 0.0)):.1f} "
            f"total={1e3*float(prof.get('total_lsq_func_s', 0.0)):.1f}",
            flush=True,
        )
        # Remove private key to keep logs/results clean.
        prof.pop('t0', None)

    # import sys; sys.exit(0)

    # return shapes consistent with MATLAB-style (column vectors)
    return (
        z2.reshape((-1, 1), order='F'),
        zc.reshape((-1, 1), order='F'),
        params,
        t,
    )
