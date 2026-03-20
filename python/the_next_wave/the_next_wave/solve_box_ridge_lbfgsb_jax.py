# JAX/jaxopt L-BFGS-B backend. Drop-in for solve_box_ridge_lbfgsb.
# Requires jax[cuda13] + jaxopt. ~10-15x slower than scipy on CPU.

import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxopt
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# JIT cache keyed on problem shape + all hyperparameters that affect the compiled kernel.
# First call per unique key pays ~5-6s compile cost; subsequent calls reuse the kernel.
# Data arrays (Ps, bw, edge indices, …) are arguments so JAX traces shapes only.
JIT_CACHE: dict = {}


def build_smooth_edges(good_indices, grid_shape):
    """
    Build frequency- and direction-neighbor edge lists for the smoothness penalty.

    Coefficients are laid out in the flattened (F-order) grid of shape
    (n_k, n_theta): flat_idx = k_idx + n_k * t_idx.  Only components in
    `good_indices` are active.  Returns local index pairs (src, dst) into the
    pruned active set (not global grid indices).

    An edge-list approach is used rather than rectangular reshape because the
    active set may form an irregular region of the (n_k, n_theta) grid —
    rectangular reshape would only work if all active components fill a complete
    rectangle, which is not guaranteed for real spectra.

    Parameters
    ----------
    good_indices : array-like of int
        Global flat (F-order) indices of active components.
    grid_shape : (int, int)
        Full base grid shape (n_k, n_theta).

    Returns
    -------
    src_f, dst_f : (N_f,) int32 arrays — freq-adjacent active pairs
    src_t, dst_t : (N_t,) int32 arrays — theta-adjacent active pairs
    """
    good_indices = np.asarray(good_indices, dtype=int).reshape(-1)
    n_k, n_theta = int(grid_shape[0]), int(grid_shape[1])

    # map global grid position -> local index in the pruned active set
    pos_to_local = {}
    for local_i, gi in enumerate(good_indices):
        k_idx = int(gi) % n_k
        t_idx = int(gi) // n_k
        pos_to_local[(k_idx, t_idx)] = local_i

    src_f, dst_f = [], []
    src_t, dst_t = [], []
    for (k_idx, t_idx), local_i in pos_to_local.items():
        # freq neighbor: next higher-frequency bin
        nb = (k_idx + 1, t_idx)
        if nb in pos_to_local:
            src_f.append(local_i)
            dst_f.append(pos_to_local[nb])
        # theta neighbor: next direction bin
        nb = (k_idx, t_idx + 1)
        if nb in pos_to_local:
            src_t.append(local_i)
            dst_t.append(pos_to_local[nb])

    return (
        np.array(src_f, dtype=np.int32),
        np.array(dst_f, dtype=np.int32),
        np.array(src_t, dtype=np.int32),
        np.array(dst_t, dtype=np.int32),
    )


def get_jit_solve(
    n_rows: int,
    n_cols: int,
    use_scale2: bool,
    has_x_prev: bool,
    has_freq_smooth: bool,
    has_theta_smooth: bool,
    n_edges_f: int,
    n_edges_t: int,
    ridge: float,
    lambda_time: float,
    lambda_freq_smooth: float,
    lambda_theta_smooth: float,
    max_iter: int,
):
    """
    Return (and cache) a JIT-compiled solve function.

    The compiled function always has the same 12-arg signature:

      solve(Ps_j, bw_j, xs0_j, lb_j, ub_j,
            scale2_j,              # ridge per-component weights (or dummy)
            x_prev_s_j,            # scaled warm-start for time penalty (or dummy)
            inv_colrms_j,          # 1/col_rms for time penalty transform (or dummy)
            src_f_j, dst_f_j,      # freq-neighbor edge indices (or dummy)
            src_t_j, dst_t_j)      # theta-neighbor edge indices (or dummy)

    Disabled optional terms use dummy length-1 arrays; JAX statically eliminates
    dead `if` branches (Python booleans are closured, not traced) so there is no
    runtime cost for unused terms in the compiled kernel.
    """
    key = (
        int(n_rows), int(n_cols),
        bool(use_scale2), bool(has_x_prev),
        bool(has_freq_smooth), bool(has_theta_smooth),
        int(n_edges_f) if has_freq_smooth else 0,
        int(n_edges_t) if has_theta_smooth else 0,
        float(ridge), float(lambda_time),
        float(lambda_freq_smooth) if has_freq_smooth else 0.0,
        float(lambda_theta_smooth) if has_theta_smooth else 0.0,
        int(max_iter),
    )
    if key in JIT_CACHE:
        return JIT_CACHE[key]

    ridge_float  = float(ridge)
    lam_t        = float(lambda_time)
    lam_f        = float(lambda_freq_smooth)
    lam_th       = float(lambda_theta_smooth)
    max_iter_int = int(max_iter)
    # x layout: [A_cos_0 .. A_cos_{n_good-1}, A_sin_0 .. A_sin_{n_good-1}]
    n_good = int(n_cols) // 2

    def solve(
        Ps_j, bw_j, xs0_j, lb_j, ub_j,
        scale2_j, x_prev_s_j, inv_colrms_j,
        src_f_j, dst_f_j, src_t_j, dst_t_j,
    ):
        def fun(xs):
            r = Ps_j @ xs - bw_j
            loss = 0.5 * jnp.dot(r, r)

            # ridge (isotropic or spectrum-weighted)
            if use_scale2:
                loss = loss + 0.5 * ridge_float * jnp.sum(xs * xs * scale2_j)
            else:
                loss = loss + 0.5 * ridge_float * jnp.dot(xs, xs)

            # temporal continuity: 0.5*lam_t*||x - x0||^2
            # in scaled space: 0.5*lam_t*||(xs - x0s) * inv_colrms||^2
            if has_x_prev:
                diff = (xs - x_prev_s_j) * inv_colrms_j
                loss = loss + 0.5 * lam_t * jnp.dot(diff, diff)

            # frequency smoothness on both cos and sin halves
            if has_freq_smooth:
                d_cos = xs[src_f_j] - xs[dst_f_j]
                d_sin = xs[n_good + src_f_j] - xs[n_good + dst_f_j]
                loss = loss + 0.5 * lam_f * (jnp.dot(d_cos, d_cos) + jnp.dot(d_sin, d_sin))

            # theta smoothness on both cos and sin halves
            if has_theta_smooth:
                d_cos = xs[src_t_j] - xs[dst_t_j]
                d_sin = xs[n_good + src_t_j] - xs[n_good + dst_t_j]
                loss = loss + 0.5 * lam_th * (jnp.dot(d_cos, d_cos) + jnp.dot(d_sin, d_sin))

            return loss

        solver = jaxopt.LBFGSB(
            fun=fun, maxiter=max_iter_int, tol=1e-5,
            linesearch='zoom', history_size=10, implicit_diff=False,
        )
        return solver.run(xs0_j, bounds=(lb_j, ub_j))

    jit_fn = jax.jit(solve)
    JIT_CACHE[key] = jit_fn
    return jit_fn


def solve_box_ridge_lbfgsb_jax(
    P,
    b,
    lb,
    ub,
    x0=None,
    ridge=1e-6,
    max_iter=80,
    ridge_sigma_x=None,
    lambda_time=0.0,
    lambda_freq_smooth=0.0,
    lambda_theta_smooth=0.0,
    good_indices=None,
    grid_shape=None,
    print_losses=False,
    use_rank_reduction=False,
    rank_tol=1e-3,
    max_rank=None,
    use_row_scale=True,
    use_col_scale=True,
):
    """
    JAX/jaxopt drop-in for solve_box_ridge_lbfgsb. Same (x, info) return contract.

    Parameters
    ----------
    lambda_time : float
        Weight for the temporal continuity penalty 0.5*lambda_time*||x - x0||^2.
        x0 serves as both the warm-start initial guess and the previous-solution
        anchor for continuity.  Set to 0 (default) to disable.
    lambda_freq_smooth : float
        Weight for frequency-direction smoothness penalty:
          0.5 * lambda_freq_smooth * sum_{freq edges} (x[f+1,t] - x[f,t])^2
        Applied independently to the cosine and sine coefficient halves.
        Requires good_indices and grid_shape.  Set to 0 (default) to disable.
    lambda_theta_smooth : float
        Weight for direction smoothness penalty:
          0.5 * lambda_theta_smooth * sum_{theta edges} (x[f,t+1] - x[f,t])^2
        Applied independently to the cosine and sine coefficient halves.
        Requires good_indices and grid_shape.  Set to 0 (default) to disable.
    good_indices : array-like of int or None
        Flat F-order indices of the active (non-zero energy) components in the
        (n_k, n_theta) base grid.  Required when either lambda_*_smooth > 0.
    grid_shape : (int, int) or None
        Full base grid shape (n_k, n_theta).  Required when either
        lambda_*_smooth > 0.
    print_losses : bool
        Print itemized loss breakdown after solving (physical coefficient space):
        total, data, ridge, time, freq_smooth, theta_smooth.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "jax and jaxopt are required for the 'jax' backend. "
            "Install with: pip install 'jax[cuda13]' jaxopt"
        )

    try:
        on_gpu = len(jax.devices('gpu')) > 0
    except Exception:
        on_gpu = False

    if not on_gpu:
        warnings.warn(
            "backend='jax' but no JAX GPU device found — ~10-15x slower than scipy on CPU.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Use float32 throughout. jaxopt.LBFGSB + jax_enable_x64 causes int64/int32
    # index mismatches in XLA scatter ops. Row/col scaling keeps float32 precision
    # sufficient (verified max_diff vs scipy ~3e-6). TODO(andermi) report upstream.

    lambda_time_f  = float(lambda_time)
    lambda_freq_f  = float(lambda_freq_smooth)
    lambda_theta_f = float(lambda_theta_smooth)

    # -- time-penalty anchor (reuse x0) --
    has_x_prev = False
    x_prev_arr = None
    if x0 is not None and lambda_time_f != 0.0:
        x_prev_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x_prev_arr.shape[0] == P.shape[1] and np.all(np.isfinite(x_prev_arr)):
            has_x_prev = True
        else:
            x_prev_arr = None

    # -- smoothness edge lists (numpy, cheap) --
    # Edge indices are local to the pruned active set: src/dst in [0, n_good).
    # Penalty is applied to both cos and sin halves: xs[src], xs[n_good+src], etc.
    _dummy_edge = np.zeros(1, dtype=np.int32)
    src_f = dst_f = src_t = dst_t = _dummy_edge
    n_edges_f = n_edges_t = 0
    if (lambda_freq_f != 0.0 or lambda_theta_f != 0.0) \
            and good_indices is not None and grid_shape is not None:
        src_f, dst_f, src_t, dst_t = build_smooth_edges(good_indices, grid_shape)
        n_edges_f = len(src_f)
        n_edges_t = len(src_t)
    has_freq_smooth  = n_edges_f > 0 and lambda_freq_f  != 0.0
    has_theta_smooth = n_edges_t > 0 and lambda_theta_f != 0.0

    # Keep original ridge_sigma_x (physical space) for loss reporting.
    ridge_sigma_x_orig = (
        np.asarray(ridge_sigma_x, dtype=np.float64).reshape(-1)
        if ridge_sigma_x is not None else None
    )

    # -- row scaling --
    if use_row_scale:
        row_rms = np.sqrt(np.mean(P * P, axis=1))
        row_rms[row_rms == 0.0] = 1.0
        w = 1.0 / row_rms
        Pw = P * w[:, None]
        bw = b * w
    else:
        Pw = P
        bw = b

    # column scaling
    if use_col_scale:
        col_rms = np.sqrt(np.mean(Pw * Pw, axis=0))
        col_rms[col_rms == 0.0] = 1.0
        Ps = Pw / col_rms[None, :]
    else:
        col_rms = np.ones(P.shape[1], dtype=np.float64)
        Ps = Pw

    lb_s = lb * col_rms
    ub_s = ub * col_rms

    if x0 is None:
        xs0 = np.zeros(P.shape[1])
    else:
        xs0 = np.asarray(x0, dtype=np.float64).reshape(-1) * col_rms

    xs0 = np.minimum(np.maximum(xs0, lb_s), ub_s)

    ridge_xs_scale2 = None
    use_scale2 = ridge_sigma_x is not None
    if use_scale2:
        ridge_sigma_x_np = np.asarray(ridge_sigma_x, dtype=np.float64).reshape(-1)
        denom = col_rms * ridge_sigma_x_np
        denom[denom == 0.0] = 1.0
        ridge_xs_scale2 = 1.0 / (denom * denom)

    # Scaled x0 and 1/col_rms for the time penalty.
    # Penalty in physical space: 0.5*lam*||x - x0||^2
    # In scaled variables (x = xs/col_rms): 0.5*lam*||(xs-x0s)*inv_colrms||^2
    inv_col_rms = 1.0 / col_rms  # kept for both time penalty and loss reporting
    x_prev_s = x_prev_arr * col_rms if has_x_prev else None

    # -- to JAX arrays --
    _DUMMY_F   = jnp.zeros(1, dtype=jnp.float32)
    _DUMMY_I32 = jnp.zeros(1, dtype=jnp.int32)

    Ps_j  = jnp.array(Ps,   dtype=jnp.float32)
    bw_j  = jnp.array(bw,   dtype=jnp.float32)
    lb_j  = jnp.array(lb_s, dtype=jnp.float32)
    ub_j  = jnp.array(ub_s, dtype=jnp.float32)
    xs0_j = jnp.array(xs0,  dtype=jnp.float32)

    scale2_j     = jnp.array(ridge_xs_scale2, dtype=jnp.float32) if use_scale2  else _DUMMY_F
    x_prev_s_j   = jnp.array(x_prev_s,        dtype=jnp.float32) if has_x_prev  else _DUMMY_F
    inv_colrms_j = jnp.array(inv_col_rms,      dtype=jnp.float32) if has_x_prev  else _DUMMY_F
    src_f_j      = jnp.array(src_f, dtype=jnp.int32) if has_freq_smooth  else _DUMMY_I32
    dst_f_j      = jnp.array(dst_f, dtype=jnp.int32) if has_freq_smooth  else _DUMMY_I32
    src_t_j      = jnp.array(src_t, dtype=jnp.int32) if has_theta_smooth else _DUMMY_I32
    dst_t_j      = jnp.array(dst_t, dtype=jnp.int32) if has_theta_smooth else _DUMMY_I32

    n_rows, n_cols = Ps.shape

    # ------------------------------------------------------------------
    # Rank-reduction closed-form path (JAX/GPU SVD).
    # Bypass L-BFGS-B entirely: compute the truncated SVD of the scaled
    # design matrix on the GPU, then solve the diagonal normal equations
    # in closed form.  O(n_rows * n_cols * min(n_rows,n_cols)) for the SVD
    # itself, but a single GPU kernel — much faster than iterative methods
    # when n_cols is large and the effective rank is small.
    # Box constraints are enforced by clipping in scaled coefficient space
    # after back-projection (approximate — see leastSquaresWavePropagation
    # docstring for full discussion).
    # ------------------------------------------------------------------
    if use_rank_reduction:
        # ---------------------------------------------------------------
        # Build the augmented system incorporating ALL quadratic penalties
        # as extra rows, so the closed-form SVD solve matches the full
        # objective used by the L-BFGS-B path.
        #
        # All extra rows are expressed in column-scaled (xs) space so they
        # concatenate cleanly with Ps and bw.  We do NOT apply additional
        # row-scaling to penalty rows since their scale is already set by
        # the lambda values.
        #
        # Temporal continuity  0.5*lambda_t*||x - x0||^2
        #   = 0.5*lambda_t*||(xs - x_prev_s)*inv_col_rms||^2
        #   → append  diag(sqrt(lambda_t)*inv_col_rms)  and target
        #             sqrt(lambda_t)*inv_col_rms*x_prev_s = sqrt(lambda_t)*x0
        #
        # Freq / theta smoothness  0.5*lam*||L x||^2  where L is the
        #   edge-difference operator in physical x space.
        #   In xs space: (x[i]-x[j]) = (xs[i]/cr[i] - xs[j]/cr[j]), so
        #   each edge contributes a row with sqrt(lam)/cr[i] at col i and
        #   -sqrt(lam)/cr[j] at col j, applied to both cos and sin halves.
        # ---------------------------------------------------------------
        Ps_np = np.array(Ps_j, dtype=np.float32)   # (n_rows, n_cols)
        bw_np = np.array(bw_j, dtype=np.float32)   # (n_rows,)
        rows_list = [Ps_np]
        b_list    = [bw_np]
        n_good = n_cols // 2

        # ---- temporal continuity ----
        if has_x_prev and lambda_time_f > 0.0:
            sqrt_lt   = np.float32(np.sqrt(lambda_time_f))
            wt        = (sqrt_lt * inv_col_rms).astype(np.float32)   # (n_cols,)
            P_time    = np.diag(wt)                                   # (n_cols, n_cols)
            b_time    = (sqrt_lt * inv_col_rms * np.asarray(x_prev_s,
                          dtype=np.float64)).astype(np.float32)       # (n_cols,)
            rows_list.append(P_time)
            b_list.append(b_time)

        # ---- freq smoothness ----
        if has_freq_smooth and lambda_freq_f > 0.0:
            sqrt_lf  = np.float32(np.sqrt(lambda_freq_f))
            src_f_np = np.asarray(src_f, dtype=int)
            dst_f_np = np.asarray(dst_f, dtype=int)
            n_ef     = len(src_f_np)
            L_f      = np.zeros((2 * n_ef, n_cols), dtype=np.float32)
            idx      = np.arange(n_ef)
            # cos half
            L_f[idx, src_f_np]           =  sqrt_lf / col_rms[src_f_np].astype(np.float32)
            L_f[idx, dst_f_np]           = -sqrt_lf / col_rms[dst_f_np].astype(np.float32)
            # sin half
            L_f[n_ef + idx, n_good + src_f_np] =  sqrt_lf / col_rms[n_good + src_f_np].astype(np.float32)
            L_f[n_ef + idx, n_good + dst_f_np] = -sqrt_lf / col_rms[n_good + dst_f_np].astype(np.float32)
            rows_list.append(L_f)
            b_list.append(np.zeros(2 * n_ef, dtype=np.float32))

        # ---- theta smoothness ----
        if has_theta_smooth and lambda_theta_f > 0.0:
            sqrt_lth = np.float32(np.sqrt(lambda_theta_f))
            src_t_np = np.asarray(src_t, dtype=int)
            dst_t_np = np.asarray(dst_t, dtype=int)
            n_et     = len(src_t_np)
            L_t      = np.zeros((2 * n_et, n_cols), dtype=np.float32)
            idx      = np.arange(n_et)
            # cos half
            L_t[idx, src_t_np]           =  sqrt_lth / col_rms[src_t_np].astype(np.float32)
            L_t[idx, dst_t_np]           = -sqrt_lth / col_rms[dst_t_np].astype(np.float32)
            # sin half
            L_t[n_et + idx, n_good + src_t_np] =  sqrt_lth / col_rms[n_good + src_t_np].astype(np.float32)
            L_t[n_et + idx, n_good + dst_t_np] = -sqrt_lth / col_rms[n_good + dst_t_np].astype(np.float32)
            rows_list.append(L_t)
            b_list.append(np.zeros(2 * n_et, dtype=np.float32))

        Ps_aug_np = np.concatenate(rows_list, axis=0)   # (n_aug, n_cols)
        bw_aug_np = np.concatenate(b_list,    axis=0)   # (n_aug,)

        # SVD on the augmented matrix (GPU)
        Ps_aug_j = jnp.array(Ps_aug_np, dtype=jnp.float32)
        bw_aug_j = jnp.array(bw_aug_np, dtype=jnp.float32)
        U_j, s_j, Vt_j = jnp.linalg.svd(Ps_aug_j, full_matrices=False)
        s_np = np.array(s_j, dtype=np.float64)

        if s_np[0] <= 0.0:
            r = 0
            x = np.zeros(n_cols, dtype=np.float64)
        else:
            thresh = float(rank_tol) * s_np[0]
            r = int(np.sum(s_np >= thresh))
            if max_rank is not None:
                r = min(r, int(max_rank))
            r = max(r, 1)
            s_r  = s_j[:r]
            U_r  = U_j[:, :r]
            Vt_r = Vt_j[:r, :]
            Utb  = U_r.T @ bw_aug_j
            z    = (s_r * Utb) / (s_r * s_r + float(ridge))
            xs_j = jnp.clip(Vt_r.T @ z, lb_j, ub_j)
            xs   = np.array(xs_j, dtype=np.float64)
            x    = xs / col_rms

        if print_losses:
            r_phys     = P @ x - b
            data_loss  = 0.5 * float(np.dot(r_phys, r_phys))
            ridge_loss = 0.5 * float(ridge) * float(np.dot(x, x))

            x_cos = x[:n_good]
            x_sin = x[n_good:]

            if has_x_prev and lambda_time_f > 0.0:
                diff      = x - x_prev_arr
                time_loss = 0.5 * lambda_time_f * float(np.dot(diff, diff))
            else:
                time_loss = 0.0

            if has_freq_smooth and lambda_freq_f > 0.0:
                src_f_np = np.asarray(src_f, dtype=int)
                dst_f_np = np.asarray(dst_f, dtype=int)
                dc = x_cos[src_f_np] - x_cos[dst_f_np]
                ds = x_sin[src_f_np] - x_sin[dst_f_np]
                freq_smooth_loss = 0.5 * lambda_freq_f * (float(np.dot(dc, dc)) + float(np.dot(ds, ds)))
            else:
                freq_smooth_loss = 0.0

            if has_theta_smooth and lambda_theta_f > 0.0:
                src_t_np = np.asarray(src_t, dtype=int)
                dst_t_np = np.asarray(dst_t, dtype=int)
                dc = x_cos[src_t_np] - x_cos[dst_t_np]
                ds = x_sin[src_t_np] - x_sin[dst_t_np]
                theta_smooth_loss = 0.5 * lambda_theta_f * (float(np.dot(dc, dc)) + float(np.dot(ds, ds)))
            else:
                theta_smooth_loss = 0.0

            total_loss = data_loss + ridge_loss + time_loss + freq_smooth_loss + theta_smooth_loss
            print(
                f'SVD-rank-{r} loss:  total={total_loss:.6e}  '
                f'data={data_loss:.6e}  ridge={ridge_loss:.6e}  '
                f'time={time_loss:.6e}  freq_smooth={freq_smooth_loss:.6e}  '
                f'theta_smooth={theta_smooth_loss:.6e}',
                flush=True,
            )

        class _RankInfo:
            pass
        info = _RankInfo()
        info.x = x
        info.nit = 1
        info.fun = float(np.dot(P @ x - b, P @ x - b)) * 0.5
        info.success = True
        info.status = 0
        info.message = f'rank-reduced closed-form (r={r}, JAX/GPU)'
        info.rank_used = r
        info.singular_values_used = s_np[:r]
        return x, info

    jit_fn = get_jit_solve(
        n_rows, n_cols,
        use_scale2, has_x_prev,
        has_freq_smooth, has_theta_smooth,
        n_edges_f, n_edges_t,
        float(ridge), lambda_time_f, lambda_freq_f, lambda_theta_f,
        int(max_iter),
    )

    result = jit_fn(
        Ps_j, bw_j, xs0_j, lb_j, ub_j,
        scale2_j, x_prev_s_j, inv_colrms_j,
        src_f_j, dst_f_j, src_t_j, dst_t_j,
    )

    xs = np.array(result.params)
    x  = xs / col_rms

    # ------------------------------------------------------------------
    # Itemized loss in the original (physical) coefficient space.
    # ------------------------------------------------------------------
    if print_losses:
        r_phys    = P @ x - b
        data_loss = 0.5 * float(np.dot(r_phys, r_phys))

        if ridge_sigma_x_orig is not None and ridge_sigma_x_orig.shape[0] == x.shape[0]:
            safe_sigma = ridge_sigma_x_orig.copy()
            safe_sigma[safe_sigma == 0.0] = 1.0
            ridge_loss = 0.5 * float(ridge) * float(np.sum((x / safe_sigma) ** 2))
        else:
            ridge_loss = 0.5 * float(ridge) * float(np.dot(x, x))

        if has_x_prev:
            diff = x - x_prev_arr
            time_loss = 0.5 * lambda_time_f * float(np.dot(diff, diff))
        else:
            time_loss = 0.0

        n_good   = n_cols // 2
        x_cos    = x[:n_good]
        x_sin    = x[n_good:]

        if has_freq_smooth:
            dc = x_cos[src_f] - x_cos[dst_f]
            ds = x_sin[src_f] - x_sin[dst_f]
            freq_smooth_loss = 0.5 * lambda_freq_f * (float(np.dot(dc, dc)) + float(np.dot(ds, ds)))
        else:
            freq_smooth_loss = 0.0

        if has_theta_smooth:
            dc = x_cos[src_t] - x_cos[dst_t]
            ds = x_sin[src_t] - x_sin[dst_t]
            theta_smooth_loss = 0.5 * lambda_theta_f * (float(np.dot(dc, dc)) + float(np.dot(ds, ds)))
        else:
            theta_smooth_loss = 0.0

        total_loss = data_loss + ridge_loss + time_loss + freq_smooth_loss + theta_smooth_loss
        print(
            f'JAX loss:  total={total_loss:.6e}  '
            f'data={data_loss:.6e}  '
            f'ridge={ridge_loss:.6e}  '
            f'time={time_loss:.6e}  '
            f'freq_smooth={freq_smooth_loss:.6e}  '
            f'theta_smooth={theta_smooth_loss:.6e}',
            flush=True,
        )

    class Result:
        pass

    info = Result()
    info.x = x
    info.nit = int(result.state.iter_num)
    info.fun = float(result.state.value)
    info.success = bool(result.state.error < 1e-5)
    info.status = 0 if info.success else 1
    info.message = (
        "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
        if info.success
        else "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
    )

    return x, info
