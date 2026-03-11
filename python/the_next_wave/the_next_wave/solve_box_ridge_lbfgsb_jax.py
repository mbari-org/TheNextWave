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


# JIT cache keyed on (n_rows, n_cols, use_scale2, ridge, max_iter).
# First call per shape pays ~5-6s compile cost; subsequent calls reuse the kernel.
# Ps/bw are arguments (not closure constants) so JAX traces shapes only and
# the compiled kernel accepts any data with those shapes.
JIT_CACHE: dict = {}


def get_jit_solve(n_rows: int, n_cols: int, use_scale2: bool, ridge: float, max_iter: int):
    key = (n_rows, n_cols, bool(use_scale2), float(ridge), int(max_iter))
    if key in JIT_CACHE:
        return JIT_CACHE[key]

    ridge_float = float(ridge)
    max_iter_int = int(max_iter)

    if use_scale2:
        def solve(Ps_j, bw_j, xs0_j, lb_j, ub_j, scale2_j):
            def fun(xs):
                r = Ps_j @ xs - bw_j
                return 0.5 * jnp.dot(r, r) + 0.5 * ridge_float * jnp.sum(xs * xs * scale2_j)
            solver = jaxopt.LBFGSB(
                fun=fun, maxiter=max_iter_int, tol=1e-5,
                linesearch='zoom', history_size=10, implicit_diff=False)
            return solver.run(xs0_j, bounds=(lb_j, ub_j))
    else:
        def solve(Ps_j, bw_j, xs0_j, lb_j, ub_j):
            def fun(xs):
                r = Ps_j @ xs - bw_j
                return 0.5 * jnp.dot(r, r) + 0.5 * ridge_float * jnp.dot(xs, xs)
            solver = jaxopt.LBFGSB(
                fun=fun, maxiter=max_iter_int, tol=1e-5,
                linesearch='zoom', history_size=10, implicit_diff=False)
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
):
    """JAX/jaxopt drop-in for solve_box_ridge_lbfgsb. Same (x, info) return contract."""
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

    # row scaling
    row_rms = np.sqrt(np.mean(P * P, axis=1))
    row_rms[row_rms == 0.0] = 1.0
    w = 1.0 / row_rms
    Pw = P * w[:, None]
    bw = b * w

    # column scaling
    col_rms = np.sqrt(np.mean(Pw * Pw, axis=0))
    col_rms[col_rms == 0.0] = 1.0
    Ps = Pw / col_rms[None, :]

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
        ridge_sigma_x = np.asarray(ridge_sigma_x, dtype=np.float64).reshape(-1)
        denom = col_rms * ridge_sigma_x
        denom[denom == 0.0] = 1.0
        ridge_xs_scale2 = 1.0 / (denom * denom)

    Ps_j  = jnp.array(Ps,  dtype=jnp.float32)
    bw_j  = jnp.array(bw,  dtype=jnp.float32)
    lb_j  = jnp.array(lb_s, dtype=jnp.float32)
    ub_j  = jnp.array(ub_s, dtype=jnp.float32)
    xs0_j = jnp.array(xs0, dtype=jnp.float32)

    n_rows, n_cols = Ps.shape
    jit_fn = get_jit_solve(n_rows, n_cols, use_scale2, float(ridge), int(max_iter))

    if use_scale2:
        scale2_j = jnp.array(ridge_xs_scale2, dtype=jnp.float32)
        result = jit_fn(Ps_j, bw_j, xs0_j, lb_j, ub_j, scale2_j)
    else:
        result = jit_fn(Ps_j, bw_j, xs0_j, lb_j, ub_j)

    xs = np.array(result.params)
    x = xs / col_rms

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
