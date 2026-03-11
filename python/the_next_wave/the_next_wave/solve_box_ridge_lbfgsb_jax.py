"""
JAX-based L-BFGS-B solver matching the scipy solve_box_ridge_lbfgsb interface.

⚠️  GPU-only recommendation
    JAX/XLA on CPU is ~10-15x SLOWER than scipy for the matrix sizes used here
    (~300 rows x 2000 cols).  The JAX backend is only beneficial when JAX is
    backed by a CUDA GPU device.  Use backend='scipy' (the default) on CPU.

Requires: jax (>=0.4), jaxopt (>=0.8)
  CPU-only:  pip install jax jaxopt           (not recommended — slower than scipy)
  CUDA 13:   pip install "jax[cuda13]" jaxopt  (recommended use case)

Performance note — JIT caching
    The previous implementation closed over Ps_j/bw_j inside `fun`, making each
    call look like a brand-new function to XLA and triggering a full ~5-6 s
    recompilation on every window iteration.

    Now Ps_j/bw_j/scale2_j are passed as explicit *arguments* to `fun`, and the
    jaxopt.LBFGSB solver is cached at module level keyed on
    (n_rows, n_cols, use_scale2, ridge, max_iter).  JAX compiles once on the
    first call for a given problem shape and reuses the XLA kernel on all
    subsequent calls with the same shape, bringing warm-call time well below 1 s.
"""

import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxopt
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level JIT-function cache.
# key  = (n_rows, n_cols, use_scale2: bool, ridge: float, max_iter: int)
# value = jax.jit-compiled solve function
#
# Strategy: wrap the entire solver.run() call inside jax.jit, with Ps_j/bw_j
# etc. as *arguments* to the JITted function.  When JAX traces the function
# for the first time (per input shape), it sees these as abstract traced
# values — not Python constants — so the compiled XLA kernel works for any
# data with those shapes.  All subsequent calls with the same shape reuse the
# compiled kernel without retracing, making warm calls fast.
#
# Why not pass extra args through solver.run(*args)?  jaxopt.LBFGSB.init_state
# has `bounds` as the second positional argument, so mixing extra fun-args and
# bounds in the same solver.run() call causes a "multiple values" TypeError.
# ---------------------------------------------------------------------------
_JIT_CACHE: dict = {}


def _get_jit_solve(n_rows: int, n_cols: int, use_scale2: bool, ridge: float, max_iter: int):
    """Return a cached jax.jit-compiled solve function for the given problem spec."""
    key = (n_rows, n_cols, bool(use_scale2), float(ridge), int(max_iter))
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    ridge_float = float(ridge)
    max_iter_int = int(max_iter)

    # Inside the jax.jit trace, Ps_j/bw_j/scale2_j are traced abstract values.
    # `fun` closes over them, so JAX sees their shapes (not values) and compiles
    # a single XLA kernel that works for any concrete data of those shapes.
    if use_scale2:
        def _solve(Ps_j, bw_j, xs0_j, lb_j, ub_j, scale2_j):
            def fun(xs):
                r = Ps_j @ xs - bw_j
                return 0.5 * jnp.dot(r, r) + 0.5 * ridge_float * jnp.sum(xs * xs * scale2_j)
            solver = jaxopt.LBFGSB(
                fun=fun, maxiter=max_iter_int, tol=1e-5,
                linesearch='zoom', history_size=10, implicit_diff=False)
            return solver.run(xs0_j, bounds=(lb_j, ub_j))
    else:
        def _solve(Ps_j, bw_j, xs0_j, lb_j, ub_j):
            def fun(xs):
                r = Ps_j @ xs - bw_j
                return 0.5 * jnp.dot(r, r) + 0.5 * ridge_float * jnp.dot(xs, xs)
            solver = jaxopt.LBFGSB(
                fun=fun, maxiter=max_iter_int, tol=1e-5,
                linesearch='zoom', history_size=10, implicit_diff=False)
            return solver.run(xs0_j, bounds=(lb_j, ub_j))

    jit_fn = jax.jit(_solve)
    _JIT_CACHE[key] = jit_fn
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
    """
    Solve: min 0.5||P x - b||^2 + 0.5*ridge*||x||^2  s.t. lb <= x <= ub.

    Drop-in JAX/jaxopt replacement for the scipy L-BFGS-B backend.
    Returns (x, info) with the same attribute contract as scipy.optimize.OptimizeResult.

    First call for a given (n_rows, n_cols, use_scale2, ridge, max_iter) triggers
    JIT compilation (~5-6 s).  Subsequent calls with the same problem shape reuse
    the compiled XLA kernel and run in GPU time only (~0.1-0.5 s).
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError(
            "jax and jaxopt are required for the 'jax' backend.\n"
            "  CPU-only:  pip install jax jaxopt\n"
            "  CUDA 13:   pip install 'jax[cuda13]' jaxopt"
        )

    # Warn if no GPU is available — JAX on CPU is ~10-15x slower than scipy
    # for the matrix sizes used here.  This is not an error; the results will
    # still be correct, just slow.
    try:
        gpu_devices = jax.devices('gpu')
        _on_gpu = len(gpu_devices) > 0
    except Exception:
        _on_gpu = False

    if not _on_gpu:
        warnings.warn(
            "backend='jax' selected but no JAX GPU device was found. "
            "JAX/XLA on CPU is typically 10-15x slower than backend='scipy' "
            "for this problem size. Consider using backend='scipy' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    #
    # Background: jaxopt.LBFGSB uses jnp.argsort internally for the Cauchy
    # point computation. When jax_enable_x64=True is active, jnp.argsort
    # returns int64 indices, which causes an XLA HLO type mismatch in the
    # downstream scatter/reduce ops (s32[] vs s64[]).  This is a bug in
    # jaxopt that should be reported at:
    #   https://github.com/google/jaxopt/issues
    #
    # Workaround: keep JAX in its default float32 mode and pass pre-scaled
    # float32 arrays.  The row/column scaling applied below brings all values
    # into a well-conditioned range where float32 precision is sufficient for
    # L-BFGS-B to converge to the same solution as the float64 scipy backend
    # (verified: max_diff ~3e-6, rel_diff ~2e-5).

    # --- row scaling ---
    row_rms = np.sqrt(np.mean(P * P, axis=1))
    row_rms[row_rms == 0.0] = 1.0
    w = 1.0 / row_rms
    Pw = P * w[:, None]
    bw = b * w

    # --- column scaling ---
    col_rms = np.sqrt(np.mean(Pw * Pw, axis=0))
    col_rms[col_rms == 0.0] = 1.0
    Ps = Pw / col_rms[None, :]

    # --- scaled bounds & warm start ---
    lb_s = lb * col_rms
    ub_s = ub * col_rms

    if x0 is None:
        xs0 = np.zeros(P.shape[1])
    else:
        xs0 = np.asarray(x0, dtype=np.float64).reshape(-1) * col_rms

    xs0 = np.minimum(np.maximum(xs0, lb_s), ub_s)

    # --- optional spectrum-weighted ridge ---
    ridge_xs_scale2_np = None
    use_scale2 = ridge_sigma_x is not None
    if use_scale2:
        ridge_sigma_x = np.asarray(ridge_sigma_x, dtype=np.float64).reshape(-1)
        denom = col_rms * ridge_sigma_x
        denom[denom == 0.0] = 1.0
        ridge_xs_scale2_np = 1.0 / (denom * denom)

    # --- move to JAX arrays (float32 — see note above re: jax_enable_x64) ---
    Ps_j  = jnp.array(Ps,  dtype=jnp.float32)
    bw_j  = jnp.array(bw,  dtype=jnp.float32)
    lb_j  = jnp.array(lb_s, dtype=jnp.float32)
    ub_j  = jnp.array(ub_s, dtype=jnp.float32)
    xs0_j = jnp.array(xs0, dtype=jnp.float32)

    # --- retrieve (or create+compile) cached JIT-compiled solve function ---
    # The JITted function closes over Ps_j/bw_j/scale2_j as traced (abstract)
    # values, so the compiled XLA kernel is reused for all calls with the same
    # array shapes.  Only the first call per shape pays compilation cost.
    n_rows, n_cols = Ps.shape
    jit_fn = _get_jit_solve(n_rows, n_cols, use_scale2, float(ridge), int(max_iter))

    if use_scale2:
        scale2_j = jnp.array(ridge_xs_scale2_np, dtype=jnp.float32)
        result = jit_fn(Ps_j, bw_j, xs0_j, lb_j, ub_j, scale2_j)
    else:
        result = jit_fn(Ps_j, bw_j, xs0_j, lb_j, ub_j)

    xs = np.array(result.params)
    x = xs / col_rms

    # --- wrap in a scipy-compatible result object ---
    class _Result:
        pass

    info = _Result()
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
