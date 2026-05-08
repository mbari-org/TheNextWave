"""JAX/jaxopt L-BFGS-B backend for the simple bounded solve."""

import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxopt
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# JIT cache keyed on problem shape + iteration limit.
# First call per unique key pays the compile cost; later calls reuse the kernel.
JIT_CACHE: dict = {}


def get_jit_solve(n_rows: int, n_cols: int, max_iter: int):
    key = (int(n_rows), int(n_cols), int(max_iter))
    if key in JIT_CACHE:
        return JIT_CACHE[key]

    max_iter_int = int(max_iter)

    def solve(P_j, b_j, x0_j, lb_j, ub_j):
        def fun(x):
            r = P_j @ x - b_j
            return 0.5 * jnp.dot(r, r)

        solver = jaxopt.LBFGSB(
            fun=fun,
            maxiter=max_iter_int,
            tol=1e-5,
            linesearch='zoom',
            history_size=10,
            implicit_diff=False,
        )
        return solver.run(x0_j, bounds=(lb_j, ub_j))

    jit_fn = jax.jit(solve)
    JIT_CACHE[key] = jit_fn
    return jit_fn


def solve_box_lbfgsb_jax(
    P,
    b,
    lb,
    ub,
    x0=None,
    max_iter=80,
    print_losses=False,
):
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
            "backend='jax' but no JAX GPU device found — slower than scipy on CPU.",
            RuntimeWarning,
            stacklevel=2,
        )

    P = np.asarray(P, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)

    if x0 is None:
        x_init = np.zeros(P.shape[1], dtype=np.float64)
    else:
        x_init = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x_init.size != P.shape[1] or not np.all(np.isfinite(x_init)):
            x_init = np.zeros(P.shape[1], dtype=np.float64)
    x_init = np.minimum(np.maximum(x_init, lb), ub)

    P_j = jnp.array(P, dtype=jnp.float32)
    b_j = jnp.array(b, dtype=jnp.float32)
    x0_j = jnp.array(x_init, dtype=jnp.float32)
    lb_j = jnp.array(lb, dtype=jnp.float32)
    ub_j = jnp.array(ub, dtype=jnp.float32)

    jit_fn = get_jit_solve(P.shape[0], P.shape[1], int(max_iter))
    result = jit_fn(P_j, b_j, x0_j, lb_j, ub_j)

    x = np.array(result.params, dtype=np.float64)

    if print_losses:
        r = P @ x - b
        data_loss = 0.5 * float(r @ r)
        print(
            f'JAX loss:  total={data_loss:.6e}  data={data_loss:.6e}',
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
        'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
        if info.success
        else 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
    )
    return x, info
