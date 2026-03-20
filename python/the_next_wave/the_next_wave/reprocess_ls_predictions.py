from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .swift import LSQWavePropParams, Prediction

ArrayLike = npt.ArrayLike


def coerce_windowed_inputs(
    x: ArrayLike,
    y: ArrayLike,
    t: ArrayLike,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Broadcast `x/y/t` to a common shape and coerce to `(window, sample)` arrays."""
    x_arr, y_arr, t_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(t, dtype=float),
    )
    original_shape = x_arr.shape

    if x_arr.ndim == 0:
        if n_windows != 1:
            raise ValueError(
                'Scalar x/y/t inputs require exactly one parameter window. '
                f'Got {n_windows} windows.'
            )
        return (
            x_arr.reshape((1, 1)),
            y_arr.reshape((1, 1)),
            t_arr.reshape((1, 1)),
            original_shape,
        )

    if x_arr.ndim == 1:
        if x_arr.shape[0] != n_windows:
            raise ValueError(
                'For 1D x/y/t inputs, len(x) must equal the number of parameter windows. '
                f'Got len(x)={x_arr.shape[0]} and n_windows={n_windows}.'
            )
        return (
            x_arr.reshape((n_windows, 1)),
            y_arr.reshape((n_windows, 1)),
            t_arr.reshape((n_windows, 1)),
            original_shape,
        )

    if x_arr.shape[0] != n_windows:
        raise ValueError(
            'The first dimension of x/y/t must match the number of parameter windows. '
            f'Got x.shape[0]={x_arr.shape[0]} and n_windows={n_windows}.'
        )

    return (
        x_arr.reshape((x_arr.shape[0], -1)),
        y_arr.reshape((y_arr.shape[0], -1)),
        t_arr.reshape((t_arr.shape[0], -1)),
        original_shape,
    )


def reshape_output(arr2d: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | float:
    if shape == ():
        return float(arr2d.reshape((-1,))[0])
    return arr2d.reshape(shape)


def params_to_sequence(
    params: Prediction | Sequence[LSQWavePropParams] | Sequence[Any],
) -> list[Any]:
    if isinstance(params, Prediction):
        return list(params.params)
    if hasattr(params, 'params') and not isinstance(params, Sequence):
        return list(getattr(params, 'params'))
    return list(params)


def reprocess_ls_predictions(
    x: ArrayLike,
    y: ArrayLike,
    t: ArrayLike,
    params: Prediction | Sequence[LSQWavePropParams] | Sequence[Any],
    *,
    amplitude_average_windows: int = 3,
    return_velocities: bool = False,
) -> np.ndarray | float | tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """
    Re-evaluate solved LSQ wave parameters at new target positions/times.

    This ports the MATLAB workflow in [reprocess_LS_predictions.m](../../reprocess_LS_predictions.m):
    for each window `i`, evaluate the current window's `kx/ky/omega` basis at the
    requested `x/y/t`, but use the mean amplitude vector across the last
    `amplitude_average_windows` solves.

    Parameters
    ----------
    x, y, t : array-like
        Target positions/times. These must broadcast to the same shape.
        The first dimension is interpreted as window index and must match the
        number of parameter windows. For 1D inputs, each element is treated as
        one target sample for one window.
    params : Prediction or sequence of parameter objects
        A `Prediction` instance or sequence whose items expose `A`, `kx`, `ky`,
        `omega`, and `use_vel` like `LSQWavePropParams`.
    amplitude_average_windows : int, optional
        Number of consecutive windows used when averaging amplitudes.
        Default is 3, matching the MATLAB file. With the default, the first two
        output windows are left as `NaN`.
    return_velocities : bool, optional
        If `True`, also return `(u, v)` computed with the same basis. When
        `False`, return only `z`, matching the MATLAB function intent.

    Returns
    -------
    z : ndarray or float
        Reprocessed heave prediction with the same shape as `x/y/t`.
    (z, u, v) : tuple, optional
        Returned when `return_velocities=True`.

    Notes
    -----
    - Windows with missing or incompatible parameter vectors are left as `NaN`.
    - `use_vel=False` windows still produce `z`; their `u/v` outputs remain `NaN`.
    """
    params_seq = params_to_sequence(params)
    n_windows = len(params_seq)
    if n_windows == 0:
        raise ValueError('`params` must contain at least one parameter window.')

    amplitude_average_windows = int(amplitude_average_windows)
    if amplitude_average_windows <= 0:
        raise ValueError('`amplitude_average_windows` must be >= 1.')

    x2d, y2d, t2d, original_shape = coerce_windowed_inputs(x, y, t, n_windows)

    z_out = np.full(x2d.shape, np.nan, dtype=float)
    u_out = np.full(x2d.shape, np.nan, dtype=float)
    v_out = np.full(x2d.shape, np.nan, dtype=float)

    for i in range(amplitude_average_windows - 1, n_windows):
        p = params_seq[i]
        if p is None:
            continue

        kx = np.asarray(getattr(p, 'kx', np.array([])), dtype=float).reshape((-1,))
        ky = np.asarray(getattr(p, 'ky', np.array([])), dtype=float).reshape((-1,))
        omega = np.asarray(getattr(p, 'omega', np.array([])), dtype=float).reshape((-1,))
        if kx.size == 0 or ky.size != kx.size or omega.size != kx.size:
            continue

        A_stack = []
        expected_size = 2 * kx.size
        for j in range(i - amplitude_average_windows + 1, i + 1):
            pj = params_seq[j]
            if pj is None:
                continue
            Aj = np.asarray(getattr(pj, 'A', np.array([])), dtype=float).reshape((-1,))
            if Aj.size != expected_size:
                continue
            A_stack.append(Aj)

        if not A_stack:
            continue

        with np.errstate(invalid='ignore'):
            A_mean = np.nanmean(np.stack(A_stack, axis=1), axis=1)
        if A_mean.size != expected_size or not np.any(np.isfinite(A_mean)):
            continue

        xw = x2d[i, :].reshape((-1, 1))
        yw = y2d[i, :].reshape((-1, 1))
        tw = t2d[i, :].reshape((-1, 1))

        phi = xw @ kx.reshape((1, -1)) + yw @ ky.reshape((1, -1)) - tw @ omega.reshape((1, -1))
        c = np.cos(phi)
        s = np.sin(phi)

        Ac = A_mean[:kx.size]
        As = A_mean[kx.size:]
        z_out[i, :] = (c @ Ac + s @ As).reshape((-1,))

        if return_velocities and bool(getattr(p, 'use_vel', False)):
            k_norm = np.sqrt(kx * kx + ky * ky)
            k_norm[k_norm == 0.0] = np.nan
            cu = np.nan_to_num((kx / k_norm) * omega, nan=0.0, posinf=0.0, neginf=0.0)
            cv = np.nan_to_num((ky / k_norm) * omega, nan=0.0, posinf=0.0, neginf=0.0)
            u_out[i, :] = ((c * cu.reshape((1, -1))) @ Ac + (s * cu.reshape((1, -1))) @ As).reshape((-1,))
            v_out[i, :] = ((c * cv.reshape((1, -1))) @ Ac + (s * cv.reshape((1, -1))) @ As).reshape((-1,))

    z_shaped = reshape_output(z_out, original_shape)
    if not return_velocities:
        return z_shaped

    return (
        z_shaped,
        reshape_output(u_out, original_shape),
        reshape_output(v_out, original_shape),
    )
