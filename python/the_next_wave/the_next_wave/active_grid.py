"""
Active-grid selection for the wave-propagation inverse problem.

Reduces the number of unknown coefficients by keeping only (frequency, direction)
bins that have plausible energy in the input spectrum.  Building a smaller design
matrix P cuts solve time and improves conditioning.

Grid convention (matches leastSquaresWavePropagation.py):
  Ei_f_theta  shape (n_f, n_theta)
  flat F-order index = f_idx + n_f * t_idx   (same as amps.T.flatten(order='F'))

Typical defaults guidance
-------------------------
freq_energy_frac = 0.05   keeps the central 95 % of spectral energy; rarely
                          needs tuning.  Raise to 0.10 to be more aggressive.
dir_energy_frac  = 0.10   retains all directional lobes above 10 % of the local
                          peak at each frequency.  Raise to 0.20 to prune harder;
                          lower to 0.05 to keep more of the tails.
pad_freq / pad_theta = 1  one-bin halo avoids cutting off spectral shoulders.
                          Use 0 for maximum pruning, 2 for conservative padding.
"""

import numpy as np


def build_active_grid_from_wavespec(
    Ei_f_theta,
    freq_energy_frac=0.05,
    dir_energy_frac=0.10,
    pad_freq=1,
    pad_theta=1,
):
    """
    Build an active (freq, theta) mask from the interpolated directional spectrum.

    Pruning is two-pass:
    1. Drop frequency bins whose marginal energy S(f) = sum_theta E(f, theta)
       is below freq_energy_frac * max_f S(f).
    2. For each retained frequency, drop direction bins where
       E(f, theta) < dir_energy_frac * max_theta E(f, theta).
       Multiple directional lobes are preserved naturally since no assumption
       about a single dominant direction is made.
    3. Expand the retained set by +/- pad_freq bins in frequency (no wrap) and
       +/- pad_theta bins in direction (circular wrap).

    Parameters
    ----------
    Ei_f_theta : array-like, shape (n_f, n_theta)
        Interpolated directional spectrum on the solution-space grid.
        Pass Ei.T from leastSquaresWavePropagation where Ei is (n_theta, n_f).
    freq_energy_frac : float
        Fraction of peak S(f) below which a frequency bin is dropped (default 0.05).
    dir_energy_frac : float
        Fraction of the local directional peak below which a direction bin is
        dropped at a given frequency (default 0.10).
    pad_freq : int
        Padding radius in frequency bins (default 1, no wrap at boundaries).
    pad_theta : int
        Padding radius in direction bins (default 1, circular).

    Returns
    -------
    active_mask : ndarray bool, shape (n_f, n_theta)
        True where the component is retained.
    good_indices : ndarray int, shape (n_active,)
        Flat F-order indices of retained components.
        These are compatible with the 'good' array in leastSquaresWavePropagation.
    grid_shape : (int, int)
        (n_f, n_theta) — the full base grid shape.
    """
    Ei = np.asarray(Ei_f_theta, dtype=float)
    if Ei.ndim != 2:
        raise ValueError(f'Ei_f_theta must be 2-D, got shape {Ei.shape}')
    n_f, n_theta = Ei.shape

    # Degenerate: no energy → keep everything
    total = float(np.nansum(Ei))
    if total == 0.0 or not np.isfinite(total):
        active_mask = np.ones((n_f, n_theta), dtype=bool)
        good_indices = np.flatnonzero(active_mask.flatten(order='F')).astype(int)
        return active_mask, good_indices, (n_f, n_theta)

    # 1. Marginal energy over directions → frequency mask
    S_f = np.nansum(Ei, axis=1)          # (n_f,)
    peak_S = float(np.max(S_f))
    if peak_S <= 0.0:
        freq_mask = np.ones(n_f, dtype=bool)
    else:
        freq_mask = S_f >= freq_energy_frac * peak_S

    # 2. Per-frequency direction mask
    #    Zero-out suppressed frequencies first so padding can't re-activate them
    #    from a direction perspective.
    active_mask = np.zeros((n_f, n_theta), dtype=bool)
    for fi in range(n_f):
        if not freq_mask[fi]:
            continue
        row = Ei[fi, :]
        peak_dir = float(np.max(row)) if row.size > 0 else 0.0
        if peak_dir <= 0.0:
            continue
        active_mask[fi, :] = row >= dir_energy_frac * peak_dir

    # 3. Padding
    if pad_freq > 0:
        padded = active_mask.copy()
        for shift in range(1, int(pad_freq) + 1):
            # lower-index neighbor (no wrap — clamp at boundary)
            padded[shift:, :]  |= active_mask[:-shift, :]
            padded[:-shift, :] |= active_mask[shift:, :]
        active_mask = padded

    if pad_theta > 0:
        padded = active_mask.copy()
        for shift in range(1, int(pad_theta) + 1):
            # circular in direction (theta is periodic)
            padded |= np.roll(active_mask, shift,  axis=1)
            padded |= np.roll(active_mask, -shift, axis=1)
        active_mask = padded

    good_indices = np.flatnonzero(active_mask.flatten(order='F')).astype(int)
    return active_mask, good_indices, (n_f, n_theta)


def expand_active_solution(x_reduced, good_indices, grid_shape):
    """
    Expand a pruned coefficient vector back to the full (n_f, n_theta) grid.

    The solver operates on x_reduced = [A_cos (n_active,), A_sin (n_active,)].
    This function places those values back into a full grid of zeros so the
    result can be plotted or compared with the input spectrum.

    Parameters
    ----------
    x_reduced : array-like, shape (2 * n_active,)
        Cosine and sine coefficients for the active components.
    good_indices : array-like of int, shape (n_active,)
        Flat F-order indices of the active components (from build_active_grid_from_wavespec).
    grid_shape : (int, int)
        Full base grid shape (n_f, n_theta).

    Returns
    -------
    cos_grid : ndarray, shape (n_f, n_theta)
    sin_grid : ndarray, shape (n_f, n_theta)
    amp_grid : ndarray, shape (n_f, n_theta)
        Instantaneous amplitude sqrt(A_cos^2 + A_sin^2) at each grid point.
    """
    x_reduced = np.asarray(x_reduced, dtype=float).reshape(-1)
    good_indices = np.asarray(good_indices, dtype=int).reshape(-1)
    n_f, n_theta = int(grid_shape[0]), int(grid_shape[1])
    n_active = good_indices.size

    if x_reduced.size != 2 * n_active:
        raise ValueError(
            f'x_reduced length {x_reduced.size} does not match '
            f'2 * n_active = {2 * n_active}'
        )

    flat_cos = np.zeros(n_f * n_theta, dtype=float)
    flat_sin = np.zeros(n_f * n_theta, dtype=float)
    flat_cos[good_indices] = x_reduced[:n_active]
    flat_sin[good_indices] = x_reduced[n_active:]

    cos_grid = flat_cos.reshape((n_f, n_theta), order='F')
    sin_grid = flat_sin.reshape((n_f, n_theta), order='F')
    amp_grid = np.sqrt(cos_grid ** 2 + sin_grid ** 2)
    return cos_grid, sin_grid, amp_grid
