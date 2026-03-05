"""
Utilities to inspect Gram-matrix structure for the LSQ design matrix.

This module is intended for research / profiling. It provides ways to quantify
and visualize how close the normalized Gram matrix is to diagonal without
necessarily forming the full dense matrix.

Definitions
-----------
Given a design matrix P (m x n), define the (unnormalized) Gram matrix:

    G = P^T P

To compare columns independent of scaling, define D = diag(G) and the
normalized correlation matrix:

    C = D^{-1/2} G D^{-1/2}

Then diag(C) = 1 (for nonzero columns). ``C`` being close to diagonal indicates
near-orthogonality of columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GramDiag:
    n_cols: int
    n_rows: int
    offdiag_fro_norm_est: float
    offdiag_rms_est: float
    max_abs_offdiag_sample: float
    n_pairs_sampled: int


def as_float_array(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def normalize_columns(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return column-normalized matrix and column norms.

    Columns with zero norm are left unchanged and get norm=1.0.
    """
    P = as_float_array(P)
    col_norm2 = np.sum(P * P, axis=0)
    col_norm = np.sqrt(col_norm2)
    col_norm[col_norm == 0.0] = 1.0
    return P / col_norm[None, :], col_norm


def estimate_offdiag_frobenius_norm(
    Pn: np.ndarray,
    *,
    n_probes: int = 20,
    seed: int = 0,
) -> float:
    """
    Estimate ||C - I||_F where C = Pn^T Pn using Hutchinson probes.

    This avoids forming the dense n x n matrix. Each probe does two
    matrix-vector multiplies, so runtime scales with matrix size.
    """
    Pn = as_float_array(Pn)
    n = int(Pn.shape[1])
    if n <= 0:
        return 0.0

    rng = np.random.default_rng(int(seed))
    acc = 0.0
    for probe_i in range(int(n_probes)):
        v = rng.choice([-1.0, 1.0], size=n)
        y = Pn @ v
        z = Pn.T @ y
        w = z - v
        acc += float(w @ w)

    mean = acc / float(max(int(n_probes), 1))
    return float(np.sqrt(max(mean, 0.0)))


def sample_offdiag_correlations(
    Pn: np.ndarray,
    *,
    n_pairs: int = 50000,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample off-diagonal entries of C = Pn^T Pn (absolute value).

    Uses random column pairs and computes dot products directly.
    """
    Pn = as_float_array(Pn)
    n = int(Pn.shape[1])
    if n < 2:
        return np.zeros((0,), dtype=float)

    rng = np.random.default_rng(int(seed))
    i = rng.integers(0, n, size=int(n_pairs), dtype=np.int64)
    j = rng.integers(0, n, size=int(n_pairs), dtype=np.int64)

    # Avoid diagonal pairs.
    same = i == j
    if np.any(same):
        j[same] = (j[same] + 1) % n

    vals = np.einsum('ij,ij->j', Pn[:, i], Pn[:, j], optimize=True)
    return np.abs(vals.astype(float, copy=False))


def corr_subset(Pn: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute exact correlation submatrix for a subset of columns."""
    Pn = as_float_array(Pn)
    idx = np.asarray(idx, dtype=np.int64).reshape((-1,))
    return (Pn[:, idx].T @ Pn[:, idx]).astype(float, copy=False)


def write_plots(
    *,
    C_sub: np.ndarray,
    offdiag_abs_samples: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> None:
    """Write a heatmap + histogram using matplotlib (optional dependency)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    eps = 1e-12
    A = np.log10(np.abs(C_sub) + eps)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(A, origin='lower', aspect='auto', interpolation='nearest')
    ax.set_title('log10(|C_sub|)   (C = normalized Gram)')
    ax.set_xlabel('column index (subset)')
    ax.set_ylabel('column index (subset)')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f'{prefix}_corr_heatmap.png', dpi=150)
    plt.close(fig)

    if offdiag_abs_samples.size:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(offdiag_abs_samples, bins=80, range=(0.0, 1.0))
        ax.set_title('|off-diagonal C_ij| (random pairs)')
        ax.set_xlabel('|C_ij|')
        ax.set_ylabel('count')
        fig.tight_layout()
        fig.savefig(out_dir / f'{prefix}_offdiag_hist.png', dpi=150)
        plt.close(fig)


def gram_diagnostics(
    P: np.ndarray,
    *,
    subset_size: int = 256,
    n_probes: int = 20,
    n_pairs: int = 50000,
    seed: int = 0,
    out_dir: Optional[str] = None,
    prefix: str = 'lsq',
) -> GramDiag:
    """
    Compute summary diagnostics and optionally write plots.

    Parameters
    ----------
    P : ndarray
        Design matrix (m x n).
    subset_size : int, optional
        Number of columns to visualize exactly in a heatmap.
    n_probes : int, optional
        Hutchinson probes for estimating ||C - I||_F.
    n_pairs : int, optional
        Number of random off-diagonal pairs to sample for a histogram.
    seed : int, optional
        RNG seed for repeatability.
    out_dir : str or None, optional
        If provided, writes PNG plots into this directory.
    prefix : str, optional
        Prefix for output filenames.

    Returns
    -------
    GramDiag
        Summary diagnostics for normalized Gram-matrix structure.

    """
    P = as_float_array(P)
    m, n = int(P.shape[0]), int(P.shape[1])
    if n == 0 or m == 0:
        return GramDiag(
            n_cols=n,
            n_rows=m,
            offdiag_fro_norm_est=0.0,
            offdiag_rms_est=0.0,
            max_abs_offdiag_sample=0.0,
            n_pairs_sampled=0,
        )

    Pn, col_norm_unused = normalize_columns(P)

    offdiag_fro = estimate_offdiag_frobenius_norm(Pn, n_probes=int(n_probes), seed=int(seed))

    # Convert Frobenius norm to an RMS per off-diagonal entry.
    n_off = float(n * (n - 1))
    offdiag_rms = offdiag_fro / float(np.sqrt(max(n_off, 1.0)))

    samples = sample_offdiag_correlations(Pn, n_pairs=int(n_pairs), seed=int(seed))
    max_sample = float(np.max(samples)) if samples.size else 0.0

    k = int(min(max(int(subset_size), 2), n))
    # Visualize a deterministic subset (first k columns) to make runs comparable.
    idx = np.arange(k, dtype=np.int64)
    C_sub = corr_subset(Pn, idx)

    if out_dir is not None:
        write_plots(
            C_sub=C_sub,
            offdiag_abs_samples=samples,
            out_dir=Path(str(out_dir)),
            prefix=str(prefix),
        )

    return GramDiag(
        n_cols=n,
        n_rows=m,
        offdiag_fro_norm_est=float(offdiag_fro),
        offdiag_rms_est=float(offdiag_rms),
        max_abs_offdiag_sample=float(max_sample),
        n_pairs_sampled=int(samples.size),
    )
