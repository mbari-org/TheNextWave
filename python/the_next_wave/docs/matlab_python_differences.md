# MATLAB → Python port: differences

Semantic comparison of the Python port against the MATLAB originals. A textual
diff does not line up; this is a hand audit of behaviour.

Reference pairs:

| MATLAB | Python |
| --- | --- |
| `leastSquaresWavePropagation.m` | `the_next_wave/leastSquaresWavePropagation.py` |
| `SWIFTdirectionalspectra.m` | `the_next_wave/SWIFTdirectionalspectra.py` |
| `MEM_directionalestimator.m` | `the_next_wave/mem_directionalestimator.py` |
| `SBGWaves.m` | `the_next_wave/sbg_waves.py` |

## 1. Differences that change numbers

**Solver.** MATLAB uses `lsqlin` with `trust-region-reflective`; Python uses
L-BFGS-B (SciPy, or jaxopt on GPU). Different algorithm, different stopping
rule. Both solve the same box-constrained problem with the same bounds, so
results should agree to solver tolerance, not to machine precision.

**Warm start.** MATLAB passes `[]` as `x0` — every window is a cold start.
Python seeds each solve with the previous window's `A`
(`the_next_wave.py`, `self.A0 = params.A`). Python solves are therefore
history-dependent: a single window is not reproducible in isolation without
supplying the same `A0`.

**Bin width on the solve grid.** MATLAB `diff([0 f2(1,:)./(2*pi)])` (backward
difference, first bin = `f[0]`); Python `np.gradient(f2_hz)` (central
difference). For a typical 40-point log-spaced grid the lowest bin differs by
~21× (0.0400 vs 0.0019 Hz) and the rest by ≤2.4%. This feeds `amps`, so the
amplitude bound on the lowest-frequency component is ~4.6× looser in MATLAB.
It also feeds the `params.Etheta` normalisation on the way out.

**Energy renormalisation denominator.** Both rescale `Ei` so the interpolated
spectrum carries the source spectrum's `m0`. MATLAB integrates with `trapz` in
both dimensions; Python uses `trapz` for the numerator but a rectangular sum
for the denominator (`np.sum(Ei * df2_hz) * dtheta_mode_deg`).

**Precision.** MATLAB is double throughout. Python builds `phi`/`P1`/`P2` in
float32 and casts to float64 for the solve.

## 2. Python-only additions

**MEM moment cap** (`SWIFTdirectionalspectra.py`, `mem_moment_cap`). Clamps
`|c1|` and `|c2|` below a cap before the MEM call. Not in MATLAB, off by
default. Required for unidirectional seas (`spreading_deg: 0`), where
`|c1| = 1` exactly and MEM's `1 - |c1|²` denominator is zero.

**Interpolation cache.** The spectrum → solution-space `griddata` result is
cached on the wavespec object and reused while the spectrum is unchanged.
Optimisation only; no effect on results.

## 3. Same behaviour, restructured

**Zero-energy pruning.** MATLAB builds `P1`/`P2` at full width, then deletes
columns where `amps == 0` (`leastSquaresWavePropagation.m:131-134`). Python
prunes `kx/ky/omega/amps_base` *before* building `phi` — avoiding trig on
columns about to be dropped — then keeps a second pass over `P1`/`P2` for any
remaining exact zeros. Same columns selected. Consequence: Python must not
re-index `kx/ky/omega` with `good` at the end, because it already did.

## 4. Quirks preserved deliberately — do not "fix"

These look like bugs and are faithful to MATLAB. Changing them breaks parity.

- **`1.4142`, not `sqrt(2)`.** Bounds are `±amps/1.4142` in both. ~1e-5
  relative difference from the exact value.
- **Two gravities.** `k = omega²/9.81`, but the interpolation grid uses
  `sqrt(k * 9.8)`. Present in MATLAB; carried over verbatim. The propagator
  itself is self-consistent at 9.81; only the interpolation frequencies are
  offset (~0.05%).
- **`gradient` for the source spectrum's `df`.** Both use a central
  difference here — the divergence in §1 is only on the *solve* grid.
- **Direction frame flips twice.** Input `theta` is compass degrees FROM; the
  solver adds 180° into the TO frame; `params.theta` adds 180° back.
- **Column-major ordering.** Python uses `order='F'` throughout to match
  MATLAB's flattening, including the component index
  `base = freq_idx + 40 * dir_idx`.
- **`unique(theta, 'last')`** semantics on duplicate directions.
