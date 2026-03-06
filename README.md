# TheNextWave
Deterministic ocean wave prediction from sparse buoy data. The code reconstructs phase-resolved ocean waves over short time-space scales using measurements from sparse Surface Wave Instrument Floats with Tracking (SWIFT) arrays.

## Frames and conventions

TheNextWave assumes a consistent local Cartesian frame for both sensor positions and horizontal velocity components.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s). These must be consistent with $x/y$.
- **Wave directions**: reported as compass degrees True (0°=North, 90°=East). Bulk directions such as $D_p$ are the direction waves come **FROM**. Propagation direction (**TO**) is $D_{to} = D_{from} + 180$ (wrapped to 0–360).
- **Rotation (`rotation_deg`)**: the Python port rotates positions during lat/lon → x/y projection (clockwise-positive). If `rotation_deg != 0`, x/y are no longer East/North; rotate (u,v) the same way or keep `rotation_deg: 0.0`.
- **Heave sign**: real SWIFT SBG heave may be inverted due to sensor mounting (handled by configurable `flip_z_sign`). Simulation data is typically already correct and should not be flipped.
