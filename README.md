# TheNextWave
Deterministic ocean wave prediction from sparse buoy data.  This methodology and codes generate phase-resolved reconstructions of ocean waves over short time-space scales using measurements of wave motion collected by sparse arrays of Surface Wave Instrument Floats w/ Tracking (SWIFTs). 

## Frames and conventions

TheNextWave assumes a consistent local Cartesian frame for both sensor positions and horizontal velocity components.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s). These must be consistent with $x/y$.
- **Rotation (`rotation_deg`)**: The Python port rotates *positions* during the lat/lon → x/y projection (clockwise-positive). If `rotation_deg != 0`, then x/y are no longer East/North; you must rotate (u,v) consistently or keep `rotation_deg: 0.0`.
- **Heave sign**: Real SWIFT SBG heave may be inverted due to sensor mounting (handled in the Python port via a configurable `flip_z_sign`). Simulation data should typically not be flipped.
