# TheNextWave (Python Port)
Deterministic ocean wave prediction from sparse buoy data. This package reconstructs
phase-resolved ocean waves over short time-space scales using measurements from sparse SWIFT arrays.

## Frames and conventions

This package uses a local Cartesian frame for both buoy positions and horizontal velocity components.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s).
- **Wave directions**: compass degrees True (0°=North, 90°=East). Bulk directions such as $D_p$ are direction waves come **FROM**. Propagation direction is **TO** = FROM + 180 (wrapped to 0–360).
- **Simulation (`mbari_wec_gz`)**: the incident-wave latent data is published in world coordinates and already uses $u/v$ = East/North. Do not apply SWIFT SBG mounting corrections to sim.
- **Heave sign (`flip_z_sign`)**: set `true` for real SWIFT SBG streams (upside-down mount correction), and `false` for sim sources that already publish up-positive $z$.
- **Rotation (`rotation_deg`)**:
   - In the Python port, `rotation_deg` is applied in the lat/lon → x/y projection (clockwise-positive).
   - If `rotation_deg != 0`, the resulting x/y axes are *not* East/North anymore.
   - The current pipeline does **not** rotate $u/v$ when `rotation_deg` is nonzero, so the safest default is `rotation_deg: 0.0` unless you rotate $u/v$ consistently upstream.
   - The MATLAB SWIFTcodes `GenericCoordinateTransform` uses a different convention, so you may see the offline MATLAB example use `rotation=180` while the Python example uses `rotation=0` for the same physical layout.

## ROS 2
Prerequisite: install ROS 2 on your machine.  
(To skip ROS 2 install and just run the example, use `Quick Start` below.)  
  
This folder is a ROS 2 package. You may navigate to the parent directory and run

``` bash
colcon build --packages-select the_next_wave
```

to build it, or move it to your own workspace.

## Example
Install `pooch` to download example data:

``` bash
sudo apt update && sudo apt install python3-pooch
```
  
An example is included that processes an SBD burst from each of four SWIFT buoys and forecasts the
wave height at a certain location for a window of time.  
  
After building the ROS 2 package as mentioned above (from the parent directory or your own
workspace), you can then source the install folder that was created by `colcon` and run the example
using

``` bash
source install/setup.bash
ros2 run the_next_wave the_next_wave_example 
```

To generate a video instead of plotting, add the `--movie out.mp4`
argument.

## Production (SBG Ethernet Bridge)

For field/production notes on wiring SWIFT SBG streams over a Digi Ethernet bridge into the ROS 2 SBG TCP bridge server, see:

- [docs/ethernet_bridge_production_setup.md](docs/ethernet_bridge_production_setup.md)

#### Quick Start (Skip ROS 2 Install)
Prerequisite: [Install](https://docs.astral.sh/uv/getting-started/installation/) `uv`.  
  
To run the example without ROS 2 or `colcon`, use `uv` to sandbox dependencies.
In the current directory, run:

``` bash
uv sync
uv run python -m the_next_wave.example
```

To generate a video instead of plotting, add the `--movie out.mp4`
argument.

## Notes

- The real-time ROS 2 pipeline (windowing → wavespec → least-squares prediction) is implemented in this repo.
- Known limitation: if `rotation_deg != 0`, rotate $u/v$ consistently upstream; this pipeline does not rotate $u/v$ automatically.

