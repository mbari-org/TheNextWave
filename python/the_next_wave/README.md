# TheNextWave (Python Port)
Deterministic ocean wave prediction from sparse buoy data. This methodology and codes generate
phase-resolved reconstructions of ocean waves over short time-space scales using measurements of
wave motion collected by sparse arrays of Surface Wave Instrument Floats w/ Tracking (SWIFTs). 

## Frames and conventions

This package uses a local Cartesian frame for both buoy positions and horizontal velocity components.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s).
- **Wave directions**: compass degrees True (0°=North, 90°=East). Bulk directions such as $D_p$ are the direction waves are coming **FROM**. Propagation direction is **TO** = FROM + 180 (wrapped to 0–360).
- **Simulation (`mbari_wec_gz`)**: the incident-wave latent data is published in world coordinates and already uses $u/v$ = East/North. Do not apply SWIFT SBG mounting corrections to sim.
- **Heave sign (`flip_z_sign`)**: set `true` for real SWIFT SBG streams (upside-down mount correction), and `false` for sim sources that already publish up-positive $z$.
- **Rotation (`rotation_deg`)**:
   - In the Python port, `rotation_deg` is applied in the lat/lon → x/y projection (clockwise-positive).
   - If `rotation_deg != 0`, the resulting x/y axes are *not* East/North anymore.
   - The current pipeline does **not** rotate $u/v$ when `rotation_deg` is nonzero, so the safest default is `rotation_deg: 0.0` unless you rotate $u/v$ consistently upstream.
   - The MATLAB SWIFTcodes `GenericCoordinateTransform` uses a different convention, so you may see the offline MATLAB example use `rotation=180` while the Python example uses `rotation=0` for the same physical layout.

## ROS 2
Prerequisite: Follow installation steps to install ROS 2 on your machine.  
(If you would like to skip installing ROS 2 for now, and just run an example, you may skip to
the `Quick Start` section below)  
  
This folder is a ROS 2 package. You may navigate to the parent directory and run

``` bash
colcon build --packages-select the_next_wave
```

to build it, or move it to your own workspace.

## Example
Prerequisite: Install `pooch` Python3 package to download example data using

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

If you would like to generate a video instead of plotting, you can add the `--movie out.mp4`
argument.

## Production (SBG Ethernet Bridge)

For field/production notes on wiring SWIFT SBG streams over a Digi Ethernet bridge into the ROS 2 SBG TCP bridge server, see:

- [docs/ethernet_bridge_production_setup.md](docs/ethernet_bridge_production_setup.md)

#### Quick Start (Skip ROS 2 Install)
Prerequisite: [Install](https://docs.astral.sh/uv/getting-started/installation/) `uv` Python3 package  
  
To quickly run the example without installing ROS 2 or building with `colcon`, you may instead use
the `uv` Python3 package to sandbox dependencies. In the current directory, use `uv` to sync
required packages to a sandbox and run the example following:

``` bash
uv sync
uv run python -m the_next_wave.example
```

If you would like to generate a video instead of plotting, you can add the `--movie out.mp4`
argument.

## Notes / TODO

- The real-time ROS 2 pipeline (windowing → wavespec → least-squares prediction) is implemented in this repo.
- Known limitation: if you use `rotation_deg != 0`, you must rotate $u/v$ consistently upstream; the pipeline does not currently rotate $u/v$ automatically.

