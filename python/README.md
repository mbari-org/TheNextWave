# TheNextWave (Python Port)
Deterministic ocean wave prediction from sparse buoy data. This package reconstructs
phase-resolved ocean waves over short time-space scales using measurements from sparse SWIFT arrays.

## Frames and conventions

The Python port expects positions and velocity components to share a consistent local Cartesian frame.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s).
- **Wave directions**: compass degrees True (0°=North, 90°=East). Bulk directions such as $D_p$ are direction waves come **FROM**. Propagation direction is **TO** = FROM + 180 (wrapped to 0–360).
- **Simulation (`mbari_wec_gz`)**: incident-wave outputs are already in world ENU with $u/v$ = East/North; do not apply SWIFT-specific sign flips to sim data.
- **Rotation (`rotation_deg`)**: applied to the lat/lon → x/y projection (clockwise-positive). If you set `rotation_deg != 0`, x/y are rotated but u/v are not automatically rotated in the pipeline; keep `rotation_deg: 0.0` unless you rotate u/v consistently.

## ROS 2
Prerequisite: install ROS 2 on your machine.  
(To skip ROS 2 install and just run the example, use `Quick Start` below.)
  
`the_next_wave` folder located in this directory is a ROS 2 package. You may use

``` bash
colcon build --packages-select the_next_wave
```

to build it in this directory, or move it to your own workspace.

## Example
Install `pooch` to download example data:

``` bash
sudo apt update && sudo apt install python3-pooch
```

An example is included in `the_next_wave` that processes an SBD burst from each of four SWIFT buoys
and forecasts the wave height at a certain location for a window of time.  
  
After building the ROS 2 package as mentioned above, you can then source the install folder that
was created by `colcon` and run the example using

``` bash
source install/setup.bash
ros2 run the_next_wave the_next_wave_example 
```

To generate a video instead of plotting, add the `--movie out.mp4`
argument.

#### Quick Start (Skip ROS 2 Install)
Prerequisite: [Install](https://docs.astral.sh/uv/getting-started/installation/) `uv`.

To run the example without ROS 2 or `colcon`, use `uv` to sandbox dependencies.
Navigate to `the_next_wave`, then run:

``` bash
cd the_next_wave
uv sync
uv run python -m the_next_wave.example
```

To generate a video instead of plotting, add the `--movie out.mp4`
argument.


## Notes

- The real-time ROS 2 pipeline (windowing → wavespec → least-squares prediction) is implemented in this repo.
- Known limitation: if `rotation_deg != 0`, rotate $u/v$ consistently upstream; this pipeline does not rotate $u/v$ automatically.

