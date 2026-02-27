# TheNextWave (Python Port)
Deterministic ocean wave prediction from sparse buoy data. This methodology and codes generate
phase-resolved reconstructions of ocean waves over short time-space scales using measurements of
wave motion collected by sparse arrays of Surface Wave Instrument Floats w/ Tracking (SWIFTs). 

## Frames and conventions

The Python port expects positions and velocity components to share a consistent local Cartesian frame.

- **Local horizontal frame**: $x$ is East (meters), $y$ is North (meters). $z$ is up-positive (meters).
- **Horizontal velocities**: $u$ is East (m/s), $v$ is North (m/s).
- **Simulation (`mbari_wec_gz`)**: incident-wave outputs are already in world ENU with $u/v$ = East/North; do not apply SWIFT-specific sign flips to sim data.
- **Rotation (`rotation_deg`)**: applied to the lat/lon → x/y projection (clockwise-positive). If you set `rotation_deg != 0`, x/y are rotated but u/v are not automatically rotated in the pipeline; keep `rotation_deg: 0.0` unless you rotate u/v consistently.

## ROS 2
Prerequisite: Follow installation steps to install ROS 2 on your machine.  
(If you would like to skip installing ROS 2 for now, and just run an example, you may skip to
the `Quick Start` section below)
  
`the_next_wave` folder located in this directory is a ROS 2 package. You may use

``` bash
colcon build --packages-select the_next_wave
```

to build it in this directory, or move it to your own workspace.

## Example
Prerequisite: Install `pooch` Python3 package to download example data using

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

If you would like to generate a video instead of plotting, you can add the `--movie out.mp4`
argument.

#### Quick Start (Skip ROS 2 Install)
Prerequisite: [Install](https://docs.astral.sh/uv/getting-started/installation/) `uv` Python3 package

To quickly run the example without installing ROS 2 or building with `colcon`, you may instead use
the `uv` Python3 package to sandbox dependencies. Navigate to `the_next_wave` package, and use `uv`
to sync required packages to a sandbox and run the example following:

``` bash
cd the_next_wave
uv sync
uv run python -m the_next_wave.example
```

If you would like to generate a video instead of plotting, you can add the `--movie out.mp4`
argument.


## TODO

Develop real-time prediction processing pipeline as ROS 2 node:
1. a. portAndDecodeFromEthernetBridge.py --> sbgMessageParse.py --> raw SBG data
   b. read the sim SBG data (u,v,heave) from mbari_wec sim ROS 2 messages
2. collect a window of raw SBG data
3. pass raw SBG window to reprocess_SBG.py (still needs to be ported from matlab) --> SBGwaves.py (needs port from matlab) --> wavespectra --> SWIFTdirectionalspectra.py --> bulk wave params
4. pass raw SBG window and bulk wave params to leastSquaresWavePropagation.py and associated preprocessing for a prediction window.

