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

If you are reading this from the parent [python/README.md](../README.md), the
commands below assume you have already changed into this `the_next_wave`
directory.

### Python dependencies for ROS 2 runs

When running the ROS 2 nodes with `ros2 run` / `ros2 launch`, do not rely on the
`uv` virtual environment. In practice, the ROS 2 environment typically uses the
system Python plus your user site packages, so install this package with `pip`
into `~/.local` from this directory.

If you want the JAX backend available:

``` bash
python3 -m pip install --user '.[jax]' --break-system-packages
```

Notes:

- `--break-system-packages` is required on some newer distro Python setups, but
   this command still installs to your user site (`~/.local`), not into the
   system package manager directories.
- To preview what would happen without installing, add `--dry-run`.
- If you do not need the JAX backend, you can install without extras:

``` bash
python3 -m pip install --user . --break-system-packages
```
  
This folder is a ROS 2 package. You may navigate to the parent directory and run

``` bash
colcon build --packages-select the_next_wave
```

to build it, or move it to your own workspace.

### ROS 2 launch files, modes, and arguments

The main entrypoint is:

``` bash
ros2 launch the_next_wave the_next_wave.launch.py
```

To see the currently exposed launch arguments, run:

``` bash
ros2 launch the_next_wave the_next_wave.launch.py -s
```

#### Main launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `enable_plotter` | `false` | Also start the standalone plotter node. |
| `enable_nextwave_node` | `true` | Start the main predictor node. |
| `enable_mbari_wec` | `true` | Include the MBARI WEC Gazebo simulator launch. |
| `gzsim_headless` | `false` | Run Gazebo without the GUI. |
| `gzsim_verbose` | `true` | Enable verbose Gazebo logging. |
| `enable_sbg_bridge` | `false` | Enable the SBG TCP bridge inside the predictor node. |
| `enable_sbg_tcp_replay` | `false` | Start the 4 SBG TCP replay nodes and enable the bridge. |
| `params_file` | empty | Override the default YAML config file. |

#### Common launch modes

- **Full sim mode (default)**: predictor + MBARI WEC sim. The plotter is **not** implied by defaults; add `enable_plotter:=true` if you want it.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py
   ```

- **Full sim mode with plotter**: predictor + plotter + MBARI WEC sim.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py enable_plotter:=true
   ```

- **Headless sim mode**: same as above, but without the Gazebo GUI.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py gzsim_headless:=true
   ```

- **Bridge-only deployment mode**: use real TCP-fed SBG streams, no Gazebo sim.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py enable_sbg_bridge:=true
   ```

- **Replay mode**: replay the packaged TCP examples and run the predictor.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py enable_sbg_tcp_replay:=true
   ```

- **Replay-only helpers**: start only the replay nodes, without the predictor.

   ``` bash
   ros2 launch the_next_wave the_next_wave.launch.py \
      enable_sbg_tcp_replay:=true \
      enable_nextwave_node:=false
   ```

#### Which config file is used?

By default, the launch file selects:

- [config/config.yaml](config/config.yaml) for normal sim runs
- [config/config_deployment.yaml](config/config_deployment.yaml) when `enable_sbg_bridge:=true`
- [config/config_sbg_tcp_replay.yaml](config/config_sbg_tcp_replay.yaml) when `enable_sbg_tcp_replay:=true`

You can always override that choice explicitly:

``` bash
ros2 launch the_next_wave the_next_wave.launch.py \
   params_file:=/absolute/path/to/my_config.yaml
```

#### How to specify incident-wave overrides in ROS launch mode

For the top-level TheNextWave launch, the simulator overrides are read from
[config/config.yaml](config/config.yaml):

- `wave_dir`: incident-wave direction in compass degrees True, waves coming **FROM**
- `swift_coords.x` / `swift_coords.y`: simulated SWIFT buoy sample locations in local meters; mapped to simulator `inc_wave_height_points`
- `bretschneider`: Bretschneider incident-wave settings; mapped to simulator `inc_wave_spectrum`

Example snippet:

```yaml
/the_next_wave_node:
   ros__parameters:
      wave_dir: 270.0
      swift_coords:
         x: [-125.0, -185.0, -100.0, 45.0]
         y: [-25.0, -115.0, -135.0, -175.0]
      bretschneider:
         Hs: 3.0
         Tp: 14.0
         n_phases: 500
         spreading_deg: 22.0
```

Those values are converted at launch time into the MBARI WEC simulator arguments
`inc_wave_dir`, `inc_wave_height_points`, and `inc_wave_spectrum`.

#### How to specify directional spreading in ROS/Gazebo runs

`spreading_deg` is still a simulator setting, but the top-level TheNextWave launch
now forwards it from `config/config.yaml` when provided through `bretschneider`.
You can also use one of these paths:

1. **Direct simulator launch** with [src/mbari_wec_gz/buoy_gazebo/launch/mbari_wec.launch.py](../../../../src/mbari_wec_gz/buoy_gazebo/launch/mbari_wec.launch.py):

    ``` bash
    ros2 launch buoy_gazebo mbari_wec.launch.py \
       inc_wave_dir:=270 \
       inc_wave_height_points:='-125:-25;-185:-115;-100:-135;45:-175' \
       inc_wave_spectrum:='inc_wave_spectrum_type:Bretschneider;Hs:3.0;Tp:14.0;n_phases:500;spreading_deg:22.0'
    ```

2. **Batch sim YAML** via [example_sim_params.yaml](../../../../example_sim_params.yaml):

    - `wave_dir`
    - `IncidentWaveHeightPoints`
    - `IncidentWaveSpectrumType -> Bretschneider -> Hs/Tp/n_phases/spreading_deg`

Example batch snippet:

```yaml
wave_dir: [270.0]
IncidentWaveHeightPoints:
   - x: [-125.0, -185.0, -100.0, 45.0]
      y: [-25.0, -115.0, -135.0, -175.0]
IncidentWaveSpectrumType:
   - Bretschneider:
         Hs: 3.0
         Tp: 14.0
         n_phases: 500
         spreading_deg: 22.0
```

Notes:

- `spreading_deg` uses the SWIFT / Spotter first-moment spread convention.
- `spreading_deg: 0` disables directional spreading and falls back to a 1D Bretschneider spectrum.
- `n_phases` is the number of random phases used for the Bretschneider realization; it is not the same as the number of spreading sectors.

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

To use a specific packaged dataset folder, add `--example-name <folder-name>`.

To generate a video instead of plotting, add the `--movie out.mp4`
argument.

### `uv run` modes and arguments

The standalone `uv` workflow is mainly for the packaged offline example tools,
not for launching Gazebo.

The most common entrypoint is:

``` bash
uv run python -m the_next_wave.example
```

Equivalent console-script form:

``` bash
uv run the_next_wave_example
```

Useful related entrypoints are:

- `uv run the_next_wave_example`
- `uv run the_next_wave_example_study`
- `uv run the_next_wave_example_loo`

To see the example CLI arguments:

``` bash
uv run python -m the_next_wave.example --help
```

#### Main `uv run` example arguments

| Argument | Meaning |
| --- | --- |
| `--example-name <folder>` | Choose a packaged dataset under `example_data/`. |
| `--movie <path>` | Save an animation as `.mp4` or `.gif`. |
| `--fps <float>` | Movie frame rate. |
| `--dpi <int>` | Movie DPI. |
| `--fig-width <float>` | Figure width in inches. |
| `--fig-height <float>` | Figure height in inches. |
| `--matlab-pred-nc <path>` | Overlay a MATLAB NetCDF prediction file. |
| `--matlab-window-warn-sec <float>` | Warning threshold for MATLAB/Python window mismatch. |
| `--solver-max-iter <int>` | LSQ solver iteration cap. |
| `--solver-backend {auto,scipy,jax}` | Select solver backend. |
| `--wavespec-swift {1,2,3,4}` | Use only one SWIFT for wavespec generation. |
| `--reproj-swift {1,2,3,4}` | Show measured vs reprojected `z/u/v` for one SWIFT. |

Example commands:

``` bash
uv run the_next_wave_example --example-name ExampleData1
uv run the_next_wave_example --example-name ExampleData1 --movie out.mp4
uv run the_next_wave_example --solver-backend jax --solver-max-iter 10
```

#### Config files in `uv run` mode

The offline example script does **not** use the ROS 2 YAML config files in
[config/](config). Its inputs come from the packaged example datasets plus the
CLI arguments above.

So if you want to change:

- simulator `wave_dir`
- simulator `spreading_deg`
- simulated SWIFT locations
- ROS node parameter YAMLs

do that through the ROS/Gazebo launch path described above, not through
`uv run python -m the_next_wave.example`.

## Production (SBG Ethernet Bridge)

For field/production notes on wiring SWIFT SBG streams over a Digi Ethernet bridge into the ROS 2 SBG TCP bridge server, see:

- [docs/ethernet_bridge_production_setup.md](docs/ethernet_bridge_production_setup.md)

#### Quick Start (Skip ROS 2 Install)
Prerequisite: [Install](https://docs.astral.sh/uv/getting-started/installation/) `uv`.  
  
To run the example without ROS 2 or `colcon`, use `uv` to sandbox dependencies.
In this `the_next_wave` directory, run:

``` bash
uv sync
uv run python -m the_next_wave.example
```

If you also want the optional JAX backend in the `uv` environment, install the
extra dependency set:

``` bash
uv sync --extra jax
uv run python -m the_next_wave.example
```

Notes:

- `uv` is recommended for standalone / non-ROS workflows.
- For ROS 2 execution, prefer the `pip install --user ...` workflow above rather
   than the `uv` environment.

To use a specific packaged dataset folder, add `--example-name <folder-name>`.

To generate a video instead of plotting, add the `--movie out.mp4`
argument.

## Notes

- The real-time ROS 2 pipeline (windowing → wavespec → least-squares prediction) is implemented in this repo.
- Known limitation: if `rotation_deg != 0`, rotate $u/v$ consistently upstream; this pipeline does not rotate $u/v$ automatically.

