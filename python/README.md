# TheNextWave Python workspace

The canonical Python package documentation now lives in:

- [the_next_wave/README.md](the_next_wave/README.md)

That README covers:

- ROS 2 usage
- `pip --user` installation for ROS 2 runs
- `uv` workflows for standalone use
- optional JAX installation
- examples and frame conventions

## Quick pointers

- The actual Python / ROS 2 package is the [the_next_wave](the_next_wave) folder.
- If you are starting from this directory and want the package-level instructions,
  first `cd the_next_wave` and then follow [the_next_wave/README.md](the_next_wave/README.md).
- To build the ROS 2 package from this directory, you can still run:

``` bash
colcon build --packages-select the_next_wave
```

