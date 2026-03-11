from glob import glob
import os

from setuptools import setup

package_name = 'the_next_wave'

setup(
    name=package_name,
    version='0.1.0',
    packages=['the_next_wave'],
    include_package_data=True,
    package_data={
        'the_next_wave': [
            '*.so',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    zip_safe=False,
    description=(
        'Deterministic ocean wave prediction from sparse buoy data. This '
        'methodology and codes generate phase-resolved reconstructions of ocean '
        'waves over short time-space scales using measurements of wave motion '
        'collected by sparse arrays of Surface Wave Instrument Floats w/ '
        'Tracking (SWIFTs)'
    ),
    tests_require=['pytest'],
    # NOTE: Keep console scripts / executables out of setup.py.
    # They are defined in pyproject.toml (works with uv and our current build flow).
    # entry_points={
    #     'console_scripts': [
    #         'the_next_wave_example = the_next_wave.example:main',
    #         'the_next_wave_node = the_next_wave.the_next_wave_node:main',
    #         'the_next_wave_sim_node = the_next_wave.the_next_wave_node:main',
    #         'the_next_wave_plotter_node = the_next_wave.the_next_wave_plotter_node:main',
    #         'the_next_wave_sbg_tcp_replay_node = the_next_wave.sbg_tcp_replay_node:main',
    #     ],
    # },
)
