#!/usr/bin/python3

import numpy as np
import xarray as xr

from .swift import SWIFTArray, SWIFTData
from .run_LS_prediction_SWIFTS import run_LS_prediction_SWIFTS
from .sbg_positions import sbg_positions


if __name__=='__main__':
    swift22_nc = xr.load_dataset('test_data/SWIFT22_DIGIFLOAT_fall2022_all_reprocessedSBG.nc')
    swift23_1_nc = xr.load_dataset('test_data/SWIFT23_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    swift23_2_nc = xr.load_dataset('test_data/SWIFT23_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    swift23_nc = xr.concat([swift23_1_nc, swift23_2_nc], dim='time')
    swift24_1_nc = xr.load_dataset('test_data/SWIFT24_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    swift24_2_nc = xr.load_dataset('test_data/SWIFT24_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    swift24_nc = xr.concat([swift24_1_nc, swift24_2_nc], dim='time')
    swift25_1_nc = xr.load_dataset('test_data/SWIFT25_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    swift25_2_nc = xr.load_dataset('test_data/SWIFT25_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    swift25_nc = xr.concat([swift25_1_nc, swift25_2_nc], dim='time')

    array = SWIFTArray()
    array.swift22 = SWIFTData.from_dataset(swift22_nc)
    sbg_positions(array.swift22)
    array.swift23 = SWIFTData.from_dataset(swift23_nc)
    sbg_positions(array.swift23)
    array.swift24 = SWIFTData.from_dataset(swift24_nc)
    sbg_positions(array.swift24)
    array.swift25 = SWIFTData.from_dataset(swift25_nc)
    sbg_positions(array.swift25)

    array, prediction = run_LS_prediction_SWIFTS(array, False)
