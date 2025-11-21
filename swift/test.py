#!/usr/bin/python3

import numpy as np
import xarray as xr

from .swift import SWIFTArray, SWIFTData
from .run_LS_prediction_SWIFTS import run_LS_prediction_SWIFTS
from .align_swift_array import align_swift_bursts_swiftarray


if __name__=='__main__':
    #swift22_nc = xr.load_dataset('test_data/SWIFT22_DIGIFLOAT_fall2022_all_reprocessedSBG.nc')
    #swift23_1_nc = xr.load_dataset('test_data/SWIFT23_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    #swift23_2_nc = xr.load_dataset('test_data/SWIFT23_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    #swift23_nc = xr.concat([swift23_1_nc, swift23_2_nc], dim='time')
    #swift24_1_nc = xr.load_dataset('test_data/SWIFT24_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    #swift24_2_nc = xr.load_dataset('test_data/SWIFT24_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    #swift24_nc = xr.concat([swift24_1_nc, swift24_2_nc], dim='time')
    #swift25_1_nc = xr.load_dataset('test_data/SWIFT25_DIGIFLOAT_fall2022_part1_reprocessedSBG.nc')
    #swift25_2_nc = xr.load_dataset('test_data/SWIFT25_DIGIFLOAT_fall2022_part2_reprocessedSBG.nc')
    #swift25_nc = xr.concat([swift25_1_nc, swift25_2_nc], dim='time')

    full_array = SWIFTArray()
    full_array.swift22 = SWIFTData.from_mat('test_data/mat/SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat', 'swift22')
    full_array.swift23 = SWIFTData.from_mat('test_data/mat/SWIFT23_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat', 'swift23')
    full_array.swift24 = SWIFTData.from_mat('test_data/mat/SWIFT24_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat', 'swift24')
    full_array.swift25 = SWIFTData.from_mat('test_data/mat/SWIFT25_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG_displacements.mat', 'swift25')

    aligned_array = align_swift_bursts_swiftarray(full_array, 'swift25', 0.25/24.)
    for burst_idx, array in enumerate(aligned_array.bursts()):
        print(f'Processing Burst {burst_idx}...')
        _, prediction = run_LS_prediction_SWIFTS(array, True)
        nc_path = f'output_data/burst_prediction_{burst_idx}.nc'
        print(f'...saving prediction as {nc_path}')
        prediction.to_netcdf(nc_path)
        #print(prediction)
