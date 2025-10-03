
import numpy as np
from pyproj import Transformer


def sbg_positions(SWIFT):

    #print(f'{SWIFT.lat.shape=}')
    #print(f'{SWIFT.lon.shape=}')
    #print(f'{SWIFT.lat[30]=}')
    #print(f'{SWIFT.lon[30]=}')
    medlat = np.nanmedian(SWIFT.lat)
    medlon = np.nanmedian(SWIFT.lon)
    #print(f'{medlat=}')
    #print(f'{medlon=}')

    zone = np.floor((medlon + 180.) / 6.) + 1.
    zone = zone.astype(int)
    #print(f'{zone=}')
    SWIFT.utmzone = zone

    epsg = 32600 + zone if medlat >= 0.0 else 32700 + zone
    #print(f'{epsg=}')

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    SWIFT.sbg_x, SWIFT.sbg_y = transformer.transform(SWIFT.lon, SWIFT.lat)  # (lon, lat) in that order

    return SWIFT
