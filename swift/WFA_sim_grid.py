import numpy as np
import utm


def WFA_sim_grid():
    # -> [x,y,lon,lat,x0,y0,lon0,lat0]:
    lat0 = 41. + (41. + 11.704 / 60.) / 60.
    lon0 = -(9. + (3. + 0.777 / 60.) / 60.)

    x0, y0, z_num, z_letter = utm.from_latlon(lat0, lon0)

    a = np.arange(3, 51, 3)
    cosb = np.cos(np.radians(np.arange(0, 361, 5)))
    sinb = np.sin(np.radians(np.arange(0, 361, 5)))

    x = x0 + np.outer(a, cosb)
    y = y0 + np.outer(a, sinb)

    lat = []
    lon = []
    for xi, yi in zip(x, y):
        lati, loni = utm.to_latlon(xi, yi, z_num, z_letter)
        lat.append(lati)
        lon.append(loni)

    lat = np.array(lat)
    lat = lat.reshape(x.shape)
    lon = np.array(lon)
    lon = lon.reshape(x.shape)

    return x, y, lon, lat, x0, y0, lon0, lat0
