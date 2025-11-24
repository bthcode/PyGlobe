import numpy as np

def latlon_to_tile(lat:float, lon:float, zoom:int)->[int,int]:
    """Convert lat/lon to tile coordinates

    Parameters
    ----------
    lat : float
        latitude in WGS84 degrees
    lon : float
        longitude in WGS84 degrees
    zoom: int
        TMS zoom level

    Returns
    -------
    x : int
        TMS x
    y : int
        TMS y
    """
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = np.radians(lat)
    y = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
    x = x % n
    y = np.clip(y, 0, n - 1)
    return x, y

def tile_y_to_lat(y, zoom):
    """Convert TMS tile Y coordinate to latitude"""
    n = 2 ** zoom
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    return np.degrees(lat_rad)



