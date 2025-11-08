import numpy as np
import math
import os

# Constants
WGS84_A = 6378137.0        # semi-major axis (m)
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)
#-------------------------------------------------------
# TILE UTILITIES
#-------------------------------------------------------
def clamp(a,b,c): 
    return max(b, min(c, a))

import math

def latlon_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """
    Convert geodetic coordinates (latitude, longitude, altitude)
    to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    Parameters:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        alt_m:   Altitude in meters (default: 0)

    Returns:
        (X, Y, Z) in meters
    """
    # WGS84 ellipsoid constants
    a = 6378137.0             # semi-major axis (m)
    f = 1 / 298.257223563     # flattening
    e2 = f * (2 - f)          # eccentricity squared

    # Convert angles to radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    # Prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)

    # Results
    X = (N + alt_m) * math.cos(lat) * math.cos(lon)
    Y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - e2) + alt_m) * math.sin(lat)

    return X, Y, Z

def latlon_to_tile(lat:float, lon:float, zoom:float)->[int,int]:
    '''Returns y,x of tile for a lat lon and zoom level'''
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return ytile, xtile

def tile2lon(x:int, z:int)->float:
     n = 2 ** z
     return x / n * 360.0 - 180.0
 
def tile2lat(y:int, z:int)->float:
    n = 2 ** z
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad)

def tile_path(cache_root:str, z:int,x:int,y:int)->str:
    '''Tile index to path where it should be in cache'''
    return os.path.join(cache_root, str(z), str(x), f"{y}.png")

def latlon_to_app_xyz(lat_deg, lon_deg, alt_m=0.0, R=1.0):
    """
    Convert WGS84 (lat, lon, alt) to application caresian

    NOTE - longitude is flipped and rotate by 180 degrees
    """
    # Convert geodetic → ECEF meters
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = WGS84_A
    e2 = WGS84_E2
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)

    xe = (N + alt_m) * math.cos(lat) * math.cos(lon)
    ye = (N + alt_m) * math.cos(lat) * math.sin(lon)
    ze = (N * (1 - e2) + alt_m) * math.sin(lat)

    # Scale Earth radius to match your scene (R corresponds to a=6378137)
    scale = R / WGS84_A

    # Convert ECEF → app’s coordinate frame:
    # Equivalent to lon' = -(lon + 180)
    lon_app = math.radians(-(lon_deg + 180))
    lat_app = math.radians(lat_deg)
    x = R * math.cos(lat_app) * math.cos(lon_app)
    y = R * math.sin(lat_app)
    z = R * math.cos(lat_app) * math.sin(lon_app)

    # Apply altitude offset (in meters → scaled units)
    x *= ( 1+ alt_m / WGS84_A)
    y *= ( 1+ alt_m / WGS84_A)
    z *= ( 1+ alt_m / WGS84_A)

    return x, y, z
