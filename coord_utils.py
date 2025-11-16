import numpy as np
import math
import os

# Constants
WGS84_A = 6378137.0        # semi-major axis (m)
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)

#============================================================
# COORDINATE SYSTEMS
#============================================================
# This application uses TWO coordinate systems:
#
# 1. OLD SYSTEM (OpenGL/Legacy):
#    - Used by OpenGL camera rotations (rot_x, rot_y)
#    - Z-axis points to Prime Meridian
#    - X-axis points to Bay of Bengal (90°E)
#    - Y-axis points to North Pole
#    Functions: latlon_to_xyz_old(), old_to_new_coords()
#
# 2. NEW SYSTEM (Standard Geographic/ECEF-like):
#    - X-axis points to Prime Meridian (0°E)
#    - Y-axis points to North Pole (90°N)
#    - Z-axis points to Bay of Bengal (90°E)
#    - Standard right-handed coordinate system
#    Functions: latlon_to_xyz_v2(), xyz_to_latlon_v2()
#
# ADAPTERS:
#    - old_to_new_coords(x,y,z) -> (z,y,x)
#    - new_to_old_coords(x,y,z) -> (z,y,x)
#    These convert between the two systems via axis permutation
#============================================================

def old_to_new_coords(x_old, y_old, z_old):
    """
    Transform from OLD app coords to NEW standard coords.
    
    Analysis from verification:
    - Old Prime Meridian (0,0,1) should become New Prime Meridian (1,0,0)
    - Old Bay of Bengal (1,0,0) should become New Bay of Bengal (0,0,1)
    - Old North Pole (0,1,0) should stay New North Pole (0,1,0)
    
    Direct mapping:
    x_new = z_old  (old Z-axis → new X-axis)
    y_new = y_old  (old Y-axis → new Y-axis)  
    z_new = x_old  (old X-axis → new Z-axis)
    """
    x_new = z_old
    y_new = y_old
    z_new = x_old
    return x_new, y_new, z_new


def new_to_old_coords(x_new, y_new, z_new):
    """
    Transform from NEW standard coords to OLD app coords.
    
    Inverse mapping:
    x_old = z_new  (new Z-axis → old X-axis)
    y_old = y_new  (new Y-axis → old Y-axis)
    z_old = x_new  (new X-axis → old Z-axis)
    """
    x_old = z_new
    y_old = y_new
    z_old = x_new
    return x_old, y_old, z_old


# These use standard ECEF-like conventions
def latlon_to_xyz_v2(lat_deg, lon_deg, alt_m=0.0, R=1.0):
    """
    Standard spherical to Cartesian conversion (ECEF-like).

    Conventions:
    - X-axis points to (0°N, 0°E) - Prime Meridian at Equator
    - Y-axis points to North Pole (90°N)
    - Z-axis points to (0°N, 90°E) - Bay of Bengal at Equator

    This is the standard right-handed coordinate system used in
    geospatial applications, flight simulators, etc.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    x = R * math.cos(lat) * math.cos(lon)
    y = R * math.sin(lat)
    z = R * math.cos(lat) * math.sin(lon)

    scale = 1.0 + (alt_m / WGS84_A)
    return x * scale, y * scale, z * scale


def xyz_to_latlon_v2(x, y, z):
    """
    Inverse of latlon_to_xyz_v2.
    Convert Cartesian coordinates back to lat/lon.
    """
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0:
        return 0.0, 0.0

    # Normalize
    x, y, z = x/r, y/r, z/r

    # Clamp y to avoid numerical errors in asin
    y = max(-1.0, min(1.0, y))

    lat = math.degrees(math.asin(y))
    lon = math.degrees(math.atan2(z, x))

    return lat, wrap_lon_deg(lon)

def clamp(a,b,c): 
    return max(b, min(c, a))

# Normalize -180..180 if you prefer
def norm_angle(a):
    while a <= -180.0: a += 360.0
    while a > 180.0: a -= 360.0
    return a

def wrap_lon_deg(lon):
    """Wrap longitude to [-180, +180)."""
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon


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
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    # Shift so lon=0 sits on +Z axis (instead of -X)
    lon_app = math.radians(-lon_deg + 90)  # <── CHANGE HERE
    lat_app = math.radians(lat_deg)

    x = R * math.cos(lat_app) * math.cos(lon_app)
    y = R * math.sin(lat_app)
    z = R * math.cos(lat_app) * math.sin(lon_app)

    x *= (1 + alt_m / WGS84_A)
    y *= (1 + alt_m / WGS84_A)
    z *= (1 + alt_m / WGS84_A)
    return x, y, z


def ned_to_ecef_velocity(v_ned, lat_deg, lon_deg):
    """
    Convert a velocity vector from local NED (North-East-Down) frame to ECEF.

    v_ned : iterable of [vn, ve, vd] in m/s
    lat_deg, lon_deg : location where NED frame is defined
    Returns np.array([vx, vy, vz]) in ECEF
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # NED to ECEF rotation matrix
    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
        [-np.sin(lat) * np.sin(lon),  np.cos(lon), -np.cos(lat) * np.sin(lon)],
        [ np.cos(lat),                0.0,         -np.sin(lat)]
    ])

    v_ecef = R.T @ np.array(v_ned, dtype=float)
    return v_ecef


