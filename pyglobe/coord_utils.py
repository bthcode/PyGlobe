import sys
import os
from collections import OrderedDict
import numpy as np

def spherical_to_ecef(lat, lon, r):
    """Convert spherical coordinates to ECEF
    r is distance from Earth center (for camera positioning)"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def lla_to_ecef(lat, lon, alt):
    """Convert latitude, longitude, altitude to ECEF coordinates
    alt is height above Earth surface (not distance from center)"""

    earth_radius = 6371000
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Total distance from Earth center = radius + altitude
    r = earth_radius + alt
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def get_enu_to_ecef_matrix(lat, lon):
    """Get rotation matrix from local ENU to ECEF frame"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # ENU basis vectors in ECEF
    east = np.array([
        -np.sin(lon_rad),
        np.cos(lon_rad),
        0
    ])
    
    north = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad)
    ])
    
    up = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])
    
    # Rotation matrix: columns are ENU basis vectors in ECEF
    R = np.column_stack([east, north, up])
    return R

