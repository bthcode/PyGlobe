import numpy as np

def spherical_to_ecef(lat:float, lon:float, r:float) -> [float,float,float]:
    """Convert spherical coordinates to ECEF

    Parameters
    ----------
    lat : float
        Latitude in WGS84 degrees
    lon : float
        Longitude in WG84 degrees
    r : float
        Distance from Earth center

    Returns
    -------
    x : float
        ECEF x
    y : float
        ECEF y
    z : float
        ECEF z
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def lla_to_ecef(lat: float, lon: float, alt: float) -> [float,float,float]:
    """Convert latitude, longitude, altitude to ECEF coordinates

    Parameters
    ----------
    lat : float
        Latitude in WGS84 Degrees
    lon : float
        Longitude in WGS84 Degrees
    alt : float
        Altitude above earth surface in meters

    Returns
    -------
    x : float
        ECEF x
    y : float
        ECEF y
    z : float
        ECEF z
    """

    earth_radius = 6371000
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Total distance from Earth center = radius + altitude
    r = earth_radius + alt
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def get_enu_to_ecef_matrix(lat: float, lon: float) -> np.ndarray:
    """Get rotation matrix from local ENU to ECEF frame

    Parameters
    ----------
    lat : float
        Latitude in WGS84 Degrees
    lon : float
        Longitude in WGS84 Degrees

    Returns
    -------
    R : np.ndarray
        3x3 Rotation Matrix
    """
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

