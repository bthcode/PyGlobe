import math, os
#-------------------------------------------------------
# TILE UTILITIES
#-------------------------------------------------------
def clamp(a,b,c): 
    return max(b, min(c, a))

def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return ytile, xtile

def latlon_to_xyz(lat, lon, R=1.0):
    lon += 180
    la = math.radians(lat)
    lo = math.radians(-lon)  # ← flip sign to restore east-positive orientation
    x = R * math.cos(la) * math.cos(lo)
    y = R * math.sin(la)
    z = R * math.cos(la) * math.sin(lo)
    return x, y, z

def xyz_to_latlon(x, y, z):
    """Inverse of above: returns (lat, lon) with lon in (-180,180]."""
    r = math.sqrt(x*x + y*y + z*z)
    lat = math.degrees(math.asin(y / r))
    lon = math.degrees(math.atan2(z, x))
    # normalize lon to (-180,180]
    if lon > 180: lon -= 360
    if lon <= -180: lon += 360
    return lat, lon

def tile2lon(x, z):
     n = 2 ** z
     return x / n * 360.0 - 180.0
 
def tile2lat(y, z):
    n = 2 ** z
    ##y = n - 1 - y
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad)

def tile_path(cache_root, z,x,y):
    return os.path.join(cache_root, str(z), str(x), f"{y}.png")


