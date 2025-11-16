# globe_offline_tile_aligned.py
# Offline, disk-only globe with tile-aligned quads, LOD blending, and thread-safe GL uploads.
#
# Requirements:
#   pip install PySide6 PyOpenGL Pillow numpy
#
# Cache layout:
#   ./cache/{z}/{x}/{y}.png
#
# Place tiles up to z=5 (or whatever you have) in that layout and run this script.


# system packages 
import math, os, pathlib, pprint, queue, sys, requests, threading, time
from collections import OrderedDict
from io import BytesIO

# additional packages
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# this project packages 
from tile_fetcher import TileFetcher

# pyside and opengl
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtNetwork import QNetworkAccessManager
from PySide6.QtCore import Qt, QTimer, Signal, QThread, Slot
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


# ----------------- Config -----------------

DOWNLOAD_TIMEOUT = 10
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
CACHE_ROOT= "./osm_cache"

USER_AGENT = "PyGlobe Example/1.0 (your_email@example.com)"  # set a sensible UA
TILE_SIZE = 512
MIN_Z = 2
MAX_Z = 9    # set to highest zoom level you have in cache
MAX_GPU_TEXTURES = 512

# Checkerboard fallback
CHECKER_COLOR_A = 200
CHECKER_COLOR_B = 60


# -- obj_loader -- #
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
from coord_utils import *

from contextlib import contextmanager

# -- coord_utils --#
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



#-------------------------------------
# Math utils
#-------------------------------------
def clamp(a,b,c): 
    return max(b, min(c, a))


# Normalize -180..180 if you prefer
def norm_angle(a):
    while a <= -180.0: a += 360.0
    while a > 180.0: a -= 360.0
    return a
#-------------------------------------
# Math utils
#-------------------------------------


#--------------------------------------------------------------
# Testing New Conversions
#--------------------------------------------------------------

# =========================================================
# CONSTANTS
# =========================================================
R_EARTH = 6378137.0   # WGS-84 meters, used to scale ECEF → GL

# =========================================================
# ROTATION MATRICES AND COORDINATE TRANSFORMS
# =========================================================

# --- ENU → ECEF rotation matrix ---
def enu_to_ecef_matrix(lat_rad, lon_rad):
    sphi, cphi = np.sin(lat_rad), np.cos(lat_rad)
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)

    return np.array([
        [-slon,            -sphi*clon,   cphi*clon],
        [ clon,            -sphi*slon,   cphi*slon],
        [  0.0,                 cphi,         sphi]
    ])


# --- Your custom ECEF → OpenGL rotation ---
# OpenGL +Z = ECEF +X
# OpenGL +X = ECEF -Y
# OpenGL +Y = ECEF -Z
R_ecef_to_gl = np.array([
    [0, -1,  0],   # Xgl = -Yecef
    [0,  0, 1],   # Ygl = -Zecef
    [1,  0,  0]    # Zgl = +Xecef
])



# =========================================================
# VELOCITY: ENU → ECEF → OpenGL
# velocity is NOT divided by R_earth (pure direction)
# =========================================================

def enu_velocity_to_gl(vel_enu, lat_deg, lon_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    # ENU → ECEF
    R_enu2ecef = enu_to_ecef_matrix(lat, lon)
    v_ecef = R_enu2ecef @ vel_enu

    # ECEF → GL
    v_gl = R_ecef_to_gl @ v_ecef
    return v_gl


def draw_arrow_at_position(pos_gl, v_gl, scale=1.0):
    """Draw a simple arrow from pos_gl in direction v_gl."""
    vx, vy, vz = (v_gl * scale).tolist()
    px, py, pz = pos_gl.tolist()

    # Line
    glLineWidth(8.0)
    glBegin(GL_LINES)
    glColor3f(1, 1, 0)
    glVertex3f(px, py, pz)
    glVertex3f(px + vx, py + vy, pz + vz)
    glEnd()

    # Tip
    glPointSize(8.0)
    glBegin(GL_POINTS)
    glColor3f(1, 0.3, 0.3)
    glVertex3f(px + vx, py + vy, pz + vz)
    glEnd()


def draw_gl_velocity_arrow(lat_deg, lon_deg, alt_m,
                           vel_e, vel_n, vel_u,
                           scale=1.0):
    """
    Draws the rotated velocity vector at the correct satellite
    position in your OpenGL radius-1 Earth world.
    """

    # Velocity in ENU (m/s)
    vel_enu = np.array([-vel_e, vel_n, vel_u], dtype=float)

    # ENU → GL direction
    v_gl = enu_velocity_to_gl(vel_enu, lat_deg, lon_deg)

    # Position: LLA → ECEF (meters)
    pos_ecef = np.array(latlon_to_ecef(lat_deg, lon_deg, alt_m))

    # Scale ECEF to match OpenGL radius=1 Earth
    pos_ecef_scaled = pos_ecef / WGS84_A

    # Rotate position into your GL frame
    #pos_gl = R_ecef_to_gl @ pos_ecef_scaled
    #pos_gl[0] *= -1
    pos_gl = np.array(latlon_to_app_xyz_old(lat_deg, lon_deg))


    # Draw arrow
    draw_arrow_at_position(pos_gl, v_gl, scale)


# =========================================================
# MAIN ENTRY POINT — CALL THIS FROM paintGL()
# =========================================================


#--------------------------------------------------------------
# Old Conversions
#--------------------------------------------------------------

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

def wrap_lon_deg(lon):
    """Wrap longitude to [-180, +180)."""
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon

def latlon_to_app_xyz_old(lat_deg, lon_deg, alt_m=0.0, R=1.0, origin_lon_deg=0.0, origin_lat_deg=0.0):
    """
    Convert lat/lon/alt to app X,Y,Z with radius R.
    origin_lon_deg, origin_lat_deg define the new central meridian/latitude.
    - Uses same app mapping as you already have: lon_app = 90 - lon_rel
    - Handles wrapping and applies origin with the correct sign so moving origin_lon
      moves the globe in the expected direction.
    """

    # compute relative longitude (longitude of point measured from origin)
    lon_rel = wrap_lon_deg(lon_deg - origin_lon_deg)
    lat_rel = lat_deg - origin_lat_deg  # usually you only change lon, but included for completeness

    # app convention: lon_app = 90 - lon_rel (preserves your existing behaviour)
    lon_app = math.radians(90.0 - lon_rel)
    lat_app = math.radians(lat_rel)

    x = R * math.cos(lat_app) * math.cos(lon_app)
    y = R * math.sin(lat_app)
    z = R * math.cos(lat_app) * math.sin(lon_app)

    scale = 1.0 + (alt_m / WGS84_A)
    return x * scale, y * scale, z * scale


# -- end coord utils -- #
#--------------------------------------------------------------
# END Old Conversions
#--------------------------------------------------------------



#--------------------------------------------------------------
# Map Entities
#--------------------------------------------------------------
@contextmanager
def gl_state_guard(save_current_color=True,
                   save_point_size=True,
                   save_line_width=True):
    '''Caches opengl state and returns it on exit'''
    # Save
    prev_color = None
    prev_point_size = None
    prev_line_width = None

    try:
        if save_current_color:
            # GL_CURRENT_COLOR returns 4 floats
            prev_color = glGetFloatv(GL_CURRENT_COLOR)
        if save_point_size:
            prev_point_size = glGetFloatv(GL_POINT_SIZE)
        if save_line_width:
            prev_line_width = glGetFloatv(GL_LINE_WIDTH)

        yield  # user code runs here

    finally:
        # Restore in reverse (order not critical)
        if save_line_width and prev_line_width is not None:
            glLineWidth(float(prev_line_width))
        if save_point_size and prev_point_size is not None:
            # glPointSize expects a float
            glPointSize(float(prev_point_size))
        if save_current_color and prev_color is not None:
            # prev_color is an array-like of 4 floats
            glColor4fv(prev_color)
# end gl_state_guard

class Mesh:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.materials = {}
# end class Mesh

class Material:
    def __init__(self, name):
        self.name = name
        self.diffuse = [0.8, 0.8, 0.8]
# end class Material

class OBJLoader:
    @staticmethod
    def load(path):
        mesh = Mesh()
        material = None
        vertices, normals = [], []
        current_material = None

        mtl_path = None
        base_dir = os.path.dirname(path)

        with open(path, "r") as f:
            for line in f:
                if line.startswith("mtllib"):
                    mtl_path = os.path.join(base_dir, line.split()[1].strip())
                elif line.startswith("usemtl"):
                    current_material = line.split()[1].strip()
                elif line.startswith("v "):
                    vertices.append(list(map(float, line.split()[1:])))
                elif line.startswith("vn "):
                    normals.append(list(map(float, line.split()[1:])))
                elif line.startswith("f "):
                    face = []
                    for v in line.split()[1:]:
                        parts = v.split("/")
                        vi = int(parts[0]) - 1
                        ni = int(parts[-1]) - 1 if len(parts) > 2 and parts[-1] else 0
                        face.append((vi, ni, current_material))
                    mesh.faces.append(face)

        if mtl_path and os.path.exists(mtl_path):
            mesh.materials = OBJLoader.load_mtl(mtl_path)

        mesh.vertices = np.array(vertices)
        mesh.normals = np.array(normals)
        return mesh

    @staticmethod
    def load_mtl(path):
        materials = {}
        current = None
        with open(path, "r") as f:
            for line in f:
                if line.startswith("newmtl"):
                    current = Material(line.split()[1].strip())
                    materials[current.name] = current
                elif line.startswith("Kd") and current:
                    current.diffuse = list(map(float, line.split()[1:4]))
        return materials
# end class OBJLoader

class SceneObject:
    def draw(self):
        pass
    def on_click(self):
        print(f"{self.__class__.__name__} clicked")
# end class ScenObject
   
class SceneModel(SceneObject):
    def __init__(self, lat_deg, lon_deg, alt_m, scale, obj_path):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.alt_m = alt_m
        self.scale = scale
        self.position = np.array(latlon_to_app_xyz_old(self.lat_deg, self.lon_deg, self.alt_m))
        self.rotation = self.calc_rotation()
        self.mesh = OBJLoader.load(obj_path)
        self.points_syz = self.position
    def calc_rotation(self):
        dir_vec = -self.position.astype(float)
        norm = np.linalg.norm(dir_vec)
        dir_vec /= norm
        dx, dy, dz = dir_vec[0], dir_vec[1], -dir_vec[2]


        # yaw: rotation around Y so forward (+Z local) points toward dir_vec
        yaw = np.degrees(np.atan2(dx, dz))

        # pitch: rotation around local X so forward axis tilts up/down toward dir_vec
        pitch = np.degrees(np.atan2(dy, np.sqrt(dx*dx + dz*dz)))

        # optional small self-spin about local forward axis (roll) or around model Z:
        roll = 0.0
        # assign into your satect rotation (X=pitch, Y=yaw, Z=roll)
        rotation = np.array([pitch, yaw, roll])
        return rotation

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)

        glBegin(GL_TRIANGLES)
        for face in self.mesh.faces:
            for vi, ni, matname in face:
                mat = self.mesh.materials.get(matname)
                if mat:
                    glColor3fv(mat.diffuse)
                if len(self.mesh.normals) > 0:
                    glNormal3fv(self.mesh.normals[ni])
                glVertex3fv(self.mesh.vertices[vi])
        glEnd()

        glPopMatrix()
# end class SceneModel



# ---------------------------------------------------------------------
# Point primitive
# ---------------------------------------------------------------------
class PointSceneObject(SceneObject):
    def __init__(self, lat_deg, lon_deg, alt_m=0.0, color=(1.0, 0.0, 0.0), size=15.0):
        super().__init__()
        self.lat = lat_deg
        self.lon = lon_deg
        self.alt = alt_m
        self.color = color
        self.size = size
        # Precompute world coordinates
        self.xyz = np.array(latlon_to_app_xyz_old(self.lat, self.lon, self.alt, R=1.0))

    def draw(self):
        glPushMatrix()
        with gl_state_guard():
            glColor3f(*self.color)
            glPointSize(self.size)

            glBegin(GL_POINTS)
            glVertex3f(*self.xyz)
            glEnd()
        glPopMatrix()

    def pick_points(self):
        return [self.xyz]
# end class PointSceneObject


# ---------------------------------------------------------------------
# Polyline primitive
# ---------------------------------------------------------------------
class PolyLineSceneObject(SceneObject):
    def __init__(self, points_wgs84, color=(1.0, 1.0, 0.0), width=4.0):
        """
        points_wgs84: list of (lat, lon, alt_m)
        """
        super().__init__()
        self.points_wgs84 = points_wgs84
        self.color = color
        self.width = width

        # Precompute xyz points in app coordinates
        self.points_xyz = [
            np.array(latlon_to_app_xyz_old(lat, lon, alt, R=1.0))
            for lat, lon, alt in self.points_wgs84
        ]

    def draw(self):
        glPushMatrix()

        with gl_state_guard():
            glColor3f(*self.color)
            glLineWidth(self.width)

            glBegin(GL_LINE_STRIP)
            for p in self.points_xyz:
                glVertex3f(*p)
            glEnd()

        glPopMatrix()
        
    def pick_points(self):
        return self.points_xyz
# end class PolyLineSceneObject


class Scene:
    def __init__(self):
        self.objects = []
        self.last_pick_ray = None  # (origin, dir) for debug drawing

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

#----------------------------------------------------
# END Map Entities
#----------------------------------------------------




#-------------------------------------------------------
# OpenGL Widget
#-------------------------------------------------------
class GlobeOfflineTileAligned(QOpenGLWidget):
    infoSig = Signal(dict)
    requestTile = Signal(int, int, int, str)
    setAimpoint = Signal(int, int, int)
    resetFetcher = Signal()
    sigShutdownFetcher = Signal()
    def __init__(self):
        super().__init__()
        self.setMinimumSize(900,600)

        # Current geometry
        self.earth_radius = 1.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.distance = 3.2
        self.last_pos = None
        self.zoom_level = 3
        self.center_lla = {'lat' : 0, 'lon': 0, 'alt': 0}

        # Object holder
        self.scene = Scene()

        # Picking stuff
        self.modelview = None
        self.viewport = None
        self.projection = None

        # Test drawings
        self.load_satellite()
        self.add_track()


        #-------------------------------------------------
        # Life cycle of a tile:
        #-------------------------------------------------
        #  1. gets added into screen_tiles
        #  2. add to inflight and req_q
        #  3. Diskloader puts img data into res_q
        #  4. Img data put into pending
        #  5. Pending converted to textures
        #  6. When textures overflow, they are removed
        #-------------------------------------------------
        # Tiles that have been requested but not received
        self.inflight = {}
        # Tiles currently on screen (estimated)
        self.screen_tiles = {} 
        # Holds img data that is not yet a texture
        self.pending = {}
        # Pending serviced by a timer, so needs a lock
        self.pending_lock = threading.Lock()
        # Handles to opengl textures
        self.textures = OrderedDict() 
        # Handles to opengl texture (kept separate so they won't be deleted)
        self.base_textures = OrderedDict() 


        #-------------------------------------------------
        # Fetcher Design
        #-------------------------------------------------
        # - Fetcher in a thread
        # - Gui Moves, 
        #    - sets setAimpoint -> cached in fetcher
        #    - emits tile requests 
        # - Fetcher sorts tile requests by distance from aimpoint
        # - when it loads a tile, it emits tileReady
        #------------------------------------------------

        # background loader queues
        self.fetcher = TileFetcher(cache_dir=CACHE_ROOT)
        self.fetcher_thread = QThread()
        self.fetcher.moveToThread(self.fetcher_thread)
        self.fetcher_thread.start()

        # Fetcher Control
        self.requestTile.connect(self.fetcher.requestTile)
        self.setAimpoint.connect(self.fetcher.setAimpoint)
        self.resetFetcher.connect(self.fetcher.reset)
        self.sigShutdownFetcher.connect(self.fetcher.shutdown)
        self.fetcher.tileReady.connect(self.onTileReady)

        #----------------------------
        # Load base layer
        #----------------------------
        self.load_base_textures()

        # Image data waiting to be turned into textures
        self.max_gpu_textures = MAX_GPU_TEXTURES

        # Publish info to display on a timer
        self.info_timer = QtCore.QTimer(self)
        self.info_timer.timeout.connect(self.publish_display_info)
        self.info_timer.start(1000)

      

    def shutdownFetcher(self):
        '''Shut Down Tile Fetching Thread'''
        self.sigShutdownFetcher.emit()
        time.sleep(1)
        self.fetcher_thread.quit()
        self.fetcher_thread.wait()

    def closeEvent(self, ev)->None:
        event.accept()
        super().closeEvent(ev)


    def initializeGL(self)->None:
        # This enables lighting
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CW)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.07,0.08,0.1,1.0)

    def resizeGL(self, w:int, h:int)->None:
        glViewport(0,0,w,h)

    def set_center_lla(self, lat: float, lon:float, alt:float=0)->None:
        self.center_lla = { 'lat' : lat,
                            'lon' : lon,
                            'alt' : alt }


    def get_center_latlon(self):
        """
        Return the latitude and longitude that the camera is looking at.
        Uses NEW v2 coordinate system for calculations.
        """
        # Convert rotation angles to radians
        rx = math.radians(self.rot_x)
        ry = math.radians(self.rot_y)

        # Compute camera position in OLD OpenGL coordinate space
        # (this is still driven by the old rotation system)
        cx = self.distance * math.sin(ry) * math.cos(rx)
        cy = -self.distance * math.sin(rx)
        cz = self.distance * math.cos(ry) * math.cos(rx)
        camera_pos_old = np.array([cx, cy, cz], dtype=float)

        # Convert camera position to NEW coordinate space
        camera_pos_new = np.array(old_to_new_coords(*camera_pos_old))

        # Store in new coordinate space for display
        self.camera_pos = camera_pos_new

        # Save individual components for debugging
        self.camera_x, self.camera_y, self.camera_z = camera_pos_new

        # Direction toward origin (in NEW space)
        camera_dir = -camera_pos_new / np.linalg.norm(camera_pos_new)

        # Ray-sphere intersection (R = 1, in NEW space)
        a = np.dot(camera_dir, camera_dir)
        b = 2 * np.dot(camera_pos_new, camera_dir)
        c = np.dot(camera_pos_new, camera_pos_new) - 1
        disc = b*b - 4*a*c

        if disc < 0:
            return None

        t = (-b - math.sqrt(disc)) / (2*a)
        point = camera_pos_new + t * camera_dir

        # Convert intersection point to lat/lon using NEW system
        lat, lon = xyz_to_latlon_v2(*point)

        # CORRECTION: The old rotation system has inverted lat/lon relationships
        # (cy = -distance * sin(rx) and the rot_y orientation)
        # So we need to flip both to match the working behavior
        lat = -lat
        lon = -lon

        return lat, wrap_lon_deg(lon)



    def load_base_textures(self):
        '''Ensure lowest level of map is always loaded'''
        # Always draw level3
        base_z = 3
        n = 2**base_z
        xs = np.arange(n)
        ys = np.arange(n)
        for x in xs:
            for y in ys:
                key = (base_z,x,y)
                self.request_tile(key)

    def paintGL(self):
        self.makeCurrent()
        self._upload_pending_textures()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, max(1, self.width()/self.height()), 0.1, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,0,self.distance, 
                  0,0,0, 
                  0,1,0)
        glRotatef(self.rot_x,1,0,0)
        glRotatef(self.rot_y,0,1,0)

        # For picking
        self.modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self.viewport = glGetIntegerv(GL_VIEWPORT)

        #---------------------------------------------
        # Base Layer
        #---------------------------------------------
        for key, tex in self.base_textures.items():
            z = key[0]
            x = key[1]
            y = key[2]
            lon0, lon1 = tile2lon(x, z), tile2lon(x+1, z)
            lat0, lat1 = tile2lat(y+1, z), tile2lat(y, z)
            self._draw_spherical_tile_v2(lat0, lat1, lon0, lon1, tex, alpha=1.0)  # <- NEW
   
        #---------------------------------------------
        # Current Layer, if available
        #---------------------------------------------
        for key in self.screen_tiles.keys():
            level_z = key[0]
            x = key[1]
            y = key[2]
            tex = self.get_tile_texture(key)
            if tex is None:
                self.request_tile(key)
                continue
            lon0, lon1 = tile2lon(x, level_z), tile2lon(x+1, level_z)
            lat0, lat1 = tile2lat(y+1, level_z), tile2lat(y, level_z)
            self._draw_spherical_tile_v2(lat0, lat1, lon0, lon1, tex, alpha=1.0)  # <- NEW
        self.scene.draw()

    # ----------------- pending/result handling -----------------
    @Slot(int,int,int,bytes)
    def onTileReady(self, z: int, x:int, y:int, data:bytes) ->None:
        '''Event handler from TileFetcher::tileReady : transfer to pending queue'''
        key = (z,x,y)
        del self.inflight[key]
        with self.pending_lock:
            self.pending[key] = data
        self.update()

    def _upload_pending_textures(self)->None:
        '''Transfer from pending to OpenGL texture, pruning old textures '''
        with self.pending_lock:
            items = list(self.pending.items())
            self.pending.clear()
        for key, arr in items:
            if arr is None:
                continue

            try:
                pil = Image.open(BytesIO(arr)).convert("RGB")
                np_img = np.asarray(pil, dtype=np.uint8)
            except Exception as e:
                print("failed to open tile", z, x, y, e)
                return

            # make sure it's contiguous 3-channel
            if np_img.ndim == 3 and np_img.shape[2] > 3:
                np_img = np_img[:, :, :3]
            arr = np.ascontiguousarray(np_img, dtype=np.uint8)
            h, w = arr.shape[0], arr.shape[1]

            arr = np.ascontiguousarray(arr, dtype=np.uint8)
            h,w = arr.shape[0], arr.shape[1]
            texid = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glPixelStorei(GL_UNPACK_ALIGNMENT,1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w,h,0,GL_RGB,GL_UNSIGNED_BYTE, arr)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D,0)
            # replace previous texture
            if key in self.textures:
                old = self.textures.pop(key)
                try: 
                    glDeleteTextures([old])
                except: 
                    pass
            if key in self.base_textures:
                old = self.base_textures.pop(key)
                try: 
                    glDeleteTextures([old])
                except: 
                    pass

            # Protected base layer
            if key[0] == 3:
                self.base_textures[key] = texid
            # Dynamic zoom layers
            else:
                self.textures[key] = texid
            # pruning
            while len(self.textures) > self.max_gpu_textures:
                oldk, oldtex = self.textures.popitem(last=False)
                try: 
                    glDeleteTextures([oldtex])
                except: 
                    pass

    # ----------------- texture & request helpers -----------------
    def get_tile_texture(self, key: [int,int,int]) -> np.uint32:
        '''Retrieve a texture
        Params
        ------
        key : (z, x, y)

        Returns
        -------
        tex_id : np.uint32 : OpenGL texture ID
        '''

        # - Level 3 is in base_textures
        # - All others in textures
        if key[0] == 3:
            return self.base_textures.get(key)
        else:
            return self.textures.get(key)

    def request_tile(self,key: [int,int,int])->None:
        '''If a tile is not already loaded, emit a request
        Params
        ------
        key: (z,x,y)
        '''
        if key in self.inflight: 
            return
        if key in self.textures:
            return
        if key in self.base_textures:
            return
        if key in self.pending:
            return
        z,x,y = key
        if z<MIN_Z or z>MAX_Z: 
            return
        self.inflight[key] = True
        self.requestTile.emit(z,x,y,TILE_URL)

    def load_satellite(self)->None:
        #------------------------------------------------------
        # Load a satellite
        base = os.path.dirname(__file__)
        sat_path = os.path.join(base, "assets/satellite/satellite.obj")
        sat = SceneModel(30, -100, 1250_000, np.array([0.1,0.1,0.1]), sat_path)
        self.scene.add(sat)

        if 0:
            model = SceneModel(lat_deg=30, lon_deg=20, alt_m=500_000,
                       scale=(0.1,0.1,0.1),
                       obj_path="assets/satellite/satellite.obj",)
            #ned_vel = [ 1000, 0, 0 ]
            #velocity = ned_to_ecef_velocity(ned_vel, 10, 20)
            #model.set_orientation_from_velocity(velocity)

            self.scene.add(model)

    def add_track(self)->None:
        # Add a single point over radar site
        self.scene.add(PointSceneObject(lat_deg=45.0, lon_deg=-93.0, alt_m=0.0,
                                   color=(0, 0, 0), size=16))

        # Add a polyline (track) connecting positions
        track_points = [
            (45.0, -93.0, 0.0),
            (46.0, -92.5, 0.0),
            (47.0, -91.8, 0.0)
        ]
        self.scene.add(PolyLineSceneObject(track_points, color=(0, 0, 0), width=8))


    # CLAUDE
    # Step 3: Replace _draw_spherical_tile with this new version
    # This uses the new v2 coordinate system internally, then adapts back to old OpenGL space
    def _draw_spherical_tile_v2(self, lat0, lat1, lon0, lon1, texid, alpha=1.0):
        """
        Draw a tile on the curved Earth using NEW coordinate system.

        This version:
        1. Computes vertices using latlon_to_xyz_v2 (new standard coords)
        2. Transforms them back to old OpenGL space using new_to_old_coords
        3. Should render IDENTICALLY to the original version
        """
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glColor4f(1.0, 1.0, 1.0, alpha)

        # Create an array that approximates the earth
        n_sub = 6
        latitudes = np.linspace(lat0, lat1, n_sub+1)
        longitudes = np.linspace(lon0, lon1, n_sub+1)

        self.debug_mode = True

        #------------------------------------------
        # Draw the outline of the box for debug
        #------------------------------------------
        if self.debug_mode:
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.0, 0.0, 0.0)  # black borders
            glBegin(GL_LINE_LOOP)
            for i in range(n_sub + 1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon0
                # bottom edge - use NEW coords, then adapt
                v = latlon_to_xyz_v2(lat_a, lon_a)
                v_old = new_to_old_coords(*v)
                glVertex3f(*v_old)
            for j in range(n_sub + 1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat1
                # right edge
                v = latlon_to_xyz_v2(lat_a, lon_a)
                v_old = new_to_old_coords(*v)
                glVertex3f(*v_old)
            for i in range(n_sub, -1, -1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon1
                # top edge
                v = latlon_to_xyz_v2(lat_a, lon_a)
                v_old = new_to_old_coords(*v)
                glVertex3f(*v_old)
            for j in range(n_sub, -1, -1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat0
                # left edge
                v = latlon_to_xyz_v2(lat_a, lon_a)
                v_old = new_to_old_coords(*v)
                glVertex3f(*v_old)
            glEnd()
            glEnable(GL_TEXTURE_2D)
            glColor3f(1, 1, 1)

        #------------------------------------------
        # Draw the actual tile
        #------------------------------------------
        for i in range(n_sub):
            for j in range(n_sub):
                la0 = latitudes[i]; la1 = latitudes[i+1]
                lo0 = longitudes[j]; lo1 = longitudes[j+1]

                # standard UVs (u: 0->1 left->right; v: 1->0 top->bottom)
                t00 = (j / n_sub, 1 - i / n_sub)
                t01 = ((j+1) / n_sub, 1 - i / n_sub)
                t11 = ((j+1) / n_sub, 1 - (i+1) / n_sub)
                t10 = (j / n_sub, 1 - (i+1) / n_sub)

                # NEW: Use v2 coordinate system
                v00_new = latlon_to_xyz_v2(la0, lo0)
                v01_new = latlon_to_xyz_v2(la0, lo1)
                v11_new = latlon_to_xyz_v2(la1, lo1)
                v10_new = latlon_to_xyz_v2(la1, lo0)

                # Transform back to old OpenGL space
                v00 = new_to_old_coords(*v00_new)
                v01 = new_to_old_coords(*v01_new)
                v11 = new_to_old_coords(*v11_new)
                v10 = new_to_old_coords(*v10_new)

                glBegin(GL_QUADS)
                glTexCoord2f(*t00); glVertex3f(*v00)
                glTexCoord2f(*t01); glVertex3f(*v01)
                glTexCoord2f(*t11); glVertex3f(*v11)
                glTexCoord2f(*t10); glVertex3f(*v10)
                glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glColor4f(1.0, 1.0, 1.0, 1.0)


    def publish_display_info(self)->None:
        '''Emit debug info'''
        self.infoSig.emit({'level' : self.zoom_level,
                           'center_lla' : self.center_lla,
                           'current_tile_x' : self.current_tile_x,
                           'current_tile_y' : self.current_tile_y,
                           'camera_pos' : { 'x' : self.camera_pos[0],
                                            'y' : self.camera_pos[1],
                                            'z' : self.camera_pos[2] 
                                           },
                           'rot' : { 'x' : self.rot_x,
                                     'y' : self.rot_y }
                           }
                          )


    def updateScene(self)->None:
        ''' Figoure out basic geometry of current view'''

        # 1. Get current center point, zoom
        distance = self.distance
        if distance > 3: 
            level_z = 3
        elif distance > 1.5:
            level_z = 4
        elif distance > 1.3:
            level_z = 5
        elif distance > 1.2:
            level_z = 6
        elif distance > 1.1:
            level_z = 7
        else:
            level_z = 8
        self.zoom_level = level_z

        # 2. Calculate scene center
        lat, lon = self.get_center_latlon()
        try:
            self.current_tile_y, self.current_tile_x = latlon_to_tile(lat, lon, level_z)
        except:
            pass
        self.set_center_lla(lat, lon, alt=0)
        self.setAimpoint.emit(int(level_z), int(self.current_tile_x), int(self.current_tile_y))

        #----------------------------------------------------------
        # 3. Get desired tile list
        # Scale x as we get near the poles
        #  TODO - this algorithm is a very rough approximation
        #----------------------------------------------------------
        R = 3
        n = 2**self.zoom_level
        l = abs(lat)
        XR = R
        if l > 55: 
            XR = 7
        elif l > 30: 
            XR = 5
        elif l > 20: 
            XR = 3
        else:
            pass

        #XR = min(n//2, XR)
        xs = np.arange(self.current_tile_x - XR, self.current_tile_x + XR+1, 1)
        xs[xs < 0] += n # wrap
        xs[xs >= n] -= n # wrap
        xs = np.unique(xs)

        ys = np.arange(self.current_tile_y - 5, self.current_tile_y + 5+1, 1)
        ys[ys < 0] += n # wrap
        ys[ys >= n] -= n # 2rap
        ys = np.unique(ys)
        
        # 4. Set current tiles on the screen
        self.screen_tiles.clear()
        for x in xs:
            for y in ys:
                x = int(x); y=int(y)
                key = (self.zoom_level, x, y)
                self.screen_tiles[(key)] = True
                self.request_tile((self.zoom_level, x, y))

    # ----------------- interaction -----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            x, y = event.x(), event.y()

            w = self.width()
            h = self.height()
            self.scene.pick(x,y, self)


        else:
            self.last_pos = event.pos()

    def mouseMoveEvent(self, ev)->None:
        '''End of mouse move'''
        if self.last_pos is None: self.last_pos=ev.pos(); return
        dx = ev.x()-self.last_pos.x()
        dy = ev.y()-self.last_pos.y()
        #self.resetFetcher.emit()
        if ev.buttons() & QtCore.Qt.LeftButton:
            self.rot_y += dx*3/self.zoom_level
            self.rot_x += dy*3/self.zoom_level
            self.updateScene()
            self.update()
        self.last_pos=ev.pos()


    def wheelEvent(self, ev)->None:
        '''Zoom'''
        delta = ev.angleDelta().y()/120.0
        self.distance -= delta*0.10
        self.distance = clamp(self.distance, 1.1, 8.0)
        self.updateScene()
        self.update()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        hbox = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()
        self.text = QtWidgets.QLabel('Label')
        vbox.addWidget(self.text)
        hbox.addLayout(vbox)
        self.globe = GlobeOfflineTileAligned()
        hbox.addWidget(self.globe)
        self.setLayout(hbox)
        self.globe.infoSig.connect(self.on_window)
        self.globe.updateScene()

    def on_window(self, info_dict: dict):
        s =  f"Level:    {info_dict['level']}\n"
        s += f"Tile X:   {info_dict['current_tile_x']}\n"
        s += f"Tile Y:   {info_dict['current_tile_y']}\n\n"

        s += f"Lat:      {info_dict['center_lla']['lat']:.2f}\n"
        s += f"Lon:      {info_dict['center_lla']['lon']:.2f}\n"
        s += f"Alt:      {info_dict['center_lla']['alt']:.2f}\n\n"

        X,Y,Z = latlon_to_ecef( info_dict['center_lla']['lat'],
                                info_dict['center_lla']['lon'],
                                0 )

        s += f"ECEF X:   {X:.1f}\n"
        s += f"ECEF Y:   {Y:.1f}\n"
        s += f"ECEF Z:   {Z:.1f}\n\n"

        s += f"CAM  X:   {info_dict['camera_pos']['x']:.1f}\n"
        s += f"CAM  Y:   {info_dict['camera_pos']['y']:.1f}\n"
        s += f"CAM  Z:   {info_dict['camera_pos']['z']:.1f}\n\n"

        s += f"Rot X:    {info_dict['rot']['x']:.1f}\n"
        s += f"Rot Y:    {info_dict['rot']['y']:.1f}\n"

        self.text.setText(s)


# ----------------- main -----------------
if __name__=="__main__":
    os.makedirs(CACHE_ROOT, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow() #GlobeOfflineTileAligned()
    win.setWindowTitle("Example Globe")
    win.resize(1200,800)
    app.aboutToQuit.connect(win.globe.shutdownFetcher)
    win.show()
    sys.exit(app.exec_())

