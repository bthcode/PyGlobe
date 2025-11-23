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
CACHE_ROOT= "./cache"

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
from obj_loader import *



def get_enu_to_ecef_matrix(lat_deg, lon_deg):
    """
    Get rotation matrix to convert from local ENU frame to ECEF/v2 frame.

    ENU Frame (local tangent plane at lat/lon):
    - E (East): Points east along the tangent
    - N (North): Points north along the tangent
    - U (Up): Points radially outward from Earth

    This matrix transforms vectors from ENU to v2/ECEF coordinates.

    Parameters:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees

    Returns:
        3x3 numpy rotation matrix
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Standard ENU to ECEF rotation matrix
    # Each column is where ENU basis vector points in ECEF
    R = np.array([
        [-sin_lon,           -sin_lat * cos_lon,  cos_lat * cos_lon],  # X_ecef
        [ cos_lon,           -sin_lat * sin_lon,  cos_lat * sin_lon],  # Y_ecef
        [ 0.0,                cos_lat,            sin_lat          ]   # Z_ecef
    ])

    return R


def enu_vector_to_v2(vel_enu, lat_deg, lon_deg):
    """
    Convert a vector from local ENU frame to standard v2/ECEF frame.

    Parameters:
        vel_enu: [east, north, up] vector in m/s (or any units)
        lat_deg: Latitude where ENU frame is defined
        lon_deg: Longitude where ENU frame is defined

    Returns:
        [x, y, z] vector in v2 coordinate frame
    """
    R = get_enu_to_ecef_matrix(lat_deg, lon_deg)
    vel_v2 = R @ np.array(vel_enu, dtype=float)
    return vel_v2


def draw_velocity_arrow(lat_deg, lon_deg, alt_m,
                       vel_east, vel_north, vel_up,
                       scale=0.0001,
                       color=(1.0, 1.0, 0.0),
                       line_width=4.0,
                       point_size=8.0):
    """
    Draw a velocity arrow in the local ENU frame at a given location.

    This function:
    1. Converts lat/lon/alt to position in v2 space
    2. Converts ENU velocity to v2 space
    3. Adapts both to old OpenGL space for rendering
    4. Draws the arrow

    Parameters:
        lat_deg: Latitude of arrow origin (degrees)
        lon_deg: Longitude of arrow origin (degrees)
        alt_m: Altitude of arrow origin (meters)
        vel_east: Eastward velocity component (m/s)
        vel_north: Northward velocity component (m/s)
        vel_up: Upward velocity component (m/s)
        scale: Scale factor for arrow length (adjust to make visible)
        color: RGB tuple (0-1 range)
        line_width: Width of arrow shaft
        point_size: Size of arrow head point

    Example:
        # Draw arrow for satellite moving 7000 m/s east, 1000 m/s north
        draw_velocity_arrow(
            lat_deg=30.0,
            lon_deg=-100.0,
            alt_m=400_000,
            vel_east=7000,
            vel_north=1000,
            vel_up=0,
            scale=0.00005
        )
    """

    # 1. Get position in STANDARD v2 space
    pos_v2 = np.array(latlon_to_xyz_v2(lat_deg, lon_deg, alt_m, R=1.0))

    # 2. Convert ENU velocity to v2 space
    vel_enu = np.array([vel_east, vel_north, vel_up], dtype=float)
    vel_v2 = enu_vector_to_v2(vel_enu, lat_deg, lon_deg)

    # 3. Scale velocity for visibility
    vel_v2_scaled = vel_v2 * scale

    # 4. Adapt to OLD OpenGL coordinate space
    pos_old = np.array(new_to_old_coords(*pos_v2))
    vel_old = np.array(new_to_old_coords(*vel_v2_scaled))

    # 5. Calculate arrow endpoint
    endpoint_old = pos_old + vel_old

    # 6. Draw the arrow
    # Save OpenGL state
    prev_line_width = glGetFloatv(GL_LINE_WIDTH)
    prev_point_size = glGetFloatv(GL_POINT_SIZE)
    prev_color = glGetFloatv(GL_CURRENT_COLOR)

    try:
        # Draw arrow shaft (line)
        glLineWidth(line_width)
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(*pos_old)
        glVertex3f(*endpoint_old)
        glEnd()

        # Draw arrow head (point)
        glPointSize(point_size)
        glColor3f(1,0,0)
        glBegin(GL_POINTS)
        glVertex3f(*endpoint_old)
        glEnd()

    finally:
        # Restore OpenGL state
        glLineWidth(float(prev_line_width))
        glPointSize(float(prev_point_size))
        glColor4fv(prev_color)


def draw_enu_basis_vectors(lat_deg, lon_deg, alt_m=0.0, scale=0.1):
    """
    Draw the ENU coordinate frame axes at a location (for debugging).

    Red = East
    Green = North
    Blue = Up

    Parameters:
        lat_deg: Latitude (degrees)
        lon_deg: Longitude (degrees)
        alt_m: Altitude (meters) - use positive value to lift above surface
        scale: Length of basis vectors
    """
    # Position (with altitude to see above tiles)
    pos_v2 = np.array(latlon_to_xyz_v2(lat_deg, lon_deg, alt_m, R=1.0))
    pos_old = np.array(new_to_old_coords(*pos_v2))

    # Get ENU basis vectors in v2 space
    R_enu_to_v2 = get_enu_to_ecef_matrix(lat_deg, lon_deg)

    # Extract basis vectors (columns of rotation matrix)
    east_v2 = R_enu_to_v2[:, 0] * scale
    north_v2 = R_enu_to_v2[:, 1] * scale
    up_v2 = R_enu_to_v2[:, 2] * scale

    # DEBUG: Print what's happening at (0, 0)
    ##if abs(lat_deg) < 0.1 and abs(lon_deg) < 0.1:
    ##    print(f"\n=== ENU DEBUG at ({lat_deg:.1f}°, {lon_deg:.1f}°) ===")
    ##    print(f"Position v2: {pos_v2}")
    ##    print(f"Position old: {pos_old}")
    ##    print(f"Up vector v2: {up_v2}")
    ##    print(f"Up vector old: {new_to_old_coords(*up_v2)}")
    ##    print(f"Does up point away from origin? {np.dot(pos_v2, up_v2) > 0}")

    # Adapt to old space
    east_old = np.array(new_to_old_coords(*east_v2))
    north_old = np.array(new_to_old_coords(*north_v2))
    up_old = np.array(new_to_old_coords(*up_v2))

    # Save state
    prev_line_width = glGetFloatv(GL_LINE_WIDTH)
    prev_color = glGetFloatv(GL_CURRENT_COLOR)

    try:
        glLineWidth(3.0)

        # Draw East (Red)
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(*pos_old)
        glVertex3f(*(pos_old + east_old))
        glEnd()

        # Draw North (Green)
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(*pos_old)
        glVertex3f(*(pos_old + north_old))
        glEnd()

        # Draw Up (Blue)
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(*pos_old)
        glVertex3f(*(pos_old + up_old))
        glEnd()

    finally:
        glLineWidth(float(prev_line_width))
        glColor4fv(prev_color)



# Example 1: Satellite moving east and north
def test_satellite_velocity():
    """
    Add this call in your paintGL() method to test:
    """
    draw_velocity_arrow(
        lat_deg=30.0,
        lon_deg=30.0,
        alt_m=400_000,  # 400 km altitude
        vel_east=0,   # 7 km/s eastward
        vel_north=0,  # 1 km/s northward
        vel_up=1000,
        scale=0.00005,
        color=(1.0, 1.0, 0.0)
    )

# Example 2: Show ENU frame at a location
def test_enu_frame():
    """
    Add this call in your paintGL() method to visualize ENU axes:
    Test multiple locations to verify correctness
    """
    # Test at Prime Meridian
    draw_enu_basis_vectors(
        lat_deg=0.0,
        lon_deg=0.0,
        alt_m=500_000,  # 500 km altitude
        scale=0.2
    )

    # Test at North Pole
    draw_enu_basis_vectors(
        lat_deg=89.0,  # Close to pole (90° can be numerically unstable)
        lon_deg=0.0,
        alt_m=500_000,
        scale=0.2
    )

    # Test at Bay of Bengal (90°E)
    draw_enu_basis_vectors(
        lat_deg=0.0,
        lon_deg=90.0,
        alt_m=500_000,
        scale=0.2
    )

    # Test at your location
    draw_enu_basis_vectors(
        lat_deg=45.0,
        lon_deg=-93.0,
        alt_m=500_000,
        scale=0.2
    )

# Example 3: Aircraft with heading and climb
def test_aircraft():
    """
    Aircraft at 10km altitude, heading northeast, climbing
    """
    # Convert heading to ENU components
    heading_deg = 45  # Northeast
    speed_horizontal = 250  # m/s
    climb_rate = 10  # m/s
    
    vel_east = speed_horizontal * np.sin(np.radians(heading_deg))
    vel_north = speed_horizontal * np.cos(np.radians(heading_deg))
    vel_up = climb_rate
    
    draw_velocity_arrow(
        lat_deg=42.0,
        lon_deg=-71.0,
        alt_m=10_000,
        vel_east=vel_east,
        vel_north=vel_north,
        vel_up=vel_up,
        scale=0.0002,
        color=(1.0, 0.5, 0.0)
    )
#-------------------------------------------------------
# OpenGL Widget
#-------------------------------------------------------
class GlobeOfflineTileAligned(QOpenGLWidget):
    infoSig = Signal(dict)
    requestTile = Signal(int, int, int, str)
    setAimpoint = Signal(int, int, int)
    resetFetcher = Signal()
    sigShutdownFetcher = Signal()
    sigStartFetcher = Signal()
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
        self.sigShutdownFetcher.connect(self.fetcher.shutdown)
        self.sigStartFetcher.connect(self.fetcher.start)
        self.fetcher_thread = QThread()
        self.fetcher.moveToThread(self.fetcher_thread)
        self.fetcher_thread.start()


        # Fetcher Control
        self.requestTile.connect(self.fetcher.requestTile)
        self.setAimpoint.connect(self.fetcher.setAimpoint)
        self.resetFetcher.connect(self.fetcher.reset)
        self.fetcher.tileReady.connect(self.onTileReady)
        self.sigStartFetcher.emit()

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
        time.sleep(2)
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
        test_satellite_velocity()
        test_aircraft()
        test_enu_frame()

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
            #x, y = event.x(), event.y()
            pass

            #w = self.width()
            #h = self.height()
            #self.scene.pick(x,y, self)


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

