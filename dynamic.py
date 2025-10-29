# globe_offline_tile_aligned.py
# Offline, disk-only globe with tile-aligned quads, LOD blending, and thread-safe GL uploads.
#
# Requirements:
#   pip install PyQt5 PyOpenGL Pillow numpy
#
# Cache layout:
#   ./cache/{z}/{x}/{y}.png
#
# Place tiles up to z=5 (or whatever you have) in that layout and run this script.

import sys, os, math, threading, queue, pprint
from collections import OrderedDict
import requests
from io import BytesIO
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtNetwork import QNetworkAccessManager
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

import math
import numpy as np


from tile_fetcher import TileFetcher
# ----------------- Config -----------------

DOWNLOAD_TIMEOUT = 10
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
CACHE_ROOT= "./osm_cache"
#TILE_URL = "https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=oqjJNjlgIemNU8MjyXFj"
#CACHE_ROOT = "./sat_cache"

USER_AGENT = "pyvista-globe-example/1.0 (your_email@example.com)"  # set a sensible UA
TILE_SIZE = 256
MIN_Z = 2
MAX_Z = 9    # set to highest zoom level you have in cache
MAX_GPU_TEXTURES = 512

# Checkerboard fallback
CHECKER_COLOR_A = 200
CHECKER_COLOR_B = 60

WGS84_A = 6378137.0        # semi-major axis (m)
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)


#-------------------------------------------------------
# TILE UTILITIES
#-------------------------------------------------------
def clamp(a,b,c): 
    return max(b, min(c, a))

def latlon_to_tile(lat:float, lon:float, zoom:float)->[int,int]:
    '''Returns y,x of tile for a lat lon and zoom level'''
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return ytile, xtile

##def latlon_to_app_xyzZ(lat: float, lon:float, R:float=1.0):
##    '''Convert to lat, lon to XYZ space - note that longitude is flipped to match OpenGL'''
##    lon += 180
##    la = math.radians(lat)
##    lo = math.radians(-lon)  # ← flip sign to restore east-positive orientation
##    x = R * math.cos(la) * math.cos(lo)
##    y = R * math.sin(la)
##    z = R * math.cos(la) * math.sin(lo)
##    return x, y, z

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

#-------------------------------------------------------
# OpenGL Widget
#-------------------------------------------------------
class GlobeOfflineTileAligned(QOpenGLWidget):
    infoSig = pyqtSignal(dict)
    requestTile = pyqtSignal(int, int, int, str)
    setAimpoint = pyqtSignal(int, int, int)
    resetFetcher = pyqtSignal()
    sigShutdownFetcher = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setMinimumSize(900,600)

        # Current geometry
        self.earth_radius = 1.0
        self.rot_x = 0.0
        self.rot_y = 90.0
        self.distance = 3.2
        self.last_pos = None
        self.zoom_level = 3
        self.center_lla = {'lat' : 0, 'lon': 0, 'alt': 0}

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
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CW)
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.07,0.08,0.1,1.0)

    def resizeGL(self, w:int, h:int)->None:
        glViewport(0,0,w,h)

    def set_center_lla(self, lat: float, lon:float, alt:float=0)->None:
        self.center_lla = { 'lat' : lat,
                            'lon' : lon,
                            'alt' : alt }

    def get_center_latlon(self) -> [float,float]:
        '''Calculate where the camera is currently pointing'''
        # Compute camera position in world coords
        rx = math.radians(self.rot_x)
        ry = math.radians(self.rot_y)
        cx = self.distance * math.sin(ry) * math.cos(rx)
        cy = -self.distance * math.sin(rx)
        cz = self.distance * math.cos(ry) * math.cos(rx)
        camera_pos = np.array([cx, cy, cz], dtype=float)

        # Direction vector toward origin
        camera_dir = -camera_pos / np.linalg.norm(camera_pos)

        # Ray-sphere intersection (R = 1)
        a = np.dot(camera_dir, camera_dir)
        b = 2 * np.dot(camera_pos, camera_dir)
        c = np.dot(camera_pos, camera_pos) - 1
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / (2*a)
        point = camera_pos + t * camera_dir

        x, y, z = point
        lat = math.degrees(math.asin(y))
        lon = -math.degrees(math.atan2(z, x))  # flip sign to match your conversion

        return -lat, -lon

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

        #---------------------------------------------
        # Base Layer
        #---------------------------------------------
        for key, tex in self.base_textures.items():
            #if key[1] != 4 or key[2] != 4:
            #    continue
            #print (f"plotting {key}")
            z = key[0]
            x = key[1]
            y = key[2]
            lon0, lon1 = tile2lon(x, z), tile2lon(x+1, z)
            lat0, lat1 = tile2lat(y+1, z), tile2lat(y, z)
            self._draw_spherical_tile(lat0, lat1, lon0, lon1, tex, alpha=1.0)

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
            self._draw_spherical_tile(lat0, lat1, lon0, lon1, tex, alpha=1.0)

        self.debug_place_markers()

    def debug_place_markers(self):
        ''' Draw a test dot in boston '''
        self.earth_radius = 1.0
        self.earth_radius_m =  WGS84_A
        self.draw_sphere(42.5,-70.8,alt = 115_000, radius_m = 10_000)


    # ----------------- pending/result handling -----------------
    @pyqtSlot(int,int,int,bytes)
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
    def get_tile_texture(self, key: tuple(int,int,int)) -> np.uint32:
        '''Retrieve a texture
        Params
        ------
        key : (z, x, y)

        Returns
        -------
        tex_id : np.uint32 : OpenGL texture ID
        '''
        if key[0] == 3:
            return self.base_textures.get(key)
        else:
            return self.textures.get(key)

    def request_tile(self,key: tuple(int,int,int))->None:
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


    def draw_sphere(self, lat, lon, alt=0.0, radius_m=100000.0, color=(1.0, 0.0, 0.0)):
        """
        Draw a sphere using WGS84 coordinates, aligned to the globe’s tile frame.
        """
        x, y, z = latlon_to_app_xyz(lat, lon, alt, R=self.earth_radius)

        #print (x,y,z)

        # Scale radius from meters to world units
        scale = self.earth_radius / WGS84_A

        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        quadric = gluNewQuadric()
        gluSphere(quadric, radius_m * scale, 24, 24)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    
    # ----------------- drawing helpers -----------------
    def _draw_spherical_tile(self, lat0:float, lat1:float, lon0:float, lon1:float, texid:np.uint32, alpha:float=1.0)->None:
        '''Draw a tile on the curved Earth'''
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glColor4f(1.0,1.0,1.0,alpha)

        # Create an array that approximates the earth 
        n_sub = 6  
        latitudes = np.linspace(lat0, lat1, n_sub+1)
        longitudes = np.linspace(lon0, lon1, n_sub+1)

        #------------------------------------------
        # Draw the outline of the box for debug
        #------------------------------------------
        self.debug_mode = True
        if self.debug_mode:
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.0, 0.0, 0.0)  # red borders
            glBegin(GL_LINE_LOOP)
            for i in range(n_sub + 1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon0
                lon_b = lon1
                # bottom edge
                glVertex3f(*latlon_to_app_xyz(lat_a, lon_a))
            for j in range(n_sub + 1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat1
                # right edge
                glVertex3f(*latlon_to_app_xyz(lat_a, lon_a))
            for i in range(n_sub, -1, -1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon1
                lon_b = lon0
                # top edge
                glVertex3f(*latlon_to_app_xyz(lat_a, lon_a))
            for j in range(n_sub, -1, -1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat0
                # left edge
                glVertex3f(*latlon_to_app_xyz(lat_a, lon_a))
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

                v00 = latlon_to_app_xyz(la0, lo0)
                v01 = latlon_to_app_xyz(la0, lo1)
                v11 = latlon_to_app_xyz(la1, lo1)
                v10 = latlon_to_app_xyz(la1, lo0)

                glBegin(GL_QUADS)
                glTexCoord2f(*t00); glVertex3f(*v00)
                glTexCoord2f(*t01); glVertex3f(*v01)
                glTexCoord2f(*t11); glVertex3f(*v11)
                glTexCoord2f(*t10); glVertex3f(*v10)
                glEnd()

        glBindTexture(GL_TEXTURE_2D,0)
        glColor4f(1.0,1.0,1.0,1.0)
        

    def publish_display_info(self)->None:
        '''Emit debug info'''
        self.infoSig.emit({'level' : self.zoom_level,
                           'center_lla' : self.center_lla,
                           'current_tile_x' : self.current_tile_x,
                           'current_tile_y' : self.current_tile_y} )


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
    def mousePressEvent(self, ev)->None: 
        '''Beginning of mouse move'''
        self.last_pos = ev.pos()

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
        QtWidgets.QMainWindow.__init__(self)
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
        s = f"Level: {info_dict['level']}\n"
        s += f"Lat: {info_dict['center_lla']['lat']:.2f}\n"
        s += f"Lon: {info_dict['center_lla']['lon']:.2f}\n"
        s += f"Alt: {info_dict['center_lla']['alt']:.2f}\n"
        s += f"X:   {info_dict['current_tile_x']}\n"
        s += f"Y:   {info_dict['current_tile_y']}\n"
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

