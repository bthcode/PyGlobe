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
from coord_utils import *
from obj_loader import OBJLoader, SceneObject, Scene, PointSceneObject, PolyLineSceneObject, SceneModel, draw_pick_ray, draw_ray_origin

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


# ---------- drawing helpers ----------
def draw_sphere_at(pos, radius=0.03, color=(1.0, 1.0, 0.0)):
    """Small sphere marker (color default yellow)."""
    glPushAttrib(GL_CURRENT_BIT)
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    quad = gluNewQuadric()
    gluSphere(quad, radius, 12, 12)
    gluDeleteQuadric(quad)
    glPopMatrix()
    glPopAttrib()

def draw_pick_ray(ray_origin, ray_dir, length=2.0, color=(1.0, 0.0, 0.0)):
    glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT)
    glColor3f(*color)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3fv(ray_origin)
    glVertex3fv(ray_origin + ray_dir * length)
    glEnd()
    glPopAttrib()
#---------------------------------------------



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

    def get_center_latlon(self) -> [float, float]:
        """Return the latitude and longitude that the camera is looking at."""
        # Convert rotation angles to radians
        rx = math.radians(self.rot_x)
        ry = math.radians(self.rot_y)

        # Compute camera position in world coordinates
        cx = self.distance * math.sin(ry) * math.cos(rx)
        cy = -self.distance * math.sin(rx)
        cz = self.distance * math.cos(ry) * math.cos(rx)
        camera_pos = np.array([cx, cy, cz], dtype=float)
        self.camera_pos = camera_pos

        # Save for debugging
        self.camera_x, self.camera_y, self.camera_z = camera_pos

        # Direction toward origin
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

        # Convert to lat/lon matching latlon_to_app_xyz()
        lat = -math.degrees(math.asin(y))
        lon = -math.degrees(math.atan2(x, z)) 

        # Normalize longitude to [-180, 180]
        if lon < -180:
            lon += 360
        if lon > 180:
            lon -= 360

        return lat, lon

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

        self.scene.draw()

        if getattr(self.scene, "last_pick_debug", None):
            dbg = self.scene.last_pick_debug
            draw_sphere_at(dbg["cam_world"], 0.03, (1,1,0))  # yellow camera
            draw_sphere_at(dbg["ray_origin"], 0.02, (0,1,0))  # green ray origin
            draw_pick_ray(dbg["ray_origin"], dbg["ray_dir"], 2.5, (1,0,0))  # red line
            if "hit_point" in dbg and dbg["hit_point"] is not None:
                draw_sphere_at(dbg["hit_point"], 0.02, (0,0,1))  # blue intersection



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

