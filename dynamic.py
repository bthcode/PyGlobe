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

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
#from PyQt5.QtOpenGL import QOpenGLWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# ----------------- Config -----------------

DOWNLOAD_TIMEOUT = 10
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
#TILE_URL = "https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=oqjJNjlgIemNU8MjyXFj"

USER_AGENT = "pyvista-globe-example/1.0 (your_email@example.com)"  # set a sensible UA
CACHE_ROOT = "./osm.cache"
TILE_SIZE = 256
MIN_Z = 2
MAX_Z = 9    # set to highest zoom level you have in cache
MAX_GPU_TEXTURES = 512

# Checkerboard fallback
CHECKER_COLOR_A = 200
CHECKER_COLOR_B = 60


##def approximate_visible_bbox(camera_pos, camera_dir, fov_y_deg, aspect):
##    """
##    Approximate the visible lat/lon rectangle of the Earth.
##    camera_pos: np.array([x, y, z]) in Earth-centered coordinates (meters or normalized radius)
##    camera_dir: unit vector pointing toward the Earth center
##    fov_y_deg: vertical field of view in degrees
##    aspect: viewport width / height
##    Returns (min_lat, max_lat, min_lon, max_lon)
##    """
##    R = 1.0  # assume unit sphere
##
##    # Find where the camera looks (intersection with Earth)
##    d = -np.dot(camera_pos, camera_dir)
##    closest_point = camera_pos + d * camera_dir
##    lat0 = np.degrees(np.arcsin(closest_point[1] / R))
##    lon0 = np.degrees(np.arctan2(closest_point[0], closest_point[2]))
##
##    # Approximate visible angular radius on the globe
##    # Half the angular width visible from the camera altitude:
##    h = np.linalg.norm(camera_pos)
##    theta = np.degrees(np.arccos(R / h))  # horizon angle
##    fov_y = np.radians(fov_y_deg)
##    fov_x = np.arctan(np.tan(fov_y / 2) * aspect) * 2
##    half_angle = np.degrees(fov_y / 2) + theta * 0.5
##
##    lat_extent = half_angle
##    lon_extent = half_angle * aspect
##
##    return (
##        lat0 - lat_extent,
##        lat0 + lat_extent,
##        lon0 - lon_extent,
##        lon0 + lon_extent,
##    )

def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return ytile, xtile


# ----------------- Utility -----------------
def clamp(a,b,c): return max(b, min(c, a))

# ORIG
def latlon_to_xyz(lat, lon, R=1.0):
    lon += 180
    la = math.radians(lat)
    lo = math.radians(-lon)  # ← flip sign to restore east-positive orientation
    x = R * math.cos(la) * math.cos(lo)
    y = R * math.sin(la)
    z = R * math.cos(la) * math.sin(lo)
    return x, y, z

##def other_latlon_to_xyz(lat, lon, R=1.0):
##    """Standard: lon positive east, lat positive north."""
##    la = math.radians(lat)
##    lo = math.radians(lon)    # <--- NO negation here
##    x = R * math.cos(la) * math.cos(lo)
##    y = R * math.sin(la)
##    z = R * math.cos(la) * math.sin(lo)
##    return x, y, z

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

def tile_path(z,x,y):
    return os.path.join(CACHE_ROOT, str(z), str(x), f"{y}.png")

# ----------------- Disk loader worker -----------------
class DiskLoader(threading.Thread):
    def __init__(self, req_q, res_q, stop_event):
        super().__init__(daemon=True)
        self.req_q = req_q
        self.res_q = res_q
        self.stop_event = stop_event
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 72)
        except Exception:
            self.font = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                z,x,y = self.req_q.get(timeout=0.1)
            except queue.Empty:
                continue
            np_img = None
            path = tile_path(z,x,y)
            if not os.path.exists(path):
                # download fot later
                url = TILE_URL.format(z=z, x=x, y=y)
                print (f"fetching: {url}")
                try:
                    s = requests.Session()
                    s.headers.update({"User-Agent": USER_AGENT})
                    resp = s.get(url, timeout=DOWNLOAD_TIMEOUT)
                    img = Image.open(BytesIO(resp.content)).convert("RGBA")
                    p = pathlib.Path(path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img.save(p)
                except Exception as err:
                    continue

            try:
                pil = Image.open(path).convert("RGB")
                np_img = np.asarray(pil, dtype=np.uint8)
            except Exception as e:
                np_img = None
                print("disk loader: failed to open", path, e)
                continue


            # keep exactly 3 channels
            if np_img.ndim == 3 and np_img.shape[2] > 3:
                np_img = np_img[:,:,:3]
            self.res_q.put((z,x,y,np_img))
            self.req_q.task_done()

# ----------------- GL widget -----------------
class GlobeOfflineTileAligned(QOpenGLWidget):
    infoSig = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.setMinimumSize(900,600)
        self.rot_x = 0.0
        self.rot_y = 90.0
        self.distance = 3.2
        self.last_pos = None

        self.zoom_level = 3

        # background loader queues
        self.req_q = queue.Queue()
        self.res_q = queue.Queue()
        self.stop_event = threading.Event()
        self.loader = DiskLoader(self.req_q, self.res_q, self.stop_event)
        self.loader.start()

        # pending uploads
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.textures = OrderedDict()
        self.base_textures = OrderedDict()
        self.inflight = {}
        self.max_gpu_textures = MAX_GPU_TEXTURES
        self.flip_horizontal_on_upload = False

        # poll timer to move results from res_q -> pending
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._transfer_results_to_pending)
        self.poll_timer.start(40)

    def closeEvent(self, ev):
        self.stop_event.set()
        self.loader.join(timeout=1.0)
        super().closeEvent(ev)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.07,0.08,0.1,1.0)

    def resizeGL(self, w, h):
        glViewport(0,0,w,h)

    def get_center_latlon(self):
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



    def paintGL(self):
        self.makeCurrent()
        self._upload_pending_textures()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, max(1, self.width()/self.height()), 0.1, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,0,self.distance, 0,0,0, 0,1,0)
        glRotatef(self.rot_x,1,0,0)
        glRotatef(self.rot_y,0,1,0)

        # zoom / blend
        ## TODO: write a better level - it should just be a function of distance, not min z max z
        ##   - it should also scale quadratically
        distance = self.distance
        if distance > 3: 
            level_z = 3
        elif distance > 1.5:
            level_z = 4
        elif distance > 1.3:
            level_z = 5
        elif distance > 1.2:
            level_z = 6
        else:
            level_z = 7

        self.zoom_level = level_z

        #print ("distance: ", self.distance)
        #level_z = int(math.floor(zoom_float)) 

        # Always draw level3
        base_z = 3
        n = 2**base_z
        xs = np.arange(n)
        ys = np.arange(n)
        for x in xs:
            for y in ys:
                key = (base_z,x,y)
                tex = self._get_texture_for_key(key)
                if tex is None:
                    #print (f"asking for {key}")
                    self._ensure_request(key)
                    continue
                lon0, lon1 = tile2lon(x, base_z), tile2lon(x+1, base_z)
                lat0, lat1 = tile2lat(y+1, base_z), tile2lat(y, base_z)
                self._draw_spherical_tile(lat0, lat1, lon0, lon1, tex, alpha=1.0)
        

        #--------------------------------------------
        # Center lat lon as a test
        lat, lon = self.get_center_latlon()
        #print(f"Camera center: lat={lat:.2f}, lon={lon:.2f}")

        #-------------------------------------------
        # TEST: can we figure out tiles on screen
        current_tile_y, current_tile_x = latlon_to_tile(lat, lon, level_z)
        #print (current_tile_y, current_tile_x)

        # draw base tiles
        n = 2**level_z

        #---------------------------------------------------
        # estimate a list of tiles that are on screen
        #  - box wit radius R
        #  - if near a pole, make X extent bigger
        #  - handle wrap conditions
        #
        # TODO:
        #  - these should go to a thread requesting tiles
        #  - remove tiles that are not on screen
        #------------------------------------------------
        # Scale x as we get near the poles
        R = 3
        l = abs(lat)
        XR = R
        if l > 55: 
            XR = n//2
        elif l > 30: 
            XR = 4
        elif l > 20: 
            XR = 7
        else:
            pass

        XR = min(n//2, XR)
        xs = np.arange(current_tile_x - XR, current_tile_x + XR+1, 1)
        xs[xs < 0] += n # wrap
        xs[xs >= n] -= n # wrap
        xs = np.unique(xs)

        ys = np.arange(current_tile_y - 3, current_tile_y + 3+1, 1)
        ys[ys < 0] += n # wrap
        ys[ys >= n] -= n # 2rap
        ys = np.unique(ys)
        # END TEST

        #print (XR, xs, ys)

        for x in xs:
            for y in ys:
                key = (level_z,x,y)
                tex = self._get_texture_for_key(key)
                if tex is None:
                    self._ensure_request(key)
                    #if key[1] < 0 or key[0] < 0:
                    #    import ipdb; ipdb.set_trace()
                    continue
                lon0, lon1 = tile2lon(x, level_z), tile2lon(x+1, level_z)
                lat0, lat1 = tile2lat(y+1, level_z), tile2lat(y, level_z)
                self._draw_spherical_tile(lat0, lat1, lon0, lon1, tex, alpha=1.0)

        #
        self.debug_place_markers()

    def debug_place_markers(self):
        tests = [ (42.5, -70.8, "Boston") ]
        for lat, lon, label in tests:
            x,y,z = latlon_to_xyz(lat, lon)   # use the corrected function
            glPushMatrix()
            glTranslatef(x,y,z)
            # small marker sphere
            quad = gluNewQuadric()
            glColor3f(1.0, 0.0, 0.0)
            gluSphere(quad, 0.01, 8, 6)
            gluDeleteQuadric(quad)
            glPopMatrix()
            # (optionally draw label using your existing text overlay)

    # ----------------- pending/result handling -----------------
    def _transfer_results_to_pending(self):
        while True:
            try:
                z,x,y,arr = self.res_q.get_nowait()
            except queue.Empty:
                break
            key = (z,x,y)
            del self.inflight[key]
            with self.pending_lock:
                self.pending[key] = arr
            self.res_q.task_done()
        self.update()

    def _upload_pending_textures(self):
        with self.pending_lock:
            items = list(self.pending.items())
            self.pending.clear()
        if not items:
            return
        for key, arr in items:
            if arr is None:
                continue
            if self.flip_horizontal_on_upload:
                arr = np.flip(arr, axis=1)
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
                try: glDeleteTextures([old])
                except: pass
            if key in self.base_textures:
                old = self.base_textures.pop(key)
                try: glDeleteTextures([old])
                except: pass

            # Protected base layer
            if key[0] == 3:
                self.base_textures[key] = texid
            # Dynamic zoom layers
            else:
                self.textures[key] = texid
            # pruning
            while len(self.textures) > self.max_gpu_textures:
                oldk, oldtex = self.textures.popitem(last=False)
                #print (f"pruning {oldk}")
                try: glDeleteTextures([oldtex])
                except: pass

    # ----------------- texture & request helpers -----------------
    def _get_texture_for_key(self, key):
        if key[0] == 3:
            return self.base_textures.get(key)
        else:
            return self.textures.get(key)

    def _ensure_request(self,key):
        #print (f"looking for {key}")
        if key in self.inflight: 
            #print ("..in flight")
            return
        z,x,y = key
        if z<MIN_Z or z>MAX_Z: 
            #print (".. bad z")
            return
        #print ("..adding to q")
        self.inflight[key] = True
        self.req_q.put(key)

    # ----------------- drawing helpers -----------------
    def _draw_spherical_tile(self, lat0, lat1, lon0, lon1, texid, alpha=1.0):
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glColor4f(1.0,1.0,1.0,alpha)

        
        n_sub = 6  # you can increase for smoother curvature; can be adaptive by zoom
        latitudes = np.linspace(lat0, lat1, n_sub+1)
        longitudes = np.linspace(lon0, lon1, n_sub+1)

        self.debug_mode = True
        if self.debug_mode:
            glDisable(GL_TEXTURE_2D)
            glColor3f(1.0, 0.0, 0.0)  # red borders
            glBegin(GL_LINE_LOOP)
            for i in range(n_sub + 1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon0
                lon_b = lon1
                # bottom edge
                glVertex3f(*latlon_to_xyz(lat_a, lon_a))
            for j in range(n_sub + 1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat1
                # right edge
                glVertex3f(*latlon_to_xyz(lat_a, lon_a))
            for i in range(n_sub, -1, -1):
                t = i / n_sub
                lat_a = lat0 + (lat1 - lat0) * t
                lon_a = lon1
                lon_b = lon0
                # top edge
                glVertex3f(*latlon_to_xyz(lat_a, lon_a))
            for j in range(n_sub, -1, -1):
                s = j / n_sub
                lon_a = lon0 + (lon1 - lon0) * s
                lat_a = lat0
                # left edge
                glVertex3f(*latlon_to_xyz(lat_a, lon_a))
            glEnd()
            glEnable(GL_TEXTURE_2D)
            glColor3f(1, 1, 1)


        for i in range(n_sub):
            for j in range(n_sub):
                la0 = latitudes[i]; la1 = latitudes[i+1]
                lo0 = longitudes[j]; lo1 = longitudes[j+1]

                # standard UVs (u: 0->1 left->right; v: 1->0 top->bottom)
                t00 = (j / n_sub, 1 - i / n_sub)
                t01 = ((j+1) / n_sub, 1 - i / n_sub)
                t11 = ((j+1) / n_sub, 1 - (i+1) / n_sub)
                t10 = (j / n_sub, 1 - (i+1) / n_sub)

                v00 = latlon_to_xyz(la0, lo0)
                v01 = latlon_to_xyz(la0, lo1)
                v11 = latlon_to_xyz(la1, lo1)
                v10 = latlon_to_xyz(la1, lo0)

                glBegin(GL_QUADS)
                glTexCoord2f(*t00); glVertex3f(*v00)
                glTexCoord2f(*t01); glVertex3f(*v01)
                glTexCoord2f(*t11); glVertex3f(*v11)
                glTexCoord2f(*t10); glVertex3f(*v10)
                glEnd()

        glBindTexture(GL_TEXTURE_2D,0)
        glColor4f(1.0,1.0,1.0,1.0)
        


    # ----------------- interaction -----------------
    def mousePressEvent(self, ev): self.last_pos = ev.pos()
    def mouseMoveEvent(self, ev):
        # TODO - scale how much this moves based on zoom level
        if self.last_pos is None: self.last_pos=ev.pos(); return
        dx = ev.x()-self.last_pos.x()
        dy = ev.y()-self.last_pos.y()
        if ev.buttons() & QtCore.Qt.LeftButton:
            self.rot_y += dx*3/self.zoom_level
            self.rot_x += dy*3/self.zoom_level
            self.update()
        self.last_pos=ev.pos()
        self.infoSig.emit({'level' : self.zoom_level} )
    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()/120.0
        self.distance -= delta*0.35
        self.distance = clamp(self.distance, 1.1, 8.0)
        self.update()
        self.infoSig.emit({'level' : self.zoom_level} )
    def keyPressEvent(self, ev):
        if ev.key()==QtCore.Qt.Key_Escape: self.close()
        else: super().keyPressEvent(ev)

    def __del__(self):
        try: self.stop_event.set()
        except: pass

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
        #self.setCentralWidget(self.hbox)
    def on_window(self, info_dict: dict):
        self.text.setText(pprint.pformat(info_dict))


# ----------------- main -----------------
if __name__=="__main__":
    os.makedirs(CACHE_ROOT, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow() #GlobeOfflineTileAligned()
    win.setWindowTitle("Example Globe")
    win.resize(1200,800)
    win.show()
    sys.exit(app.exec_())

