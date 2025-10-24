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

import sys, os, math, threading, queue
from collections import OrderedDict
import requests
timeout = 3
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt5 import QtCore, QtWidgets, QtGui
#from PyQt5.QtOpenGL import QOpenGLWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# ----------------- Config -----------------

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
USER_AGENT = "pyvista-globe-example/1.0 (your_email@example.com)"  # set a sensible UA
CACHE_ROOT = "./cache"
TILE_SIZE = 256
MIN_Z = 2
MAX_Z = 6    # set to highest zoom level you have in cache
MAX_GPU_TEXTURES = 2048

# LOD blending params
BLEND_SHARPNESS = 3.0

# Checkerboard fallback
CHECKER_COLOR_A = 200
CHECKER_COLOR_B = 60


def approximate_visible_bbox(camera_pos, camera_dir, fov_y_deg, aspect):
    """
    Approximate the visible lat/lon rectangle of the Earth.
    camera_pos: np.array([x, y, z]) in Earth-centered coordinates (meters or normalized radius)
    camera_dir: unit vector pointing toward the Earth center
    fov_y_deg: vertical field of view in degrees
    aspect: viewport width / height
    Returns (min_lat, max_lat, min_lon, max_lon)
    """
    R = 1.0  # assume unit sphere

    # Find where the camera looks (intersection with Earth)
    d = -np.dot(camera_pos, camera_dir)
    closest_point = camera_pos + d * camera_dir
    lat0 = np.degrees(np.arcsin(closest_point[1] / R))
    lon0 = np.degrees(np.arctan2(closest_point[0], closest_point[2]))

    # Approximate visible angular radius on the globe
    # Half the angular width visible from the camera altitude:
    h = np.linalg.norm(camera_pos)
    theta = np.degrees(np.arccos(R / h))  # horizon angle
    fov_y = np.radians(fov_y_deg)
    fov_x = np.arctan(np.tan(fov_y / 2) * aspect) * 2
    half_angle = np.degrees(fov_y / 2) + theta * 0.5

    lat_extent = half_angle
    lon_extent = half_angle * aspect

    return (
        lat0 - lat_extent,
        lat0 + lat_extent,
        lon0 - lon_extent,
        lon0 + lon_extent,
    )

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

def make_checkerboard(size=TILE_SIZE, squares=8):
    arr = np.zeros((size,size,3), dtype=np.uint8)
    s = size//squares
    for yy in range(squares):
        for xx in range(squares):
            color = CHECKER_COLOR_A if (xx+yy)%2 else CHECKER_COLOR_B
            arr[yy*s:(yy+1)*s, xx*s:(xx+1)*s, :] = color
    return arr

CHECKER = make_checkerboard()

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
            if os.path.exists(path):
                try:
                    pil = Image.open(path).convert("RGB")
                    np_img = np.asarray(pil, dtype=np.uint8)
                except Exception as e:
                    print("disk loader: failed to open", path, e)
            else:

                # download fot later
                url = TILE_URL.format(z=z, x=x, y=y)
                s = requests.Session()
                s.headers.update({"User-Agent": USER_AGENT})
                resp = s.get(url, timeout=timeout)
                img = Image.open(BytesIO(resp.content)).convert("RGBA")
                import pathlib
                p = pathlib.Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)

                arr = CHECKER.copy()
                pil = Image.fromarray(arr)
                draw = ImageDraw.Draw(pil)
                draw.text((6,6), f"missing {z}/{x}/{y}", fill=(255,0,0), font=self.font)
                np_img = np.asarray(pil, dtype=np.uint8)
            # keep exactly 3 channels
            if np_img.ndim == 3 and np_img.shape[2] > 3:
                np_img = np_img[:,:,:3]
            self.res_q.put((z,x,y,np_img))
            self.req_q.task_done()

# ----------------- GL widget -----------------
class GlobeOfflineTileAligned(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(900,600)
        self.rot_x = 0.0
        self.rot_y = 90.0
        self.distance = 2.8
        self.last_pos = None

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
        self.inflight = set()
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
        dir = -camera_pos / np.linalg.norm(camera_pos)

        # Ray-sphere intersection (R = 1)
        a = np.dot(dir, dir)
        b = 2 * np.dot(camera_pos, dir)
        c = np.dot(camera_pos, camera_pos) - 1
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / (2*a)
        point = camera_pos + t * dir

        x, y, z = point
        lat = math.degrees(math.asin(y))
        lon = -math.degrees(math.atan2(z, x))  # flip sign to match your conversion

        #fov_y_deg = 10 
        #aspect = self.width() / self.height()
        #print (approximate_visible_bbox(camera_pos, dir, fov_y_deg, aspect))
        #if lat < 0:
        #    lat = abs(lat)

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
        #zoom_float = clamp( (8-self.distance)/(8-1.2)*(MAX_Z-MIN_Z)+MIN_Z, MIN_Z, MAX_Z)
        distance = self.distance
        if distance > 3: 
            level_z = 3
        elif distance > 2.6:
            level_z = 4
        elif distance > 1.4:
            level_z = 5
        else:
            level_z = 6

        print ("distance: ", self.distance)
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
                    self._ensure_request(key)
                    continue
                lon0, lon1 = tile2lon(x, base_z), tile2lon(x+1, base_z)
                lat0, lat1 = tile2lat(y+1, base_z), tile2lat(y, base_z)
                self._draw_spherical_tile(lat0, lat1, lon0, lon1, tex, alpha=1.0)

        

        #--------------------------------------------
        # Center lat lon as a test
        lat, lon = self.get_center_latlon()
        print(f"Camera center: lat={lat:.2f}, lon={lon:.2f}")

        #-------------------------------------------
        # TEST: can we figure out tiles on screen
        current_tile_y, current_tile_x = latlon_to_tile(lat, lon, level_z)
        print (current_tile_y, current_tile_x)

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
            XR *=4
        elif l > 20: 
            XR *=2
        else:
            pass
        xs = np.arange(current_tile_x - XR, current_tile_x + XR+1, 1)
        xs[xs < 0] += n # wrap
        xs[xs >= n] -= n # wrap
        xs = np.unique(xs)

        ys = np.arange(current_tile_y - XR, current_tile_y + XR+1, 1)
        ys[ys < 0] += n # wrap
        ys[ys >= n] -= n # 2rap
        ys = np.unique(ys)
        # END TEST


        for x in xs:
            for y in ys:
                key = (level_z,x,y)
                tex = self._get_texture_for_key(key)
                if tex is None:
                    self._ensure_request(key)
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
            if key in self.textures:
                old = self.textures.pop(key)
                try: glDeleteTextures([old])
                except: pass
            self.textures[key] = texid
            while len(self.textures) > self.max_gpu_textures:
                oldk, oldtex = self.textures.popitem(last=False)
                try: glDeleteTextures([oldtex])
                except: pass

    # ----------------- texture & request helpers -----------------
    def _get_texture_for_key(self, key):
        return self.textures.get(key)
    def _ensure_request(self,key):
        if key in self.inflight: return
        z,x,y = key
        if z<MIN_Z or z>MAX_Z: return
        self.inflight.add(key)
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
            self.rot_y += dx*0.5
            self.rot_x += dy*0.5
            self.update()
        self.last_pos=ev.pos()
    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()/120.0
        self.distance -= delta*0.35
        self.distance = clamp(self.distance, 1.2, 8.0)
        self.update()
    def keyPressEvent(self, ev):
        if ev.key()==QtCore.Qt.Key_Escape: self.close()
        else: super().keyPressEvent(ev)

    def __del__(self):
        try: self.stop_event.set()
        except: pass

# ----------------- main -----------------
if __name__=="__main__":
    # TODO - remove available cache logic - was temporary
    os.makedirs(CACHE_ROOT, exist_ok=True)
    available = {}
    for z in range(MIN_Z, MAX_Z+1):
        zd = os.path.join(CACHE_ROOT,str(z))
        if os.path.isdir(zd):
            available[z] = sum(len(files) for _,_,files in os.walk(zd))
        else: available[z]=0
    print("Cache availability (tiles found):", available)

    app = QtWidgets.QApplication(sys.argv)
    win = GlobeOfflineTileAligned()
    win.setWindowTitle("Offline Globe — Tile-Aligned LOD")
    win.resize(1200,800)
    win.show()
    sys.exit(app.exec_())

