"""
globe_tms.py
Simple desktop 3D globe using PyQt5 + PyOpenGL that loads TMS/Web-Mercator tiles,
stitches them into an world_image (single zoom level) and maps the world_image onto a sphere.
"""

import sys
import math
import os
import threading
from functools import lru_cache
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
import requests

from PyQt5 import QtWidgets, QtGui, QtCore
from OpenGL.GL import *
from OpenGL.GLU import *


# ----------------------
# Config
# ----------------------
TILE_SIZE = 256
ZOOM = 3  # start small; increase for more detail (but cost grows as (2^z)^2 tiles)
TMS_IS_TOP_ORIGIN = True
# Example TMS/XYZ templates:
# - XYZ (Google/OpenStreetMap): "https://tile.openstreetmap.org/{z}/{x}/{y}.png" (y top origin)
# - TMS: "https://tile-server/{z}/{x}/{tms_y}.png" (y bottom origin)
# Choose the template you want. If server uses XYZ (most do), set tms_top_origin=False and use xyz template.
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
USER_AGENT = "pyvista-globe-example/1.0 (your_email@example.com)"  # set a sensible UA
# If using a true TMS server where y origin is bottom, set TMS_IS_TOP_ORIGIN=False and TILE_URL accordingly.

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ----------------------
# Tile math helpers
# ----------------------
def num_tiles(z):
    return 2 ** z


def tile_bounds_lonlat(x, y, z, tms=True):
    """Return (lon_left, lat_bottom, lon_right, lat_top) in degrees for that tile.
    If tms=True interpret y as TMS (origin bottom)."""
    n = 2 ** z
    if tms:
        # convert TMS y-> XYZ y: XYZ_y = n - 1 - TMS_y
        y_xyz = n - 1 - y
    else:
        y_xyz = y

    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0

    # lat from tile y in Web Mercator
    def tile_y_to_lat(y_tile):
        # returns latitude of the tile edge in degrees
        n = 2.0 ** z
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
        return math.degrees(lat_rad)

    lat_top = tile_y_to_lat(y_xyz)
    lat_bottom = tile_y_to_lat(y_xyz + 1)
    return lon_left, lat_bottom, lon_right, lat_top


def latlon_to_mercator_uv(lat_deg, lon_deg):
    """Map latitude/longitude to normalized Web Mercator texture coordinates (u,v) in [0,1].
       u: 0->1 left to right (lon -180..180).
       v: 0->1 top->bottom for XYZ tile layout (so consistent with tile y in XYZ).
    """
    # u straightforward
    u = (lon_deg + 180.0) / 360.0
    # v using mercator:
    lat_rad = math.radians(lat_deg)
    # clamp latitude for mercator
    max_lat = 85.0511287798066
    lat_rad = math.radians(max(min(lat_deg, max_lat), -max_lat))
    merc_n = math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))
    v = 0.5 - merc_n / (2.0 * math.pi)
    return u, v


# ----------------------
# Tile download + cache
# ----------------------
def tile_cache_path(z, x, y):
    return os.path.join(CACHE_DIR, f"{z}/{x}/{y}.png")


def fetch_tile(z, x, y, url_template=TILE_URL, tms_origin=False, session=None, timeout=10.0):
    """Fetch tile and cache locally. If server uses XYZ but you have TMS y, convert appropriately externally."""
    cache_p = tile_cache_path(z, x, y)
    if os.path.exists(cache_p):
        try:
            return Image.open(cache_p).convert("RGBA")
        except Exception:
            print (f"Error fetching tile: {cache_p}")
            # corrupted cache, remove
            os.remove(cache_p)

    # Many servers expect XYZ (top origin). If user supplies TMS coordinates (origin bottom),
    # convert to XYZ here if the server template expects XYZ. We'll attempt to be flexible:
    n = 2 ** z
    y_for_url = (n - 1 - y) if tms_origin else y

    url = url_template.format(z=z, x=x, y=y_for_url)
    print (f"url: {url}")
    s = session or requests.Session()
    resp = s.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGBA")
    import pathlib
    p = pathlib.Path(cache_p)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache_p)
    return img


def build_global_world_image(z, url_template=TILE_URL, tms_origin=False, max_workers=8, progress_cb=None):
    """Download every tile at zoom z and stitch into a single square world_image image.
       Returns a PIL Image and the tiles across/rows (=2**z).
    """
    n = 2 ** z
    world_image_w = TILE_SIZE * n
    world_image_h = TILE_SIZE * n
    world_image = Image.new("RGBA", (world_image_w, world_image_h), (0, 0, 0, 255))

    tiles = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    # download in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for y in range(n):
            for x in range(n):
                futures[ex.submit(fetch_tile, z, x, y, url_template, tms_origin, session)] = (x, y)

        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            x, y = futures[fut]
            try:
                img = fut.result()
            except Exception as e:
                print(f"Failed to fetch tile {x},{y}: {e}")
                # fill with blank
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (100, 100, 100, 255))
            world_image.paste(img, (x * TILE_SIZE, y * TILE_SIZE))
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return world_image, n


# ----------------------
# OpenGL + Qt widget
# ----------------------
class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.rot_x = -20.0
        self.rot_y = 30.0
        self.distance = 3.0

        # interaction state
        self.last_pos = None

        # sphere mesh
        self.lat_steps = 90
        self.lon_steps = 180
        self.vertex_data = None
        self.index_data = None
        self.texture_id = None

        # start tile download & world_image build on a background thread.
        self.world_image_ready = False
        self.loading_text = "Downloading tiles..."
        self._start_build_world_image(ZOOM)

    def _start_build_world_image(self, zoom):
        def progress(done, total):
            self.loading_text = f"Downloading tiles {done}/{total}"
            self.update()

        def worker():
            try:
                print("Building world_image (this may take a minute for high zoom)...")
                world_image, n = build_global_world_image(zoom, TILE_URL, tms_origin=not TMS_IS_TOP_ORIGIN, progress_cb=progress)
                world_image_path = os.path.join(CACHE_DIR, f"world_image_z{zoom}.png")
                world_image.save(world_image_path)
                # convert to OpenGL texture on the GUI thread
                QtCore.QMetaObject.invokeMethod(self, "load_world_image_from_file", QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(str, world_image_path),
                                                )
            except Exception as e:
                print("Atlas build failed:", e)
                self.loading_text = "Failed to build world_image: " + str(e)
                self.update()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    @QtCore.pyqtSlot(str)
    def load_world_image_from_file(self, path):
        # create OpenGL texture from the world_image image
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        data = img.tobytes("raw", "RGBA", 0, -1)
        self.makeCurrent()
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.world_image_w = w
        self.world_image_h = h
        self.world_image_ready = True
        self.make_sphere_mesh()
        self.update()

    def make_sphere_mesh(self):
        # create lat-long sphere and compute per-vertex texture coords using Web Mercator mapping.
        lat_steps = self.lat_steps
        lon_steps = self.lon_steps
        verts = []
        normals = []
        uvs = []
        for i in range(lat_steps + 1):
            lat = -90.0 + 180.0 * i / lat_steps  # -90..90
            lat_rad = math.radians(lat)
            for j in range(lon_steps + 1):
                lon = -180.0 + 360.0 * j / lon_steps
                lon_rad = math.radians(lon)
                x = math.cos(lat_rad) * math.cos(lon_rad)
                y = math.sin(lat_rad)
                z = math.cos(lat_rad) * math.sin(lon_rad)
                verts.extend([x, y, z])
                normals.extend([x, y, z])
                # compute mercator uv (u in [0,1], v in [0,1]) then map to world_image pixels
                u, v = latlon_to_mercator_uv(lat, lon)
                # world_image coordinates (u_pixels, v_pixels) if needed; but for GL we use normalized coords relative to world_image size
                # Note: our stitched world_image uses the XYZ tile row order from fetching function; we've stitched tiles with y increasing downward.
                uvs.extend([u, v])
        verts = np.array(verts, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)

        # indices
        idx = []
        for i in range(lat_steps):
            for j in range(lon_steps):
                p0 = i * (lon_steps + 1) + j
                p1 = p0 + 1
                p2 = p0 + (lon_steps + 1)
                p3 = p2 + 1
                idx.extend([p0, p2, p1])
                idx.extend([p1, p2, p3])
        idx = np.array(idx, dtype=np.uint32)

        self.vertex_data = (verts, normals, uvs)
        self.index_data = idx

    # Qt / OpenGL lifecycle
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.1, 0.12, 0.15, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.width = w
        self.height = h

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # camera
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, (self.width / float(self.height)) if self.height else 1.0, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, self.distance, 0, 0, 0, 0, 1, 0)

        # apply rotations
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)

        if not self.world_image_ready:
            # Clear background
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Draw loading text using QPainter overlay
            painter = QtGui.QPainter(self)
            painter.setPen(QtCore.Qt.white)
            painter.setFont(QtGui.QFont("Helvetica", 14))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.loading_text)
            painter.end()
            return

        # Draw textured sphere
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        verts, normals, uvs = self.vertex_data
        idx = self.index_data

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, normals)
        glTexCoordPointer(2, GL_FLOAT, 0, uvs)

        glDrawElements(GL_TRIANGLES, idx.size, GL_UNSIGNED_INT, idx)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

        glBindTexture(GL_TEXTURE_2D, 0)


    # interaction
    def mousePressEvent(self, ev):
        self.last_pos = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.last_pos is None:
            self.last_pos = ev.pos()
            return
        dx = ev.x() - self.last_pos.x()
        dy = ev.y() - self.last_pos.y()
        if ev.buttons() & QtCore.Qt.LeftButton:
            self.rot_y -= dx * 0.5
            self.rot_x += dy * 0.5
            self.update()
        self.last_pos = ev.pos()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() / 120.0  # typical step
        self.distance -= delta * 0.3
        self.distance = max(1.2, min(20.0, self.distance))
        self.update()


# ----------------------
# Main app
# ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    glw = GLWidget()
    window.setCentralWidget(glw)
    window.setWindowTitle("Desktop TMS Globe (Python)")
    window.resize(1024, 768)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

