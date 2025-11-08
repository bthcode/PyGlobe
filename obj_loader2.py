import sys, math
import numpy as np
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------------------------------------------------------
# Coordinate conversion (you already have)
# ---------------------------------------------------------------
def latlon_to_app_xyz(lat_deg, lon_deg, alt_m=0.0, R=1.0):
    la = math.radians(lat_deg)
    lo = math.radians(lon_deg)
    x = (R + alt_m / 6371000.0) * math.cos(la) * math.cos(lo)
    y = (R + alt_m / 6371000.0) * math.cos(la) * math.sin(lo)
    z = (R + alt_m / 6371000.0) * math.sin(la)
    return (x, y, z)

# ---------------------------------------------------------------
# Scene objects (Point + Polyline)
# ---------------------------------------------------------------
class SceneObject:
    def draw(self):
        pass
    def on_click(self):
        print(f"{self.__class__.__name__} clicked")

class PointSceneObject(SceneObject):
    def __init__(self, lat, lon, alt=0.0, color=(1,0,0), size=6):
        self.lat, self.lon, self.alt = lat, lon, alt
        self.color = color
        self.size = size
        self.xyz = np.array(latlon_to_app_xyz(lat, lon, alt, R=1.0))

    def draw(self):
        glColor3f(*self.color)
        glPointSize(self.size)
        glBegin(GL_POINTS)
        glVertex3f(*self.xyz)
        glEnd()

class PolyLineSceneObject(SceneObject):
    def __init__(self, points_wgs84, color=(1,1,0), width=2):
        self.points_wgs84 = points_wgs84
        self.color = color
        self.width = width
        self.points_xyz = [np.array(latlon_to_app_xyz(lat, lon, alt, R=1.0))
                           for lat, lon, alt in self.points_wgs84]

    def draw(self):
        glColor3f(*self.color)
        glLineWidth(self.width)
        glBegin(GL_LINE_STRIP)
        for p in self.points_xyz:
            glVertex3f(*p)
        glEnd()

# ---------------------------------------------------------------
# Scene container with picking
# ---------------------------------------------------------------
class Scene:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

    def pick(self, x, y, viewport):
        """Return the clicked object or None"""
        ray_origin, ray_dir = screen_to_world_ray(x, y, viewport)
        min_dist = float("inf")
        picked = None

        for obj in self.objects:
            if isinstance(obj, PointSceneObject):
                d = ray_point_distance(ray_origin, ray_dir, obj.xyz)
                if d < 0.02 and d < min_dist:  # threshold in world units
                    picked, min_dist = obj, d

            elif isinstance(obj, PolyLineSceneObject):
                for p1, p2 in zip(obj.points_xyz[:-1], obj.points_xyz[1:]):
                    d = ray_segment_distance(ray_origin, ray_dir, p1, p2)
                    if d < 0.02 and d < min_dist:
                        picked, min_dist = obj, d

        return picked

# ---------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------
def screen_to_world_ray(x, y, viewport):
    """Compute a ray from camera through screen point (x,y)."""
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    winY = viewport[3] - float(y)  # invert Y for OpenGL
    near = np.array(gluUnProject(x, winY, 0.0, modelview, projection, viewport))
    far  = np.array(gluUnProject(x, winY, 1.0, modelview, projection, viewport))
    dir_vec = far - near
    dir_vec /= np.linalg.norm(dir_vec)
    return near, dir_vec

def ray_point_distance(ray_origin, ray_dir, point):
    v = point - ray_origin
    t = np.dot(v, ray_dir)
    closest = ray_origin + t * ray_dir
    return np.linalg.norm(point - closest)

def ray_segment_distance(ray_origin, ray_dir, p1, p2):
    v = p2 - p1
    w0 = ray_origin - p1
    a = np.dot(v, v)
    b = np.dot(v, ray_dir)
    c = np.dot(ray_dir, ray_dir)
    d = np.dot(v, w0)
    e = np.dot(ray_dir, w0)
    denom = a * c - b * b
    if denom == 0:
        return float("inf")
    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom
    sc = np.clip(sc, 0.0, 1.0)
    pc = p1 + sc * v
    qc = ray_origin + tc * ray_dir
    return np.linalg.norm(pc - qc)

# ---------------------------------------------------------------
# OpenGL Widget
# ---------------------------------------------------------------
class GlobeWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.scene = Scene()
        # Example objects
        self.scene.add(PointSceneObject(45, -93, 0.0, color=(1,0,0)))
        track = [(45, -93, 0), (46, -92.5, 0), (47, -91.8, 0)]
        self.scene.add(PolyLineSceneObject(track, color=(0,1,1)))

        # Camera
        self.zoom = 2.5
        self.rot_x = 20
        self.rot_y = -30

    # ---------------- Rendering ----------------
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.15, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Simple orbit camera
        gluLookAt(0, 0, self.zoom, 0, 0, 0, 0, 0, 1)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 0, 1)

        self.scene.draw()

    # ---------------- Mouse interaction ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            viewport = glGetIntegerv(GL_VIEWPORT)
            picked = self.scene.pick(x, y, viewport)
            if picked:
                picked.on_click()
            else:
                print("Nothing picked")

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom -= delta * 0.1
        self.zoom = max(1.2, min(10.0, self.zoom))
        self.update()

# ---------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = GlobeWidget()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())

