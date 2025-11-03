from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
from coord_utils import *

from contextlib import contextmanager

def draw_ray_origin(pos, radius=0.02, color=(1.0, 0.2, 0.2)):
    """Draw a small red sphere showing the ray’s origin."""
    glPushAttrib(GL_CURRENT_BIT)
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*pos)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 12, 12)
    gluDeleteQuadric(quad)
    glPopMatrix()
    glPopAttrib()

def draw_pick_ray(ray_origin, ray_dir, length=2.0, color=(1.0, 0.0, 0.0)):
    """Draw a debug line representing the picking ray."""
    glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT)
    glColor3f(*color)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3fv(ray_origin)
    glVertex3fv(ray_origin + ray_dir * length)
    glEnd()
    glPopAttrib()


def draw_point_sphere(pos, radius=0.02, color=(0.0, 1.0, 0.0)):
    """Draw a small sphere at the picked point."""
    glPushAttrib(GL_CURRENT_BIT)
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*pos)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 12, 12)
    gluDeleteQuadric(quad)
    glPopMatrix()
    glPopAttrib()



# ---------------------------------------------------------------
# Ray helpers for object selection
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

import numpy as np
from OpenGL.GLU import gluUnProject

def ray_segment_distance(ray_origin, ray_dir, p0, p1):
    v = p1 - p0
    w0 = ray_origin - p0
    a = np.dot(ray_dir, ray_dir)
    b = np.dot(ray_dir, v)
    c = np.dot(v, v)
    d = np.dot(ray_dir, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return np.linalg.norm(np.cross(w0, ray_dir)), 0.0, 0.0

    t_ray = (b * e - c * d) / denom
    t_seg = np.clip((a * e - b * d) / denom, 0.0, 1.0)

    closest_ray = ray_origin + t_ray * ray_dir
    closest_seg = p0 + t_seg * v
    dist = np.linalg.norm(closest_ray - closest_seg)
    return dist, t_ray, t_seg, closest_seg





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


class Mesh:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.materials = {}

class Material:
    def __init__(self, name):
        self.name = name
        self.diffuse = [0.8, 0.8, 0.8]

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


class SceneObject:
    def draw(self):
        pass
    def on_click(self):
        print(f"{self.__class__.__name__} clicked")

class SceneModel(SceneObject):
    def __init__(self, lat_deg, lon_deg, alt_m, scale, obj_path):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.alt_m = alt_m
        self.scale = scale
        self.position = np.array(latlon_to_app_xyz(self.lat_deg, self.lon_deg, self.alt_m))
        self.rotation = self.calc_rotation()
        self.mesh = OBJLoader.load(obj_path)
        self.points_syz = self.position
    def calc_rotation(self):
        dir_vec = -self.position.astype(float)
        norm = np.linalg.norm(dir_vec)
        dir_vec /= norm
        dx, dy, dz = dir_vec[0], dir_vec[1], dir_vec[2]

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
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
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
        self.xyz = np.array(latlon_to_app_xyz(self.lat, self.lon, self.alt, R=1.0))

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
            np.array(latlon_to_app_xyz(lat, lon, alt, R=1.0))
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

##class Scene:
##    '''Container for scene objects'''
##    def __init__(self):
##        self.objects = []
##
##    def add(self, obj):
##        self.objects.append(obj)
##
##    def draw(self):
##        for obj in self.objects:
##            obj.draw()
##
##    def pick(self, x, y, viewport):
##        """Return the clicked object or None"""
##        ray_origin, ray_dir = screen_to_world_ray(x, y, viewport)
##        min_dist = float("inf")
##        picked = None
##
##        for obj in self.objects:
##            # NOTE Can't select a satellite yet
##            if isinstance(obj, PointSceneObject):
##                d = ray_point_distance(ray_origin, ray_dir, obj.xyz)
##                if d < 0.02 and d < min_dist:  # threshold in world units
##                    picked, min_dist = obj, d
##
##            elif isinstance(obj, PolyLineSceneObject):
##                for p1, p2 in zip(obj.points_xyz[:-1], obj.points_xyz[1:]):
##                    d = ray_segment_distance(ray_origin, ray_dir, p1, p2)
##                    print (d)
##                    if d < 0.02 and d < min_dist:
##                        picked, min_dist = obj, d
##
##        return picked


# ---------------------------------------------------------------------
# Scene class with picking support
# ---------------------------------------------------------------------
## class Scene:
##     def __init__(self):
##         self.objects = []
## 
##     def add(self, obj):
##         self.objects.append(obj)
## 
##     def draw(self):
##         for obj in self.objects:
##             obj.draw()
## 
##     def draw_pick_ray(self,ray_origin, ray_dir, length=2.0, color=(1.0, 0.0, 0.0)):
##         """
##         Draw a debug line representing the picking ray in world coordinates.
## 
##         Parameters
##         ----------
##         ray_origin : np.array (3,)
##             Start of the ray in world coordinates
##         ray_dir : np.array (3,)
##             Normalized ray direction
##         length : float
##             How long to draw the ray
##         color : tuple
##             RGB color of the line
##         """
##         glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT)
##         glColor3f(*color)
##         glLineWidth(2.0)
## 
##         glBegin(GL_LINES)
##         glVertex3fv(ray_origin)
##         glVertex3fv(ray_origin + ray_dir * length)
##         glEnd()
## 
##         glPopAttrib()
## 
## 
##     def pick(self, x, y, view_width, view_height):
##         """
##         Convert a mouse (x, y) click into a 3D picking ray and test against objects.
## 
##         Returns: (hit_obj, hit_distance) or (None, None)
##         """
##         # Get current matrices
##         modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
##         projection = glGetDoublev(GL_PROJECTION_MATRIX)
##         viewport = glGetIntegerv(GL_VIEWPORT)
## 
##         # Flip y (Qt has origin top-left, OpenGL bottom-left)
##         y_gl = viewport[3] - y
## 
##         # Unproject to world coordinates
##         near_eye = gluUnProject(x, y_gl, 0.0, modelview, projection, viewport)
##         far_eye  = gluUnProject(x, y_gl, 1.0, modelview, projection, viewport)
## 
##         mv_inv = np.linalg.inv(modelview)
##         p0_eye = np.array(list(near_eye) + [1.0])
##         p1_eye = np.array(list(far_eye) + [1.0])
##         p0_world = mv_inv @ p0_eye
##         p1_world = mv_inv @ p1_eye
## 
##         ray_origin = p0_world[:3] / p0_world[3]
##         ray_dir = (p1_world[:3] / p1_world[3]) - ray_origin
##         ray_dir /= np.linalg.norm(ray_dir)
## 
##         # 🧭 Correct for your flipped lon_app coordinate frame
##         ray_origin[2] *= -1
##         ray_dir[2] *= -1
## 
##         self.draw_pick_ray(ray_origin, ray_dir, length=2.0)
##         
##         best_dist = np.inf
##         best_obj = None
## 
##         for obj in self.objects:
##             if not hasattr(obj, "pick_points"):
##                 continue
## 
##             pts = obj.pick_points()
##             if len(pts) == 1:
##                 dist = np.linalg.norm(np.cross(ray_origin - pts[0], ray_dir))
##                 if dist < best_dist:
##                     best_dist = dist
##                     best_obj = obj
## 
##             else:
##                 for i in range(len(pts) - 1):
##                     dist, t_ray, _ = ray_segment_distance(ray_origin, ray_dir, pts[i], pts[i + 1])
##                     if dist < best_dist:
##                         best_dist = dist
##                         best_obj = obj
## 
##         if best_obj is not None:
##             return best_obj, best_dist
##         return None, None

import numpy as np
from OpenGL.GLU import gluUnProject

def ray_segment_distance(ray_origin, ray_dir, p0, p1):
    v = p1 - p0
    w0 = ray_origin - p0
    a = np.dot(ray_dir, ray_dir)
    b = np.dot(ray_dir, v)
    c = np.dot(v, v)
    d = np.dot(ray_dir, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return np.linalg.norm(np.cross(w0, ray_dir)), 0.0, 0.0

    t_ray = (b * e - c * d) / denom
    t_seg = np.clip((a * e - b * d) / denom, 0.0, 1.0)

    closest_ray = ray_origin + t_ray * ray_dir
    closest_seg = p0 + t_seg * v
    dist = np.linalg.norm(closest_ray - closest_seg)
    return dist, t_ray, t_seg, closest_seg


import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

def ray_segment_distance(ray_origin, ray_dir, p0, p1):
    """Compute shortest distance between infinite ray and finite segment."""
    v = p1 - p0
    w0 = ray_origin - p0
    a = np.dot(ray_dir, ray_dir)
    b = np.dot(ray_dir, v)
    c = np.dot(v, v)
    d = np.dot(ray_dir, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-8:
        # Nearly parallel
        sc = 0.0
        tc = e / c
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom

    tc = np.clip(tc, 0.0, 1.0)
    p_ray = ray_origin + sc * ray_dir
    p_seg = p0 + tc * v
    return np.linalg.norm(p_ray - p_seg)


class Scene:
    def __init__(self):
        self.objects = []
        self.last_pick_ray = None  # (origin, dir) for debug drawing

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

    # ----------------------------------------------------------
    # ✅ Picking with correct world-space ray
    # ----------------------------------------------------------

    def pick(self, x, y, widget):
        # === 1. Get matrices and viewport ===
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        y_gl = viewport[3] - y

        # === 2. Compute true camera position from modelview ===
        mv = np.array(modelview, dtype=np.float64).reshape((4,4)).T
        mv_inv = np.linalg.inv(mv)
        camera_pos = mv_inv[:3, 3]

        # === 3. Compute unprojected near/far points ===
        near = np.array(gluUnProject(x, y_gl, 0.0, modelview, projection, viewport))
        far  = np.array(gluUnProject(x, y_gl, 1.0, modelview, projection, viewport))
        ray_dir = far - near
        ray_dir /= np.linalg.norm(ray_dir)

        print(f"Camera pos (world): {camera_pos}")
        print(f"Ray origin (world): {near}")
        print(f"Ray dir (world):    {ray_dir}")

        # === 4. Save for debug visualization ===
        self.last_pick_debug = {
            "cam_world": camera_pos,
            "ray_origin": near,
            "ray_dir": ray_dir,
        }

        # === 5. Run intersection tests (same as before) ===
        hit_obj, hit_dist, hit_point = None, float('inf'), None
        for obj in self.objects:
            if not hasattr(obj, "get_segments"):
                continue
            for p0, p1 in obj.get_segments():
                dist, closest = ray_segment_distance(near, ray_dir, p0, p1)
                if dist < hit_dist:
                    hit_obj, hit_dist, hit_point = obj, dist, closest

        if hit_obj:
            print(f"Picked {hit_obj} at distance {hit_dist}")
            self.last_pick_debug["hit_point"] = hit_point
        else:
            print("No pick")

        return hit_obj

