from OpenGL.GL import *
import numpy as np
import os
from coord_utils import *

from contextlib import contextmanager

@contextmanager
def gl_state_guard(save_current_color=True,
                   save_point_size=True,
                   save_line_width=True):
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
    def __init__(self, mesh):
        self.mesh = mesh
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.scale = np.ones(3)

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

class SceneModel:
    def __init__(self, lat_deg, lon_deg, alt_m, scale, obj_path):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.alt_m = alt_m
        self.scale = scale
        self.position = np.array(latlon_to_app_xyz(self.lat_deg, self.lon_deg, self.alt_m))
        self.rotation = self.calc_rotation()
        self.mesh = OBJLoader.load(obj_path)
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
        super().__init__(mesh=None)
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


# ---------------------------------------------------------------------
# Polyline primitive
# ---------------------------------------------------------------------
class PolyLineSceneObject(SceneObject):
    def __init__(self, points_wgs84, color=(1.0, 1.0, 0.0), width=4.0):
        """
        points_wgs84: list of (lat, lon, alt_m)
        """
        super().__init__(mesh=None)
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

class Scene:
    '''Container for scene objects'''
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

##    def draw_sphere(self, lat, lon, alt=0.0, radius_m=100000.0, color=(1.0, 0.0, 0.0)):
##        """
##        Draw a sphere using WGS84 coordinates, aligned to the globe’s tile frame.
##        """
##        x, y, z = latlon_to_app_xyz(lat, lon, alt, R=self.earth_radius)
##
##        #print ("sphere: ", x,y,z)
##
##        # Scale radius from meters to world units
##        scale = self.earth_radius / WGS84_A
##
##        glPushMatrix()
##        glTranslatef(x, y, z)
##        glColor3f(*color)
##        quadric = gluNewQuadric()
##        gluSphere(quadric, radius_m * scale, 24, 24)
##        gluDeleteQuadric(quadric)
##        glPopMatrix()
