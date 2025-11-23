from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
from coord_utils import *

from contextlib import contextmanager


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

    def intersect_ray(self, ray_origin, ray_direction):
        """
        Test if ray intersects this object.

        Parameters:
            ray_origin: numpy array [x, y, z] in ECEF
            ray_direction: normalized numpy array [x, y, z] in ECEF

        Returns:
            distance (float) if hit, None if miss
        """
        raise NotImplementedError


class SceneModel(SceneObject):
    def __init__(self, lat_deg, lon_deg, alt_m, scale, obj_path, roll, pitch, yaw):
        self.lat = lat_deg
        self.lon = lon_deg
        self.alt = alt_m
        self.scale = scale
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.ecef_pos = lla_to_ecef(self.lat,self.lon,self.alt)
        self.pick_radius = 200_000

        #self.mesh = OBJLoader.load(obj_path)
        #self.points_syz = self.position
        vertices = []
        faces = []
        
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        parts = line.split()
                        # Handle faces like "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                        face = []
                        for p in parts[1:]:
                            face.append(int(p.split('/')[0]) - 1)  # OBJ indices start at 1
                        faces.append(face)
            
            self.mesh = {
                'vertices': np.array(vertices),
                'faces': faces
            }
            print(f"Loaded satellite mesh: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"Error loading satellite mesh: {e}")

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

    def intersect_ray(self, ray_origin, ray_direction):
        """Ray-sphere intersection for picking"""
        # Vector from ray origin to sphere center
        oc = ray_origin - self.ecef_pos

        # Quadratic equation coefficients
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        self.pick_radius = 150000
        c = np.dot(oc, oc) - self.pick_radius ** 2

        discriminant = b * b - 4 * a * c
        print (f"NEW: discriminant={discriminant}")


        # Closest approach
        #t_closest = -np.dot(oc, ray_direction) / np.dot(ray_direction, ray_direction)
        #closest_point = cam_pos + t_closest * ray_dir_world
        #closest_dist = np.linalg.norm(closest_point - sat_pos)

        #print(f"   Closest approach: {closest_dist}")
        if discriminant < 0:
            return None


        # Return closest intersection distance
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None

    def draw(self):
        self.draw_mesh(self.lat, self.lon, self.alt, self.roll, self.pitch, self.yaw)

    def draw_mesh(self, lat, lon, alt, roll=0, pitch=0, yaw=0):
        """Draw satellite at specified location with ENU orientation
        roll, pitch, yaw are in degrees in local ENU frame
        roll: rotation about east axis
        pitch: rotation about north axis  
        yaw: rotation about up axis
        """
        
        # Get ECEF position
        px, py, pz = lla_to_ecef(lat, lon, alt)
        
        # Get ENU to ECEF rotation matrix
        R_enu_to_ecef = get_enu_to_ecef_matrix(lat, lon)
        
        # Create rotation matrix for orientation in ENU frame
        # Apply rotations in order: yaw (Z), pitch (Y), roll (X) in ENU
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Roll (rotation about East - X axis in ENU)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Pitch (rotation about North - Y axis in ENU)
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Yaw (rotation about Up - Z axis in ENU)
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation in ENU
        R_enu = Rz @ Ry @ Rx
        
        # Total rotation: ENU orientation -> ECEF
        R_total = R_enu_to_ecef @ R_enu
        
        # Scale the mesh (OBJ files are often in arbitrary units)
        scale = 200000  # Scale to ~50km size
        
        # Convert to OpenGL 4x4 matrix (column-major)
        gl_matrix = np.eye(4)
        gl_matrix[:3, :3] = R_total * scale  # Include scale in rotation
        gl_matrix[:3, 3] = [px, py, pz]
        
        glDisable(GL_TEXTURE_2D)
        glPushMatrix()
        
        # Apply transformation matrix
        glMultMatrixf(gl_matrix.T.flatten())  # Transpose for OpenGL column-major
        
        # Draw the mesh
        glColor3f(0.8, 0.8, 0.8)  # Light gray
        
        for face in self.mesh['faces']:
            glBegin(GL_POLYGON)
            for vertex_idx in face:
                v = self.mesh['vertices'][vertex_idx]
                glVertex3f(v[0], v[1], v[2])
            glEnd()
        
        glPopMatrix()
        glEnable(GL_TEXTURE_2D)
     


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
        #self.xyz = np.array(latlon_to_app_xyz(self.lat, self.lon, self.alt, R=1.0))
        self.xyz = np.array(lla_to_ecef(self.lat, self.lon, self.alt))

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
            np.array(lla_to_ecef(lat, lon, alt))
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



class Scene:
    def __init__(self):
        self.objects = []
        self.last_pick_ray = None  # (origin, dir) for debug drawing

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

    def pick(self, ray_origin, ray_direction):
        """Find closest object hit by ray"""
        closest_obj = None
        closest_dist = float('inf')

        for obj in self.objects:
            try:
                dist = obj.intersect_ray(ray_origin, ray_direction)
                print (f'distance: {dist}')
            except NotImplementedError:
                continue
            if dist is not None and dist < closest_dist:
                closest_dist = dist
                closest_obj = obj

        return closest_obj, closest_dist if closest_obj else None





