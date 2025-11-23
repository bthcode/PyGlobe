from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
from pyglobe.coord_utils import lla_to_ecef, spherical_to_ecef, get_enu_to_ecef_matrix
from contextlib import contextmanager


@contextmanager
def gl_state_guard():
    """Context manager to save/restore OpenGL state"""
    depth_test = glIsEnabled(GL_DEPTH_TEST)
    lighting = glIsEnabled(GL_LIGHTING)
    texture_2d = glIsEnabled(GL_TEXTURE_2D)
    blend = glIsEnabled(GL_BLEND)
    line_width = glGetFloatv(GL_LINE_WIDTH)
    color = glGetFloatv(GL_CURRENT_COLOR)

    try:
        yield
    finally:
        if depth_test: glEnable(GL_DEPTH_TEST)
        else: glDisable(GL_DEPTH_TEST)
        if lighting: glEnable(GL_LIGHTING)
        else: glDisable(GL_LIGHTING)
        if texture_2d: glEnable(GL_TEXTURE_2D)
        else: glDisable(GL_TEXTURE_2D)
        if blend: glEnable(GL_BLEND)
        else: glDisable(GL_BLEND)
        glLineWidth(float(line_width))
        glColor4fv(color)


# =============================================================================
# Base Scene Object
# =============================================================================

class SceneObject:
    """Base class for all scene objects"""
    
    def draw(self):
        """Draw this object"""
        raise NotImplementedError
    
    def intersect_ray(self, ray_origin, ray_direction):
        """
        Test if ray intersects this object.
        
        Parameters:
            ray_origin: numpy array [x, y, z] in ECEF (meters)
            ray_direction: normalized numpy array [x, y, z] in ECEF
            
        Returns:
            distance (float) if hit, None if miss
        """
        raise NotImplementedError
    
    def on_click(self):
        """Called when object is clicked"""
        print(f"{self.__class__.__name__} clicked")
    
    def __str__(self):
        return self.__class__.__name__


# =============================================================================
# 3D Model with OBJ file
# =============================================================================

class SceneModel(SceneObject):
    """3D model loaded from OBJ file, positioned at lat/lon/alt with ENU orientation"""
    
    def __init__(self, lat_deg, lon_deg, alt_m, scale, obj_path, roll=0, pitch=0, yaw=0, pick_radius=200000):
        super().__init__()
        self.lat = lat_deg
        self.lon = lon_deg
        self.alt = alt_m
        self.scale = scale
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.pick_radius = pick_radius
        
        # Compute ECEF position
        self.ecef_pos = np.array(lla_to_ecef(self.lat, self.lon, self.alt))
        
        # Load mesh
        self.mesh = self._load_obj(obj_path)
    
    def _load_obj(self, obj_path):
        """Load OBJ file"""
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
                        face = []
                        for p in parts[1:]:
                            face.append(int(p.split('/')[0]) - 1)
                        faces.append(face)
            
            print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            return {'vertices': np.array(vertices), 'faces': faces}
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return None
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Ray-sphere intersection for picking"""
        oc = ray_origin - self.ecef_pos
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.pick_radius ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None
    
    def draw(self):
        """Draw the mesh with ENU orientation"""
        # Get ECEF position
        px, py, pz = self.ecef_pos
        
        # Get ENU to ECEF rotation matrix
        R_enu_to_ecef = get_enu_to_ecef_matrix(self.lat, self.lon)
        
        # Create rotation matrix for orientation in ENU frame
        roll_rad = np.radians(self.roll)
        pitch_rad = np.radians(self.pitch)
        yaw_rad = np.radians(self.yaw)
        
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
        
        # Combined rotation
        R_enu = Rz @ Ry @ Rx
        R_total = R_enu_to_ecef @ R_enu
        
        # Create OpenGL transformation matrix
        gl_matrix = np.eye(4)
        gl_matrix[:3, :3] = R_total * self.scale
        gl_matrix[:3, 3] = [px, py, pz]
        
        glDisable(GL_TEXTURE_2D)
        glPushMatrix()
        glMultMatrixf(gl_matrix.T.flatten())
        
        glColor3f(0.8, 0.8, 0.8)
        for face in self.mesh['faces']:
            glBegin(GL_POLYGON)
            for vertex_idx in face:
                v = self.mesh['vertices'][vertex_idx]
                glVertex3f(v[0], v[1], v[2])
            glEnd()
        
        glPopMatrix()
        glEnable(GL_TEXTURE_2D)


# =============================================================================
# Point Marker
# =============================================================================

class PointSceneObject(SceneObject):
    """Single point marker"""
    
    def __init__(self, lat, lon, alt=0.0, color=(1.0, 0.0, 0.0), size=15.0, pick_radius=50000):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.color = color
        self.size = size
        self.pick_radius = pick_radius
        self.xyz = np.array(lla_to_ecef(lat, lon, alt))
    
    def draw(self):
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glColor3f(*self.color)
            glPointSize(self.size)
            
            glBegin(GL_POINTS)
            glVertex3f(*self.xyz)
            glEnd()
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Ray-sphere intersection"""
        oc = ray_origin - self.xyz
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.pick_radius ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None


# =============================================================================
# Polyline (Track)
# =============================================================================

class PolyLineSceneObject(SceneObject):
    """Connected line segments"""
    
    def __init__(self, points_wgs84, color=(1.0, 1.0, 0.0), width=4.0, 
                 altitude_offset=0.0, pick_radius=25000):
        super().__init__()
        self.points_wgs84 = points_wgs84
        self.color = color
        self.width = width
        self.pick_radius = pick_radius
        
        self.points_xyz = [
            np.array(lla_to_ecef(lat, lon, alt + altitude_offset))
            for lat, lon, alt in self.points_wgs84
        ]
    
    def draw(self):
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            
            glColor3f(*self.color)
            glLineWidth(self.width)
            
            glBegin(GL_LINE_STRIP)
            for p in self.points_xyz:
                glVertex3f(*p)
            glEnd()
            
            glDisable(GL_POLYGON_OFFSET_LINE)
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Ray-cylinder intersection for each line segment"""
        min_dist = None
        
        for i in range(len(self.points_xyz) - 1):
            p0 = self.points_xyz[i]
            p1 = self.points_xyz[i + 1]
            
            dist = self._ray_cylinder_intersect(ray_origin, ray_direction, p0, p1, self.pick_radius)
            if dist is not None:
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        
        return min_dist
    
    def _ray_cylinder_intersect(self, ray_origin, ray_dir, cyl_p0, cyl_p1, radius):
        """Intersect ray with cylinder around line segment"""
        cyl_axis = cyl_p1 - cyl_p0
        cyl_len = np.linalg.norm(cyl_axis)
        if cyl_len < 1e-6:
            return None
        cyl_axis = cyl_axis / cyl_len
        
        delta = ray_origin - cyl_p0
        dot_ray_axis = np.dot(ray_dir, cyl_axis)
        dot_delta_axis = np.dot(delta, cyl_axis)
        
        a = 1 - dot_ray_axis ** 2
        b = 2 * (np.dot(delta, ray_dir) - dot_ray_axis * dot_delta_axis)
        c = np.dot(delta, delta) - dot_delta_axis ** 2 - radius ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t < 0:
            t = (-b + np.sqrt(discriminant)) / (2 * a)
        if t < 0:
            return None
        
        hit_point = ray_origin + t * ray_dir
        projection = np.dot(hit_point - cyl_p0, cyl_axis)
        
        if 0 <= projection <= cyl_len:
            return t
        return None


# =============================================================================
# Circle
# =============================================================================

class CircleSceneObject(SceneObject):
    """Circle on the ground"""
    
    def __init__(self, center_lat, center_lon, radius_meters, 
                 color=(0.0, 1.0, 0.0), width=2.0, num_points=64, altitude_offset=10.0):
        super().__init__()
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_meters = radius_meters
        self.color = color
        self.width = width
        self.altitude_offset = altitude_offset
        
        # Generate circle points
        earth_radius = 6371000
        angular_radius = radius_meters / earth_radius
        
        points_wgs84 = []
        for i in range(num_points + 1):
            angle = 2 * np.pi * i / num_points
            dlat = angular_radius * np.cos(angle) * 180 / np.pi
            dlon = angular_radius * np.sin(angle) * 180 / np.pi / np.cos(np.radians(center_lat))
            
            lat = center_lat + dlat
            lon = center_lon + dlon
            points_wgs84.append((lat, lon, 0.0))
        
        self.points_xyz = [
            np.array(lla_to_ecef(lat, lon, alt + altitude_offset))
            for lat, lon, alt in points_wgs84
        ]
        
        self.center_xyz = np.array(lla_to_ecef(center_lat, center_lon, altitude_offset))
    
    def draw(self):
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            
            glColor3f(*self.color)
            glLineWidth(self.width)
            
            glBegin(GL_LINE_LOOP)
            for p in self.points_xyz:
                glVertex3f(*p)
            glEnd()
            
            glDisable(GL_POLYGON_OFFSET_LINE)
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Approximate as sphere for picking"""
        oc = ray_origin - self.center_xyz
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - (self.radius_meters * 1.2) ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None


# =============================================================================
# Polygon
# =============================================================================

class PolygonSceneObject(SceneObject):
    """Polygon on the ground defined by corner points"""
    
    def __init__(self, points_wgs84, color=(1.0, 0.5, 0.0), fill_color=None, 
                 width=2.0, altitude_offset=10.0, alpha=0.5):
        super().__init__()
        self.points_wgs84 = points_wgs84
        self.color = color
        self.fill_color = fill_color if fill_color else (*color, alpha)
        self.width = width
        self.altitude_offset = altitude_offset
        
        self.points_xyz = [
            np.array(lla_to_ecef(lat, lon, alt + altitude_offset))
            for lat, lon, alt in self.points_wgs84
        ]
        
        self.center_xyz = np.mean(self.points_xyz, axis=0)
        self.bounding_radius = max([np.linalg.norm(p - self.center_xyz) for p in self.points_xyz])
    
    def draw(self):
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)
            
            # Draw filled polygon
            if self.fill_color:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(*self.fill_color)
                
                glBegin(GL_POLYGON)
                for p in self.points_xyz:
                    glVertex3f(*p)
                glEnd()
                
                glDisable(GL_BLEND)
            
            # Draw outline
            glColor3f(*self.color)
            glLineWidth(self.width)
            
            glBegin(GL_LINE_LOOP)
            for p in self.points_xyz:
                glVertex3f(*p)
            glEnd()
            
            glDisable(GL_POLYGON_OFFSET_FILL)
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Approximate as sphere for picking"""
        oc = ray_origin - self.center_xyz
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.bounding_radius ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None


# =============================================================================
# Image Overlay
# =============================================================================

class ImageOverlaySceneObject(SceneObject):
    """Textured image on the ground with 4 corner points"""
    
    def __init__(self, corners_wgs84, image_path, altitude_offset=10.0, alpha=1.0):
        """
        corners_wgs84: list of 4 tuples [(lat, lon, alt), ...] in order: 
                       bottom-left, bottom-right, top-right, top-left
        """
        super().__init__()
        self.corners_wgs84 = corners_wgs84
        self.image_path = image_path
        self.altitude_offset = altitude_offset
        self.alpha = alpha
        self.texture_id = None
        
        self.corners_xyz = [
            np.array(lla_to_ecef(lat, lon, alt + altitude_offset))
            for lat, lon, alt in self.corners_wgs84
        ]
        
        self.center_xyz = np.mean(self.corners_xyz, axis=0)
        self.bounding_radius = max([np.linalg.norm(p - self.center_xyz) for p in self.corners_xyz])
        
        #self._load_texture()
    
    def _load_texture(self):
        """Load image as OpenGL texture"""
        if not os.path.exists(self.image_path):
            print(f"Warning: Image not found: {self.image_path}")
            return
        
        from PySide6.QtGui import QImage
        
        image = QImage(self.image_path)
        if image.isNull():
            print(f"Warning: Failed to load image: {self.image_path}")
            return
        
        image = image.convertToFormat(QImage.Format_RGBA8888)
        image = image.mirrored()
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(),
                    0, GL_RGBA, GL_UNSIGNED_BYTE, image.bits())
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def draw(self):
        if not self.texture_id:
            self._load_texture()
        if not self.texture_id:
            return
        
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(1.0, 1.0, 1.0, self.alpha)
            
            # Draw textured quad (BL, BR, TR, TL)
            tex_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
            
            glBegin(GL_QUADS)
            for i in range(4):
                glTexCoord2f(*tex_coords[i])
                glVertex3f(*self.corners_xyz[i])
            glEnd()
            
            glDisable(GL_POLYGON_OFFSET_FILL)
            glBindTexture(GL_TEXTURE_2D, 0)
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Approximate as sphere for picking"""
        oc = ray_origin - self.center_xyz
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.bounding_radius ** 2
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t >= 0 else None


# =============================================================================
# Scene Container
# =============================================================================

class Scene:
    """Container for all scene objects"""
    
    def __init__(self):
        self.objects = []
    
    def add(self, obj):
        """Add an object to the scene"""
        self.objects.append(obj)
    
    def remove(self, obj):
        """Remove an object from the scene"""
        if obj in self.objects:
            self.objects.remove(obj)
    
    def clear(self):
        """Remove all objects"""
        self.objects.clear()
    
    def draw(self):
        """Draw all objects"""
        for obj in self.objects:
            obj.draw()
    
    def pick(self, ray_origin, ray_direction):
        """
        Find closest object hit by ray.
        
        Returns:
            (object, distance) if hit, (None, None) if no hit
        """
        closest_obj = None
        closest_dist = float('inf')
        
        for obj in self.objects:
            try:
                dist = obj.intersect_ray(ray_origin, ray_direction)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    closest_obj = obj
            except NotImplementedError:
                continue
        
        return (closest_obj, closest_dist) if closest_obj else (None, None)

# TEST 
def add_test_objects(scene, satellite_obj_path="assets/satellite/satellite.obj",
                     image_path="sar.jpeg"):
    """
    Add one of each scene object type for testing.

    Parameters:
        scene: Scene instance to add objects to
        satellite_obj_path: Path to OBJ file for SceneModel
        image_path: Path to image file for ImageOverlaySceneObject
    """

    # 1. Point - Red marker over New York
    point = PointSceneObject(
        lat=40.7128,
        lon=-74.0060,
        alt=0,
        color=(1.0, 0.0, 0.0),  # Red
        size=20.0,
        pick_radius=50000
    )
    scene.add(point)
    print("Added: Red point over New York")

    # 2. Polyline - Track from Boston to Washington DC
    track_points = [
        (42.3601, -71.0589, 0),      # Boston
        (40.7128, -74.0060, 0),      # New York
        (39.9526, -75.1652, 0),      # Philadelphia
        (38.9072, -77.0369, 0)       # Washington DC
    ]
    track = PolyLineSceneObject(
        points_wgs84=track_points,
        color=(1.0, 1.0, 0.0),  # Yellow
        width=6.0,
        altitude_offset=0,
        pick_radius=25000
    )
    scene.add(track)
    print("Added: Yellow track (Boston to DC)")

    # 3. Circle - 200km radius around Denver
    circle = CircleSceneObject(
        center_lat=39.7392,
        center_lon=-104.9903,
        radius_meters=200000,  # 200 km
        color=(0.0, 1.0, 0.0),  # Green
        width=3.0,
        altitude_offset=10.0
    )
    scene.add(circle)
    print("Added: Green circle around Denver (200km radius)")

    # 4. Polygon - Square over Texas
    polygon_points = [
        (30.0, -100.0, 0),  # Southwest corner
        (30.0, -95.0, 0),   # Southeast corner
        (35.0, -95.0, 0),   # Northeast corner
        (35.0, -100.0, 0)   # Northwest corner
    ]
    polygon = PolygonSceneObject(
        points_wgs84=polygon_points,
        color=(1.0, 0.5, 0.0),  # Orange outline
        fill_color=(1.0, 0.5, 0.0, 0.3),  # Semi-transparent orange fill
        width=3.0,
        altitude_offset=10.0
    )
    scene.add(polygon)
    print("Added: Orange polygon over Texas")

    if 1:
        # 5. Image Overlay - SAR image over California
        if os.path.exists(image_path):
            image_corners = [
                (34.0, -120.0, 0),  # Bottom-left
                (34.0, -118.0, 0),  # Bottom-right
                (36.0, -118.0, 0),  # Top-right
                (36.0, -120.0, 0)   # Top-left
            ]
            image_overlay = ImageOverlaySceneObject(
                corners_wgs84=image_corners,
                image_path=image_path,
                altitude_offset=10.0,
                alpha=0.7
            )
            scene.add(image_overlay)
            print(f"Added: Image overlay over California ({image_path})")
        else:
            print(f"Warning: Image not found: {image_path}")

    # 6. SceneModel - Satellite over Miami
    if os.path.exists(satellite_obj_path):
        satellite = SceneModel(
            lat_deg=25.7617,
            lon_deg=-80.1918,
            alt_m=500000,  # 500 km altitude
            scale=50000,   # 50 km apparent size
            obj_path=satellite_obj_path,
            roll=0,
            pitch=0,
            yaw=0,
            pick_radius=200000
        )
        scene.add(satellite)
        print(f"Added: Satellite over Miami at 500km altitude")
    else:
        print(f"Warning: OBJ file not found: {satellite_obj_path}")

    print(f"\nTotal objects in scene: {len(scene.objects)}")
    print("All objects are pickable - click on them to test!")
