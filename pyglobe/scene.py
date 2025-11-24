from OpenGL.GL import *
from OpenGL.GLU import *
from PySide6.QtCore import QObject, Signal
import numpy as np
import os
from pyglobe.coord_utils import lla_to_ecef, get_enu_to_ecef_matrix
from contextlib import contextmanager


@contextmanager
def gl_state_guard() -> None:
    """Context manager to save/restore OpenGL state when drawing scene objects"""
    depth_test = glIsEnabled(GL_DEPTH_TEST)
    lighting = glIsEnabled(GL_LIGHTING)
    texture_2d = glIsEnabled(GL_TEXTURE_2D)
    blend = glIsEnabled(GL_BLEND)
    line_width = glGetFloatv(GL_LINE_WIDTH)
    color = glGetFloatv(GL_CURRENT_COLOR)

    try:
        yield
    finally:
        if depth_test: 
            glEnable(GL_DEPTH_TEST)
        else: 
            glDisable(GL_DEPTH_TEST)
        if lighting: 
            glEnable(GL_LIGHTING)
        else: 
            glDisable(GL_LIGHTING)
        if texture_2d: 
            glEnable(GL_TEXTURE_2D)
        else: 
            glDisable(GL_TEXTURE_2D)
        if blend: 
            glEnable(GL_BLEND)
        else: 
            glDisable(GL_BLEND)
        glLineWidth(float(line_width))
        glColor4fv(color)


class SceneObject:
    """Base class for all scene objects
    
    Attributes
    ----------
    label : str
        Name for this object

    """

    def __init__(self, label):
        self.label = label
    
    def draw(self) -> None:
        """Draw this object"""
        raise NotImplementedError
    
    def intersect_ray(self, ray_origin: np.ndarray, ray_direction:np.ndarray) -> float | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """
        raise NotImplementedError
    
    def __str__(self):
        return f'{self.__class__.__name__} : {self.label}'


# =============================================================================
# 3D Model with OBJ file
# =============================================================================

class SceneModel(SceneObject):
    """3D model loaded from OBJ file, positioned at lat/lon/alt with ENU orientation"""
    
    def __init__(self, label : str, lat_deg : float, lon_deg: float, alt_m: float, 
                 scale: float|int, obj_path:str, roll:float=0, pitch:float=0, yaw:float=0, 
                 pick_radius:float|int=200000):
        '''
        Parameters
        ----------
        label : str
            name for this object
        lat_deg : float
            WGS84 latitude
        lon_deg : float
            WGS84 longitude
        alt_m : float
            WGS84 altitude
        scale : float
            scaling factor visualization size
        obj_path : str
            Path to mesh .obj file
        roll : float 
            Degrees of rotation around east in local ENU
        pitch : float
            Degrees of rotation around north in local ENU
        yaw : float
            Degrees of rotation around up in local ENU
        pick_radius: float
            Distance around object that will return a successful mouse click
        '''
        super().__init__(label)
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
    
    def _load_obj(self, obj_path:str) ->None:
        """Load OBJ file
        
        Parameters
        ----------
        obj_path : str
            Path to .obj file
        """
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
    
    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray) -> float | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """

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
    
    def __init__(self, label:str, lat:float, lon:float, alt:float=0.0, 
                 color=(1.0, 0.0, 0.0), size:float=15.0, pick_radius=50000):
        '''
        Parameters
        ----------
        label : str
            Name of this object
        lat : float
            Latitude WGS84 degrees
        lon : float
            Longitude WGS84 degrees
        alt : float
            Altitude WGS84 meters
        color : [float,float,float]
            Color (0:1, 0:1, 0:1)
        size : float
            Size to draw the point
        pick_radius : number
            Meters within which a click is 'successful'
        '''

        super().__init__(label)
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.color = color
        self.size = size
        self.pick_radius = pick_radius
        self.xyz = np.array(lla_to_ecef(lat, lon, alt))
    
    def draw(self) -> None:
        '''Drow this object in OpenGL'''
        with gl_state_guard():
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glColor3f(*self.color)
            glPointSize(self.size)
            
            glBegin(GL_POINTS)
            glVertex3f(*self.xyz)
            glEnd()
    
    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray) -> float | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """

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
    
    def __init__(self, label:str, points_wgs84, color=(1.0, 1.0, 0.0), width:float=4.0, 
                 altitude_offset:float=0.0, pick_radius=25000):
        '''
        Parameters
        ----------
        label : str
            Name of this object
        points_wgs84 : iterable[float]
            Set of points making up this polygon
        color : [float,float,float]
            Color of outline of circle
        fill_color : None | [float,float,float]
            Color of center of circle (optional)
        width : float
            Width of outline of circcle
        altitude_offset : float
            Distance above the surface of the earth to draw this object
        pick_radius : int | float
            Area around object that is considered a successful 'click' on it
        '''

        super().__init__(label)
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
    
    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray)->float | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """
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
    
    def __init__(self, label:str, center_lat:float, center_lon:float, radius_meters:float, 
                 color=(0.0, 1.0, 0.0), fill_color=None, width:float=2.0, num_points:int=64, 
                 altitude_offset:float=10.0):
        '''
        Parameters
        ----------
        label : str
            Name of this object
        center_lat : float
            Latitude of circle center in WGS84 Degrees
        center_lon : float
            Longitude of circle center in WGS84 Degrees
        radius_meters : float
            Radius of circle in meters
        color : [float,float,float]
            Color of outline of circle
        fill_color : None | [float,float,float]
            Color of center of circle (optional)
        width : float
            Width of outline of circcle
        num_points : int
            Number of points to use for approximating the circle
        altitude_offset : float
            Distance above the surface of the earth to draw this object
        '''
        super().__init__(float)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_meters = radius_meters
        self.color = color
        self.width = width
        self.altitude_offset = altitude_offset
        self.fill_color=fill_color
        
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

            
            glColor3f(*self.color)
            glLineWidth(self.width)
            
            glBegin(GL_LINE_LOOP)
            for p in self.points_xyz:
                glVertex3f(*p)
            glEnd()
            
            glDisable(GL_POLYGON_OFFSET_LINE)
    
    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray) -> np.ndarray | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """

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
    
    def __init__(self, label:str, points_wgs84, color:[float,float,float]=(1.0, 0.5, 0.0), 
                 fill_color=None, 
                 width:float=2.0, altitude_offset:float=10.0, alpha:float=0.5):
        '''
        Parameters
        ----------
        label : str
            Name of this object
        points_wgs84 : iterable[float]
            Set of points making up this polygon
        color : [float,float,float]
            Color of outline of circle
        fill_color : None | [float,float,float]
            Color of center of circle (optional)
        width : float
            Width of outline of circcle
        altitude_offset : float
            Distance above the surface of the earth to draw this object
        alpha : float
            Transparency of this polygon (0=totally transparent, 1=not transparent)
        '''
        super().__init__(label)
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
    
    def draw(self) -> None:
        '''OpenGL Drawing Function for this object'''
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

    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray) -> np.ndarray | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """
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
    
    def __init__(self, label : str, corners_wgs84, image_path:str, altitude_offset:float=10.0, alpha:float=1.0):
        """
        label : str
            name for this object
        corners_wgs84: tuple
            list of 4 tuples [(lat, lon, alt), ...] in order: 
            bottom-left, bottom-right, top-right, top-left
        image_path : str
            path to image to load
        altitude_offset : float
            distance above the earth's surface to draw image
        alpha : float
            transparancy to apply.  0 = fully transparnt, 1=not transparent
        """
        super().__init__(label)
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
    
    def draw(self) -> None:
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

    def intersect_ray(self, ray_origin:np.ndarray, ray_direction:np.ndarray) -> np.ndarray | None:
        """
        Test if ray intersects this object.
        
        Parameters
        ----------
            ray_origin: np.ndarray
                numpy array [x, y, z] in ECEF (meters)
            ray_direction: np.ndarray
                normalized numpy array [x, y, z] in ECEF
            
        Returns
        -------
            distance: float | None
                if hit, None if miss
        """
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

class Scene(QObject):
    """Container for all scene objects"""

    sigClicked = Signal(SceneObject)
    
    def __init__(self):
        super().__init__()
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
        
        if closest_obj is not None:
            self.sigClicked.emit(closest_obj)

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
        'Example Point',
        lat=42.3601,
        lon=-71.0589,
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
        'Example Track',
        points_wgs84=track_points,
        color=(0.0, 0.0, 0.0),  # Yellow
        width=8.0,
        altitude_offset=2000, # earth curvature
        pick_radius=25000
    )
    scene.add(track)
    print("Added: Yellow track (Boston to DC)")

    # 3. Circle - 200km radius around Denver
    circle = CircleSceneObject(
        'Example Circle',
        center_lat=39.7392,
        center_lon=-104.9903,
        radius_meters=100000,
        color=(0.0, 0.0, 0.0),
        fill_color=(0.0, 0.5, 0.0, 0.3),
        width=3.0,
        altitude_offset=2000 # Earth curvature
    )
    scene.add(circle)
    print("Added: Green circle around Denver (200km radius)")

    # 4. Polygon - Square over Texas
    polygon_points = [
        (33.0, -96.0, 0),  # Southwest corner
        (33.0, -95.0, 0),   # Southeast corner
        (34.0, -95.0, 0),   # Northeast corner
        (34.0, -96.0, 0)   # Northwest corner
    ]
    polygon = PolygonSceneObject(
        'Example Polygon',
        points_wgs84=polygon_points,
        color=(1.0, 0.5, 0.0),  # Orange outline
        fill_color=(1.0, 0.5, 0.0, 0.3),  # Semi-transparent orange fill
        width=3.0,
        altitude_offset=2000.0 # Account for earth curvature
    )
    scene.add(polygon)
    print("Added: Orange polygon over Texas")

    # 5. Image Overlay - SAR image over California
    if os.path.exists(image_path):
        image_corners = [
            (34.0, -120.0, 0),  # Bottom-left
            (34.0, -118.0, 0),  # Bottom-right
            (36.0, -118.0, 0),  # Top-right
            (36.0, -120.0, 0)   # Top-left
        ]
        image_overlay = ImageOverlaySceneObject(
            'Example Image',
            corners_wgs84=image_corners,
            image_path=image_path,
            altitude_offset=2000.0, # Account for earth curvature
            alpha=0.7
        )
        scene.add(image_overlay)
        print(f"Added: Image overlay over California ({image_path})")
    else:
        print(f"Warning: Image not found: {image_path}")

    # 6. SceneModel - Satellite over Miami
    if os.path.exists(satellite_obj_path):
        satellite = SceneModel(
            'Example Mesh Object',
            lat_deg=25.7617,
            lon_deg=-80.1918,
            alt_m=500000,  # 500 km altitude
            scale=200000,   # 50 km apparent size
            obj_path=satellite_obj_path,
            roll=0,
            pitch=0,
            yaw=-90,
            pick_radius=200000
        )
        scene.add(satellite)
        print(f"Added: Satellite over Miami at 500km altitude")
    else:
        print(f"Warning: OBJ file not found: {satellite_obj_path}")

    print(f"\nTotal objects in scene: {len(scene.objects)}")
    print("All objects are pickable - click on them to test!")
