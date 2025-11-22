import sys
import os
import numpy as np
import threading
import time
from collections import OrderedDict
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, Signal, QThread, Slot
from PySide6.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GLU import *

# this project packages 
from tile_fetcher import TileFetcher

MIN_Z = 2
MAX_Z = 9    # set to highest zoom level you have in cache
DOWNLOAD_TIMEOUT = 10
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # common XYZ server (y top origin)
CACHE_ROOT= "./cache"
USER_AGENT = "PyGlobe Example/1.0 (your_email@example.com)"  # set a sensible UA
TILE_SIZE = 512
MIN_Z = 2
MAX_Z = 9    # set to highest zoom level you have in cache
MAX_GPU_TEXTURES = 512

class GlobeWidget(QOpenGLWidget):
    infoSig = Signal(dict)
    requestTile = Signal(int, int, int, str)
    setAimpoint = Signal(int, int, int)
    resetFetcher = Signal()
    sigShutdownFetcher = Signal()
    sigStartFetcher = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.debug = False
        self.camera_distance = 15000000  # 15,000 km from center
        self.camera_lon = 0.0  # degrees
        self.camera_lat = 0.0  # degrees
        self.last_pos = None
        
        # Earth radius in meters
        self.earth_radius = 6371000
        
        # Tile textures
        #self.tile_textures = {}
        #self.tile_cache_path = "cache/3"

        # Handles to opengl textures
        self.textures = OrderedDict() 
        # Handles to opengl texture (kept separate so they won't be deleted)
        self.base_textures = OrderedDict() 


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
        self.sigShutdownFetcher.connect(self.fetcher.shutdown)
        self.sigStartFetcher.connect(self.fetcher.start)
        self.fetcher_thread = QThread()
        self.fetcher.moveToThread(self.fetcher_thread)
        self.fetcher_thread.start()

        # Fetcher Control
        self.requestTile.connect(self.fetcher.requestTile)
        self.setAimpoint.connect(self.fetcher.setAimpoint)
        self.resetFetcher.connect(self.fetcher.reset)
        self.fetcher.tileReady.connect(self.onTileReady)
        self.sigStartFetcher.emit()

        #----------------------------
        # Load base layer
        #----------------------------
        self.load_base_textures()

        
        # Satellite mesh
        self.satellite_mesh = None
        self.satellite_obj_path = "assets/satellite/satellite.obj"
        
        # Satellite state (for click detection)
        self.satellite_lat = 42.0
        self.satellite_lon = -71.0
        self.satellite_alt = 500000
        self.satellite_roll = 0
        self.satellite_pitch = 0
        self.satellite_yaw = 0
        
        # Debug ray casting
        self.debug_ray_origin = None
        self.debug_ray_end = None

    def shutdownFetcher(self):
        '''Shut Down Tile Fetching Thread'''
        self.sigShutdownFetcher.emit()
        time.sleep(1)
        self.fetcher_thread.quit()
        self.fetcher_thread.wait()

    def closeEvent(self, ev)->None:
        event.accept()
        super().closeEvent(ev)

        
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_CULL_FACE)  # Disable culling to see all tiles
        
        # Light position (stationary in ECEF)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        glClearColor(0.0, 0.0, 0.1, 1.0)
        
        # Load OSM tiles
        self.load_tiles()
        
        # Load satellite mesh
        self.load_satellite_mesh()
        
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 100000, 50000000)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Calculate camera position in ECEF (camera_distance is from center)
        cam_x, cam_y, cam_z = self.spherical_to_ecef(
            self.camera_lat, self.camera_lon, self.camera_distance
        )
        
        # Look at origin (Earth center)
        gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                  0, 0, 0,                # Look at origin
                  0, 0, 1)                # Up vector (Z-axis)
        
        # Draw Earth at origin
        self.draw_earth()
        
        # Draw coordinate axes
        if self.debug:
            self.draw_axes()
        
            # Debug: Draw ENU frame at a test location
            self.draw_coordinate_frame(42.0, -71.0, 2000000)  # Boston area, 2000km altitude
        
        # Draw satellite with orientation in local ENU
        # Orientation: (roll, pitch, yaw) in degrees in local ENU frame
        self.draw_satellite(self.satellite_lat, self.satellite_lon, self.satellite_alt, 
                          roll=self.satellite_roll, pitch=self.satellite_pitch, yaw=self.satellite_yaw)
        
        if self.debug:
            # Draw debug ray if available
            if self.debug_ray_origin is not None and self.debug_ray_end is not None:
                self.draw_debug_ray()
        
    def draw_earth(self):
        glPushMatrix()
        glColor3f(1.0, 1.0, 1.0)  # White to show texture colors
        
        # Draw tiles
        # At zoom level 3, we have 8x8 tiles
        for x in range(8):
            for y in range(8):
                if (x, y) in self.textures:
                    self.draw_tile(x, y)
        glPopMatrix()
        
    def draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        axis_length = self.earth_radius * 1.5
        
        glBegin(GL_LINES)
        # X-axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y-axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        # Z-axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        glEnd()
        
        glEnable(GL_LIGHTING)

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



    
    ##def load_tiles(self):
    ##    """Load OSM tiles from cache"""
    ##    if not os.path.exists(self.tile_cache_path):
    ##        print(f"Warning: Tile cache path '{self.tile_cache_path}' not found")
    ##        return
    ##    
    ##    # Level 3 has 8x8 tiles
    ##    for x in range(8):
    ##        x_path = os.path.join(self.tile_cache_path, str(x))
    ##        if not os.path.exists(x_path):
    ##            continue
    ##            
    ##        for y in range(8):
    ##            tile_path = os.path.join(x_path, f"{y}.png")
    ##            if os.path.exists(tile_path):
    ##                texture_id = self.load_texture(tile_path)
    ##                if texture_id:
    ##                    self.tile_textures[(x, y)] = texture_id
    ##    
    ##    print(f"Loaded {len(self.tile_textures)} tiles")
    
    def load_texture(self, filepath):
        """Load a PNG texture"""
        image = QImage(filepath)
        if image.isNull():
            return None
        
        image = image.convertToFormat(QImage.Format_RGBA8888)
        image = image.mirrored()  # Flip vertically for OpenGL
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, image.bits())
        
        return texture_id
    
    def draw_tile(self, x, y):
        """Draw a single tile on the sphere"""
        texture_id = self.tile_textures.get((x, y))
        if not texture_id:
            return
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Calculate tile bounds in spherical coordinates
        # OSM tiles use Web Mercator, but we'll approximate with lat/lon
        n = 2 ** 3  # zoom level 3
        
        # Longitude is linear
        lon_min = (x / n) * 360.0 - 180.0
        lon_max = ((x + 1) / n) * 360.0 - 180.0
        
        # Latitude needs inverse Web Mercator calculation
        lat_max = self.tile_y_to_lat(y, 3)
        lat_min = self.tile_y_to_lat(y + 1, 3)
        
        # Draw tile as a quad mesh on the sphere
        steps = 10  # subdivisions for better sphere approximation
        
        glBegin(GL_QUADS)
        for i in range(steps):
            for j in range(steps):
                # Calculate lat/lon for this quad
                lat0 = lat_min + (lat_max - lat_min) * i / steps
                lat1 = lat_min + (lat_max - lat_min) * (i + 1) / steps
                lon0 = lon_min + (lon_max - lon_min) * j / steps
                lon1 = lon_min + (lon_max - lon_min) * (j + 1) / steps
                
                # Texture coordinates
                s0 = j / steps
                s1 = (j + 1) / steps
                t0 = i / steps
                t1 = (i + 1) / steps
                
                # Four corners of the quad
                vertices = [
                    (lat0, lon0, s0, t0),
                    (lat0, lon1, s1, t0),
                    (lat1, lon1, s1, t1),
                    (lat1, lon0, s0, t1)
                ]
                
                for lat, lon, s, t in vertices:
                    px, py, pz = self.lla_to_ecef(lat, lon, 0)  # Tiles on surface (alt=0)
                    # Normal points outward from sphere center
                    nx, ny, nz = px / self.earth_radius, py / self.earth_radius, pz / self.earth_radius
                    glTexCoord2f(s, t)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(px, py, pz)
        
        glEnd()
    
    def tile_y_to_lat(self, y, zoom):
        """Convert OSM tile Y coordinate to latitude"""
        n = 2 ** zoom
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        return np.degrees(lat_rad)
    
    def draw_coordinate_frame(self, lat, lon, alt):
        """Draw ENU (East-North-Up) coordinate frame at specified location"""
        # Get ECEF position
        px, py, pz = self.lla_to_ecef(lat, lon, alt)
        
        # Calculate ENU basis vectors
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # East vector (tangent to latitude circle, pointing east)
        east = np.array([
            -np.sin(lon_rad),
            np.cos(lon_rad),
            0
        ])
        
        # North vector (tangent to meridian, pointing north)
        north = np.array([
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad)
        ])
        
        # Up vector (radial, pointing away from Earth center)
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])
        
        # Normalize (should already be normalized, but just to be safe)
        east = east / np.linalg.norm(east)
        north = north / np.linalg.norm(north)
        up = up / np.linalg.norm(up)
        
        # Scale for visibility
        axis_length = 500000  # 500 km
        
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        
        # East axis (Red)
        glColor3f(1, 0, 0)
        glVertex3f(px, py, pz)
        glVertex3f(px + east[0] * axis_length, 
                   py + east[1] * axis_length, 
                   pz + east[2] * axis_length)
        
        # North axis (Green)
        glColor3f(0, 1, 0)
        glVertex3f(px, py, pz)
        glVertex3f(px + north[0] * axis_length, 
                   py + north[1] * axis_length, 
                   pz + north[2] * axis_length)
        
        # Up axis (Blue)
        glColor3f(0, 0, 1)
        glVertex3f(px, py, pz)
        glVertex3f(px + up[0] * axis_length, 
                   py + up[1] * axis_length, 
                   pz + up[2] * axis_length)
        
        glEnd()
        
        # Draw a small sphere at the origin point
        glColor3f(1, 1, 0)  # Yellow
        glPushMatrix()
        glTranslatef(px, py, pz)
        quadric = gluNewQuadric()
        gluSphere(quadric, 50000, 10, 10)  # 50km radius sphere
        gluDeleteQuadric(quadric)
        glPopMatrix()
        
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
    
    def get_enu_to_ecef_matrix(self, lat, lon):
        """Get rotation matrix from local ENU to ECEF frame"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # ENU basis vectors in ECEF
        east = np.array([
            -np.sin(lon_rad),
            np.cos(lon_rad),
            0
        ])
        
        north = np.array([
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad)
        ])
        
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])
        
        # Rotation matrix: columns are ENU basis vectors in ECEF
        R = np.column_stack([east, north, up])
        return R
    
    def load_satellite_mesh(self):
        """Load satellite mesh from OBJ file"""
        if not os.path.exists(self.satellite_obj_path):
            print(f"Warning: Satellite mesh '{self.satellite_obj_path}' not found")
            return
        
        vertices = []
        faces = []
        
        try:
            with open(self.satellite_obj_path, 'r') as f:
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
            
            self.satellite_mesh = {
                'vertices': np.array(vertices),
                'faces': faces
            }
            print(f"Loaded satellite mesh: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"Error loading satellite mesh: {e}")
    
    def draw_satellite(self, lat, lon, alt, roll=0, pitch=0, yaw=0):
        """Draw satellite at specified location with ENU orientation
        roll, pitch, yaw are in degrees in local ENU frame
        roll: rotation about east axis
        pitch: rotation about north axis  
        yaw: rotation about up axis
        """
        if self.satellite_mesh is None:
            # Draw a simple sphere as fallback
            px, py, pz = self.lla_to_ecef(lat, lon, alt)
            glDisable(GL_TEXTURE_2D)
            glColor3f(1, 0, 0)  # Red sphere if no mesh
            glPushMatrix()
            glTranslatef(px, py, pz)
            quadric = gluNewQuadric()
            gluSphere(quadric, 100000, 16, 16)  # 100km radius
            gluDeleteQuadric(quadric)
            glPopMatrix()
            glEnable(GL_TEXTURE_2D)
            return
        
        # Get ECEF position
        px, py, pz = self.lla_to_ecef(lat, lon, alt)
        
        # Get ENU to ECEF rotation matrix
        R_enu_to_ecef = self.get_enu_to_ecef_matrix(lat, lon)
        
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
        scale = 50000  # Scale to ~50km size
        
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
        
        for face in self.satellite_mesh['faces']:
            glBegin(GL_POLYGON)
            for vertex_idx in face:
                v = self.satellite_mesh['vertices'][vertex_idx]
                glVertex3f(v[0], v[1], v[2])
            glEnd()
        
        glPopMatrix()
        glEnable(GL_TEXTURE_2D)
        
    def lla_to_ecef(self, lat, lon, alt):
        """Convert latitude, longitude, altitude to ECEF coordinates
        alt is height above Earth surface (not distance from center)"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Total distance from Earth center = radius + altitude
        r = self.earth_radius + alt
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return x, y, z
    
    def spherical_to_ecef(self, lat, lon, r):
        """Convert spherical coordinates to ECEF
        r is distance from Earth center (for camera positioning)"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return x, y, z
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Force a render to ensure matrices are current
            self.makeCurrent()
            # Start dragging camera
            self.last_pos = event.pos()
            
        elif event.button() == Qt.RightButton:
            # Check if satellite was clicked
            if self.check_satellite_click(event.pos().x(), event.pos().y()):
                print(f"✓✓✓ SATELLITE CLICKED! ✓✓✓")
            
    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            self.last_pos = event.pos()
            return
            
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.camera_lon -= dx * 0.5
            self.camera_lat = np.clip(self.camera_lat + dy * 0.5, -89, 89)
            self.update()
            
        self.last_pos = event.pos()
        
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:  
            return
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.camera_distance *= zoom_factor
        self.camera_distance = np.clip(
            self.camera_distance, 
            self.earth_radius * 1.1, 
            self.earth_radius * 10
        )
        self.update()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.auto_rotate = not self.auto_rotate
        elif event.key() == Qt.Key_R:
            self.camera_lon = 0
            self.camera_lat = 0
            self.camera_distance = 15000000
            self.auto_rotate = True
            self.update()
    
    def check_satellite_click(self, mouse_x, mouse_y):
        """Check if mouse click intersects with satellite using ray casting"""
        print(f"\n{'='*60}")
        print(f"CLICK DEBUG - STEP BY STEP")
        print(f"{'='*60}")
        
        # Handle high DPI scaling
        dpr = self.devicePixelRatio()
        mx = mouse_x * dpr
        my = mouse_y * dpr
        
        # Get satellite position in ECEF
        sat_x, sat_y, sat_z = self.lla_to_ecef(
            self.satellite_lat, self.satellite_lon, self.satellite_alt
        )
        sat_pos = np.array([sat_x, sat_y, sat_z])
        
        # Get camera position
        cam_x, cam_y, cam_z = self.spherical_to_ecef(
            self.camera_lat, self.camera_lon, self.camera_distance
        )
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        print(f"\n1. POSITIONS")
        print(f"   Mouse: ({mouse_x}, {mouse_y})")
        print(f"   Camera: ({cam_x/1e6:.2f}, {cam_y/1e6:.2f}, {cam_z/1e6:.2f}) Mm")
        print(f"   Satellite: ({sat_x/1e6:.2f}, {sat_y/1e6:.2f}, {sat_z/1e6:.2f}) Mm")
        print(f"   Distance: {np.linalg.norm(sat_pos - cam_pos)/1e6:.2f} Mm")
        
        # Get viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = viewport[2]
        height = viewport[3]
        
        # Convert to NDC
        #ndc_x = (2.0 * mouse_x) / width - 1.0
        #ndc_y = 1.0 - (2.0 * mouse_y) / height
        # Convert to NDC using PHYSICAL coordinates
        ndc_x = (2.0 * mx) / width - 1.0
        ndc_y = 1.0 - (2.0 * my) / height
        
        print(f"\n2. NORMALIZED DEVICE COORDS")
        print(f"   Viewport: {width} x {height}")
        print(f"   Mouse: ({mouse_x}, {mouse_y})")
        print(f"   Center would be: ({width/2:.1f}, {height/2:.1f})")
        print(f"   NDC: ({ndc_x:.3f}, {ndc_y:.3f}) - center is (0, 0)")
        
        # Camera basis vectors (matching gluLookAt behavior)
        # gluLookAt(eye, center, up) builds a camera matrix
        # Forward: from eye toward center
        center = np.array([0.0, 0.0, 0.0])
        eye = cam_pos
        world_up = np.array([0.0, 0.0, 1.0])
        
        forward = center - eye  # Direction from eye to center
        forward = forward / np.linalg.norm(forward)
        
        # Right: cross product of forward and up
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)
        
        # Up: cross product of right and forward (to make orthonormal)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        print(f"\n3. CAMERA BASIS VECTORS")
        print(f"   Forward: ({forward[0]:7.4f}, {forward[1]:7.4f}, {forward[2]:7.4f})")
        print(f"   Right:   ({right[0]:7.4f}, {right[1]:7.4f}, {right[2]:7.4f})")
        print(f"   Up:      ({up[0]:7.4f}, {up[1]:7.4f}, {up[2]:7.4f})")
        
        # Build ray in camera space
        fov = 45.0
        aspect = width / height if height > 0 else 1
        fov_rad = np.radians(fov)
        tan_half_fov = np.tan(fov_rad / 2.0)
        
        # Camera space: +X=right, +Y=up, -Z=forward
        ray_cam_x = ndc_x * aspect * tan_half_fov
        ray_cam_y = ndc_y * tan_half_fov
        ray_cam_z = -1.0
        
        # DON'T normalize yet!
        ray_dir_cam = np.array([ray_cam_x, ray_cam_y, ray_cam_z])
        
        print(f"\n4. RAY IN CAMERA SPACE")
        print(f"   ray_dir_cam (unnormalized): ({ray_dir_cam[0]:7.4f}, {ray_dir_cam[1]:7.4f}, {ray_dir_cam[2]:7.4f})")
        
        # Transform to world space
        # In camera space: -Z is forward
        # ray_dir_cam = (x, y, -1) should map to: x*right + y*up + forward
        # Since ray_dir_cam[2] = -1, we need to negate it: -ray_dir_cam[2] = 1
        ray_dir_world = (ray_dir_cam[0] * right + 
                        ray_dir_cam[1] * up + 
                        (-ray_dir_cam[2]) * forward)  # Negate the Z component!
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
        
        # What direction SHOULD center of screen point?
        expected_center = forward
        
        print(f"\n5. RAY IN WORLD SPACE")
        print(f"   Formula: ray_cam.x*right + ray_cam.y*up + (-ray_cam.z)*forward")
        print(f"   Values: {ray_dir_cam[0]:.3f}*right + {ray_dir_cam[1]:.3f}*up + {-ray_dir_cam[2]:.3f}*forward")
        print(f"   ray_dir_world: ({ray_dir_world[0]:7.4f}, {ray_dir_world[1]:7.4f}, {ray_dir_world[2]:7.4f})")
        print(f"   Expected (center): ({expected_center[0]:7.4f}, {expected_center[1]:7.4f}, {expected_center[2]:7.4f})")
        
        # CRITICAL TEST: Does (0,0,-1) map to forward?
        test_cam = np.array([0.0, 0.0, -1.0])
        test_world = (test_cam[0] * right + test_cam[1] * up + (-test_cam[2]) * forward)
        test_world_norm = test_world / np.linalg.norm(test_world)
        dot_test = np.dot(test_world_norm, forward)
        print(f"   VERIFY: (0,0,-1) -> ({test_world_norm[0]:.4f}, {test_world_norm[1]:.4f}, {test_world_norm[2]:.4f})")
        print(f"   Dot with forward: {dot_test:.6f} <<< SHOULD BE 1.0")
        
        # Store for visualization
        self.debug_ray_origin = cam_pos
        self.debug_ray_end = cam_pos + ray_dir_world * 10000000
        self.update()
        
        # Ray-sphere intersection
        sphere_radius = 150000  # 150km radius - reasonable click target
        
        oc = cam_pos - sat_pos
        a = np.dot(ray_dir_world, ray_dir_world)
        b = 2.0 * np.dot(oc, ray_dir_world)
        c = np.dot(oc, oc) - sphere_radius ** 2
        
        discriminant = b * b - 4 * a * c
        
        # Closest approach
        t_closest = -np.dot(oc, ray_dir_world) / np.dot(ray_dir_world, ray_dir_world)
        closest_point = cam_pos + t_closest * ray_dir_world
        closest_dist = np.linalg.norm(closest_point - sat_pos)
        
        print(f"\n6. INTERSECTION TEST")
        print(f"   Sphere radius: {sphere_radius/1000:.0f} km")
        print(f"   Discriminant: {discriminant:.2e}")
        print(f"   Closest approach: {closest_dist/1000:.2f} km")
        
        if discriminant >= 0:
            print(f"\n   ✓✓✓ HIT! ✓✓✓")
            return True
        else:
            print(f"\n   ✗ MISS")
            return False
    
    def draw_debug_ray(self):
        """Draw the last cast ray for debugging"""
        if self.debug_ray_origin is None or self.debug_ray_end is None:
            return
        
        print(f"Drawing debug ray from {self.debug_ray_origin/1e6} to {self.debug_ray_end/1e6} Mm")
            
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(5.0)
        
        glBegin(GL_LINES)
        glColor3f(1, 1, 0)  # Yellow
        glVertex3f(self.debug_ray_origin[0], self.debug_ray_origin[1], self.debug_ray_origin[2])
        glVertex3f(self.debug_ray_end[0], self.debug_ray_end[1], self.debug_ray_end[2])
        glEnd()
        
        # Camera marker
        glColor3f(0, 1, 1)  # Cyan
        glPushMatrix()
        glTranslatef(self.debug_ray_origin[0], self.debug_ray_origin[1], self.debug_ray_origin[2])
        quadric = gluNewQuadric()
        gluSphere(quadric, 50000, 10, 10)
        gluDeleteQuadric(quadric)
        glPopMatrix()
        
        # Draw a magenta sphere at ray end for debugging
        glColor3f(1, 0, 1)  # Magenta
        glPushMatrix()
        glTranslatef(self.debug_ray_end[0], self.debug_ray_end[1], self.debug_ray_end[2])
        quadric = gluNewQuadric()
        gluSphere(quadric, 100000, 10, 10)
        gluDeleteQuadric(quadric)
        glPopMatrix()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 OpenGL 3D Globe (ECEF)")
        self.setGeometry(100, 100, 1024, 768)
        
        self.globe = GlobeWidget(self)
        self.setCentralWidget(self.globe)
        
        self.statusBar().showMessage(
            "Drag to rotate | Scroll to zoom | Space to pause | R to reset"
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.globe.shutdownFetcher)
    window.show()
    sys.exit(app.exec())
