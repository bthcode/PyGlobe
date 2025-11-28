# STDLIB Imports
from collections import OrderedDict
import numpy as np
import sys
import os

# Pyside Imports
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage

# OpenGL Imports
from OpenGL.GL import *
from OpenGL.GLU import *

# This Project Imports
from pyglobe.tile_fetcher import TileManager
from pyglobe.scene import *
from pyglobe.coord_utils import *
from pyglobe.tile_utils import latlon_to_tile, tile_y_to_lat

class GlobeWidget(QOpenGLWidget):
    '''PySide6 OpenGL Widget for displaying a 3D Globe'''

    # Signals for TileFetcher
    requestTile = Signal(int, int, int, str)
    setAimpoint = Signal(int, int, int)
    infoSig = Signal(dict)
    sigObjectClicked = Signal(SceneObject)
    
    def __init__(self, parent=None, tile_cache_dir:str='cache'):
        super().__init__(parent)
        self.setMinimumSize(1000,600)
        self.camera_distance = 20000000  # 15,000 km from center
        self.camera_lon = 0.0  # degrees
        self.camera_lat = 0.0  # degrees
        self.last_pos = None # For mouse dragging
        self.max_gpu_textures = 1024 # Max number of GPU textures to hold in memory
        
        # Earth radius in meters
        self.earth_radius = 6371000
        
        # Tile textures
        self.tile_textures = OrderedDict()
        # Level 3 - Keep it separate so we always have it
        self.base_textures = OrderedDict()
        self.tile_cache_path = tile_cache_dir

        self.inflight_tiles = {}  # Tiles currently being fetched
        self.pending_tile_data = {}  # Tiles waiting to be uploaded to GPU
        self.screen_tiles = {}  # Tiles that should be visible
        
        # OSM tile URL template
        self.tile_url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        
        # Scene contains objects to be drawn on the map
        self.scene = Scene()
        self.scene.sigClicked.connect(self.on_object_clicked)
        self.scene.sigUpdated.connect(self.update)
        
        # Publish info to display on a timer
        self.info_timer = QTimer(self)
        self.info_timer.timeout.connect(self.publish_display_info)
        self.info_timer.start(1000)

        self.tile_manager = None
        self.init_tile_manager(self.tile_cache_path, self.tile_url_template)

        
        self.base_layer_loaded = False

    def init_tile_manager(self, cache_dir, url_template):
        if self.tile_manager is not None:
            self.tile_manager.stop()
            del self.tile_manager

        self.tile_manager = TileManager(cache_dir, url_template)
        self.tile_manager.tileReady.connect(self.on_tile_ready)
        self.tile_manager.start()

        
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
        
        # Load satellite mesh
        #self.load_satellite_mesh()
        
        # Request initial tiles
        self.request_visible_tiles()
        
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 100000, 50000000)
        glMatrixMode(GL_MODELVIEW)
        #self.width = w
        #self.height = h
        
    def paintGL(self):
        self.makeCurrent()

        if not self.base_layer_loaded:
            self.load_base_textures()
            self.base_layer_loaded = True

        # Upload any pending tile data to GPU
        self.upload_pending_tiles()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Calculate camera position in ECEF (camera_distance is from center)
        cam_x, cam_y, cam_z = spherical_to_ecef(
            self.camera_lat, self.camera_lon, self.camera_distance
        )
        
        # Look at origin (Earth center)
        gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                  0, 0, 0,                # Look at origin
                  0, 0, 1)                # Up vector (Z-axis)
        
        # Draw Earth at origin
        self.draw_earth()
        
        # Draw objects on the map        
        self.scene.draw()

    def draw_earth(self):
        '''Draw earth and tiles'''
        glPushMatrix()
        glColor3f(1.0, 1.0, 1.0)  # White to show texture colors

        # Draw base layer
        for tile, _ in self.base_textures.items():
            if tile in self.base_textures:
                self.draw_tile(*tile)
        
        # Draw current screen tiles
        for tile, _ in self.screen_tiles.items():
            if tile in self.tile_textures:
                self.draw_tile(*tile)
        
        glPopMatrix()
        
    
    #------------------------------------------------
    # Tile Handling
    #------------------------------------------------
    def calculate_zoom_level(self)->int:
        """Calculate appropriate zoom level based on camera distance"""
        altitude = self.camera_distance - self.earth_radius
        altitude_mm = altitude / 1e6
        
        if altitude_mm > 8:
            return 3
        elif altitude_mm > 5:
            return 4
        elif altitude_mm > 3:
            return 5
        elif altitude_mm > 2:
            return 6
        elif altitude_mm > 1:
            return 7
        else:
            return 8
 
    def load_base_textures(self):
        '''Ensure lowest level of map is always loaded'''
        # Always draw level3
        base_z = 3
        n = 2**base_z
        xs = np.arange(n)
        ys = np.arange(n)
        for x in xs:
            for y in ys:
                key = (base_z,int(x),int(y))
                self.request_tile(key)

    def request_visible_tiles(self):
        """Request tiles that should be visible"""
        zoom = self.calculate_zoom_level()
        lat, lon = self.camera_lat, self.camera_lon
        center_x, center_y = latlon_to_tile(lat, lon, zoom)

        self.zoom_level = zoom
        self.center_lla = {'lat' : lat, 'lon' : lon, 'alt' : 0}
        self.current_tile_x = center_x
        self.current_tile_y = center_y

        # Set aimpoint for prioritization
        self.tile_manager.setAimpoint(zoom, center_x, center_y)
        
        n = 2 ** zoom
        
        # Calculate tile radius based on latitude
        abs_lat = abs(lat)
        if abs_lat > 70:
            x_radius, y_radius = 2, 3
        elif abs_lat > 60:
            x_radius, y_radius = 3, 4
        elif abs_lat > 45:
            x_radius, y_radius = 4, 5
        else:
            x_radius, y_radius = 5, 6
        
        # Build set of tiles that should be visible
        new_screen_tiles = {}
        for dx in range(-x_radius, x_radius + 1):
            for dy in range(-y_radius, y_radius + 1):
                x = (center_x + dx) % n
                y = center_y + dy
                if 0 <= y < n:
                    tile_key = (zoom, x, y)
                    new_screen_tiles[tile_key] = True
                    self.request_tile((zoom,x,y))
        self.screen_tiles = new_screen_tiles

    def request_tile(self,key: [int,int,int])->None:
        '''If a tile is not already loaded, emit a request
        Parameters
        -----------
        key: (z,x,y)
            TMS tile z,x,y
        '''
        if key in self.inflight_tiles: 
            return
        if key in self.tile_textures:
            return
        if key in self.pending_tile_data:
            return
        if key in self.base_textures:
            return
        z,x,y = key
        self.inflight_tiles[key] = True
        self.tile_manager.requestTile(z,x,y)

    @Slot(int, int, int, bytes)
    def on_tile_ready(self, z, x, y, data):
        """Called when TileFetcher has tile data ready"""
        tile_key = (z, x, y)
        
        # Remove from inflight
        del self.inflight_tiles[tile_key]
        
        # Store data to be uploaded to GPU in next paintGL call
        self.pending_tile_data[tile_key] = data
        self.update()

    def upload_pending_tiles(self):
        """Upload pending tile data to GPU"""
        for (z, x, y), data in list(self.pending_tile_data.items()):
            # Convert bytes to QImage
            image = QImage()
            if image.loadFromData(data):
                image = image.convertToFormat(QImage.Format_RGBA8888)
                image = image.mirrored()
                
                # Create OpenGL texture
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(),
                           0, GL_RGBA, GL_UNSIGNED_BYTE, image.bits())
                
                glGenerateMipmap(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D,0)

                # replace previous texture
                key = (z,x,y)
                if key in self.tile_textures:
                    old = self.tile_textures.pop(key)
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


                if z == 3:
                    self.base_textures[(z,x,y)] = texture_id
                else:
                    self.tile_textures[(z, x, y)] = texture_id

                # pruning
                while len(self.tile_textures) > self.max_gpu_textures:
                    oldk, oldtex = self.tile_textures.popitem(last=False)
                    try: 
                        glDeleteTextures([oldtex])
                    except: 
                        pass

            
            del self.pending_tile_data[(z, x, y)]
    
    def draw_tile(self, z, x, y):
        """Draw a single tile on the sphere"""
        if z == 3:
            texture_id = self.base_textures.get((z,x,y))
        else:
            texture_id = self.tile_textures.get((z, x, y))
        if not texture_id:
            return
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Calculate tile bounds
        n = 2 ** z
        
        lon_min = (x / n) * 360.0 - 180.0
        lon_max = ((x + 1) / n) * 360.0 - 180.0
        
        lat_max = tile_y_to_lat(y, z)
        lat_min = tile_y_to_lat(y + 1, z)
        
        # Draw tile as a quad mesh on the sphere
        steps = 7  # subdivisions for better sphere approximation
        
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
                    px, py, pz = lla_to_ecef(lat, lon, 0)  # Tiles on surface (alt=0)
                    # Normal points outward from sphere center
                    nx, ny, nz = px / self.earth_radius, py / self.earth_radius, pz / self.earth_radius
                    glTexCoord2f(s, t)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(px, py, pz)
        
        glEnd()
    

    #-------------------------------------------------------
    # EVENT HANDLERS
    #-------------------------------------------------------
    def publish_display_info(self)->None:
        '''Emit debug info'''
        self.infoSig.emit({'level' : self.zoom_level,
                           'center_lla' : self.center_lla,
                           'current_tile_x' : self.current_tile_x,
                           'current_tile_y' : self.current_tile_y }
                          )


    def on_object_clicked(self, obj):
        self.sigObjectClicked.emit(obj)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Force a render to ensure matrices are current
            self.makeCurrent()
            self.last_pos = event.pos()
            self.auto_rotate = False
        elif event.button() == Qt.RightButton:
            # Check if satellite was clicked
            pos = event.pos()
            ray_origin, ray_direction = self.mouse_to_ray(pos.x(), pos.y())
            self.scene.pick(ray_origin, ray_direction)
        else:
            self.last_pos = event.pos()
            self.auto_rotate = False
            
    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            self.last_pos = event.pos()
            return
            
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()

        dx = dx *3/self.zoom_level
        dy = dy *3/self.zoom_level
        
        if event.buttons() & Qt.LeftButton:
            # Reversed for more natural movement
            self.camera_lon -= dx * 0.5
            self.camera_lat = np.clip(self.camera_lat + dy * 0.5, -89, 89)
            self.update()
            
        self.last_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """Request tiles when mouse is released after dragging"""
        if event.button() == Qt.LeftButton:
            self.request_visible_tiles()
        
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
        self.request_visible_tiles()
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

    def close(self):
        if self.tile_manager is not None:
            self.tile_manager.stop()

    
    #--------------------------------------------------------------
    # Map Object Management
    #--------------------------------------------------------------
    def add_object(self, obj : SceneObject ) -> None:
        self.scene.add(obj)
        self.update()

    def remove_object(self, obj : SceneObject) -> None:
        self.scene.remove(obj)
        self.update()

    def mouse_to_ray(self, mouse_x, mouse_y):
        """
        Convert mouse coordinates to a ray in ECEF space.

        Parameters:
            mouse_x, mouse_y: Mouse coordinates in widget space

        Returns:
            (ray_origin, ray_direction): Both as numpy arrays in ECEF coordinates
        """
        # Handle high DPI scaling
        dpr = self.devicePixelRatio()

        # Use widget dimensions, not viewport (viewport can be wrong in layouts)
        widget_w = self.width()
        widget_h = self.height()
        w = widget_w * dpr
        h = widget_h * dpr

        # Scale mouse coordinates
        mx = mouse_x * dpr
        my = mouse_y * dpr

        # Convert to NDC using widget dimensions
        ndc_x = (2.0 * mx) / w - 1.0
        ndc_y = 1.0 - (2.0 * my) / h

        # Get camera position in ECEF
        cam_x, cam_y, cam_z = spherical_to_ecef(
            self.camera_lat, self.camera_lon, self.camera_distance
        )
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Camera basis vectors (matching check_satellite_click)
        # Forward: from camera toward origin
        center = np.array([0.0, 0.0, 0.0])
        eye = cam_pos
        world_up = np.array([0.0, 0.0, 1.0])

        forward = center - eye
        forward = forward / np.linalg.norm(forward)

        # Right: cross product of forward and up
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)

        # Up: cross product of right and forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build ray in camera space
        fov = 45.0
        aspect = w / h if h > 0 else 1
        fov_rad = np.radians(fov)
        tan_half_fov = np.tan(fov_rad / 2.0)

        ray_cam_x = ndc_x * aspect * tan_half_fov
        ray_cam_y = ndc_y * tan_half_fov
        ray_cam_z = -1.0

        ray_dir_cam = np.array([ray_cam_x, ray_cam_y, ray_cam_z])

        # Transform to world space
        ray_dir_world = (ray_dir_cam[0] * right +
                        ray_dir_cam[1] * up +
                        (-ray_dir_cam[2]) * forward)
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

        return cam_pos, ray_dir_world


# end class GlobeWidget
