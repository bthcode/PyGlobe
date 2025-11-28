import sys
import os
import pathlib
from collections import OrderedDict
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GLU import *

from pyglobe import globe
from pyglobe import scene

class GlobeTestWidget(QWidget):
    startFetcher = Signal()
    
    def __init__(self):
        super().__init__()
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        # Drop-down to select map source 
        self.map_combo = QComboBox()
        self.map_combo.addItems(
            ["OSM", "Blue Marble" ]
        )
        self.map_combo.currentIndexChanged.connect(self.on_map_combo)
        vbox.addWidget(self.map_combo)

        # Text area to print display debug output
        self.text = QLabel('Label')
        vbox.addWidget(self.text)

        self.additional_text = QLabel('')
        vbox.addWidget(self.additional_text)

        # TODO - adding this column throws off the raycasting calcs
        hbox.addLayout(vbox)

        # Globe Widget
        self.globe = globe.GlobeWidget(self)
        #self.globe.tile_url_template = ("http://s3.amazonaws.com/com.modestmaps.bluemarble/{z}-r{y}-c{x}.jpg")
        hbox.addWidget(self.globe)

        # Connect globe events
        self.globe.infoSig.connect(self.on_window)
        self.setLayout(hbox)

        self.add_objects()
        self.globe.sigObjectClicked.connect(self.print_object)
        
        self.counter = 0

        self.move_examples_timer = QTimer()
        self.move_examples_timer.timeout.connect(self.on_timer)
        self.move_examples_timer.start(100)
        

    def on_map_combo(self):
        txt = self.map_combo.currentText()
        if txt == 'OSM':
            cache_dir = 'osm'
            url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        else:
            cache_dir = 'bluemarble'
            url_template = "http://s3.amazonaws.com/com.modestmaps.bluemarble/{z}-r{y}-c{x}.jpg"
        self.globe.init_tile_manager(cache_dir, url_template)


    def add_objects(self):
        this_file = pathlib.Path(__file__).resolve()
        this_dir  = this_file.parent
        assets_dir = this_dir / 'assets'


        # 1. Point - Red marker over New York
        self.point = scene.PointSceneObject(
            'Point',
            lat=12.3601,
            lon=12.0589,
            alt=0,
            color=(1.0, 0.0, 0.0),  # Red
            size=10.0,
            pick_radius=50000
        )
        self.globe.add_object(self.point)

        # 2. Polyline -
        track_points = [
            (12.3601, 12.0589, 0),    
        ]
        for i in range(12):
            track_points.append( ( track_points[0][0] + i * 0.2,
                                   track_points[0][1] + i * 0.2,
                                   0) )
        self.track = scene.PolyLineSceneObject(
            'Track',
            points_wgs84=track_points,
            color=(0.0, 0.0, 0.0),  # Yellow
            width=8.0,
            altitude_offset=2000, # earth curvature
            pick_radius=25000
        )
        self.globe.add_object(self.track)



        #----------- CIRCLE OBJECT -----------#
        self.circle = scene.CircleSceneObject(
            'Circle',
            center_lat=10,
            center_lon=10,
            radius_meters=100000,
            color=(0.0, 0.0, 0.0),
            fill_color=(0.7, 0.7, 0.7, 0.7),
            width=3.0,
            altitude_offset=2000 # Earth curvature
        )
        self.globe.add_object(self.circle)

        #---------- POLYGON -----------#
        polygon_points = [
            (3.0, -6.0, 0),  # Southwest corner
            (3.0, -5.0, 0),   # Southeast corner
            (4.0, -5.0, 0),   # Northeast corner
            (4.0, -6.0, 0)   # Northwest corner
        ]
        self.polygon = scene.PolygonSceneObject(
            'Polygon',
            points_wgs84=polygon_points,
            color=(1.0, 0.5, 0.0),  # Orange outline
            fill_color=(1.0, 0.5, 0.0, 0.3),  # Semi-transparent orange fill
            width=3.0,
            altitude_offset=2000.0 # Account for earth curvature
        )
        self.globe.add_object(self.polygon)

        #-------- IMAGE ------------#
        # 5. Image Overlay - SAR image over California
        self.images = [ assets_dir / 'images' / 'squirrel_tail_bushy_tail.jpg',	
                        assets_dir / 'images' / 'squirrel_tree_mammal_paw.jpg' ]
        self.image_idx = 0

        img_path = self.images[self.image_idx]
        if img_path.exists():
            image_corners = [
                (-14.0, -7.0, 0),  # Bottom-left
                (-14.0, -9.0, 0),  # Bottom-right
                (-16.0, -9.0, 0),  # Top-right
                (-16.0, -7.0, 0)   # Top-left
            ]
            self.image_overlay = scene.ImageOverlaySceneObject(
                'Image',
                corners_wgs84=image_corners,
                image_path=str(img_path),
                altitude_offset=2000.0, # Account for earth curvature
                alpha=0.7
            )
            self.globe.add_object(self.image_overlay)
        else:
            print(f"Warning: Image not found: {str(img_path)}")

        satellite_obj_path = assets_dir / 'satellite' / 'satellite.obj'
        if satellite_obj_path.exists():
            satellite = scene.SceneModel(
                'Satellite',
                lat_deg=15.7617,
                lon_deg=-20.1918,
                alt_m=500000,  # 500 km altitude
                scale=200000,   # 50 km apparent size
                obj_path=str(satellite_obj_path),
                roll=0,
                pitch=0,
                yaw=-90,
                pick_radius=200000
            )
            self.globe.add_object(satellite)
            self.satellite=satellite
        else:
            print(f"Warning: OBJ file not found: {satellite_obj_path}")

    def on_timer(self ):
        # Test function for moving stuff
        self.counter += 1

        # Move the satellite
        lat = self.satellite.lat
        lon = self.satellite.lon

        lon += 0.1
        if lon > 180: lon -= 360
        lat += 0.1
        # Note - this jumps you from pole to pole.  Good enough for a demo
        if lat > 90: lat -= 180
        alt = self.satellite.alt
        roll = self.satellite.roll
        pitch = self.satellite.pitch
        yaw = self.satellite.yaw
        self.satellite.set_pos( lat, lon, alt, roll, pitch, yaw)

        # Swap images
        if self.counter > 5:
            self.image_idx += 1
            self.image_overlay.set_image(self.images[self.image_idx % len(self.images) ] )
            self.counter = 0

        # Move the image
        corners = self.image_overlay.corners_wgs84
        new_corners = []
        for corner in corners:
            new_corner = (corner[0] + 0.1, corner[1] -0.1, corner[2])
            new_corners.append(new_corner)
        self.image_overlay.set_corners(new_corners)

        # Move the circle
        radius = self.circle.radius_meters
        radius += 10_000
        if radius >= 150_000:
            radius = 80_000
        self.circle.set_pos(self.circle.center_lat,
                            self.circle.center_lon,
                            radius)

        # Track and point
        polyline_points = self.track.points_wgs84
        new_points = []
        for point in polyline_points:
            newpoint = (point[0] - 0.1, point[1] - 0.1, point[2] )
            new_points.append(newpoint)
        self.track.set_points(new_points)

        self.point.set_pos(self.point.lat -0.1, self.point.lon - 0.1, self.point.alt)

        # Polygon
        polygon_points = self.polygon.points_wgs84
        new_points = []
        for point in polygon_points:
            newpoint = (point[0] + 0.1, point[1] - 0.1, point[2] )
            new_points.append(newpoint)
        self.polygon.set_points(new_points)

        self.polygon.set_points(new_points)


    def print_object(self, obj: scene.SceneObject):
        self.additional_text.setText(f'Removing: {obj.label}')
        self.globe.remove_object(obj)
        
    def on_window(self, info_dict: dict):
        s =  f"Level:    {info_dict['level']}\n"
        s += f"Tile X:   {info_dict['current_tile_x']}\n"
        s += f"Tile Y:   {info_dict['current_tile_y']}\n\n"

        s += f"Lat:      {info_dict['center_lla']['lat']:.2f}\n"
        s += f"Lon:      {info_dict['center_lla']['lon']:.2f}\n"
        s += f"Alt:      {info_dict['center_lla']['alt']:.2f}\n\n"


        self.text.setText(s)

    def close(self):
        self.move_examples_timer.stop()
        self.globe.close()


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 OpenGL 3D Globe (ECEF)")
        self.globe_widget = GlobeTestWidget()
        self.setCentralWidget(self.globe_widget)
        self.statusBar().showMessage('Left click/drag to move. Right click to select objects')

    def closeEvent(self, event):
        self.globe_widget.close()

if __name__ == '__main__':
    # Create a diretory for tile caching
    os.makedirs('cache', exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Cleanup on exit
    window.resize(1200,800) 
    window.show()
    sys.exit(app.exec())
