import sys
import os
from collections import OrderedDict
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GLU import *

from pyglobe import globe
from pyglobe import tile_fetcher
from pyglobe import scene

class MainWindow(QWidget):
    startFetcher = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 OpenGL 3D Globe (ECEF)")
        #self.setGeometry(100, 100, 1024, 768)
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        self.text = QLabel('Label')
        vbox.addWidget(self.text)

        self.additional_text = QLabel('')
        vbox.addWidget(self.additional_text)

        # TODO - adding this column throws off the raycasting calcs
        hbox.addLayout(vbox)
        self.globe = globe.GlobeWidget(self)
        hbox.addWidget(self.globe)
        self.globe.infoSig.connect(self.on_window)
        self.setLayout(hbox)

        scene.add_test_objects(self.globe.scene) 
        self.globe.sigObjectClicked.connect(self.print_object)
        
        # Set up TileFetcher in separate thread
        self.fetcher_thread = QThread()
        self.fetcher = tile_fetcher.TileFetcher(cache_dir="cache")
        self.fetcher.moveToThread(self.fetcher_thread)
        
        # Connect signals
        self.startFetcher.connect(self.fetcher.start)
        self.globe.requestTile.connect(self.fetcher.requestTile)
        self.globe.setAimpoint.connect(self.fetcher.setAimpoint)
        self.fetcher.tileReady.connect(self.globe.on_tile_ready)
        
        # Start fetcher thread
        self.fetcher_thread.start()
        self.startFetcher.emit()

    def print_object(self, obj: scene.SceneObject):
        self.additional_text.setText(f'Clicked: {obj.label}')
        
    def on_window(self, info_dict: dict):
        s =  f"Level:    {info_dict['level']}\n"
        s += f"Tile X:   {info_dict['current_tile_x']}\n"
        s += f"Tile Y:   {info_dict['current_tile_y']}\n\n"

        s += f"Lat:      {info_dict['center_lla']['lat']:.2f}\n"
        s += f"Lon:      {info_dict['center_lla']['lon']:.2f}\n"
        s += f"Alt:      {info_dict['center_lla']['alt']:.2f}\n\n"


        self.text.setText(s)


if __name__ == '__main__':
    # Create a diretory for tile caching
    os.makedirs('cache', exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Cleanup on exit
    app.aboutToQuit.connect(window.fetcher.shutdown)
    app.aboutToQuit.connect(window.fetcher_thread.quit)
    app.aboutToQuit.connect(window.fetcher_thread.wait)
    window.resize(1200,800) 
    window.show()
    sys.exit(app.exec())
