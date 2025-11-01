import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
from obj_loader import OBJLoader, SceneObject, Scene


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.scene = Scene()
        self.angle = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.03, 0.04, 0.07, 1.0)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 5.0, 1.0])

        base = os.path.dirname(__file__)
        earth_path = os.path.join(base, "assets/earth/earth.obj")
        sat_path = os.path.join(base, "assets/satellite/satellite.obj")

        self.earth = SceneObject(OBJLoader.load(earth_path))
        self.earth.scale = np.array([1.5, 1.5, 1.5])

        self.scene.add(self.earth)

        sat_mesh = OBJLoader.load(sat_path)
        for i in range(3):
            sat = SceneObject(sat_mesh)
            sat.scale = np.array([0.5, 0.5, 0.5])
            self.scene.add(sat)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 2, 6, 0, 0, 0, 0, 1, 0)
        self.scene.draw()

    def update_animation(self):
        self.angle += 1

        # Earth rotation
        self.earth.rotation[1] = self.angle * 0.5

        # Satellites orbiting Earth and facing center
        for i, obj in enumerate(self.scene.objects[1:]):  # skip Earth
            orbit_angle = np.radians(self.angle + i * 120)

            # Orbit position
            obj.position = np.array([
                2.0 * np.cos(orbit_angle),
                0.0,
                2.0 * np.sin(orbit_angle)
            ])

            # --- NEW: face Earth (center) ---
            # direction vector to Earth (from satellite)
            dir_vec = -obj.position
            dir_vec /= np.linalg.norm(dir_vec)

            # compute yaw (rotation around Y) to face the origin
            yaw = np.degrees(np.arctan2(dir_vec[0], dir_vec[2]))
            obj.rotation = np.array([0.0, yaw, 0.0])

            # optional: small spin around own axis for realism
            #obj.rotation[2] = (self.angle * 2) % 360

        self.update()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = GLWidget()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())

