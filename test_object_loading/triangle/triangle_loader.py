import sys, math
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


# -----------------------------------------------------------
# Simple OBJ loader (no materials, triangulated meshes only)
# -----------------------------------------------------------
def load_obj(path):
    vertices = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # OBJ indices are 1-based
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    return vertices, faces


# -----------------------------------------------------------
# OpenGL Widget
# -----------------------------------------------------------
class ObjViewer(QOpenGLWidget):
    def __init__(self, obj_path):
        super().__init__()
        self.obj_path = obj_path
        self.vertices = None
        self.faces = None
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rotation)
        self.timer.start(16)  # ~60fps

    def update_rotation(self):
        self.angle += 1
        if self.angle > 360:
            self.angle -= 360
        self.update()

    # --- OpenGL setup ---
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.07, 0.08, 0.1, 1.0)
        self.vertices, self.faces = load_obj(self.obj_path)

    # --- Resize handler ---
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h else 1
        gluPerspective(45, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    # --- Render ---
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Move back and rotate
        glTranslatef(0.0, 0.0, -3.0)
        glRotatef(self.angle, 0, 1, 0)

        glBegin(GL_TRIANGLES)
        glColor3f(0.8, 0.8, 0.9)
        for face in self.faces:
            for idx in face:
                glVertex3fv(self.vertices[idx])
        glEnd()


# -----------------------------------------------------------
# Main Window
# -----------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, obj_path):
        super().__init__()
        self.setWindowTitle("PyQt5 OpenGL OBJ Viewer")
        self.viewer = ObjViewer(obj_path)
        self.setCentralWidget(self.viewer)
        self.resize(800, 600)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow("triangle.obj")  # 👈 replace with your file
    window.show()
    sys.exit(app.exec_())

