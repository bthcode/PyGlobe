from PyQt5.QtWidgets import QApplication, QOpenGLWidget, QMainWindow
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class OpenGLSphereWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_angle = 0.0

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH) # For smoother lines
        glLineWidth(1.5) # Adjust line thickness

        # Compile shaders (simplified for brevity, actual shaders needed)
        # self.shader_program = self.compile_shaders(...)

        self.setup_sphere_data()

    def setup_sphere_data(self):
        # Generate sphere vertices and indices (example for a simple sphere)
        self.vertices = []
        self.indices = []
        radius = 1.0
        stacks = 30
        slices = 20

        for i in range(stacks + 1):
            phi = np.pi / stacks * i
            for j in range(slices + 1):
                theta = 2 * np.pi / slices * j
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                self.vertices.extend([x, y, z])

        # Generate indices for wireframe (connecting adjacent vertices)
        for i in range(stacks):
            for j in range(slices):
                p1 = i * (slices + 1) + j
                p2 = p1 + (slices + 1)
                p3 = p1 + 1
                p4 = p2 + 1

                # Lines for latitude
                self.indices.extend([p1, p3])
                self.indices.extend([p2, p4])
                # Lines for longitude
                self.indices.extend([p1, p2])
                self.indices.extend([p3, p4])

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)

        # Create VBOs (Vertex Buffer Objects) (modern OpenGL)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #---- Set camera position and where it's pointing
        # eye x, y, z, center x, y, z, up x, y, z  [ up seems to be vector direction of up in the scene ]
        gluLookAt(0, 5, 5, 0, 0, 0, 0, 1, 0) # Camera position

        #---- Rotate the scene?  
        # angle, x, y, z
        glRotatef(self.rotation_angle, 0, 1, 0) # Rotate around Y-axis

        # Use shaders (if implemented)
        # glUseProgram(self.shader_program)

        # Draw wireframe sphere using VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        # glUseProgram(0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def animate(self):
        self.rotation_angle += 0.5
        if self.rotation_angle > 360:
            self.rotation_angle -= 360
        self.update() # Request a repaint

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 OpenGL Wireframe Sphere")
        self.opengl_widget = OpenGLSphereWidget(self)
        self.setCentralWidget(self.opengl_widget)

        # Timer for animation
        from PyQt5.QtCore import QTimer
        #self.timer = QTimer(self)
        #self.timer.timeout.connect(self.opengl_widget.animate)
        #self.timer.start(20) # Update every 20 milliseconds

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
