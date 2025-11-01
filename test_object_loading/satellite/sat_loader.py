import sys, math
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *

class OBJModel:
    def __init__(self, path):
        self.vertices = []
        self.faces = []  # list of (material, [vertex_indices])
        self.materials = {}
        self.current_mtl = None
        self.load(path)

    def load(self, path):
        base = path.rsplit("/", 1)[0]
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.split()
                    self.vertices.append((float(x), float(y), float(z)))
                elif line.startswith("usemtl "):
                    self.current_mtl = line.split()[1].strip()
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    idxs = [int(p.split("/")[0]) - 1 for p in parts]
                    self.faces.append((self.current_mtl, idxs))
                elif line.startswith("mtllib "):
                    #mtl_path = f"{base}/{line.split()[1].strip()}"
                    mtl_path = "satellite.mtl"
                    self.load_mtl(mtl_path)

        # Normalize to fit in unit cube
        if self.vertices:
            xs, ys, zs = zip(*self.vertices)
            cx, cy, cz = [(max(v) + min(v)) / 2 for v in (xs, ys, zs)]
            scale = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            self.vertices = [((x - cx) / scale * 2,
                              (y - cy) / scale * 2,
                              (z - cz) / scale * 2) for x, y, z in self.vertices]

    def load_mtl(self, path):
        current = None
        with open(path, "r") as f:
            for line in f:
                if line.startswith("newmtl "):
                    current = line.split()[1].strip()
                    self.materials[current] = {"Kd": (0.8, 0.8, 0.8)}
                elif line.startswith("Kd ") and current:
                    _, r, g, b = line.split()
                    self.materials[current]["Kd"] = (float(r), float(g), float(b))

    def draw(self):
        for mtl, idxs in self.faces:
            color = self.materials.get(mtl, {}).get("Kd", (0.8, 0.8, 0.8))
            glColor3fv(color)
            glBegin(GL_TRIANGLES)
            for i in range(0, len(idxs), 3):
                for j in range(3):
                    glVertex3fv(self.vertices[idxs[i+j]])
            glEnd()


class GLWidget(QOpenGLWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.rot_x = -20
        self.rot_y = -30
        self.zoom = -4

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.07, 0.08, 0.1, 1.0)

        # light direction
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 1.0, 0.0])

        self.model = OBJModel(self.model_path)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        if self.model:
            self.model.draw()

    def mousePressEvent(self, e):
        self.last_pos = e.pos()

    def mouseMoveEvent(self, e):
        dx = e.x() - self.last_pos.x()
        dy = e.y() - self.last_pos.y()
        if e.buttons() & Qt.LeftButton:
            self.rot_x += dy * 0.5
            self.rot_y += dx * 0.5
            self.update()
        self.last_pos = e.pos()

    def wheelEvent(self, e):
        self.zoom += e.angleDelta().y() / 480.0
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    path = "satellite.obj"  # make sure satellite.obj + satellite.mtl are in same folder
    w = GLWidget(path)
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())

