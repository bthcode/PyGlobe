from OpenGL.GL import *
import numpy as np
import os

class Mesh:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.materials = {}

class Material:
    def __init__(self, name):
        self.name = name
        self.diffuse = [0.8, 0.8, 0.8]

class OBJLoader:
    @staticmethod
    def load(path):
        mesh = Mesh()
        material = None
        vertices, normals = [], []
        current_material = None

        mtl_path = None
        base_dir = os.path.dirname(path)

        with open(path, "r") as f:
            for line in f:
                if line.startswith("mtllib"):
                    mtl_path = os.path.join(base_dir, line.split()[1].strip())
                elif line.startswith("usemtl"):
                    current_material = line.split()[1].strip()
                elif line.startswith("v "):
                    vertices.append(list(map(float, line.split()[1:])))
                elif line.startswith("vn "):
                    normals.append(list(map(float, line.split()[1:])))
                elif line.startswith("f "):
                    face = []
                    for v in line.split()[1:]:
                        parts = v.split("/")
                        vi = int(parts[0]) - 1
                        ni = int(parts[-1]) - 1 if len(parts) > 2 and parts[-1] else 0
                        face.append((vi, ni, current_material))
                    mesh.faces.append(face)

        if mtl_path and os.path.exists(mtl_path):
            mesh.materials = OBJLoader.load_mtl(mtl_path)

        mesh.vertices = np.array(vertices)
        mesh.normals = np.array(normals)
        return mesh

    @staticmethod
    def load_mtl(path):
        materials = {}
        current = None
        with open(path, "r") as f:
            for line in f:
                if line.startswith("newmtl"):
                    current = Material(line.split()[1].strip())
                    materials[current.name] = current
                elif line.startswith("Kd") and current:
                    current.diffuse = list(map(float, line.split()[1:4]))
        return materials


class SceneObject:
    def __init__(self, mesh):
        self.mesh = mesh
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.scale = np.ones(3)

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(*self.scale)

        glBegin(GL_TRIANGLES)
        for face in self.mesh.faces:
            for vi, ni, matname in face:
                mat = self.mesh.materials.get(matname)
                if mat:
                    glColor3fv(mat.diffuse)
                if len(self.mesh.normals) > 0:
                    glNormal3fv(self.mesh.normals[ni])
                glVertex3fv(self.mesh.vertices[vi])
        glEnd()

        glPopMatrix()


class Scene:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def draw(self):
        for obj in self.objects:
            obj.draw()

