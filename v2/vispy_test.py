import sys

import numpy as np
from vispy import scene
from vispy.geometry import create_sphere
from vispy.scene.visuals import Mesh
from vispy.visuals import MarkersVisual

canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                           size=(800, 600), show=True)

view = canvas.central_widget.add_view()
view.camera = 'turntable'

sphere = create_sphere(rows=50, cols=50, radius=1)
vertices = sphere.get_vertices()

mesh = Mesh(meshdata=sphere, parent=view.scene)

sphere = create_sphere(rows=50, cols=50, radius=5)
vertices = sphere.get_vertices()
colors = (vertices + 1) / 2
sphere.set_vertex_colors(colors)
mesh.set_data(meshdata=sphere)
# sphere = Sphere(radius=1, method='latitude', parent=view.scene)

# print(sphere.vertices)

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()