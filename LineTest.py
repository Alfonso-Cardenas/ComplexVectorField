import sys

from vispy import app, visuals, scene

import numpy as np
import time
from numpy.polynomial import polynomial as P
import time
import warnings
warnings.filterwarnings("ignore")

i = 0
def update(event):
    global i
    l.set_data(connect = connectionPoints[:i])

    i += 1
    i = i%(len(xPoints))


# build your visuals, that's all
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
line = scene.visuals.create_visual_node(visuals.LineVisual)

# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor = "white")

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov = 45
view.camera.distance = 500

xPoints = np.linspace(-100, 100, 1000)
linePoints = np.array([[x, 0, 0] for x in xPoints])
connectionPoints = np.array([[i, i+1] for i in range(len(linePoints - 1))])

l = line(parent=view.scene)
l.set_gl_state(depth_test=True)
l.set_data(pos = linePoints, color = "black", width = 2, connect = connectionPoints)

timer = app.Timer()
timer.connect(update)
timer.start()

# run
if __name__ == "__main__":
    if sys.flags.interactive != 1:
        app.run()