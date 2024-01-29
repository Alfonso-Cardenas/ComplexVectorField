import sys

from vispy import app, visuals, scene, keys
from vispy.scene.visuals import Text

import numpy as np
import warnings
warnings.filterwarnings("ignore")

colorMatrix = np.load("colorMatrix.npy")

pointMatrixHistory = np.load("pointMatrixHistory.npy")
# zplots = np.load("deadlines.npy")
# x2 = np.load("x2.npy")

nframes = pointMatrixHistory.shape[0]
sampleSize = pointMatrixHistory.shape[1]

allPoints = np.reshape(pointMatrixHistory, (nframes*sampleSize,3))

centerPointHistory = np.load("centerPointHistory.npy")

deadPoints = np.load("deadPoints.npy")

lineColors = np.load("lineColors.npy") #[c for i in range(nframes) for c in colorMatrix]
lineConnections = np.load("lineConnections.npy")
                            # np.array([[i*sampleSize + j, (i+1)*sampleSize+j] 
                            # if np.linalg.norm(allPoints[i*sampleSize + j] - allPoints[(i+1)*sampleSize+j]) < 2
                            # else
                            # [i*sampleSize + j, i*sampleSize + j] 
                            # for i in range(nframes - 1) for j in range(sampleSize)])

renderIndices = np.load("renderIndices.npy")
        #list(range(int(nframes/2)))[::-1] + list(range(int(nframes/2))) +\
        #list(range(int(nframes/2),nframes)) + list(range(int(nframes/2),nframes))[::-1]

print("done reading matrices!")

lineIndices = []
i = 0 #ultimo frame antes de desaparecer 4659
focusOnPoint = False
reverse = True
def update(event):
    global i
    global lineIndices
    global view
    global reverse
    t1.text = f"Coords of centerpoint = {centerPointHistory[i,0]:.5f}, {centerPointHistory[i,1]:.5f}"
    p1.set_data(pos = 
    pointMatrixHistory[i], face_color=colorMatrix, symbol="o", size=5, edge_width=0
    )
    lineIndices.append(i)
    if len(lineIndices) > 1:
        lineIndices = lineIndices[1:]
    traceStart = min(lineIndices)
    traceEnd = max(lineIndices)
    traces.set_data(connect = lineConnections[i*sampleSize:(i+1)*sampleSize])
    if(focusOnPoint): view.camera.center = pointMatrixHistory[i][0]
    if(i == (nframes - 1) or i == 0):
        reverse = not reverse
        timer.stop()
    #print(pointMatrixHistory[i][0])
    #view.canvas.measure_fps()
    #traceStart = max(0, i-20)
    #traces.set_data(connect = lineConnections[traceStart*sampleSize:i*sampleSize])
    
    if(reverse):
        i -= 1
    else:
        i += 1

# build your visuals, that's all
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
line = scene.visuals.create_visual_node(visuals.LineVisual)

def pause():
    if timer.running:
        timer.stop()
    else:
        timer.start()

def exit():
    canvas.close()

def focus():
    global focusOnPoint
    global view
    focusOnPoint = not focusOnPoint
    if(not focusOnPoint):
        view.camera.center = [0,0,0]

def revFunc():
    global reverse
    reverse = not reverse

def fullScreen():
    global canvas
    canvas.fullscreen = True

def toggleAxis():
    global xaxis
    global yaxis
    global zaxis
    xaxis.visible = not xaxis.visible 
    yaxis.visible = not yaxis.visible 
    zaxis.visible = not zaxis.visible 

def togglePoints():
    global p1
    p1.visible = not p1.visible

def toggleLines():
    global traces
    traces.visible = not traces.visible

def toggleDeadPoints():
    global p2
    p2.visible = not p2.visible

key_dict = {"space" : pause, "escape" : exit, "f" : focus, "r" : revFunc, "d" : toggleDeadPoints,
             "f1" : fullScreen, "a" : toggleAxis, "p" : togglePoints, "l" : toggleLines}

canvas = scene.SceneCanvas(keys=key_dict, show=True, bgcolor = "white", autoswap=False, vsync=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov = 45 # type: ignore
view.camera.distance = 10 # type: ignore

p1 = Scatter3D(parent=view.scene)
#p1.set_gl_state(depth_test=True)
# arr = np.array([*pointMatrixHistory[int(nframes/2)-1]])
# p1.set_data(
#     arr, face_color="black", symbol="o", size=10, edge_width=0, edge_color="black"
# )

p2 = Scatter3D(parent=view.scene)
# p2.set_gl_state(depth_test=True)
# p2.set_data(
#     deadPoints, face_color="black", symbol="o", size=10, edge_width=0, edge_color="black"
# )

xaxis = line(parent=view.scene)
xaxis.set_gl_state(depth_test=True)
xaxis.set_data([[-5, 0, 0], [5, 0, 0]], color="red", width = 3)
yaxis = line(parent=view.scene)
yaxis.set_gl_state(depth_test=True)
yaxis.set_data([[0, -5, 0], [0, 5, 0]], color="green", width = 3)
zaxis = line(parent=view.scene)
zaxis.set_gl_state(depth_test=True)
zaxis.set_data([[0, 0, -5], [0, 0, 5]], color="blue", width = 3)

traces = line(parent=view.scene)
traces.set_gl_state(depth_test=True)
traces.set_data(pos = allPoints, color = lineColors, width = 3, connect = np.array([[0,0]])) #lineConnections

# deadlines = []
# for zplot in zplots:
#     deadline = line(parent = view.scene)
#     deadline.set_gl_state(depth_test=True)
#     deadline.set_data(pos = np.array([[z.real, z.imag,  x] for z, x in zip(zplot,x2)]), color = "black", width = 5)

t1 = Text('', parent=canvas.scene, anchor_x='left')# type: ignore
t1.font_size = 12
t1.pos = 1, 12

timer = app.Timer()
timer.connect(update)
timer.start()

# run
if __name__ == "__main__":
    if sys.flags.interactive != 1:
        app.run()