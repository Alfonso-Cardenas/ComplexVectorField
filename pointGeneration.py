import pyopencl as cl
import numpy as np

import time

from numpy.polynomial import polynomial as P

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

#np.random.seed(2901)

start = time.time()

N = np.int32(3)
a = np.random.uniform(-1, 1, (N, N)) + 1.j*np.random.uniform(-1, 1, (N, N))
a[0,0] = 0
a[1,0] = 0
a[0,1] = 0
a[0,2] = -0.02308907-0.92907722j
a[1,1] = 0.38376201-0.35505029j
a[2,0] = -0.9692624-0.71594008j
# a[0,3] = 0
# a[1,2] = 0
# a[2,1] = 0
# a[3,0] = 0
print(np.diag(np.fliplr(a)))
a = a.reshape(-1)

b = np.random.uniform(-1, 1, (N, N)) + 1.j*np.random.uniform(-1, 1, (N, N))
b[0,0] = 0
b[1,0] = 0
b[0,1] = 0
b[0,2] = 0.34934538-0.35943337j
b[1,1] = 0.16915693-0.75952778j
b[2,0] = 0.31920888+0.45262952j
# b[0,3] = 0
# b[1,2] = 0
# b[2,1] = 0
# b[3,0] = 0
print(np.diag(np.fliplr(b)))
b = b.reshape(-1)

eps = np.float32(.001)

a = a.astype(np.csingle)
b = b.astype(np.csingle)

initialLimit = 1
nFrames = np.int32(100)
#sampleSize = np.int32(1)
pointsPerCircle = np.int32(100)
numOfCircles = 5
circle = np.array([[np.cos(theta), np.sin(theta), 0] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
circle1 = np.array([[0, np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
circle2 = np.array([[np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
# circle
pointMatrix = np.array([[circle*r, circle1*r, circle2*r] for r in np.linspace(0.01, 1, numOfCircles)], dtype=np.float32).reshape(-1, 3)    
# sphere
#pointMatrix = np.array([circle*np.sin(theta) + [0,0,np.cos(theta)] for theta in np.linspace(0.001, np.pi, numOfCircles)], dtype=np.float32).reshape(-1, 3)
#pointMatrix = np.array([circle, circle1, circle2], dtype=np.float32).reshape(-1, 3)
numOfCircles *= 3
sampleSize = np.int32(len(pointMatrix))
# for i in range(sampleSize):
#     pointMatrix[i*3+2] = (pointMatrix[i*3+2] + np.pi)/4
# pointMatrix = np.random.uniform(-1, 1, (sampleSize,3)) #inside sphere
# pointMatrix = (pointMatrix.T/np.linalg.norm(pointMatrix, axis = 1))
# randomRadius = np.random.uniform(.1, 2, sampleSize)
# pointMatrix = (pointMatrix * randomRadius).T.reshape(-1).astype(np.float32)

allPoints = np.zeros(sampleSize*3*nFrames, dtype=np.float32)

deadPointsSize = np.int32(200000)
radiuses = np.random.uniform(0,1, deadPointsSize)
thetas = np.random.uniform(0, np.pi, deadPointsSize)
alphas = np.random.uniform(0, 2*np.pi, deadPointsSize)
deadPoints = np.array([[r*np.sin(theta)*np.cos(alpha), r*np.sin(alpha)*np.cos(theta), r*np.cos(theta)] for r, theta, alpha in zip(radiuses, thetas, alphas)])
renderDeadPoints = np.empty(deadPointsSize, dtype=np.bool_)

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type = cl.device_type.GPU)
ctx = cl.Context(devices = my_gpu_devices )
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
point_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pointMatrix)
deadpoint_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=deadPoints)
allpoint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, allPoints.nbytes)
renderdeadpoints_buf = cl.Buffer(ctx, mf.WRITE_ONLY, renderDeadPoints.nbytes)

prg = cl.Program(ctx, open("updatePoints.c").read()).build()

prg.updatePoints(queue, (sampleSize,), None,
             N, nFrames, sampleSize, point_buf, a_buf,
             b_buf, allpoint_buf, eps)

cl.enqueue_copy(queue, allPoints, allpoint_buf)

end = time.time()

print("done calculating points in", "{:.2f}".format(end-start), "seconds")

np.save("pointMatrixHistory", allPoints.reshape(nFrames,sampleSize,3))

start = time.time()

prg.calcDeadPoints(queue, (deadPointsSize,), None,
             N, deadPointsSize, deadpoint_buf, a_buf,
             b_buf, renderdeadpoints_buf, np.float32(.4))

cl.enqueue_copy(queue, renderDeadPoints, renderdeadpoints_buf)

end = time.time()

deadPoints = deadPoints[np.where(renderDeadPoints)]

print(len(deadPoints))

np.save("deadPoints", deadPoints)

print("done calculating deadPoints in", "{:.2f}".format(end-start), "seconds")

# def Xcoefs(x2):
#     return np.array([sum([b[i*N+k]*pow(x2+c,k) for k in range(N-i) ]) for i in range(N)])

# x2 = np.linspace(-100, 100, 10000)
# coefMat = Xcoefs(x2).T

# roots = P.polyroots(coefMat[0])
# coefMat = coefMat[1:]
# zplots = [[root] for root in roots]
# for coefs in coefMat:
#     roots = P.polyroots(coefs)
#     for zplot in zplots:
#         goodRoot = 9999999
#         mindist = 99999999
#         for root in roots:
#             distToRoot = abs(zplot[-1] - root)
#             if distToRoot < mindist:
#                 mindist = distToRoot
#                 goodRoot = root
#         roots = np.delete(roots, np.where(roots == goodRoot))
#         zplot.append(goodRoot)

# print("done calculating roots")

# np.save("deadlines", np.array(zplots))
# np.save("x2", x2)

start = time.time()

startingColor = np.array([15,129,160])/255
endingColor = np.array([250, 218, 94])/255
#randomColorPerCircle = np.array([np.random.uniform(0,1,3) for _ in range(numOfCircles)])
colorMatrix = np.array([startingColor*ratio + endingColor*(1-ratio) for ratio in np.linspace(0, 1, numOfCircles) for _ in range(pointsPerCircle)], dtype = np.float32)
#np.array([(p + initialLimit)/(4*initialLimit) + np.array([.25, .25, .25]) for p in np.reshape(pointMatrix, (sampleSize, 3))]).astype(np.float32)
#colorMatrix = np.array([(p + 2)/4 for p in pointMatrix.reshape(sampleSize, 3)])

end = time.time()

print("done calculating colors in", "{:.2f}".format(end-start), "seconds")

np.save("colorMatrix", colorMatrix)

lineColors = np.zeros(sampleSize * 3 * nFrames).astype(np.float32)

colorMatrix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=colorMatrix)
lineColors_buf = cl.Buffer(ctx, mf.WRITE_ONLY, lineColors.nbytes)

start = time.time()

prg.calcLineColors(queue, lineColors.shape, None, sampleSize, colorMatrix_buf, lineColors_buf)

cl.enqueue_copy(queue, lineColors, lineColors_buf)

end = time.time()

print("done calculating lineColors in", "{:.2f}".format(end-start), "seconds")

np.save("lineColors", lineColors.reshape(-1, 3))

lineConnections = np.zeros(sampleSize * 2 * nFrames).astype(np.int32)

allpoint_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allPoints)
lineConnections_buf = cl.Buffer(ctx, mf.WRITE_ONLY, lineConnections.nbytes)

start = time.time()

prg.calcLineConnectionsCircles(queue, (sampleSize * nFrames,), None, pointsPerCircle, lineConnections_buf, allpoint_buf)

cl.enqueue_copy(queue, lineConnections, lineConnections_buf)

end = time.time()

print("done calculating lineConnections in", "{:.2f}".format(end-start), "seconds")

np.save("lineConnections", lineConnections.reshape(-1, 2))