import pyopencl as cl
import numpy as np
from numpy.linalg import norm

import time

from numpy.polynomial import polynomial as P

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

np.random.seed(2901)

start = time.time()

N = 3
a = np.random.uniform(-1, 1, (N, N)) + 1.j*np.random.uniform(-1, 1, (N, N))
a[0,0] = 0
a[1,0] = 0
a[0,1] = 0
# a[0,2] = -0.02308907-0.92907722j
# a[1,1] = 0.38376201-0.35505029j
# a[2,0] = -0.9692624-0.71594008j
# a[0,3] = 0
# a[1,2] = 0
# a[2,1] = 0
# a[3,0] = 0
print(np.diag(np.fliplr(a)))
#a = a.reshape(-1)

b = np.random.uniform(-1, 1, (N, N)) + 1.j*np.random.uniform(-1, 1, (N, N))
b[0,0] = 0
b[1,0] = 0
b[0,1] = 0
# b[0,2] = 0.34934538-0.35943337j
# b[1,1] = 0.16915693-0.75952778j
# b[2,0] = 0.31920888+0.45262952j
# b[0,3] = 0
# b[1,2] = 0
# b[2,1] = 0
# b[3,0] = 0
print(np.diag(np.fliplr(b)))
#b = b.reshape(-1)

a = a.astype(np.csingle)
b = b.astype(np.csingle)

initialLimit = 1
nFrames = 10000
#sampleSize = np.int32(1)
centerPoint = np.array([.4+.3j, .7-.1j], dtype=np.csingle)
centerPointHistory = np.zeros((nFrames + 1, 2), dtype=np.csingle)
centerPointHistory[0] = np.array(centerPoint, dtype=np.csingle)
vHistory = np.zeros((nFrames,2), dtype=np.csingle)
eps = .001
allPointEps = eps/70

def X(z):
    x1 = np.array([a[i,k]*pow(z[0], i)*pow(z[1],k) for i in range(N) for k in range(N-i)])
    x2 = np.array([b[i,k]*pow(z[0], i)*pow(z[1],k) for i in range(N) for k in range(N-i)])
    return np.array([sum(x1), sum(x2)])

start = time.time()

for i in range(1, nFrames + 1):
    x = X(centerPoint)
    #x /= norm(x)
    centerPoint += eps * x
    # centerPoint = centerPoint/norm(centerPoint)
    centerPointHistory[i] = np.array(centerPoint, dtype=np.csingle)
    vHistory[i-1] = x/norm(x)

# vHistory = np.array([centerPointHistory[i + 1] - centerPointHistory[i] for i in range(nFrames)])
# norms = norm(vHistory, axis=1)
# vHistory = np.array([vHistory[i]/norms[i] for i in range(nFrames)], dtype=np.csingle)

end = time.time()

print("done calculating centerPointHistory in", "{:.2f}".format(end-start), "seconds")

np.save("centerPointHistory", centerPointHistory)

pointsPerCircle = 50
numOfCircles = 30
numOfSpheres = 5
circle = np.array([[np.cos(theta), np.sin(theta), 0] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
circle1 = np.array([[0, np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
circle2 = np.array([[np.cos(theta), 0, np.sin(theta)] for theta in np.linspace(0, 2*np.pi, pointsPerCircle)])
# circle
#pointMatrix = np.array([[circle*r, circle1*r, circle2*r] for r in np.linspace(0.01, 2, numOfCircles)], dtype=np.float32).reshape(-1, 3)    
# sphere
sphere = np.array([circle*np.sin(theta) + [0,0,np.cos(theta)] for theta in np.linspace(0.001, np.pi, numOfCircles)], dtype=np.float32).reshape(-1, 3)
pointMatrix = np.array([sphere*r for r in np.linspace(.001, .01, numOfSpheres)], dtype=np.float32).reshape(-1, 3)   
#pointMatrix = np.array([circle, circle1, circle2], dtype=np.float32).reshape(-1, 3)
sampleSize = len(pointMatrix)
numOfCircles *= numOfSpheres
#numOfCircles *= 3
# for i in range(sampleSize):
#     pointMatrix[i*3+2] = (pointMatrix[i*3+2] + np.pi)/4
# pointMatrix = np.random.uniform(-1, 1, (sampleSize,3)) #inside sphere
# pointMatrix = (pointMatrix.T/np.linalg.norm(pointMatrix, axis = 1))
# randomRadius = np.random.uniform(.1, 2, sampleSize)
# pointMatrix = (pointMatrix * randomRadius).T.reshape(-1).astype(np.float32)

allPoints = np.zeros(sampleSize*3*nFrames).astype(np.float32)
#renderIndices = np.zeros(2*nFrames - 1).astype(np.int32)

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type = cl.device_type.GPU)
ctx = cl.Context(devices = my_gpu_devices )
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
point_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pointMatrix)
allpoint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, allPoints.nbytes)
centerPoint_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=centerPointHistory)
vHist_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vHistory)
#renderIndices_buf = cl.Buffer(ctx, mf.WRITE_ONLY, renderIndices.nbytes)

prg = cl.Program(ctx, open("updatePoints.c").read()).build()

prg.updatePointsTransversal(queue, (sampleSize,), None,
             np.int32(N), np.int32(nFrames), np.int32(sampleSize), point_buf, a_buf, b_buf,
             centerPoint_buf, vHist_buf, allpoint_buf, np.float32(allPointEps))

cl.enqueue_copy(queue, allPoints, allpoint_buf)

end = time.time()

print("done calculating points in", "{:.2f}".format(end-start), "seconds")

np.save("pointMatrixHistory", allPoints.reshape(nFrames,sampleSize,3))

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

prg.calcLineColors(queue, lineColors.shape, None, np.int32(sampleSize), colorMatrix_buf, lineColors_buf)

cl.enqueue_copy(queue, lineColors, lineColors_buf)

end = time.time()

print("done calculating lineColors in", "{:.2f}".format(end-start), "seconds")

np.save("lineColors", lineColors.reshape(-1, 3))

lineConnections = np.zeros(sampleSize * 2 * nFrames).astype(np.int32)

allpoint_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allPoints)
lineConnections_buf = cl.Buffer(ctx, mf.WRITE_ONLY, lineConnections.nbytes)

start = time.time()

prg.calcLineConnectionsCircles(queue, (sampleSize * nFrames,), None, np.int32(pointsPerCircle), lineConnections_buf, allpoint_buf)

cl.enqueue_copy(queue, lineConnections, lineConnections_buf)

end = time.time()

print("done calculating lineConnections in", "{:.2f}".format(end-start), "seconds")

np.save("lineConnections", lineConnections.reshape(-1, 2))