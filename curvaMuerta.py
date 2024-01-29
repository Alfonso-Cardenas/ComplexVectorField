import numpy as np
from numpy.polynomial import polynomial as P
import time

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(199340)

N = 3
b = np.random.uniform(-1, 1, (N, N)) + 1.j*np.random.uniform(-1, 1, (N, N))
b[0,0] = 0

c = 1.j


def Xcoefs(x2):
    return np.array([sum([b[i,k]*pow(x2+c,k) for k in range(N-i) ]) for i in range(N)])

x2 = np.linspace(-100, 100, 1000)
coefMat = Xcoefs(x2).T

zplot0 = []
zplot1 = []
for coefs in coefMat:
    roots = P.polyroots(coefs)
    if(len(zplot0) != 0 and abs(roots[0] - zplot0[-1]) < abs(roots[1] - zplot0[-1])):
        zplot0.append(roots[0])
        zplot1.append(roots[1])
    else:
        zplot0.append(roots[1])
        zplot1.append(roots[0])

xplot0 = [z.real for z in zplot0]
yplot0 = [z.imag for z in zplot0]
xplot1 = [z.real for z in zplot1]
yplot1 = [z.imag for z in zplot1]

plt.plot(xplot0, yplot0, x2)
plt.plot(xplot1, yplot1, x2)
plt.show()