import numpy as np
import matplotlib.pyplot as plt
from modules import point_factory

b = np.array(
    [
        [0, 1, 2],
        [2, 3, 4],
    ]
)

c = np.array([1, 1, 1, -1, 0,])


a = np.array(
    [
        [2,2,2],
        [4,2,1],
        [0,0,0],
        [1,1,1],
        [1,0,1],
        [0,2,2],
        [3,3,3],
    ]
)
print(np.mean(a, axis=1))