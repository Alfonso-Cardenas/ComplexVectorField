import numpy as np
import numpy.typing as npt
from modules import constants

import time


def _print_coefficients_aux(N: int, c: npt.NDArray) -> None:
    idx = 0
    for i in range(N):
        print()
        for j in range(i+1):
            print(f'{c[idx]:.2f}', end=', ')
            idx += 1
    print()


def print_coefficients(N: int, a: npt.NDArray, b: npt.NDArray) -> None:
    print("z1'=", end='')
    _print_coefficients_aux(N, a)

    print("z2'=", end='')
    _print_coefficients_aux(N, b)


def sterographic_projection(
    points: npt.NDArray[np.csingle],
    projection_pole: str = 'north'
) -> npt.NDArray[np.float32]:
    projection = np.zeros(points.shape[:-1] + (3, ), dtype=np.float32)

    projection[..., (0, 2)] = points.real
    projection[..., 1] = points[..., 0].imag
    d = points[..., 1].imag

    if projection_pole == 'north':
        projection = projection / (1 - d[..., None])
    elif projection_pole == 'south':
        projection = projection / (1 + d[..., None])
    else:
        raise ValueError(f'Projection {projection_pole} is not a valid projection type')

    return projection


def inverse_stereographic_projection(  # ONLY NORTH ATM
    points: npt.NDArray,
) -> npt.NDArray[np.csingle]:
    projection = np.zeros(points.shape[:-1] + (2,), dtype=np.csingle)

    normsqrd = np.sum(points * points, axis=1)
    d = (normsqrd - 1)/(normsqrd + 1)
    projection[..., 1].imag = d
    projection.real = points[..., (0, 2)] * (1 - d)[..., None]
    projection[..., 0].imag = points[..., 1] * (1 - d)

    return projection


def find_orbit_heads(trajectories: npt.NDArray, orbit_tol: float):
    orbit_heads = [trajectories[0, 0]]
    orbit_colors = [constants.COLORS[0]]
    trajectory_colors = [constants.COLORS[0]]

    for i in range(1, trajectories.shape[1]):
        new_head = True

        for idx, head in enumerate(orbit_heads):
            dist = np.linalg.norm(trajectories[:, i] - head, axis=1)
            if np.any(dist < orbit_tol):
                new_head = False
                trajectory_colors.append(constants.COLORS[idx])
                break

        if new_head:
            orbit_colors.append(constants.COLORS[len(orbit_heads)])
            trajectory_colors.append(constants.COLORS[len(orbit_heads)])
            orbit_heads.append(trajectories[0, i])

    line_colors = np.tile(trajectory_colors, (trajectories.shape[0], 1))
    return (
        np.array(orbit_heads, dtype=np.csingle),
        line_colors.astype(np.float32),
    )


def timed_function(func):
    def inner(*args, **kwargs):
        start = time.time()

        returned_val = func(*args, **kwargs)

        end = time.time()
        print(f'Finished computing function "{func.__name__}" after {end-start:.2f}s')

        return returned_val
    return inner
