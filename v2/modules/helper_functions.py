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


def calc_zs_powers(S3_sample, N):
    zs_powers = np.ones((len(S3_sample), (N * (N + 1)) // 2), dtype=np.csingle)
    idx = 1
    for power in range(1, N):
        last_power_idx = idx + power
        zs_powers[:, idx: last_power_idx] = zs_powers[:, idx - power: idx] * S3_sample[:, 0, None]
        zs_powers[:, last_power_idx] = zs_powers[:, idx - 1] * S3_sample[:, 1]
        idx = last_power_idx + 1
    return zs_powers


def calc_x(S3_sample, N, a, b):
    zs_powers = calc_zs_powers(S3_sample, N)
    x = np.zeros_like(S3_sample)
    x[:, 0] = np.sum(zs_powers * a, axis=1)
    x[:, 1] = np.sum(zs_powers * b, axis=1)
    return x


def calc_h(S3_sample, N, a, b):
    x = calc_x(S3_sample, N, a, b)
    h = 1.j * (x[:, 0].conjugate() * S3_sample[:, 0] + x[:, 1].conjugate() * S3_sample[:, 1])
    return h


def calc_dead_points(S3_sample, N, a, b, tol):
    h = calc_h(S3_sample, N, a, b)
    return S3_sample[np.where(np.abs(h) < tol)]


def timed_function(func):
    def inner(*args, **kwargs):
        start = time.time()

        returned_val = func(*args, **kwargs)

        end = time.time()
        print(f'Finished computing function "{func.__name__}" after {end-start:.2f}s')

        return returned_val
    return inner


def normalize_vector(v):
    v /= np.linalg.norm(v, axis=1)[:, None]

def calc_perp_vecs(x, radius):
    normalize_vector(x)
    v = np.zeros_like(x)
    sign = np.sign(x[:, 0])
    sign[np.where(sign == 0)] = 1
    v[:, 0] = - sign * x[:, 1] / x[:, 0]
    v[:, 1] = sign
    normalize_vector(v)
    w = np.cross(v, x, axis=1)
    v *= radius
    w *= radius
    return v, w

def calc_torus_points(orbit, points_per_step, radius):
    angles = np.linspace(0, constants.PI_TIMES_2, points_per_step)
    c = np.cos(angles)
    s = np.sin(angles)

    x = orbit[1:] - orbit[:-1]
    v, w = calc_perp_vecs(x, radius)

    circles = (v[:, None] * c[None, :, None] + w[:, None] * s[None, :, None])
    circles += orbit[:-1, None]

    return circles.reshape(-1, 3)

def calc_line_connections(trajectory_shape):
    line_connections = np.zeros((trajectory_shape[1] * (trajectory_shape[0] - 1), 2), dtype=np.int32)
    line_connections[:, 0] = np.arange(0, trajectory_shape[1] * (trajectory_shape[0] - 1))
    line_connections[:, 1] = np.arange(trajectory_shape[1], trajectory_shape[1] * trajectory_shape[0])

    return line_connections

def calc_torus_faces(step_amount, torus_points_per_step):
    return np.array(
        [
            [
                step * torus_points_per_step + idx,
                step * torus_points_per_step + (idx + 1) % torus_points_per_step,
                ((step + 1) % step_amount) * torus_points_per_step + idx,
            ]
            for step in range(step_amount)
            for idx in range(torus_points_per_step)
        ] + [
            [
                step * torus_points_per_step + (idx + 1) % torus_points_per_step,
                ((step + 1) % step_amount) * torus_points_per_step + idx,
                ((step + 1) % step_amount) * torus_points_per_step + (idx + 1) % torus_points_per_step,
            ]
            for step in range(step_amount)
            for idx in range(torus_points_per_step)
        ],
        dtype=np.int32
    )