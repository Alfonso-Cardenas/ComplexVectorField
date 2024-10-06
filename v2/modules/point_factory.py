import numpy as np
from modules import constants


def randomly_sample_S3(point_amount: int):
    initial_sample = (
        np.random.uniform(-1, 1, (point_amount, 2)) + 1.j * np.random.uniform(-1, 1, (point_amount, 2))
    ).astype(np.csingle)
    return initial_sample / np.linalg.norm(initial_sample, axis=1)[:, None]


def randomly_sample_S3_fixed_latitude(point_amount: int, latitude: float):
    initial_sample = np.random.uniform(-1, 1, (point_amount, 3))

    initial_sample = initial_sample / np.linalg.norm(initial_sample, axis=1)[:, None]
    initial_sample = initial_sample * np.sqrt(1 - latitude * latitude)

    S3_sample = np.zeros((point_amount, 2), dtype=np.csingle)

    S3_sample.real = initial_sample[:, :2]
    S3_sample[:, 0].imag = initial_sample[:, 2]
    S3_sample[:, 1].imag = latitude

    return S3_sample


def uniformly_sample_S3(
    phi_amount: int,
    psi_amount: int,
    theta_amount: int,
    hemisphere: str = 'both'
):
    phis = np.linspace(0, constants.PI_TIMES_2, phi_amount)

    if hemisphere == 'both':
        psis = np.linspace(0, constants.PI_TIMES_2, psi_amount)
    elif hemisphere == 'north':
        psis = np.linspace(0, np.pi, psi_amount)
    elif hemisphere == 'south':
        psis = np.linspace(np.pi, constants.PI_TIMES_2, psi_amount)
    else:
        raise ValueError(f'hemisphere {hemisphere} is not a valid hemisphere option')

    thetas = np.linspace(0, constants.PI_OVER_2, theta_amount)

    xs = (np.sin(thetas)[:, None] @ np.exp(1.j * phis)[None,]).reshape(-1)
    ys = (np.exp(1.j * psis)[:, None] @ np.cos(thetas)[None,]).reshape(-1)

    sample_points = np.zeros((len(phis) * len(psis) * len(thetas), 2), dtype=np.csingle)
    sample_points[:,0] = np.repeat(xs[:,None], len(psis), axis=1).T.reshape(-1)
    sample_points[:,1] = np.repeat(ys, len(phis))

    return sample_points